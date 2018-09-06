#--------------------------------
# Name:         ee_zonal_stats_by_zone_modis.py
# Purpose:      Download zonal stats by zone using Earth Engine
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
from collections import defaultdict
import datetime
from io import StringIO
import json
import logging
import math
import os
import pprint
import re
import requests
from subprocess import check_output
import sys

import ee
import numpy as np
from osgeo import ogr
import pandas as pd

# This is an awful way of getting the parent folder into the path
# We really should package this up as a module with a setup.py
# This way the ee_tools folders would be in the
#   PYTHONPATH env. variable
ee_tools_path = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.realpath(__file__))))
# if os.path.join(ee_tools_path, 'ee_tools') not in sys.path:
#     sys.path.insert(0, os.path.join(ee_tools_path, 'ee_tools'))
if ee_tools_path not in sys.path:
    sys.path.insert(0, ee_tools_path)

import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.modis
import ee_tools.utils as utils

pp = pprint.PrettyPrinter(indent=4)


def main(ini_path=None, overwrite_flag=False):
    """Earth Engine Zonal Stats Export

    Parameters
    ----------
    ini_path : str
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    """
    logging.info('\nEarth Engine zonal statistics by zone')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='EXPORT')
    inputs.parse_section(ini, section='ZONAL_STATS')

    # These may eventually be set in the INI file
    modis_fields = [
        'ZONE_NAME', 'ZONE_FID', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY']
    #     'AREA', 'PIXEL_SIZE', 'PIXEL_COUNT', 'PIXEL_TOTAL',
    #     'CLOUD_COUNT', 'CLOUD_TOTAL', 'CLOUD_PCT', 'QA']

    # Convert the shapefile to geojson
    # if not os.path.isfile(ini['ZONAL_STATS']['zone_geojson']):
    if os.path.isfile(ini['ZONAL_STATS']['zone_geojson']) or overwrite_flag:
        out_driver = ogr.GetDriverByName('GeoJSON')
        out_driver.DeleteDataSource(ini['ZONAL_STATS']['zone_geojson'])
    if not os.path.isfile(ini['ZONAL_STATS']['zone_geojson']):
        logging.info('\nConverting zone shapefile to GeoJSON')
        logging.debug('  {}'.format(ini['ZONAL_STATS']['zone_geojson']))
        check_output([
            'ogr2ogr', '-f', 'GeoJSON', '-preserve_fid',
            '-select', '{}'.format(ini['INPUTS']['zone_field']),
            # '-lco', 'COORDINATE_PRECISION=2'
            ini['ZONAL_STATS']['zone_geojson'],
            ini['INPUTS']['zone_shp_path']])

    # # Get ee features from shapefile
    # zone_geom_list = gdc.shapefile_2_geom_list_func(
    #     ini['INPUTS']['zone_shp_path'],
    #     zone_field=ini['INPUTS']['zone_field'],
    #     reverse_flag=False)
    # # zone_count = len(zone_geom_list)
    # # output_fmt = '_{0:0%sd}.csv' % str(int(math.log10(zone_count)) + 1)

    # Read in the zone geojson
    logging.debug('\nReading zone GeoJSON')
    try:
        with open(ini['ZONAL_STATS']['zone_geojson'], 'r') as f:
            zones_geojson = json.load(f)
    except Exception as e:
        logging.error('  Error reading zone geojson file, removing')
        logging.debug('  Exception: {}'.format(e))
        os.remove(ini['ZONAL_STATS']['zone_geojson'])

    # Check if the zone_names are unique
    # Eventually support merging common zone_names
    zone_names = [
        str(z['properties'][ini['INPUTS']['zone_field']]) \
            .replace(' ', '_').lower()
        for z in zones_geojson['features']]
    if len(set(zone_names)) != len(zones_geojson['features']):
        logging.error(
            '\nERROR: There appear to be duplicate zone ID/name values.'
            '\n  Currently, the values in "{}" must be unique.'
            '\n  Exiting.'.format(ini['INPUTS']['zone_field']))
        return False

    # # Check if the zone_names are unique
    # # Eventually support merging common zone_names
    # if len(set([z[1] for z in zone_geom_list])) != len(zone_geom_list):
    #     logging.error(
    #         '\nERROR: There appear to be duplicate zone ID/name values.'
    #         '\n  Currently, the values in "{}" must be unique.'
    #         '\n  Exiting.'.format(ini['INPUTS']['zone_field']))
    #     return False

    # Get projection from shapefile to build EE geometries
    # GeoJSON technically should always be EPSG:4326 so don't assume
    #  coordinates system property will be set
    zone_osr = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zone_wkt = gdc.osr_wkt(zone_osr)

    # Check that shapefile has matching spatial reference
    if not gdc.matching_spatref(zone_osr, ini['SPATIAL']['osr']):
        logging.warning('  Zone OSR:\n{}\n'.format(zone_osr))
        logging.warning('  Output OSR:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.warning('  Zone Proj4:   {}'.format(
            zone_osr.ExportToProj4()))
        logging.warning('  Output Proj4: {}'.format(
            ini['SPATIAL']['osr'].ExportToProj4()))
        logging.warning(
            '\nWARNING: \n'
            'The output and zone spatial references do not appear to match\n'
            'This will likely cause problems!')
        input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(zone_wkt))
        logging.debug('  Output Projection:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.debug('  Output Cellsize: {}'.format(
            ini['SPATIAL']['cellsize']))


    # Initialize Earth Engine API key
    logging.info('\nInitializing Earth Engine')
    ee.Initialize()
    utils.ee_request(ee.Number(1).getInfo())

    # Filter features by FID
    # Don't filter until after tile lists are built
    if ini['INPUTS']['fid_keep_list']:
        zones_geojson['features'] = [
            ftr for ftr in zones_geojson['features']
            if ftr['id'] in ini['INPUTS']['fid_keep_list']]
    if ini['INPUTS']['fid_skip_list']:
        zones_geojson['features'] = [
            ftr for ftr in zones_geojson['features']
            if ftr['id'] not in ini['INPUTS']['fid_skip_list']]

    # Merge geometries (after filtering by FID above)
    if ini['INPUTS']['merge_geom_flag']:
        logging.debug('\nMerging geometries')
        merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for zone_ftr in zones_geojson['features']:
            zone_multipolygon = ogr.ForceToMultiPolygon(
                ogr.CreateGeometryFromJson(json.dumps(zone_ftr['geometry'])))
            for zone_polygon in zone_multipolygon:
                merge_geom.AddGeometry(zone_polygon)
                zones_geojson['features'] = [{
            'type': 'Feature',
            'id': 0,
            'properties': {ini['INPUTS']['zone_field']: zones_geojson['name']},
            'geometry': json.loads(merge_geom.ExportToJson())}]


    # Calculate zonal stats for each feature separately
    logging.info('')
    # for zone_fid, zone_name, zone_json in zone_geom_list:
    #     zone['fid'] = zone_fid
    #     zone['name'] = zone_name.replace(' ', '_')
    #     zone['json'] = zone_json
    for zone_ftr in zones_geojson['features']:
        zone = {}
        zone['fid'] = zone_ftr['id']
        zone['name'] = str(zone_ftr['properties'][ini['INPUTS']['zone_field']]) \
            .replace(' ', '_')
        zone['json'] = zone_ftr['geometry']

        logging.info('ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))
        logging.debug('  Zone')

        # Build EE geometry object for zonal stats
        zone['geom'] = ee.Geometry(
            geo_json=zone['json'], opt_proj=zone_wkt, opt_geodesic=False)
        # logging.debug('  Centroid: {}'.format(
        #     zone['geom'].centroid(1).getInfo()['coordinates']))

        # Use feature geometry to build extent, transform, and shape
        zone_geom = ogr.CreateGeometryFromJson(json.dumps(zone['json']))
        if zone_geom.GetGeometryName() in ['POINT', 'MULTIPOINT']:
            # Compute area as cellsize * number of points
            point_count = 0
            for i in range(0, zone_geom.GetGeometryCount()):
                point_count += zone_geom.GetGeometryRef(i).GetPointCount()
            zone['area'] = ini['SPATIAL']['cellsize'] * point_count
        else:
            # Adjusting area up to nearest multiple of cellsize to account for
            #   polygons that were modified to avoid interior holes
            zone['area'] = ini['SPATIAL']['cellsize'] * math.ceil(
                zone_geom.GetArea() / ini['SPATIAL']['cellsize'])

        # zone['area'] = zone_geom.GetArea()
        zone['extent'] = gdc.Extent(zone_geom.GetEnvelope())
        # zone['extent'] = gdc.Extent(zone['geom'].GetEnvelope())
        zone['extent'] = zone['extent'].ogrenv_swap()
        zone['extent'] = zone['extent'].adjust_to_snap(
            'EXPAND', ini['SPATIAL']['snap_x'], ini['SPATIAL']['snap_y'],
            ini['SPATIAL']['cellsize'])
        zone['geo'] = zone['extent'].geo(ini['SPATIAL']['cellsize'])
        zone['transform'] = gdc.geo_2_ee_transform(zone['geo'])
        # zone['transform'] = '[' + ','.join(map(str, zone['transform'])) + ']'
        zone['shape'] = zone['extent'].shape(ini['SPATIAL']['cellsize'])
        logging.debug('    Zone Shape: {}'.format(zone['shape']))
        logging.debug('    Zone Transform: {}'.format(zone['transform']))
        logging.debug('    Zone Extent: {}'.format(zone['extent']))
        # logging.debug('    Zone Geom: {}'.format(zone['geom'].getInfo()))

        # Assume all pixels in all images could be reduced
        zone['max_pixels'] = zone['shape'][0] * zone['shape'][1]
        logging.debug('    Max Pixels: {}'.format(zone['max_pixels']))

        # Set output spatial reference
        # Eventually allow user to manually set these
        # output_crs = zone['proj']
        ini['EXPORT']['transform'] = zone['transform']
        logging.debug('    Output Projection: {}'.format(
            ini['SPATIAL']['crs']))
        logging.debug('    Output Transform: {}'.format(
            ini['EXPORT']['transform']))

        zone['output_ws'] = os.path.join(
            ini['ZONAL_STATS']['output_ws'], zone['name'])
        if not os.path.isdir(zone['output_ws']):
            os.makedirs(zone['output_ws'])

        if ini['ZONAL_STATS']['modis_daily_flag']:
            modis_daily_func(modis_fields, ini, zone, overwrite_flag)


def modis_daily_func(export_fields, ini, zone, overwrite_flag=False):
    """

    Function will attempt to generate export tasks only for missing DATES
    Also try to limit the products to only those with missing data

    Parameters
    ----------
    export_fields : list
    ini : dict
        Input file parameters.
    zone : dict
        Zone specific parameters.
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).
        Don't remove/replace the CSV file directly.

    """
    logging.info('  MODIS')

    # DEADBEEF - For now, hardcode transform to a standard MODIS image
    # ini['EXPORT']['transform'] = [
    #     231.656358264, 0.0, -20015109.354, 0.0, -231.656358264, 10007554.677]
    # ini['EXPORT']['transform'] = '[{}]'.format(','.join(
    #     map(str, (500.0, 0.0, 0.0, 0.0, -500.0, 0.0))))
    # logging.debug('    Output Transform: {}'.format(
    #     ini['EXPORT']['transform']))

    # Only keep daily MODIS products
    modis_products = [
        p for p in ini['ZONAL_STATS']['modis_products'][:]
        if p.split('_')[-1].upper() in ee_tools.modis.daily_collections]
    modis_fields = [f.upper() for f in modis_products]

    # Initialize the MODIS object
    args = {
        k: v for section in ['INPUTS']
        for k, v in ini[section].items()
        if k in [
            'cloud_flag',
            'start_year', 'end_year',
            'start_month', 'end_month',
            'start_doy', 'end_doy',
        ]}
    modis = ee_tools.modis.MODIS(args)

    # pprint.pprint(ee.Image(modis.get_daily_collection().first()).getInfo())
    # print(modis.get_daily_collection().aggregate_histogram('DATE').getInfo())
    # input('ENTER')

    def csv_writer(output_df, output_path, output_fields):
        """Write the dataframe to CSV with custom formatting"""
        csv_df = output_df.copy()

        # Convert float fields to objects, set NaN to None
        float_fields = modis_fields[:]
        # float_fields = modis_fields + ['CLOUD_PCT']
        for field in csv_df.columns.values:
            if field.upper() not in float_fields:
                continue
            csv_df[field] = csv_df[field].astype(object)
            null_mask = csv_df[field].isnull()
            csv_df.loc[null_mask, field] = None
            csv_df.loc[~null_mask, field] = csv_df.loc[~null_mask, field].map(
                lambda x: '{0:10.6f}'.format(x).strip())
            # csv_df.loc[~null_mask, [field]] = csv_df.loc[~null_mask, [field]].apply(
            #     lambda x: '{0:10.6f}'.format(x[0]).strip(), axis=1)

        # Set field types
        # Don't set the following since they may contain NaN/None?
        # 'QA', 'PIXEL_TOTAL', 'PIXEL_COUNT', 'CLOUD_TOTAL', 'CLOUD_COUNT']
        for field in ['ZONE_FID', 'YEAR', 'MONTH', 'DAY', 'DOY']:
            csv_df[field] = csv_df[field].astype(int)
        # if csv_df['ZONE_NAME'].dtype == np.float64:
        #     csv_df['ZONE_NAME'] = csv_df['ZONE_NAME'].astype(int).astype(str)

        # DEADBEEF
        # if csv_df['QA'].isnull().any():
        #     csv_df.loc[csv_df['QA'].isnull(), 'QA'] = 0
        # cloud_mask = csv_df['CLOUD_TOTAL'] > 0
        # if cloud_mask.any():
        #     csv_df.loc[cloud_mask, 'CLOUD_PCT'] = 100.0 * (
        #         csv_df.loc[cloud_mask, 'CLOUD_COUNT'] /
        #         csv_df.loc[cloud_mask, 'CLOUD_TOTAL'])

        csv_df.reset_index(drop=False, inplace=True)
        csv_df.sort_values(by=['DATE', 'SENSOR'], inplace=True)
        csv_df.to_csv(output_path, index=False, columns=output_fields)
    #
    # # Make copy of export field list in order to retain existing columns
    # output_fields = export_fields[:]

    # Read existing output table if possible
    # Otherwise build an empty dataframe
    # Only add new empty fields when reading in (don't remove any existing)
    output_id = output_id = '{}_modis_daily'.format(zone['name'])
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('    {}'.format(output_path))
    try:
        output_df = pd.read_csv(output_path, parse_dates=['DATE'])
        # output_df['ZONE_NAME'] = output_df['ZONE_NAME'].astype(str)

        # Move any existing columns not in export_fields to end of CSV
        output_fields.extend([
            f for f in output_df.columns.values if f not in export_fields])
        output_df = output_df.reindex(columns=output_fields)
        output_df.sort_values(by=['DATE', 'SCENE_ID'], inplace=True)
    except IOError:
        logging.debug(
            '    Output path doesn\'t exist, building empty dataframe')
        output_df = pd.DataFrame(columns=output_fields)
    except Exception as e:
        logging.exception('    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')

    # Use the DATE as the index
    output_df.set_index('DATE', inplace=True, drop=True)
    # output_df.index.name = 'DATE'

    # # For overwrite, drop all expected entries from existing output DF
    # if overwrite_flag:
    #     output_df = output_df[~output_df.index.isin(list(export_dates))]

    # # # # DEADBEEF - Reset zone area
    # # if not output_df.empty:
    # #     logging.info('    Updating zone area')
    # #     output_df.loc[
    # #         output_df.index.isin(list(export_ids)),
    # #         ['AREA']] = zone['area']
    # #     csv_writer(output_df, output_path, output_fields)
    #
    # # List of SCENE_IDs that are entirely missing
    # # This may include scenes that don't intersect the zone
    # missing_all_ids = export_ids - set(output_df.index.values)
    # # logging.info('  Dates missing all values: {}'.format(
    # #     ', '.join(sorted(missing_all_ids))))
    #
    # # If there are any fully missing scenes, identify whether they
    # #   intersect the zone or not
    # # Exclude non-intersecting SCENE_IDs from export_ids set
    # # Add non-intersecting SCENE_IDs directly to the output dataframe
    # if missing_all_ids:
    #     # Get SCENE_ID list mimicking a full extract below
    #     #   (but without products)
    #     # Start with INI path/row keep list but update based on SCENE_ID later
    #     modis.products = []
    #     modis.zone_geom = zone['geom']
    #
    #     # Get the SCENE_IDs that intersect the zone
    #     # Process each MODIS type independently
    #     # Was having issues getting the full scene list at once
    #     logging.debug('    Getting intersecting SCENE_IDs')
    #     missing_zone_ids = set()
    #     type_list = modis._modis_list[:]
    #     for type_str in type_list:
    #         modis._modis_list = [type_str]
    #         missing_zone_ids.update(set(utils.ee_getinfo(
    #             modis.get_collection().aggregate_histogram('SCENE_ID'))))
    #         logging.debug('      {} {}'.format(
    #             type_str, len(missing_zone_ids)))
    #     modis._modis_list = type_list
    #
    #     # Difference of sets are SCENE_IDs that don't intersect
    #     missing_skip_ids = missing_all_ids - missing_zone_ids
    #
    #     # Updating missing all SCENE_ID list to not include
    #     #   non-intersecting scenes
    #     missing_all_ids = set(missing_zone_ids)
    #
    #     # Remove skipped/empty SCENE_IDs from possible SCENE_ID list
    #     export_ids = export_ids - missing_skip_ids
    #     # logging.debug('  Missing Include: {}'.format(
    #     #     ', '.join(sorted(missing_zone_ids))))
    #     # logging.debug('  Missing Exclude: {}'.format(
    #     #     ', '.join(sorted(missing_skip_ids))))
    #     logging.info('    Include ID count: {}'.format(
    #         len(missing_zone_ids)))
    #     logging.info('    Exclude ID count: {}'.format(
    #         len(missing_skip_ids)))
    #
    #     if missing_skip_ids:
    #         logging.debug('    Appending empty non-intersecting SCENE_IDs')
    #         missing_df = pd.DataFrame(
    #             index=missing_skip_ids, columns=output_df.columns)
    #         missing_df.index.name = 'SCENE_ID'
    #         missing_df['ZONE_NAME'] = str(zone['name'])
    #         missing_df['ZONE_FID'] = zone['fid']
    #         missing_df['AREA'] = zone['area']
    #         missing_df['PLATFORM'] = missing_df.index.str.slice(0, 4)
    #         missing_df['DATE'] = pd.to_datetime(
    #             missing_df.index.str.slice(12, 20), format='%Y%m%d')
    #         missing_df['YEAR'] = missing_df['DATE'].dt.year
    #         missing_df['MONTH'] = missing_df['DATE'].dt.month
    #         missing_df['DAY'] = missing_df['DATE'].dt.day
    #         missing_df['DOY'] = missing_df['DATE'].dt.dayofyear.astype(int)
    #         missing_df['QA'] = np.nan
    #         # missing_df['QA'] = 0
    #         missing_df['PIXEL_SIZE'] = modis.cellsize
    #         missing_df['PIXEL_COUNT'] = 0
    #         missing_df['PIXEL_TOTAL'] = 0
    #         missing_df['CLOUD_COUNT'] = 0
    #         missing_df['CLOUD_TOTAL'] = 0
    #         missing_df['CLOUD_PCT'] = np.nan
    #         # missing_df[f] = missing_df[f].astype(int)
    #
    #         # Remove the overlapping missing entries
    #         # Then append the new missing entries
    #         if output_df.index.intersection(missing_df.index).any():
    #             output_df.drop(
    #                 output_df.index.intersection(missing_df.index),
    #                 inplace=True)
    #         output_df = output_df.append(missing_df, sort=False)
    #         csv_writer(output_df, output_path, output_fields)
    #
    # # Identify SCENE_IDs that are missing any data
    # # Filter based on product and SCENE_ID lists
    # # Check for missing data as long as PIXEL_COUNT > 0
    # missing_fields = modis_fields[:]
    # missing_id_mask = (
    #     (output_df['PIXEL_COUNT'] > 0) &
    #     output_df.index.isin(export_ids))
    # missing_df = output_df.loc[missing_id_mask, missing_fields].isnull()
    #
    # # List of SCENE_IDs and products with some missing data
    # missing_any_ids = set(missing_df[missing_df.any(axis=1)].index.values)
    #
    # # DEADBEEF - For now, skip SCENE_IDs that are only missing Ts
    # if not missing_df.empty and 'TS' in missing_fields:
    #     missing_ts_ids = set(missing_df[
    #         missing_df[['TS']].any(axis=1) &
    #         ~missing_df.drop('TS', axis=1).any(axis=1)].index.values)
    #     if missing_ts_ids:
    #         logging.info('  SCENE_IDs missing Ts only: {}'.format(
    #             ', '.join(sorted(missing_ts_ids))))
    #         missing_any_ids -= missing_ts_ids
    #         # input('ENTER')
    #
    # # logging.debug('  SCENE_IDs missing all values: {}'.format(
    # #     ', '.join(sorted(missing_all_ids))))
    # # logging.debug('  SCENE_IDs missing any values: {}'.format(
    # #     ', '.join(sorted(missing_any_ids))))
    #
    # # Check for fields that are entirely empty or not present
    # #   These may have been added but not filled
    # # Additional logic is to handle condition where
    # #   calling all on an empty dataframe returns True
    # if not missing_df.empty:
    #     missing_all_products = set(
    #         f.lower()
    #         for f in missing_df.columns[missing_df.all(axis=0)])
    #     missing_any_products = set(
    #         f.lower()
    #         for f in missing_df.columns[missing_df.any(axis=0)])
    # else:
    #     missing_all_products = set()
    #     missing_any_products = set()
    # if missing_all_products:
    #     logging.debug('    Products missing all values: {}'.format(
    #         ', '.join(sorted(missing_all_products))))
    # if missing_any_products:
    #     logging.debug('    Products missing any values: {}'.format(
    #         ', '.join(sorted(missing_any_products))))
    #
    # missing_ids = missing_all_ids | missing_any_ids
    # missing_products = missing_all_products | missing_any_products
    #
    # # If mosaic flag is set, switch IDs back to non-mosaiced
    # if ini['INPUTS']['mosaic_method'] in modis.mosaic_options:
    #     missing_scene_ids = [
    #         scene_id for mosaic_id in missing_ids
    #         for scene_id in mosaic_id_dict[mosaic_id]]
    # else:
    #     missing_scene_ids = set(missing_ids)
    # # logging.debug('  SCENE_IDs missing: {}'.format(
    # #     ', '.join(sorted(missing_scene_ids))))
    # logging.info('    Missing ID count: {}'.format(
    #     len(missing_scene_ids)))
    #
    # # Evaluate whether a subset of SCENE_IDs or products can be exported
    # # The SCENE_ID skip and keep lists cannot be mosaiced SCENE_IDs
    # if not missing_scene_ids and not missing_products:
    #     logging.info('    No missing data or products, skipping zone')
    #     return True
    # elif missing_scene_ids or missing_all_products:
    #     logging.info('    Exporting all products for specific SCENE_IDs')
    #     modis.scene_id_keep_list = sorted(list(missing_scene_ids))
    #     modis.products = modis_products[:]
    # elif missing_scene_ids and missing_products:
    #     logging.info('    Exporting specific missing products/SCENE_IDs')
    #     modis.scene_id_keep_list = sorted(list(missing_scene_ids))
    #     modis.products = list(missing_products)
    # elif not missing_scene_ids and missing_products:
    #     # This conditional will happen when images are missing Ts only
    #     # The SCENE_IDs are skipped but the missing products is not being
    #     #   updated also.
    #     logging.info(
    #         '    Missing products but no missing SCENE_IDs, skipping zone')
    #     return True
    # else:
    #     logging.error('    Unhandled conditional')
    #     input('ENTER')

    def export_update(data_df):
        """Set/modify ancillary field values in the export CSV dataframe"""
        # First remove any extra rows that were added for exporting
        data_df.drop(
            data_df[data_df['DATE'] == 'DEADBEEF'].index, inplace=True)

        # # With old Fmask data, PIXEL_COUNT can be > 0 even if all data is NaN
        # if ('NDVI_TOA' in data_df.columns.values and
        #         'TS' in data_df.columns.values):
        #     drop_mask = (
        #         data_df['NDVI_TOA'].isnull() & data_df['TS'].isnull() &
        #         (data_df['PIXEL_COUNT'] > 0))
        #     if not data_df.empty and drop_mask.any():
        #         data_df.loc[drop_mask, ['PIXEL_COUNT']] = 0

        # Add additional fields to the export data frame
        # data_df.set_index('DATE', inplace=True, drop=True)
        if not data_df.empty:
            # data_df['ZONE_NAME'] = data_df['ZONE_NAME'].astype(str)
            data_df['ZONE_FID'] = zone['fid']
            # data_df['DATE'] = pd.to_datetime(data_df.index.str, format='%Y%m%d')
            data_df['DATE'] = pd.to_datetime(data_df['DATE'].str, format='%Y%m%d')
            data_df['YEAR'] = data_df['DATE'].dt.year
            data_df['MONTH'] = data_df['DATE'].dt.month
            data_df['DAY'] = data_df['DATE'].dt.day
            data_df['DOY'] = data_df['DATE'].dt.dayofyear.astype(int)
            data_df['DATE'] = data_df['DATE'].dt.strftime('%Y-%m-%d')
            # data_df['AREA'] = zone['area']
            # data_df['PIXEL_SIZE'] = modis.cellsize

            # fmask_mask = data_df['CLOUD_TOTAL'] > 0
            # if fmask_mask.any():
            #     data_df.loc[fmask_mask, 'CLOUD_PCT'] = 100.0 * (
            #         data_df.loc[fmask_mask, 'CLOUD_COUNT'] /
            #         data_df.loc[fmask_mask, 'CLOUD_TOTAL'])
            # data_df['QA'] = 0
            # data_fields = [
            #     p.upper()
            #     for p in modis.products + ['CLOUD_PCT']]
            # data_df[data_fields] = data_df[data_fields].round(10)

        # Remove unused export fields
        if 'system:index' in data_df.columns.values:
            del data_df['system:index']
        if '.geo' in data_df.columns.values:
            del data_df['.geo']

        return data_df

    # Adjust start and end year to even multiples of year_step
    iter_start_year = ini['INPUTS']['start_year']
    iter_end_year = ini['INPUTS']['end_year'] + 1
    iter_years = ini['ZONAL_STATS']['year_step']
    if iter_years > 1:
        iter_start_year = int(math.floor(
            float(iter_start_year) / iter_years) * iter_years)
        iter_end_year = int(math.ceil(
            float(iter_end_year) / iter_years) * iter_years)

    # Process date range by year
    for year in range(iter_start_year, iter_end_year, iter_years):
        start_dt = datetime.datetime(year, 1, 1)
        end_dt = (
            datetime.datetime(year + iter_years, 1, 1) -
            datetime.timedelta(0, 1))
        start_date = start_dt.date().isoformat()
        end_date = end_dt.date().isoformat()
        start_year = max(start_dt.date().year, ini['INPUTS']['start_year'])
        end_year = min(end_dt.date().year, ini['INPUTS']['end_year'])

        # Filter by iteration date in addition to input date parameters
        modis.start_date = start_date
        modis.end_date = end_date

        for product in modis_products:
            logging.info('  Product: {}'.format(product.upper()))
            modis_coll = modis.get_daily_collection(product.upper())

            # DEBUG - Test that the MODIS collection is getting built
            # print(modis_coll.aggregate_histogram('DATE').getInfo())
            # # input('ENTER')
            # print('Bands: {}'.format(
            #     [x['id'] for x in ee.Image(modis_coll.first()).getInfo()['bands']]))
            # print('Date: {}'.format(
            #     ee.Image(modis_coll.first()).getInfo()['properties']['DATE']))
            # input('ENTER')
            # if ee.Image(modis_coll.first()).getInfo() is None:
            #     logging.info('    No images, skipping')
            #     continue

            # Calculate values and statistics
            # Build function in loop to set water year ETo/PPT values
            def zonal_stats_func(image):
                """"""
                date = ee.Date(image.get('system:time_start'))
                bands = 1

                # Using zone['geom'] as the geometry should make it
                #   unnecessary to clip also
                input_mean = ee.Image(image) \
                    .select([product.upper()]) \
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=zone['geom'],
                        crs=ini['SPATIAL']['crs'],
                        crsTransform=ini['EXPORT']['transform'],
                        bestEffort=False,
                        tileScale=1,
                        maxPixels=zone['max_pixels'] * bands)

                # # Count unmasked Fmask pixels to get pixel count
                # # Count Fmask > 1 to get Fmask count (0 is clear and 1 is water)
                # cloud_img = ee.Image(image).select(['cloud'])
                # input_count = ee.Image([
                #         cloud_img.gte(0).unmask().rename(['pixel']),
                #         cloud_img.gt(1).rename(['fmask'])]) \
                #     .reduceRegion(
                #         reducer=ee.Reducer.sum().combine(
                #             ee.Reducer.count(), '', True),
                #         geometry=zone['geom'],
                #         crs=ini['SPATIAL']['crs'],
                #         crsTransform=ini['EXPORT']['transform'],
                #         bestEffort=False,
                #         tileScale=1,
                #         maxPixels=zone['max_pixels'] * 3)

                # Standard output
                zs_dict = {
                    'ZONE_NAME': str(zone['name']),
                    # 'ZONE_FID': zone['fid'],
                    'DATE': date.format('YYYY-MM-dd'),
                    # 'AREA': zone['area'],
                    # 'PIXEL_SIZE': modis.cellsize,
                    # 'PIXEL_COUNT': input_count.get('pixel_sum'),
                    # 'PIXEL_TOTAL': input_count.get('pixel_count'),
                    # 'CLOUD_COUNT': input_count.get('cloud_sum'),
                    # 'CLOUD_TOTAL': input_count.get('cloud_count'),
                    # 'CLOUD_PCT': ee.Number(input_count.get('cloud_sum')) \
                    #     .divide(ee.Number(input_count.get('cloud_count'))) \
                    #     .multiply(100),
                    # 'QA': ee.Number(0),
                    product.upper(): input_mean.get(product.upper()),
                }

                return ee.Feature(None, zs_dict)

            stats_coll = modis_coll.map(zonal_stats_func, False)

            # # DEBUG - Test the function for a single image
            # stats_info = zonal_stats_func(
            #     ee.Image(modis_coll.first())).getInfo()
            # pp.pprint(stats_info['properties'])
            # input('ENTER')

            # # DEBUG - Print the stats info to the screen
            # stats_info = stats_coll.getInfo()
            # for ftr in stats_info['features']:
            #     pp.pprint(ftr)
            # input('ENTER')
            # # return False

            # Add a dummy entry to the stats collection
            format_dict = {
                'ZONE_NAME': 'DEADBEEF',
                # 'ZONE_FID': -9999,
                'DATE': 'DEADBEEF',
                product.upper(): 'DEADBEEF',
            }
            stats_coll = ee.FeatureCollection(ee.Feature(None, format_dict)) \
                .merge(stats_coll)

            # # DEBUG - Print the stats info to the screen
            # stats_info = stats_coll.getInfo()
            # for ftr in stats_info['features']:
            #     pp.pprint(ftr)
            # input('ENTER')

            if ini['EXPORT']['export_dest'] == 'getinfo':
                logging.debug('    Requesting data')
                export_info = utils.ee_getinfo(stats_coll)['features']
                export_df = pd.DataFrame([f['properties'] for f in export_info])
                print(export_df)
                input('ENTER')
                export_df = export_update(export_df)

                # Save data to main dataframe
                if not export_df.empty:
                    logging.debug('    Processing data')
                    if overwrite_flag:
                        # Update happens inplace automatically
                        # output_df.update(export_df)
                        output_df = output_df.append(export_df, sort=False)
                    else:
                        # Combine first doesn't have an inplace parameter
                        output_df = output_df.combine_first(export_df)

    # # Save updated CSV
    # if output_df is not None and not output_df.empty:
    #     logging.info('    Writing CSV')
    #     csv_writer(output_df, output_path, output_fields)
    # else:
    #     logging.info(
    #         '  Empty output dataframe\n'
    #         '  The exported CSV files may not be ready')


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine zonal statistics by zone',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=utils.arg_valid_file,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    args = parser.parse_args()

    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    else:
        args.ini = utils.get_ini_path(os.getcwd())
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, overwrite_flag=args.overwrite)
