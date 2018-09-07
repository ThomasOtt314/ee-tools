#--------------------------------
# Name:         ee_zonal_stats_by_zone_modis.py
# Purpose:      Download zonal stats by zone using Earth Engine
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
# from collections import defaultdict
import datetime
# from io import StringIO
import json
import logging
import math
import os
import pprint
# import re
# import requests
from subprocess import check_output
import sys

import ee
# import numpy as np
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
    modis_daily_fields = [
        'ZONE_NAME', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY']
    #     'AREA', 'PIXEL_SIZE', 'PIXEL_COUNT', 'PIXEL_TOTAL',
    #     'CLOUD_COUNT', 'CLOUD_TOTAL', 'CLOUD_PCT', 'QA']

    modis_daily_fields.extend(
        p.upper()
        for p in ini['ZONAL_STATS']['modis_products']
        if p.split('_')[-1].upper() in ee_tools.modis.daily_collections)

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
            modis_daily_func(modis_daily_fields, ini, zone, overwrite_flag)


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
    modis_fields = [p.upper() for p in modis_products]

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
        for field in ['YEAR', 'MONTH', 'DAY', 'DOY']:
            csv_df[field] = csv_df[field].astype(int)
        # if csv_df['ZONE_NAME'].dtype == np.float64:
        #     csv_df['ZONE_NAME'] = csv_df['ZONE_NAME'].astype(int).astype(str)

        # csv_df.reset_index(drop=False, inplace=True)
        csv_df.sort_values(by=['DATE'], inplace=True)
        csv_df.to_csv(output_path, index=False, columns=output_fields)

    # Make copy of export field list in order to retain existing columns
    output_fields = export_fields[:]

    # Read existing output table if possible
    # Otherwise build an empty dataframe
    # Only add new empty fields when reading in (don't remove any existing)
    output_id = output_id = '{}_modis_daily'.format(zone['name'])
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('    {}'.format(output_path))
    try:
        output_df = pd.read_csv(output_path, parse_dates=['DATE'])
        output_df['DATE'] = output_df['DATE'].dt.strftime('%Y-%m-%d')
        output_df['ZONE_NAME'] = output_df['ZONE_NAME'].astype(str)

        # Move any existing columns not in export_fields to end of CSV
        output_fields.extend([
            f for f in output_df.columns.values if f not in export_fields])
        output_df = output_df.reindex(columns=output_fields)
        output_df.sort_values(by=['DATE'], inplace=True)
    except IOError:
        logging.debug(
            '    Output path doesn\'t exist, building empty dataframe')
        output_df = pd.DataFrame(columns=output_fields)
    except Exception as e:
        logging.exception('    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')

    # # Use the DATE as the index
    # output_df.set_index('DATE', inplace=True, drop=True)
    # output_df.index.name = 'DATE'
    # output_df['YEAR'] = pd.to_datetime(output_df.index).year
    # output_df['MONTH'] = pd.to_datetime(output_df.index).month
    # output_df['DAY'] = pd.to_datetime(output_df.index).day
    # output_df['DOY'] = pd.to_datetime(output_df.index).dayofyear.astype(int)

    # Don't use DATE as the index
    output_df['DATE'] = pd.to_datetime(output_df['DATE'], format='%Y-%m-%d')
    output_df['YEAR'] = output_df['DATE'].dt.year
    output_df['MONTH'] = output_df['DATE'].dt.month
    output_df['DAY'] = output_df['DATE'].dt.day
    output_df['DOY'] = output_df['DATE'].dt.dayofyear.astype(int)
    output_df['DATE'] = output_df['DATE'].dt.strftime('%Y-%m-%d')

    # Get list of possible dates based on INI
    export_dates = set(
        date_str for date_str in utils.date_range(
            '{}-01-01'.format(ini['INPUTS']['start_year']),
            '{}-12-31'.format(ini['INPUTS']['end_year']))
        if date_str < datetime.datetime.today().strftime('%Y-%m-%d'))
    # logging.debug('  Export Dates: {}'.format(
    #     ', '.join(sorted(export_dates))))

    # For overwrite, drop all expected entries from existing output DF
    if overwrite_flag:
        # output_df = output_df[~output_df.index.isin(list(export_dates))]
        output_df = output_df[~output_df['DATE'].isin(list(export_dates))]

    # # # # DEADBEEF - Reset zone area
    # # if not output_df.empty:
    # #     logging.info('    Updating zone area')
    # #     output_df.loc[
    # #         output_df.index.isin(list(export_dates)),
    # #         ['AREA']] = zone['area']
    # #     csv_writer(output_df, output_path, output_fields)

    # List of DATES that are entirely missing
    # This may include scenes that don't intersect the zone
    missing_all_dates = export_dates - set(output_df['DATE'].values)
    logging.info('  DATES missing all values: {}'.format(
        ', '.join(sorted(missing_all_dates))))

    # if not missing_all_dates:
    #     logging.info('  No missing DATES')
    #     return True

    if missing_all_dates:
        logging.debug('    Appending empty DATES')
        # # Use DATE as index
        # missing_df = pd.DataFrame(index=missing_all_dates,
        #                           columns=output_df.columns)
        # missing_df['ZONE_NAME'] = zone['name']
        # missing_df.index.name = 'DATE'
        # missing_df.sort_index(inplace=True)
        # missing_df['YEAR'] = pd.to_datetime(missing_df.index).year
        # missing_df['MONTH'] = pd.to_datetime(missing_df.index).month
        # missing_df['DAY'] = pd.to_datetime(missing_df.index).day
        # missing_df['DOY'] = pd.to_datetime(missing_df.index).dayofyear.astype(int)

        # Don't set DATE as index
        missing_df = pd.DataFrame(index=range(len(missing_all_dates)),
                                  columns=output_df.columns)
        missing_df['DATE'] = sorted(missing_all_dates)
        missing_df['ZONE_NAME'] = zone['name']
        missing_df['DATE'] = pd.to_datetime(missing_df['DATE'], format='%Y-%m-%d')
        missing_df['YEAR'] = missing_df['DATE'].dt.year
        missing_df['MONTH'] = missing_df['DATE'].dt.month
        missing_df['DAY'] = missing_df['DATE'].dt.day
        missing_df['DOY'] = missing_df['DATE'].dt.dayofyear.astype(int)
        missing_df['DATE'] = missing_df['DATE'].dt.strftime('%Y-%m-%d')

        # missing_df['AREA'] = zone['area']
        # missing_df['QA'] = np.nan
        # missing_df['QA'] = 0
        # missing_df['PIXEL_SIZE'] = modis.cellsize
        # missing_df['PIXEL_COUNT'] = 0
        # missing_df['PIXEL_TOTAL'] = 0
        # missing_df['CLOUD_COUNT'] = 0
        # missing_df['CLOUD_TOTAL'] = 0
        # missing_df['CLOUD_PCT'] = np.nan
        # missing_df[f] = missing_df[f].astype(int)

        # Remove the overlapping missing entries
        # Then append the new missing entries
        # Drop the indices to the intersection can be computed
        print(output_df.head())
        print(missing_df.head())
        if (not output_df.empty and
                output_df['DATE'].intersection(missing_df['DATE']).any()):
            output_df.drop(
                output_df['DATE'].intersection(missing_df['DATE']),
                inplace=True)
        output_df = output_df.append(missing_df, sort=False)

        logging.debug('    Writing to CSV')
        print(output_df.head())
        csv_writer(output_df, output_path, output_fields)

        del missing_df

    # # Identify DATES that are missing any data
    # # Filter based on product and DATES lists
    # # Check for missing data as long as PIXEL_COUNT > 0
    # missing_fields = modis_fields[:]
    # missing_date_mask = (
    #     (output_df['PIXEL_COUNT'] > 0) &
    #     output_df.index.isin(export_dates))
    # missing_df = output_df.loc[missing_date_mask, missing_fields].isnull()
    #
    # # List of DATES and products with some missing data
    # missing_any_dates = set(missing_df[missing_df.any(axis=1)].index.values)
    #
    # # logging.debug('  DATES missing all values: {}'.format(
    # #     ', '.join(sorted(missing_all_dates))))
    # # logging.debug('  DATES missing any values: {}'.format(
    # #     ', '.join(sorted(missing_any_dates))))
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
    # missing_dates = missing_all_dates | missing_any_dates
    # missing_products = missing_all_products | missing_any_products
    #
    # # If mosaic flag is set, switch DATES back to non-mosaiced
    # missing_scene_dates = set(missing_dates)
    # # logging.debug('  DATES missing: {}'.format(
    # #     ', '.join(sorted(missing_scene_dates))))
    # logging.info('    Missing DATE count: {}'.format(
    #     len(missing_scene_dates)))
    #
    # # Evaluate whether a subset of SCENE_IDs or products can be exported
    # # The SCENE_ID skip and keep lists cannot be mosaiced SCENE_IDs
    # if not missing_scene_ids and not missing_products:
    #     logging.info('    No missing data or products, skipping zone')
    #     return True
    # elif missing_scene_dates or missing_all_products:
    #     logging.info('    Exporting all products for specific DATES')
    #     modis.products = modis_products[:]
    # elif missing_scene_dates and missing_products:
    #     logging.info('    Exporting specific missing products/DATES')
    #     modis.products = list(missing_products)
    # elif not missing_scene_ids and missing_products:
    #     # This conditional will happen when images are missing Ts only
    #     # The DATES are skipped but the missing products is not being
    #     #   updated also.
    #     logging.info(
    #         '    Missing products but no missing DATES, skipping zone')
    #     return True
    # else:
    #     logging.error('    Unhandled conditional')
    #     input('ENTER')

    def clean_export_df(data_df):
        """Set/modify ancillary field values in the export CSV dataframe"""
        # First remove any extra rows that were added for exporting
        data_df.drop(data_df[data_df['DATE'] == 'DEADBEEF'].index,
                     inplace=True)

        # Add additional fields to the export data frame
        # data_df.set_index('DATE', inplace=True, drop=True)
        if not data_df.empty:
            # Otherwise if DATE is not the index:
            data_df['DATE'] = pd.to_datetime(data_df['DATE'], format='%Y-%m-%d')
            data_df['YEAR'] = data_df['DATE'].dt.year
            data_df['MONTH'] = data_df['DATE'].dt.month
            data_df['DAY'] = data_df['DATE'].dt.day
            data_df['DOY'] = data_df['DATE'].dt.dayofyear.astype(int)
            data_df['DATE'] = data_df['DATE'].dt.strftime('%Y-%m-%d')

            # data_df['ZONE_NAME'] = data_df['ZONE_NAME'].astype(str)
            # data_df['AREA'] = zone['area']
            # data_df['PIXEL_SIZE'] = modis.cellsize

            # DEADBEEF
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

        # data_df.set_index('DATE', inplace=True, drop=True)

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
        # start_year = max(start_dt.date().year, ini['INPUTS']['start_year'])
        # end_year = min(end_dt.date().year, ini['INPUTS']['end_year'])

        # Filter by iteration date in addition to input date parameters
        modis.start_date = start_date
        modis.end_date = end_date
        # modis.start_year = start_year
        # modis.end_year = end_year

        for product in modis_products:
            logging.info('    Product: {}'.format(product.upper()))
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

            logging.debug('    Requesting data')
            export_info = utils.ee_getinfo(stats_coll)['features']
            export_df = pd.DataFrame([f['properties'] for f in export_info])
            export_df = clean_export_df(export_df)

            # Save data to main dataframe
            # Combine first needs DATE as the index to work
            if not export_df.empty:
                logging.debug('    Processing data')
                export_df.set_index(['DATE'], inplace=True, drop=True)
                output_df.set_index(['DATE'], inplace=True, drop=True)
                output_df = output_df.combine_first(export_df)
                # output_df = output_df.update(export_df)
                output_df.reset_index(inplace=True, drop=False)

                # if overwrite_flag:
                #     # Update happens inplace automatically
                #     # output_df.update(export_df)
                #     output_df = output_df.append(export_df, sort=False)
                # else:
                #     # Combine first doesn't have an inplace parameter
                #     output_df = output_df.combine_first(export_df)

    # Save updated CSV
    if output_df is not None and not output_df.empty:
        logging.info('    Writing CSV')
        csv_writer(output_df, output_path, output_fields)
    else:
        logging.info(
            '  Empty output dataframe\n'
            '  The exported CSV files may not be ready')


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
