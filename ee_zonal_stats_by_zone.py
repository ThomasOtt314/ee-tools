#--------------------------------
# Name:         ee_zonal_stats_by_zone.py
# Purpose:      Download zonal stats by zone using Earth Engine
# Modified:     2017-07-06
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

import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils

pp = pprint.PrettyPrinter(indent=4)


def main(ini_path=None, overwrite_flag=False):
    """Earth Engine Zonal Stats Export

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nEarth Engine zonal statistics by zone')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='EXPORT')
    inputs.parse_section(ini, section='ZONAL_STATS')

    # Zonal stats init file paths
    zone_geojson = os.path.join(
        ini['ZONAL_STATS']['output_ws'],
        os.path.basename(ini['INPUTS']['zone_shp_path']).replace(
            '.shp', '.geojson'))

    # These may eventually be set in the INI file
    landsat_daily_fields = [
        'ZONE_NAME', 'ZONE_FID', 'DATE', 'SCENE_ID', 'PLATFORM',
        'PATH', 'ROW', 'YEAR', 'MONTH', 'DAY', 'DOY',
        'AREA', 'PIXEL_SIZE', 'PIXEL_COUNT', 'PIXEL_TOTAL',
        'FMASK_COUNT', 'FMASK_TOTAL', 'FMASK_PCT', 'CLOUD_SCORE', 'QA']
    gridmet_daily_fields = [
        'ZONE_NAME', 'ZONE_FID', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY',
        'WATER_YEAR', 'ETO', 'PPT']
    gridmet_monthly_fields = [
        'ZONE_NAME', 'ZONE_FID', 'DATE', 'YEAR', 'MONTH', 'WATER_YEAR',
        'ETO', 'PPT']
    pdsi_dekad_fields = [
        'ZONE_NAME', 'ZONE_FID', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY',
        'PDSI']

    # Concert REFL_TOA, REFL_SUR, and TASSELED_CAP products to bands
    if 'refl_toa' in ini['ZONAL_STATS']['landsat_products']:
        ini['ZONAL_STATS']['landsat_products'].extend([
            'blue_toa', 'green_toa', 'red_toa',
            'nir_toa', 'swir1_toa', 'swir2_toa'])
        ini['ZONAL_STATS']['landsat_products'].remove('refl_toa')
    if 'refl_sur' in ini['ZONAL_STATS']['landsat_products']:
        ini['ZONAL_STATS']['landsat_products'].extend([
            'blue_sur', 'green_sur', 'red_sur',
            'nir_sur', 'swir1_sur', 'swir2_sur'])
        ini['ZONAL_STATS']['landsat_products'].remove('refl_sur')
    if 'tasseled_cap' in ini['ZONAL_STATS']['landsat_products']:
        ini['ZONAL_STATS']['landsat_products'].extend([
            'tc_green', 'tc_bright', 'tc_wet'])
        ini['ZONAL_STATS']['landsat_products'].remove('tasseled_cap')
    landsat_daily_fields.extend(
        [p.upper() for p in ini['ZONAL_STATS']['landsat_products']])

    # Convert the shapefile to geojson
    if not os.path.isfile(zone_geojson) or overwrite_flag:
        logging.info('\nConverting zone shapefile to GeoJSON')
        logging.debug('  {}'.format(zone_geojson))
        check_output([
            'ogr2ogr', '-f', 'GeoJSON', '-preserve_fid',
            '-select', '{}'.format(ini['INPUTS']['zone_field']),
            # '-lco', 'COORDINATE_PRECISION=2'
            zone_geojson, ini['INPUTS']['zone_shp_path']])

    # # Get ee features from shapefile
    # zone_geom_list = gdc.shapefile_2_geom_list_func(
    #     ini['INPUTS']['zone_shp_path'], zone_field=ini['INPUTS']['zone_field'],
    #     reverse_flag=False)
    # # zone_count = len(zone_geom_list)
    # # output_fmt = '_{0:0%sd}.csv' % str(int(math.log10(zone_count)) + 1)

    # Read in the zone geojson
    logging.debug('\nReading zone GeoJSON')
    try:
        with open(zone_geojson, 'r') as f:
            zones = json.load(f)
    except Exception as e:
        logging.error('  Error reading zone geojson file, removing')
        logging.debug('  Exception: {}'.format(e))
        os.remove(zone_geojson)

    # Check if the zone_names are unique
    # Eventually support merging common zone_names
    zone_names = [
        str(z['properties'][ini['INPUTS']['zone_field']]).replace(' ', '_').lower()
        for z in zones['features']]
    if len(set(zone_names)) != len(zones['features']):
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
    utils.request(ee.Number(1).getInfo())

    # Get current running tasks before getting file lists
    tasks = utils.get_ee_tasks()

    # Build separate tile lists for each zone
    # Build tile lists before filtering by FID below
    # DEADBEEF - This is a list of all "possible" tile that is
    #   independent of the INI tile settings
    ini['ZONAL_STATS']['zone_tile_json'] = {}
    ini['ZONAL_STATS']['tile_scene_json'] = {}
    if (os.path.isfile(ini['ZONAL_STATS']['zone_tile_path']) and
            not overwrite_flag):
        logging.debug('\nReading zone tile lists\n  {}'.format(
            ini['ZONAL_STATS']['zone_tile_path']))
        with open(ini['ZONAL_STATS']['zone_tile_path'], 'r') as f:
            ini['ZONAL_STATS']['zone_tile_json'] = json.load(f)
    else:
        logging.info('\nBuilding zone tile lists')
        step = 1000
        zone_n = len(zones['features'])
        for zone_i in range(0, len(zones['features']), step):
            logging.debug('  Zones: {}-{}'.format(
                zone_i, min(zone_i + step, zone_n) - 1))
            zone_ftr_sub = zones['features'][zone_i: min(zone_i + step, zone_n)]

            # Build the zones feature collection in a list comprehension
            #   in order to set the correct spatial reference
            zone_field = ini['INPUTS']['zone_field']
            zone_coll = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry(f['geometry'], zone_wkt, False),
                    {zone_field: f['properties'][zone_field]})
                for f in zone_ftr_sub])

            # Load the WRS2 custom footprint collection
            # Ingested shapefiles have lower case field names
            tile_field = 'wrs2_tile'
            wrs2_coll = ee.FeatureCollection(
                    'users/cgmorton/wrs2_descending_conus_custom') \
                .filterBounds(zone_coll.geometry())

            # Extract tile values from joined collection
            def ftr_property(ftr):
                scenes = ee.FeatureCollection(ee.List(ee.Feature(ftr).get('scenes'))) \
                    .toList(100).map(lambda tile: ee.Feature(tile).get(tile_field))
                return ee.Feature(None, {
                    zone_field: ee.String(ftr.get(zone_field)),
                    tile_field: scenes})

            # Intersect the geometry and wrs2 collections
            spatialFilter = ee.Filter.intersects(
                leftField='.geo', rightField='.geo', maxError=10)
            join_coll = ee.FeatureCollection(
                ee.Join.saveAll(matchesKey='scenes') \
                    .apply(zone_coll, wrs2_coll, spatialFilter) \
                    .map(ftr_property))

            # Build a list of tiles for each zone
            for f in utils.getinfo(join_coll)['features']:
                zone_name = str(f['properties'][ini['INPUTS']['zone_field']]) \
                    .replace(' ', '_')
                ini['ZONAL_STATS']['zone_tile_json'][zone_name] = sorted(list(set(
                    f['properties'][tile_field])))

        logging.debug('  Saving zone tile dictionary')
        logging.debug('    {}'.format(ini['ZONAL_STATS']['zone_tile_path']))
        with open(ini['ZONAL_STATS']['zone_tile_path'], 'w') as f:
            json.dump(ini['ZONAL_STATS']['zone_tile_json'], f, sort_keys=True)

    # Filter features by FID
    # Don't filter until after tile lists are built
    if ini['INPUTS']['fid_keep_list']:
        zones['features'] = [
            ftr for ftr in zones['features']
            if ftr['id'] in ini['INPUTS']['fid_keep_list']]
    if ini['INPUTS']['fid_skip_list']:
        zones['features'] = [
            ftr for ftr in zones['features']
            if ftr['id'] not in ini['INPUTS']['fid_skip_list']]

    # Merge geometries
    # Merge after filtering by FID
    if ini['INPUTS']['merge_geom_flag']:
        merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for zone_ftr in zones['features']:
            zone_multipolygon = ogr.ForceToMultiPolygon(
                ogr.CreateGeometryFromJson(json.dumps(zone_ftr['geometry'])))
            for zone_polygon in zone_multipolygon:
                merge_geom.AddGeometry(zone_polygon)
        zones['features'] = [{
            'type': 'Feature',
            'id': 0,
            'properties': {ini['INPUTS']['zone_field']: zones['name']},
            'geometry': json.loads(merge_geom.ExportToJson())}]

    # Get list of existing images/files
    if ini['EXPORT']['export_dest'] == 'cloud':
        logging.debug('\nGetting cloud storage file list')
        ini['EXPORT']['cloud_file_list'] = utils.get_bucket_files(
            ini['EXPORT']['project_name'], ini['EXPORT']['export_ws'])
        # logging.debug(ini['EXPORT']['cloud_file_list'])
    # if ini['EXPORT']['export_dest'] == 'gdrive':
    #     logging.debug('\nGetting Google drive file list')
    #     ini['EXPORT']['gdrive_file_list'] = [
    #         os.path.join(ini['ZONAL_STATS']['output_ws'], x)
    #         for x in os.listdir(ini['ZONAL_STATS']['output_ws'])]
    #     logging.debug(ini['EXPORT']['gdrive_file_list'])

    # Get end date of GRIDMET (if needed)
    # This could be moved to inside the INI function
    if ini['ZONAL_STATS']['gridmet_monthly_flag']:
        gridmet_end_dt = utils.request(ee.Date(ee.Image(
            ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .filterDate(
                    '{}-01-01'.format(ini['INPUTS']['end_year'] - 1),
                    '{}-01-01'.format(ini['INPUTS']['end_year'] + 1)) \
                .limit(1, 'system:time_start', False) \
                .first()
            ).get('system:time_start')).format('YYYY-MM-dd')).getInfo()
        gridmet_end_dt = datetime.datetime.strptime(
            gridmet_end_dt, '%Y-%m-%d')
        logging.debug('    Last GRIDMET date: {}'.format(gridmet_end_dt))


    # Calculate zonal stats for each feature separately
    logging.info('')
    # for zone_fid, zone_name, zone_json in zone_geom_list:
    #     zone['fid'] = zone_fid
    #     zone['name'] = zone_name.replace(' ', '_')
    #     zone['json'] = zone_json
    for zone_ftr in zones['features']:
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
        #     zone['geom'].centroid(100).getInfo()['coordinates']))

        # Use feature geometry to build extent, transform, and shape
        zone_poly = ogr.CreateGeometryFromJson(json.dumps(zone['json']))
        zone['area'] = zone_poly.GetArea()
        zone['extent'] = gdc.Extent(zone_poly.GetEnvelope())
        # zone['extent'] = gdc.Extent(zone['geom'].GetEnvelope())
        zone['extent'] = zone['extent'].ogrenv_swap()
        zone['extent'] = zone['extent'].adjust_to_snap(
            'EXPAND', ini['SPATIAL']['snap_x'], ini['SPATIAL']['snap_y'],
            ini['SPATIAL']['cellsize'])
        zone['geo'] = zone['extent'].geo(ini['SPATIAL']['cellsize'])
        zone['transform'] = gdc.geo_2_ee_transform(zone['geo'])
        zone['shape'] = zone['extent'].shape(ini['SPATIAL']['cellsize'])
        logging.debug('    Zone Shape: {}'.format(zone['shape']))
        logging.debug('    Zone Transform: {}'.format(zone['transform']))
        logging.debug('    Zone Extent: {}'.format(zone['extent']))
        # logging.debug('  Zone Geom: {}'.format(zone['geom'].getInfo()))

        # Assume all pixels in all 14+2 images could be reduced
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

        if ini['ZONAL_STATS']['landsat_flag']:
            landsat_func(
                landsat_daily_fields, ini, zone, tasks, overwrite_flag)
        if ini['ZONAL_STATS']['gridmet_daily_flag']:
            gridmet_daily_func(
                gridmet_daily_fields, ini, zone, tasks, overwrite_flag)
        if ini['ZONAL_STATS']['gridmet_monthly_flag']:
            gridmet_monthly_func(
                gridmet_monthly_fields, ini, zone, tasks, gridmet_end_dt,
                overwrite_flag)
        if ini['ZONAL_STATS']['pdsi_flag']:
            pdsi_func(pdsi_dekad_fields, ini, zone, tasks, overwrite_flag)


def landsat_func(export_fields, ini, zone, tasks, overwrite_flag=False):
    """

    Function will attempt to generate export tasks only for missing SCENE_IDs
    Also try to limit the products to only those with missing data

    Args:
        export_fields ():
        ini (dict): Input file parameters
        zone (dict): Zone specific parameters
        tasks ():
        gridmet_end_dt (datetime):
        overwrite_flag (bool): if True, overwrite existing values.
            Don't remove/replace the CSV file directly.
    """
    logging.info('  Landsat')

    # Pre-filter by tile
    # First get the list of possible tiles for each zone
    try:
        zone_tile_list = ini['ZONAL_STATS']['zone_tile_json'][zone['name']]
    except KeyError:
        logging.info('    No matching tiles, skipping zone')
        return True
    if ini['INPUTS']['path_keep_list']:
        zone_tile_list = [
            tile for tile in zone_tile_list
            if int(tile[1:4]) in ini['INPUTS']['path_keep_list']]
    if zone_tile_list and ini['INPUTS']['row_keep_list']:
        zone_tile_list = [
            tile for tile in zone_tile_list
            if int(tile[5:8]) in ini['INPUTS']['row_keep_list']]
    if zone_tile_list and ini['INPUTS']['tile_keep_list']:
        zone_tile_list = [
            tile for tile in zone_tile_list
            if tile in ini['INPUTS']['tile_keep_list']]
    if not zone_tile_list:
        logging.info('    No matching tiles, skipping zone')
        return True

    # Initialize the Landsat object
    # For getting SCENE_ID lists, don't use zone_geom or products
    #   and set mosaic_method to 'none' to get separate SCENE_ID lists
    #   for each tile
    # These will be applied below
    landsat_args = {
        k: v for section in ['INPUTS']
        for k, v in ini[section].items()
        if k in [
            'landsat4_flag', 'landsat5_flag',
            'landsat7_flag', 'landsat8_flag',
            'fmask_flag', 'acca_flag', 'fmask_source',
            'start_year', 'end_year',
            'start_month', 'end_month', 
            'start_doy', 'end_doy',
            'scene_id_keep_list', 'scene_id_skip_list',
            'path_keep_list', 'row_keep_list',
            'adjust_method', 'mosaic_method'
        ]}
    landsat = ee_common.Landsat(landsat_args)
    if ini['INPUTS']['tile_geom']:
        landsat.tile_geom = ini['INPUTS']['tile_geom']
    landsat.zone_geom = None
    landsat.products = []
    landsat.mosaic_method = 'none'

    # Make copy of export field list in order to retain existing columns
    output_fields = export_fields[:]

    # Read existing output table if possible
    # Otherwise build an empty dataframe
    # Only add new empty fields when reading in (don't remove any existing)
    logging.debug('    Comparing output SCENE_IDs to init SCENE_ID list')
    output_id = output_id = '{}_landsat_daily'.format(zone['name'])
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('    {}'.format(output_path))
    try:
        output_df = pd.read_csv(output_path, parse_dates=['DATE'])

        # DEADBEEF - Code to update old style SCENE_IDs
        # This should only need to be run once
        old_id_mask = ~output_df['SCENE_ID'].str.contains('_')
        if any(old_id_mask):
            output_df.loc[old_id_mask, 'SCENE_ID'] = output_df[old_id_mask].apply(
                lambda x: '{}0{}_{}{}_{}'.format(
                    x['SCENE_ID'][0:2], x['SCENE_ID'][2:3],
                    x['SCENE_ID'][3:6], x['SCENE_ID'][6:9],
                    x['DATE'].strftime('%Y%m%d')), axis=1)
        output_df['PLATFORM'] = output_df['SCENE_ID'].str.slice(0, 4)
        if 'LANDSAT' in output_df.columns.values:
            del output_df['LANDSAT']

        # Move any existing columns not in export_fields to end of CSV
        output_fields.extend([
            f for f in output_df.columns.values if f not in export_fields])
        output_df = output_df.reindex(columns=output_fields)
        output_df.sort_values(by=['DATE', 'ROW'], inplace=True)
    except IOError:
        logging.debug(
            '    Output path doesn\'t exist, building empty dataframe')
        output_df = pd.DataFrame(columns=output_fields)
    except Exception as e:
        logging.exception('    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')

    # DEADBEEF - Look for duplicate SCENE_IDs
    if not output_df.empty and output_df.duplicated(['SCENE_ID']).any():
        logging.debug('    Removing duplicate SCENE_IDs')
        output_df = output_df[output_df.duplicated(['SCENE_ID'], False)]

    # # DEADBEEF - Remove all empty entries
    # if not output_df.empty and output_df['NDVI_TOA'].isnull().any():
    #     logging.debug('    Removing empty entries')
    #     output_df.drop(
    #         output_df[output_df['NDVI_TOA'].isnull()].index, inplace=True)
    #     output_df.sort_values(by=['DATE', 'ROW'], inplace=True)
    #     output_df.to_csv(output_path, index=False, columns=output_fields)
    #     input('ENTER')

    # Use the SCENE_ID as the index  
    output_df.set_index('SCENE_ID', inplace=True, drop=True)

    if output_df.empty:
        # If output DF is empty, skip filtering by SCENE_ID or product
        logging.info(
            '    Processing all available SCENE_IDs and products '
            'based on INI parameters')
        landsat.products = ini['ZONAL_STATS']['landsat_products']
    else:
        # Filter based on the pre-computed SCENE_ID lists from the init
        # Get the list of possible SCENE_IDs for each zone tile
        logging.debug('    Getting SCENE_ID lists')
        export_ids = set()
        for tile in zone_tile_list:
            if tile in ini['ZONAL_STATS']['tile_scene_json'].keys():
                # Read the previously computed tile SCENE_ID list
                export_ids.update(
                    ini['ZONAL_STATS']['tile_scene_json'][tile])
            else:
                # Compute the SCENE_ID list for each tile if needed
                logging.debug('      {}'.format(tile))
                path_row_re = re.compile('p(?P<PATH>\d{1,3})r(?P<ROW>\d{1,3})')
                path, row = list(map(int, path_row_re.match(tile).groups()))

                # Filter the Landsat collection down to a single tile
                landsat.path_keep_list = [path]
                landsat.row_keep_list = [row]
                landsat_coll = landsat.get_collection()

                # Get new scene ID list
                ini['ZONAL_STATS']['tile_scene_json'][tile] = utils.getinfo(
                    landsat_coll.aggregate_histogram('SCENE_ID'))
                export_ids.update(ini['ZONAL_STATS']['tile_scene_json'][tile])

        # If export_ids is empty, all SCENE_IDs may have been filtered
        if not export_ids:
            logging.info(
                '    No SCENE_IDs to process after applying INI filters, '
                'skipping zone')
            return False

        # Compute mosaiced SCENE_IDs after filtering
        if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
            mosaic_id_dict = defaultdict(list)
            for scene_id in export_ids:
                mosaic_id = '{}XXX{}'.format(scene_id[:8], scene_id[11:])
                mosaic_id_dict[mosaic_id].append(scene_id)
            export_ids = set(mosaic_id_dict.keys())

        # List of SCENE_IDs that are entirely missing
        missing_all_ids = export_ids - set(output_df.index.values)

        # Identify SCENE_IDs that are missing any data
        # Filter based on product and SCENE_ID lists
        # Check for missing data as long as PIXEL_COUNT > 0
        #   (Null has not been requested/export)
        missing_fields = [
            f.upper() for f in ini['ZONAL_STATS']['landsat_products']]
        missing_id_mask = (
            (output_df['PIXEL_COUNT'] > 0) & 
            output_df.index.isin(export_ids))
        missing_df = output_df.loc[missing_id_mask, missing_fields].isnull()

        # List of SCENE_IDs and products with some missing data
        missing_any_ids = set(missing_df[missing_df.any(axis=1)].index.values)
        # logging.debug('  SCENE_IDs missing all values: {}'.format(
        #     ', '.join(sorted(missing_all_ids))))
        # logging.debug('  SCENE_IDs missing any values: {}'.format(
        #     ', '.join(sorted(missing_any_ids))))

        # Check for fields that are entirely empty or not present
        #   These may have been added but not filled
        # Additional logic is to handle condition where 
        #   calling all on an empty dataframe returns True
        if not missing_df.empty:
            missing_all_products = set(
                f.lower()
                for f in missing_df.columns[missing_df.all(axis=0)])
            missing_any_products = set(
                f.lower()
                for f in missing_df.columns[missing_df.any(axis=0)])
        else:
            missing_all_products = set()
            missing_any_products = set()
        if missing_all_products:
            logging.debug('    Products missing all values: {}'.format(
                ', '.join(sorted(missing_all_products))))
        if missing_any_products:
            logging.debug('    Products missing any values: {}'.format(
                ', '.join(sorted(missing_any_products))))

        missing_ids = missing_all_ids | missing_any_ids
        missing_products = missing_all_products | missing_any_products

        # If mosaic flag is set, switch IDs back to non-mosaiced
        if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
            missing_scene_ids = [
                scene_id for mosaic_id in missing_ids
                for scene_id in mosaic_id_dict[mosaic_id]]
        else:
            missing_scene_ids = set(missing_ids)
        logging.debug('  SCENE_IDs missing: {}'.format(
            ', '.join(sorted(missing_scene_ids))))
        logging.info('    Missing ID count: {}'.format(
            len(missing_scene_ids)))

        # Evaluate whether a subset of SCENE_IDs or products can be exported
        # The SCENE_ID skip and keep lists cannot be mosaiced SCENE_IDs
        if not missing_scene_ids and not missing_products:
            logging.info('    No missing data or products, skipping zone')
            return True
        elif missing_scene_ids:
            logging.info('    Exporting all products for specific SCENE_IDs')
            landsat.update_scene_id_keep(missing_scene_ids)
            landsat.products = ini['ZONAL_STATS']['landsat_products']
        elif missing_all_products:
            logging.info('    Exporting all products for specific SCENE_IDs')
            landsat.update_scene_id_keep(missing_scene_ids)
            landsat.products = list(missing_all_products)
        elif missing_products or missing_scene_ids:
            logging.info('    Exporting specific missing products/SCENE_IDs')
            landsat.update_scene_id_keep(missing_scene_ids)
            landsat.products = list(missing_products)
        else:
            logging.error('Unhandled conditional')
            input('ENTER')


    # Reset the Landsat collection args
    landsat.path_keep_list = landsat_args['path_keep_list']
    landsat.row_keep_list = landsat_args['row_keep_list']
    landsat.mosaic_method = landsat_args['mosaic_method']
    # landsat.zone_geom = zone['geom']
    # pp.pprint(vars(landsat))
    # input('ENTER')

    def export_update(data_df):
        """Set/modify ancillary field values in the export CSV dataframe"""
        # First remove any extra rows that were added for exporting
        data_df.drop(
            data_df[data_df.SCENE_ID == 'DEADBEEF'].index, inplace=True)

        # Add additional fields to the export data frame
        data_df.set_index('SCENE_ID', inplace=True, drop=True)
        if not data_df.empty:
            data_df['ZONE_FID'] = zone['fid']
            data_df['PLATFORM'] = data_df.index.str.slice(0, 4)
            data_df['PATH'] = data_df.index.str.slice(5, 8).astype(int)
            data_df['DATE'] = pd.to_datetime(
                data_df.index.str.slice(12, 20), format='%Y%m%d')
            data_df['YEAR'] = data_df['DATE'].dt.year
            data_df['MONTH'] = data_df['DATE'].dt.month
            data_df['DAY'] = data_df['DATE'].dt.day
            data_df['DOY'] = data_df['DATE'].dt.dayofyear.astype(int)
            data_df['AREA'] = zone['area']
            data_df['PIXEL_SIZE'] = landsat.cellsize
            fmask_mask = data_df['FMASK_TOTAL'] > 0
            if fmask_mask.any():
                data_df.loc[fmask_mask, 'FMASK_PCT'] = 100.0 * (
                    data_df.loc[fmask_mask, 'FMASK_COUNT'] /
                    data_df.loc[fmask_mask, 'FMASK_TOTAL'])
            data_df['QA'] = 0

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

        # Skip year range if SCENE_ID keep list is set and doesn't match
        if (landsat.scene_id_keep_list and not any(
                set(int(x[12:16]) for x in landsat.scene_id_keep_list) &
                set(range(start_year, end_year + 1)))):
            logging.debug('  {}  {}'.format(start_date, end_date))
            logging.debug('    No matching SCENE_IDs for year range')
            continue
        else:
            logging.info('  {}  {}'.format(start_date, end_date))

        if iter_years > 1:
            year_str = '{}_{}'.format(start_year, end_year)
        else:
            year_str = '{}'.format(year)
        # logging.debug('  {}  {}'.format(start_year, end_year))

        # Include EPSG code in export and output names
        if 'EPSG' in ini['SPATIAL']['crs']:
            crs_str = '_' + ini['SPATIAL']['crs'].replace(':', '').lower()
        else:
            crs_str = ''

        # Export Landsat zonal stats
        export_id = '{}_{}_landsat{}_{}'.format(
            ini['INPUTS']['zone_filename'], zone['name'].lower(),
            crs_str, year_str)
        # export_id = '{}_{}_landsat_{}'.format(
        #     os.path.splitext(ini['INPUTS']['zone_filename'])[0],
        #     zone_name, year_str)
        # output_id = '{}_landsat{}_{}'.format(
        #     zone['name'], crs_str, year_str)
        if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
            export_id += '_' + ini['INPUTS']['mosaic_method'].lower()
            # output_id += '_' + ini['INPUTS']['mosaic_method'].lower()

        export_path = os.path.join(
            ini['EXPORT']['export_ws'], export_id + '.csv')
        logging.debug('    Export: {}'.format(export_id + '.csv'))

        # There is an EE bug that appends "ee_export" to the end of CSV
        #   file names when exporting to cloud storage
        # Also, use the sharelink path for reading the csv directly
        if ini['EXPORT']['export_dest'] == 'cloud':
            export_cloud_name = export_id + 'ee_export.csv'
            export_cloud_path = os.path.join(
                ini['EXPORT']['export_ws'], export_cloud_name)
            export_cloud_url = 'https://storage.googleapis.com/{}/{}'.format(
                ini['EXPORT']['bucket_name'], export_cloud_name)

        if export_id in tasks.keys():
            logging.info('    Task already submitted, skipping')
            continue
        elif (ini['EXPORT']['export_dest'] == 'gdrive' and
                os.path.isfile(export_path)):
            if ini['EXPORT']['export_only']:
                logging.info('    Export CSV already exists, skipping')
                continue

            # Modify CSV while copying from Google Drive
            logging.debug('    Reading export CSV')
            try:
                export_df = pd.read_csv(export_path)
            # except pd.io.common.EmptyDataError:
            except pd.errors.EmptyDataError:
                export_df = pd.DataFrame()
                logging.debug('    Empty export CSV, skipping')
            except Exception as e:
                logging.error('  Unhandled Exception\n  {}'.format(e))
                input('ENTER')

            # Save data to main dataframe
            if not export_df.empty:
                logging.info('    Processing exported CSV')
                export_df = export_update(export_df)
                if overwrite_flag:
                    # Update happens inplace automatically
                    output_df.update(export_df)
                else:
                    # Combine_first() doesn't have an inplace parameter
                    output_df = output_df.combine_first(export_df)

            # DEADBEEF
            # logging.debug('    Removing export CSV')
            # os.remove(export_path)
            continue
        elif (ini['EXPORT']['export_dest'] == 'cloud' and
                export_cloud_name in ini['EXPORT']['cloud_file_list']):
            if ini['EXPORT']['export_only']:
                logging.debug('    Export CSV already exists, skipping')
                continue

            logging.debug('    Reading {}'.format(export_cloud_url))
            try:
                export_request = requests.get(export_cloud_url).content
                export_df = pd.read_csv(
                    StringIO(export_request.decode('utf-8')))
            # except pd.io.common.EmptyDataError:
            except pd.errors.EmptyDataError:
                export_df = pd.DataFrame()
                logging.debug('    Empty eport CSV, skipping')
            except Exception as e:
                logging.error('  Unhandled Exception\n  {}'.format(e))
                input('ENTER')

            # Save data to main dataframe
            export_df = export_update(export_df)
            if not export_df.empty:
                logging.info('    Processing exported CSV')
                if overwrite_flag:
                    # Update happens inplace automatically
                    output_df.update(export_df)
                else:
                    # Combine first doesn't have an inplace parameter
                    output_df = output_df.combine_first(export_df)

            logging.debug('    Removing {}'.format(export_cloud_path))
            try:
                check_output(['gsutil', 'rm', export_cloud_path])
            except Exception as e:
                logging.error('Unhandled Exception')
                logging.error(str(e))
            continue


        # Filter by iteration date in addition to input date parameters
        landsat.start_date = start_date
        landsat.end_date = end_date
        landsat_coll = landsat.get_collection()

        # DEBUG - Test that the Landsat collection is getting built
        # print(landsat_coll.aggregate_histogram('SCENE_ID').getInfo())
        # input('ENTER')
        # print('Bands: {}'.format(
        #     [x['id'] for x in ee.Image(landsat_coll.first()).getInfo()['bands']]))
        # print('SceneID: {}'.format(
        #     ee.Image(landsat_coll.first()).getInfo()['properties']['SCENE_ID'])
        # input('ENTER')
        # if ee.Image(landsat_coll.first()).getInfo() is None:
        #     logging.info('    No images, skipping')
        #     continue

        # Calculate values and statistics
        # Build function in loop to set water year ETo/PPT values
        def zonal_stats_func(image):
            """"""
            scene_id = ee.String(image.get('SCENE_ID'))
            date = ee.Date(image.get('system:time_start'))
            # doy = ee.Number(date.getRelative('day', 'year')).add(1)
            bands = len(landsat.products) + 3

            # Using zone['geom'] as the geomtry should make it
            #   unnecessary to clip also
            input_mean = ee.Image(image) \
                .select(landsat.products + ['cloud_score', 'row']) \
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=zone['geom'],
                    crs=ini['SPATIAL']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False,
                    tileScale=1,
                    maxPixels=zone['max_pixels'] * bands)

            # Count unmasked Fmask pixels to get pixel count
            # Count Fmask > 1 to get Fmask count (0 is clear and 1 is water)
            fmask_img = ee.Image(image).select(['fmask'])
            input_count = ee.Image([
                    fmask_img.gte(0).unmask().rename(['pixel']),
                    fmask_img.gt(1).rename(['fmask'])]) \
                .reduceRegion(
                    reducer=ee.Reducer.sum().combine(
                        ee.Reducer.count(), '', True),
                    geometry=zone['geom'],
                    crs=ini['SPATIAL']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False,
                    tileScale=1,
                    maxPixels=zone['max_pixels'] * 3)

            # Standard output
            zs_dict = {
                'ZONE_NAME': zone['name'],
                # 'ZONE_FID': zone['fid'],
                'SCENE_ID': scene_id.slice(0, 20),
                # 'PLATFORM': scene_id.slice(0, 4),
                # 'PATH': ee.Number(scene_id.slice(5, 8)),
                'ROW': ee.Number(input_mean.get('row')),
                # Compute dominant row
                # 'ROW': ee.Number(scene_id.slice(8, 11)),
                # 'DATE': date.format('YYYY-MM-dd'),
                # 'YEAR': date.get('year'),
                # 'MONTH': date.get('month'),
                # 'DAY': date.get('day'),
                # 'DOY': doy,
                # 'AREA': zone['area'],
                # 'PIXEL_SIZE': landsat.cellsize,
                'PIXEL_COUNT': input_count.get('pixel_sum'),
                'PIXEL_TOTAL': input_count.get('pixel_count'),
                'FMASK_COUNT': input_count.get('fmask_sum'),
                'FMASK_TOTAL': input_count.get('fmask_count'),
                # 'FMASK_PCT': ee.Number(input_count.get('fmask_sum')) \
                #     .divide(ee.Number(input_count.get('fmask_count'))) \
                #     .multiply(100),
                'CLOUD_SCORE': input_mean.get('cloud_score'),
                # 'QA': ee.Number(0)
            }
            # Product specific output
            zs_dict.update({
                p.upper(): input_mean.get(p.lower()) for p in landsat.products
            })
            return ee.Feature(None, zs_dict)

        stats_coll = landsat_coll.map(zonal_stats_func, False)

        # # DEBUG - Test the function for a single image
        # stats_info = zonal_stats_func(
        #     ee.Image(landsat_coll.first())).getInfo()
        # pp.pprint(stats_info['properties'])
        # input('ENTER')

        # # DEBUG - Print the stats info to the screen
        # stats_info = stats_coll.getInfo()
        # for ftr in stats_info['features']:
        #     pp.pprint(ftr)
        # input('ENTER')
        # return False

        # Add a dummy entry to the stats collection 
        format_dict = {
            'ZONE_NAME': 'DEADBEEF',
            'SCENE_ID': 'DEADBEEF',
            'ROW': -9999,
            'PIXEL_COUNT': -9999,
            'PIXEL_TOTAL': -9999,
            'FMASK_COUNT': -9999,
            'FMASK_TOTAL': -9999,
            'CLOUD_SCORE': -9999,
        }
        format_dict.update({p.upper(): -9999 for p in landsat.products})
        stats_coll = ee.FeatureCollection(ee.Feature(None, format_dict)) \
            .merge(stats_coll)

        # # DEBUG - Print the stats info to the screen
        # stats_info = stats_coll.getInfo()
        # for ftr in stats_info['features']:
        #     pp.pprint(ftr)
        # input('ENTER')

        if ini['EXPORT']['export_dest'] == 'gdrive':
            logging.debug('    Building export task')
            task = ee.batch.Export.table.toDrive(
                collection=stats_coll,
                description=export_id,
                folder=ini['EXPORT']['export_folder'],
                fileNamePrefix=export_id,
                fileFormat='CSV')
            logging.debug('    Starting export task')
            utils.request(task.start())
        elif ini['EXPORT']['export_dest'] == 'cloud':
            logging.debug('    Building export task')
            task = ee.batch.Export.table.toCloudStorage(
                collection=stats_coll,
                description=export_id,
                bucket=ini['EXPORT']['bucket_name'],
                fileNamePrefix='{}'.format(export_id.replace('-', '')),
                # fileNamePrefix=export_id,
                fileFormat='CSV')
            logging.debug('    Starting export task')
            utils.request(task.start())
        elif ini['EXPORT']['export_dest'] == 'getinfo':
            logging.debug('    Requesting data')
            export_info = utils.getinfo(stats_coll)['features']
            export_df = pd.DataFrame([ftr['properties'] for ftr in export_info])
            export_df = export_update(export_df)

            # Save data to main dataframe
            if not export_df.empty:
                logging.debug('    Processing data')
                if overwrite_flag:
                    # Update happens inplace automatically
                    output_df.update(export_df)
                else:
                    # Combine first doesn't have an inplace parameter
                    output_df = output_df.combine_first(export_df)

    # Save updated CSV
    # if output_df is not None and not output_df.empty:
    if output_df is not None:
        output_df.reset_index(drop=False, inplace=True)
        output_df.sort_values(by=['DATE', 'ROW'], inplace=True)

        # Eventually these will all be populated directly from the export
        # if 'PLATFORM' not in output_df.columns.values:
        output_df['PLATFORM'] = output_df['SCENE_ID'].str.slice(0, 4)
        if 'LANDSAT' in output_df.columns.values:
            del output_df['LANDSAT']
        output_df['AREA'] = zone['area']
        fmask_mask = output_df['FMASK_TOTAL'] > 0
        if fmask_mask.any():
            output_df.loc[fmask_mask, 'FMASK_PCT'] = 100.0 * (
                output_df.loc[fmask_mask, 'FMASK_COUNT'] /
                output_df.loc[fmask_mask, 'FMASK_TOTAL'])
        if output_df['QA'].isnull().any():
            output_df.loc[output_df['QA'].isnull(), 'QA'] = 0
        output_df['PIXEL_SIZE'] = landsat.cellsize

        # Set output types before saving
        if output_df['ZONE_NAME'].dtype == np.float64:
            output_df['ZONE_NAME'] = output_df['ZONE_NAME'].astype(int).astype(str)
        # DEADBEEF - I can't set these to int since they contain NaN
        # 'QA', 'PIXEL_TOTAL', 'PIXEL_COUNT', 'FMASK_TOTAL', 'FMASK_COUNT']:
        for field in ['ZONE_FID', 'PATH', 'YEAR', 'MONTH', 'DAY', 'DOY']:
            output_df[field] = output_df[field].astype(int)
        output_df.to_csv(output_path, index=False, columns=output_fields)
    else:
        logging.info(
            '  Empty output dataframe\n'
            '  The exported CSV files may not be ready')


def gridmet_daily_func(export_fields, ini, zone, tasks, overwrite_flag=False):
    """

    Args:
        export_fields ():
        ini (dict): Input file parameters
        zone (dict): Zone specific parameters
        tasks ():
        gridmet_end_dt (datetime):
        overwrite_flag (bool): if True, overwrite existing files
    """

    logging.info('  GRIDMET Daily ETo/PPT')

    export_id = '{}_{}_gridmet_daily'.format(
        os.path.splitext(ini['INPUTS']['zone_filename'])[0],
        zone['name'].lower())
    output_id = '{}_gridmet_daily'.format(zone['name'])

    export_path = os.path.join(
        ini['EXPORT']['export_ws'], export_id + '.csv')
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('    Export: {}'.format(export_id + '.csv'))
    logging.debug('    Output: {}'.format(output_path))

    # Get INI date range
    # Insert one additional year the beginning for water year totals
    start_date = '{}-01-01'.format(ini['INPUTS']['start_year'] - 1)
    end_date = min(
        '{:04d}-01-01'.format(ini['INPUTS']['end_year'] + 1),
        datetime.datetime.today().strftime('%Y-%m-%d'))
    export_dates = set(utils.date_range(start_date, end_date))

    # Read in existing data if possible
    try:
        output_df = pd.read_csv(output_path, parse_dates=['DATE'])
    except IOError:
        logging.debug(
            '    Output path doesn\'t exist, building empty dataframe')
        output_df = pd.DataFrame(columns=export_fields)
    except Exception as e:
        logging.exception('    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')
    output_df.set_index('DATE', inplace=True, drop=True)

    # Check for any expected dates not in the output dateframe
    # For now, if any dates are missing recompute all dates in INI range
    export_dates = export_dates - set(output_df['DATE'].values)
    if not export_dates and not overwrite_flag:
        logging.info('    All dates present, skipping')
        return True

    def negative_adjust(image):
        """GRIDMET PPT zero values are sometimes a small negative value"""
        return image.where(image.lt(0), 0)

    gridmet_coll = ee.ImageCollection(
        ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
            .select(['pet', 'pr'], ['eto', 'ppt']) \
            .filterDate(start_date, end_date) \
            .map(negative_adjust))

    # There is an EE bug that appends "ee_export" to the end of CSV
    #   file names when exporting to cloud storage
    # Also, use the sharelink path for reading the csv directly
    if ini['EXPORT']['export_dest'] == 'cloud':
        export_cloud_name = export_id + 'ee_export.csv'
        export_cloud_path = os.path.join(
            ini['EXPORT']['export_ws'], export_cloud_name)
        export_cloud_url = 'https://storage.googleapis.com/{}/{}'.format(
            ini['EXPORT']['bucket_name'], export_cloud_name)

    if export_id in tasks.keys():
        logging.debug('  Task already submitted, skipping')
        return True
    elif (ini['EXPORT']['export_dest'] == 'gdrive' and
            os.path.isfile(export_path)):
        # Modify CSV while copying from Google Drive
        logging.debug('    Reading export CSV')
        try:
            export_df = pd.read_csv(export_path)
            # export_request = requests.get(export_cloud_url).content
            # export_df = pd.read_csv(
            #     StringIO(export_request.decode('utf-8')))
        # except pd.io.common.EmptyDataError:
        except pd.errors.EmptyDataError:
            export_df = pd.DataFrame()
            logging.debug('    Empty export CSV, skipping')
        except Exception as e:
            logging.error('  Unhandled Exception\n  {}'.format(e))
            input('ENTER')

        # Save data to main dataframe
        if not export_df.empty:
            logging.info('    Processing exported CSV')
            export_df.set_index('DATE', inplace=True, drop=True)
            if overwrite_flag:
                # Update happens inplace automatically
                output_df.update(export_df)
            else:
                # Combine first doesn't have an inplace parameter
                output_df = output_df.combine_first(export_df)

        logging.debug('    Removing export CSV')
        os.remove(export_path)
        return True
    elif (ini['EXPORT']['export_dest'] == 'cloud' and
            export_cloud_name in ini['EXPORT']['cloud_file_list']):
        # Modify CSV while copying from Google Drive
        logging.debug('    Reading export CSV')
        try:
            export_request = requests.get(export_cloud_url).content
            export_df = pd.read_csv(
                StringIO(export_request.decode('utf-8')))
        # except pd.io.common.EmptyDataError:
        except pd.errors.EmptyDataError:
            export_df = pd.DataFrame()
            logging.debug('    Empty export CSV, skipping')
        except Exception as e:
            logging.error('  Unhandled Exception\n  {}'.format(e))
            input('ENTER')

        # Save data to main dataframe
        if not export_df.empty:
            logging.info('    Processing exported CSV')
            export_df.set_index('DATE', inplace=True, drop=True)
            if overwrite_flag:
                # Update happens inplace automatically
                output_df.update(export_df)
            else:
                # Combine first doesn't have an inplace parameter
                output_df = output_df.combine_first(export_df)
            output_df.reset_index(drop=False, inplace=True)
            output_df.sort_values(by=['DATE'], inplace=True)
            output_df = output_df[export_fields]
            output_df.to_csv(output_path, index=False, columns=export_fields)

        logging.debug('    Removing {}'.format(export_cloud_path))
        try:
            check_output(['gsutil', 'rm', export_cloud_path])
        except Exception as e:
            logging.error('Unhandled Exception')
            logging.error(str(e))
        return True

    # Calculate values and statistics
    # Build function in loop to set water year ETo/PPT values
    def gridmet_zonal_stats_func(image):
        """"""
        date = ee.Date(image.get('system:time_start'))
        year = ee.Number(date.get('year'))
        month = ee.Number(date.get('month'))
        doy = ee.Number(date.getRelative('day', 'year')).add(1)
        wyear = ee.Number(ee.Date.fromYMD(
            year, month, 1).advance(3, 'month').get('year'))
        input_mean = ee.Image(image) \
            .reduceRegion(
                ee.Reducer.mean(), geometry=zone['geom'],
                crs=ini['SPATIAL']['crs'],
                crsTransform=ini['EXPORT']['transform'],
                bestEffort=False, tileScale=1,
                maxPixels=zone['max_pixels'] * 3)
        return ee.Feature(
            None,
            {
                'ZONE_NAME': zone['name'],
                'ZONE_FID': zone['fid'],
                'DATE': date.format('YYYY-MM-dd'),
                'YEAR': year,
                'MONTH': month,
                'DAY': date.get('day'),
                'DOY': doy,
                'WATER_YEAR': wyear,
                'ETO': input_mean.get('eto'),
                'PPT': input_mean.get('ppt')
            })
    stats_coll = gridmet_coll.map(gridmet_zonal_stats_func)

    if ini['EXPORT']['export_dest'] == 'gdrive':
        logging.debug('    Building export task')
        task = ee.batch.Export.table.toDrive(
            collection=stats_coll,
            description=export_id,
            folder=ini['EXPORT']['export_folder'],
            fileNamePrefix=export_id,
            fileFormat='CSV')
        logging.debug('    Starting export task')
        utils.request(task.start())
    elif ini['EXPORT']['export_dest'] == 'cloud':
        logging.debug('    Building export task')
        task = ee.batch.Export.table.toCloudStorage(
            collection=stats_coll,
            description=export_id,
            bucket=ini['EXPORT']['bucket_name'],
            fileNamePrefix='{}'.format(export_id.replace('-', '')),
            # fileNamePrefix=export_id,
            fileFormat='CSV')
        logging.debug('    Starting export task')
        utils.request(task.start())
    elif ini['EXPORT']['export_dest'] == 'getinfo':
        logging.debug('    Requesting data')
        export_info = utils.request(stats_coll.getInfo())['features']
        export_df = pd.DataFrame([ftr['properties'] for ftr in export_info])
        export_df.set_index('DATE', inplace=True, drop=True)

        # Save data to main dataframe
        if not export_df.empty:
            logging.debug('    Processing data')
            if overwrite_flag:
                # Update happens inplace automatically
                output_df.update(export_df)
            else:
                # Combine first doesn't have an inplace parameter
                output_df = output_df.combine_first(export_df)

    # Save updated CSV
    if output_df is not None:
        output_df.reset_index(drop=False, inplace=True)
        output_df.sort_values(by=['DATE'], inplace=True)
        output_df = export_df[export_fields]
        output_df.to_csv(output_path, index=False, columns=export_fields)


def gridmet_monthly_func(export_fields, ini, zone, tasks, gridmet_end_dt,
                         overwrite_flag=False):
    """

    Args:
        export_fields ():
        ini (dict): Input file parameters
        zone (dict): Zone specific parameters
        tasks ():
        gridmet_end_dt (datetime):
        overwrite_flag (bool): if True, overwrite existing files
    """

    logging.info('  GRIDMET Monthly ETo/PPT')

    export_id = '{}_{}_gridmet_monthly'.format(
        os.path.splitext(ini['INPUTS']['zone_filename'])[0],
        zone['name'].lower())
    output_id = '{}_gridmet_monthly'.format(zone['name'])

    export_path = os.path.join(
        ini['EXPORT']['export_ws'], export_id + '.csv')
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('    Export: {}'.format(export_id + '.csv'))
    logging.debug('    Output: {}'.format(output_path))

    # Read in existing data if possible
    try:
        output_df = pd.read_csv(output_path)
        # output_df = pd.read_csv(output_path, index_col='DATE')
    except IOError:
        logging.debug(
            '    Output path doesn\'t exist, building empty dataframe')
        output_df = pd.DataFrame(columns=export_fields)
        # output_df.set_index('DATE', inplace=True, drop=True)
        # output_df.index.name = 'DATE'
    except Exception as e:
        logging.exception('    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')
    output_df.set_index('DATE', inplace=True, drop=True)

    # DEADBEEF - Remove negative PPT
    if any(output_df['PPT'] < 0):
        logging.info('    Removing negatives')
        output_df.drop(output_df[output_df['PPT'] < 0].index, inplace=True)
        output_df.reset_index(drop=False, inplace=True)
        output_df.sort_values(by=['DATE'], inplace=True)
        output_df.to_csv(output_path, index=False, columns=export_fields)

    # Get list of existing dates in the CSV
    if not output_df.empty:
        output_dates = set([
            (y, m) for y, m in output_df[['YEAR', 'MONTH']].values])
    else:
        output_dates = set()

    # Get list of possible dates based on INI
    export_dates = set([
        (y, m)
        for y in range(
            ini['INPUTS']['start_year'] - 1, ini['INPUTS']['end_year'] + 1)
        for m in range(1, 13)
        if datetime.datetime(y, m, 1) <= gridmet_end_dt])
    export_dates = export_dates - output_dates
    logging.debug('    Missing dates: {}'.format(len(export_dates)))

    # Skip processing if all dates already exist in the CSV
    if not export_dates and not overwrite_flag:
        logging.info('    All dates present, skipping')
        return True
    export_dt_list = ee.List([
        ee.Date.fromYMD(y, m, 1) for y, m in sorted(export_dates)])

    # Export GRIDMET zonal stats
    # Insert one additional year the beginning for water year totals
    # Compute monthly sums of GRIDMET
    def monthly_sum(start_dt):
        def negative_adjust(image):
            # GRIDMET PPT zero values are
            return image.where(image.lt(0), 0)
        gridmet_coll = ee.ImageCollection(
            ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .select(['pet', 'pr']) \
                .filterDate(
                    ee.Date(start_dt),
                    ee.Date(start_dt).advance(1, 'month')) \
                .map(negative_adjust))
        return ee.Image(gridmet_coll.sum()) \
            .select([0, 1], ['eto', 'ppt']) \
            .set('system:time_start', ee.Date(start_dt).millis())
    gridmet_coll = ee.ImageCollection.fromImages(
        export_dt_list.map(monthly_sum))


    # There is an EE bug that appends "ee_export" to the end of CSV
    #   file names when exporting to cloud storage
    # Also, use the sharelink path for reading the csv directly
    if ini['EXPORT']['export_dest'] == 'cloud':
        export_cloud_name = export_id + 'ee_export.csv'
        export_cloud_path = os.path.join(
            ini['EXPORT']['export_ws'], export_cloud_name)
        export_cloud_url = 'https://storage.googleapis.com/{}/{}'.format(
            ini['EXPORT']['bucket_name'], export_cloud_name)

    if export_id in tasks.keys():
        logging.debug('  Task already submitted, skipping')
        return True
    elif (ini['EXPORT']['export_dest'] == 'gdrive' and
            os.path.isfile(export_path)):
        # Modify CSV while copying from Google Drive
        logging.debug('    Reading export CSV')
        try:
            export_df = pd.read_csv(export_path)
            # export_request = requests.get(export_cloud_url).content
            # export_df = pd.read_csv(
            #     StringIO(export_request.decode('utf-8')))
        # except pd.io.common.EmptyDataError:
        except pd.errors.EmptyDataError:
            export_df = pd.DataFrame()
            logging.debug('    Empty export CSV, skipping')
        except Exception as e:
            logging.error('  Unhandled Exception\n  {}'.format(e))
            input('ENTER')

        # Save data to main dataframe
        if not export_df.empty:
            logging.info('    Processing exported CSV')
            export_df.set_index('DATE', inplace=True, drop=True)
            if overwrite_flag:
                # Update happens inplace automatically
                output_df.update(export_df)
            else:
                # Combine first doesn't have an inplace parameter
                output_df = output_df.combine_first(export_df)

        logging.debug('    Removing export CSV')
        os.remove(export_path)
        return True
    elif (ini['EXPORT']['export_dest'] == 'cloud' and
            export_cloud_name in ini['EXPORT']['cloud_file_list']):
        # Modify CSV while copying from Google Drive
        logging.debug('    Reading export CSV')
        try:
            export_request = requests.get(export_cloud_url).content
            export_df = pd.read_csv(
                StringIO(export_request.decode('utf-8')))
        # except pd.io.common.EmptyDataError:
        except pd.errors.EmptyDataError:
            export_df = pd.DataFrame()
            logging.debug('    Empty export CSV, skipping')
        except Exception as e:
            logging.error('  Unhandled Exception\n  {}'.format(e))
            input('ENTER')

        # Save data to main dataframe
        if not export_df.empty:
            logging.info('    Processing exported CSV')
            export_df.set_index('DATE', inplace=True, drop=True)
            if overwrite_flag:
                # Update happens inplace automatically
                output_df.update(export_df)
            else:
                # Combine first doesn't have an inplace parameter
                output_df = output_df.combine_first(export_df)

        logging.debug('    Removing {}'.format(export_cloud_path))
        try:
            check_output(['gsutil', 'rm', export_cloud_path])
        except Exception as e:
            logging.error('Unhandled Exception')
            logging.error(str(e))
        return True

    # Calculate values and statistics
    # Build function in loop to set water year ETo/PPT values
    def gridmet_zonal_stats_func(image):
        """"""
        date = ee.Date(image.get('system:time_start'))
        year = ee.Number(date.get('year'))
        month = ee.Number(date.get('month'))
        wyear = ee.Number(ee.Date.fromYMD(
            year, month, 1).advance(3, 'month').get('year'))
        input_mean = ee.Image(image) \
            .reduceRegion(
                ee.Reducer.mean(), geometry=zone['geom'],
                crs=ini['SPATIAL']['crs'],
                crsTransform=ini['EXPORT']['transform'],
                bestEffort=False, tileScale=1,
                maxPixels=zone['max_pixels'] * 3)
        return ee.Feature(
            None,
            {
                'ZONE_NAME': zone['name'],
                'ZONE_FID': zone['fid'],
                'DATE': date.format('YYYY-MM'),
                'YEAR': year,
                'MONTH': month,
                'WATER_YEAR': wyear,
                'ETO': input_mean.get('eto'),
                'PPT': input_mean.get('ppt')
            })
    stats_coll = gridmet_coll.map(gridmet_zonal_stats_func)

    if ini['EXPORT']['export_dest'] == 'gdrive':
        logging.debug('    Building export task')
        task = ee.batch.Export.table.toDrive(
            collection=stats_coll,
            description=export_id,
            folder=ini['EXPORT']['export_folder'],
            fileNamePrefix=export_id,
            fileFormat='CSV')
        logging.debug('    Starting export task')
        utils.request(task.start())
    elif ini['EXPORT']['export_dest'] == 'cloud':
        logging.debug('    Building export task')
        task = ee.batch.Export.table.toCloudStorage(
            collection=stats_coll,
            description=export_id,
            bucket=ini['EXPORT']['bucket_name'],
            fileNamePrefix='{}'.format(export_id.replace('-', '')),
            # fileNamePrefix=export_id,
            fileFormat='CSV')
        logging.debug('    Starting export task')
        utils.request(task.start())
    elif ini['EXPORT']['export_dest'] == 'getinfo':
        logging.debug('    Requesting data')
        export_info = utils.request(stats_coll.getInfo())['features']
        export_df = pd.DataFrame([ftr['properties'] for ftr in export_info])
        export_df.set_index('DATE', inplace=True, drop=True)

        # Save data to main dataframe
        if not export_df.empty:
            logging.debug('    Processing data')
            if overwrite_flag:
                # Update happens inplace automatically
                output_df.update(export_df)
            else:
                # Combine first doesn't have an inplace parameter
                output_df = output_df.combine_first(export_df)

    # Save updated CSV
    if output_df is not None:
        output_df.reset_index(drop=False, inplace=True)
        output_df.sort_values(by=['DATE'], inplace=True)
        output_df = output_df[export_fields]
        output_df.to_csv(output_path, index=False, columns=export_fields)


def pdsi_func(export_fields, ini, zone, tasks, overwrite_flag=False):
    """

    Args:
        export_fields ():
        ini (dict): Input file parameters
        zone (dict): Zone specific parameters
        tasks ():
        gridmet_end_dt (datetime):
        overwrite_flag (bool): if True, overwrite existing files
    """

    logging.info('  GRIDMET PDSI')

    pdsi_coll = ee.ImageCollection('IDAHO_EPSCOR/PDSI') \
        .select(['pdsi'], ['pdsi']) \
        .filterDate(
            '{}-01-01'.format(ini['INPUTS']['start_year']),
            '{}-01-01'.format(ini['INPUTS']['end_year'] + 1))
    export_id = '{}_{}_pdsi_dekad'.format(
        os.path.splitext(ini['INPUTS']['zone_filename'])[0],
        zone['name'].lower())
    output_id = '{}_pdsi_dekad'.format(zone['name'])

    export_path = os.path.join(
        ini['EXPORT']['export_ws'], export_id + '.csv')
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('    Export: {}'.format(export_id + '.csv'))
    logging.debug('    Output: {}'.format(output_path))

    # There is an EE bug that appends "ee_export" to the end of CSV
    #   file names when exporting to cloud storage
    # Also, use the sharelink path for reading the csv directly
    if ini['EXPORT']['export_dest'] == 'cloud':
        export_cloud_name = export_id + 'ee_export.csv'
        export_cloud_path = os.path.join(
            ini['EXPORT']['export_ws'], export_cloud_name)
        export_cloud_url = 'https://storage.googleapis.com/{}/{}'.format(
            ini['EXPORT']['bucket_name'], export_cloud_name)

    if overwrite_flag:
        if export_id in tasks.keys():
            logging.debug('  Task already submitted, cancelling')
            ee.data.cancelTask(tasks[export_id])
            del tasks[export_id]

        if (ini['EXPORT']['export_dest'] == 'gdrive' and
                os.path.isfile(export_path)):
            logging.debug('  Export CSV already exists, removing')
            os.remove(export_path)
        elif (ini['EXPORT']['export_dest'] == 'cloud' and
                export_cloud_name in ini['EXPORT']['file_list']):
            logging.debug('    Export image already exists')
            # # Files in cloud storage are easily overwritten
            # #   so it is unneccesary to manually remove them
            # # This would remove an existing file
            # check_output(['gsutil', 'rm', export_path])

        if os.path.isfile(output_path):
            logging.debug('    Output CSV already exists, removing')
            os.remove(output_path)

    # This should probably be moved into an else block
    #   to avoid lots of os.path.isfile calls when overwriting
    if export_id in tasks.keys():
        logging.debug('  Task already submitted, skipping')
        return True
    elif (ini['EXPORT']['export_dest'] == 'gdrive' and
            os.path.isfile(export_path)):
        logging.debug('  Export CSV already exists, moving')
        # Modify CSV while copying from Google Drive
        try:
            export_df = pd.read_csv(export_path)
            export_df = export_df[export_fields]
            export_df.sort_values(by=['DATE'], inplace=True)
            export_df.to_csv(
                output_path, index=False, columns=export_fields)
        except pd.io.common.EmptyDataError:
            # Save an empty dataframe to the output path
            logging.warning('    Empty dataframe')
            export_df = pd.DataFrame(columns=export_fields)
            export_df.to_csv(
                output_path, index=False, columns=export_fields)
            # logging.warning('    Empty dataframe, skipping')
            # continue
        os.remove(export_path)
        return True
    elif (ini['EXPORT']['export_dest'] == 'cloud' and
            export_cloud_name in ini['EXPORT']['cloud_file_list']):
        logging.debug('    Export file already exists, moving')
        logging.debug('    Reading {}'.format(export_cloud_url))
        try:
            export_request = requests.get(export_cloud_url).content
            export_df = pd.read_csv(
                StringIO(export_request.decode('utf-8')))
            export_df = export_df[export_fields]
            export_df.sort_values(by=['DATE'], inplace=True)
            export_df.to_csv(
                output_path, index=False, columns=export_fields)
        except pd.io.common.EmptyDataError:
            # Save an empty dataframe to the output path
            logging.warning('    Empty dataframe')
            export_df = pd.DataFrame(columns=export_fields)
            export_df.to_csv(
                output_path, index=False, columns=export_fields)
            # logging.warning('    Empty dataframe, skipping')
            # continue
        except Exception as e:
            logging.error('Unhandled Exception')
            logging.error(str(e))
            return False
        logging.debug('    Removing {}'.format(export_cloud_path))
        try:
            check_output(['gsutil', 'rm', export_cloud_path])
        except Exception as e:
            logging.error('Unhandled Exception')
            logging.error(str(e))
        return True
    elif os.path.isfile(output_path):
        logging.debug('    Output CSV already exists, skipping')
        return True

    # Calculate values and statistics
    # Build function in loop to set water year ETo/PPT values
    def pdsi_zonal_stats_func(image):
        """"""
        date = ee.Date(image.get('system:time_start'))
        doy = ee.Number(date.getRelative('day', 'year')).add(1)
        input_mean = ee.Image(image) \
            .reduceRegion(
                ee.Reducer.mean(), geometry=zone['geom'],
                crs=ini['SPATIAL']['crs'],
                crsTransform=ini['EXPORT']['transform'],
                bestEffort=False, tileScale=1,
                maxPixels=zone['max_pixels'] * 2)
        return ee.Feature(
            None,
            {
                'ZONE_NAME': zone['name'],
                'ZONE_FID': zone['fid'],
                'DATE': date.format('YYYY-MM-dd'),
                'YEAR': date.get('year'),
                'MONTH': date.get('month'),
                'DAY': date.get('day'),
                'DOY': doy,
                'PDSI': input_mean.get('pdsi'),
            })
    stats_coll = pdsi_coll.map(pdsi_zonal_stats_func)

    logging.debug('  Building export task')
    if ini['EXPORT']['export_dest'] == 'gdrive':
        task = ee.batch.Export.table.toDrive(
            collection=stats_coll,
            description=export_id,
            folder=ini['EXPORT']['export_folder'],
            fileNamePrefix=export_id,
            fileFormat='CSV')
    elif ini['EXPORT']['export_dest'] == 'cloud':
        task = ee.batch.Export.table.toCloudStorage(
            collection=stats_coll,
            description=export_id,
            bucket=ini['EXPORT']['bucket_name'],
            fileNamePrefix='{}'.format(export_id.replace('-', '')),
            # fileNamePrefix=export_id,
            fileFormat='CSV')

    # Download the CSV to your Google Drive
    logging.debug('    Starting export task')
    utils.request(task.start())


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
