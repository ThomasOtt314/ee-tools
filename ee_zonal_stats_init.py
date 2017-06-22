#--------------------------------
# Name:         ee_zonal_stats_init.py
# Purpose:      Initialize shapefile zonal stats using Earth Engine
# Created       2017-06-19
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
from collections import defaultdict
import datetime
import json
import logging
import os
import re
import subprocess
import sys
from time import sleep

import ee
import pandas as pd
from osgeo import ogr

import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils


def main(ini_path=None, overwrite_flag=False):
    """Earth Engine Zonal Stats Initialization Script

    This script will generate empty Landsat zonal stats CSV files.
    If CSV files already exists, empty entries will be added
    for any missing SCENE_ID.

    Ancillary output files
    zone_pathrow.json
        All possible path/rows for each zone (ignores path/row INI parameters).
        Using the custom WRS2 descending footprints.
    pathrow_sceneid.json
        All possible SCENE_IDs for each path/row based on the
        date range and scene_id skip/keep lists.
    sceneid_zone.json


    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nInitialize Earth Engine Zonal Stats')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    # inputs.parse_section(ini, section='EXPORT')
    inputs.parse_section(ini, section='ZONAL_STATS')

    path_row_re = re.compile('p(?P<PATH>\d{1,3})r(?P<ROW>\d{1,3})')

    # File paths
    zone_geojson = os.path.join(
        ini['ZONAL_STATS']['output_ws'],
        os.path.basename(ini['INPUTS']['zone_shp_path']).replace(
            '.shp', '.geojson'))
    zone_pr_json = zone_geojson.replace('.geojson', '_path_rows.json')
    pr_scene_json = zone_geojson.replace('.geojson', '_scene_id.json')
    pr_mosaic_json = zone_geojson.replace('.geojson', '_mosaic_id.json')

    # Convert the shapefile to geojson
    if not os.path.isfile(zone_geojson) or overwrite_flag:
        logging.info('\nConverting zone shapefile to GeoJSON')
        logging.debug('  {}'.format(zone_geojson))
        subprocess.check_output([
            'ogr2ogr', '-f', 'GeoJSON', '-preserve_fid',
            '-select', '{}'.format(ini['INPUTS']['zone_field']),
            # '-lco', 'COORDINATE_PRECISION=2'
            zone_geojson, ini['INPUTS']['zone_shp_path']])

    # Read in the zone geojson
    logging.debug('\nReading zone GeoJSON')
    try:
        with open(zone_geojson, 'r') as f:
            zones = json.load(f)
    except Exception as e:
        logging.error('  Error reading geojson file, removing')
        os.remove(zone_geojson)

    # Check if the zone_names are unique
    # Eventually support merging common zone_names
    zone_names = [
        str(z['properties'][ini['INPUTS']['zone_field']]).replace(' ', '_')
        for z in zones['features']]
    if len(set(zone_names)) != len(zones['features']):
        logging.error(
            '\nERROR: There appear to be duplicate zone ID/name values.'
            '\n  Currently, the values in "{}" must be unique.'
            '\n  Exiting.'.format(ini['INPUTS']['zone_field']))
        return False

    # Need zone_shp_path projection to build EE geometries
    zone_osr = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zone_wkt = gdc.osr_wkt(zone_osr)
    zones['crs'] = {'type': 'wkt', 'properties': zone_wkt.replace('"', '\'')}
    # zones['crs'] = {'type': 'proj4', 'properties': zone_osr.ExportToProj4()}

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
            'The INI spatial reference and zone shapefile spatial references '
            'do not appear to match\nThis will likely cause problems!')
        input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(zone_wkt))
        logging.debug('  Output Projection:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.debug('  Output Cellsize: {}'.format(
            ini['SPATIAL']['cellsize']))

    # # Merge geometries
    # if ini['INPUTS']['merge_geom_flag']:
    #     merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
    #     for zone_ftr in zones['features']:
    #         zone_multipolygon = ogr.ForceToMultiPolygon(
    #             ogr.CreateGeometryFromJson(json.dumps(zone_ftr['geometry'])))
    #         for zone_polygon in zone_multipolygon:
    #             merge_geom.AddGeometry(zone_polygon)
    #     zones['features'] = [{
    #         'type': 'Feature',
    #         'id': 0,
    #         'properties': {ini['INPUTS']['zone_field']: zones['name']},
    #         'geometry': json.loads(merge_geom.ExportToJson())}]

    # # Compute convex hull of each geometry
    # for zone_i, zone_ftr in enumerate(zones['features']):
    #     zone_geom = ogr.CreateGeometryFromJson(
    #         json.dumps(zone_ftr['geometry']))
    #     zones['features'][zone_i]['geometry'] = json.loads(
    #         zone_geom.ConvexHull().ExportToJson())

    # # Save spatial reference to each feature in GeoJSON
    # for zone_i, zone_ftr in enumerate(zones['features']):
    #     zones['features'][zone_i]['properties']['crs'] = zones_crs
    #     # zones['features'][zone_i]['properties']['crs'] = zones_wkt.replace('"', '\'')
    #     # zones['features'][zone_i]['properties']['crs'] = zones_wkt

    # Save the updated GeoJSON
    with open(zone_geojson, 'w') as f:
        json.dump(zones, f)



    # Initialize Earth Engine API key
    logging.info('\nInitializing Earth Engine')
    ee.Initialize()
    utils.getinfo(ee.Number(1))

    # Build separate path/row lists for each zone
    # DEADBEEF - This is a list of all "possible" path/rows that is
    #   independent of the INI path/row settings
    logging.info('\nBuilding zone path/row lists')
    zone_pr_dict = {}

    # Iterate by blocks of 1000 zones
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
        path_row_field = 'path_row'
        wrs2_coll = ee.FeatureCollection(
                'users/cgmorton/wrs2_descending_conus_custom') \
            .filterBounds(zone_coll.geometry())

        # Extract path/row values from joined collection
        def ftr_property(ftr):
            scenes = ee.FeatureCollection(ee.List(ee.Feature(ftr).get('scenes'))) \
                .toList(100).map(lambda pr: ee.Feature(pr).get(path_row_field))
            return ee.Feature(None, {
                zone_field: ee.String(ftr.get(zone_field)),
                path_row_field: scenes})

        # Intersect the geometry and wrs2 collections
        spatialFilter = ee.Filter.intersects(
            leftField='.geo', rightField='.geo', maxError=10)
        join_coll = ee.FeatureCollection(
            ee.Join.saveAll(matchesKey='scenes') \
                .apply(zone_coll, wrs2_coll, spatialFilter) \
                .map(ftr_property))

        # Build a list of path/rows for each zone
        for f in utils.getinfo(join_coll)['features']:
            zone_name = str(f['properties'][ini['INPUTS']['zone_field']]) \
                .replace(' ', '_')
            zone_pr_dict[zone_name] = sorted(list(set(
                f['properties'][path_row_field])))

            # DEADBEEF - This is a list of all "possible" path/rows that is
            #   independentof the INI path/row settings
            # # Filter path/rows based on INI settings
            # path_rows = f['properties'][path_row_field]
            # if ini['INPUTS']['path_keep_list']:
            #     path_rows = [
            #         pr for pr in path_rows
            #         if int(pr[1:4]) in ini['INPUTS']['path_keep_list']]
            # if ini['INPUTS']['row_keep_list']:
            #     path_rows = [
            #         pr for pr in path_rows
            #         if int(pr[5:8]) in ini['INPUTS']['row_keep_list']]
            # if ini['INPUTS']['path_row_list']:
            #     path_rows = sorted(list(
            #         set(path_rows) & set(ini['INPUTS']['path_row_list'])))
            # path_row_dict[zone_name] = path_rows

    logging.debug('  Saving zone path/row dictionary')
    logging.debug('    {}'.format(zone_pr_json))
    with open(zone_pr_json, 'w') as f:
        json.dump(zone_pr_dict, f, sort_keys=True)



    # Get SCENE_ID lists for each path/row
    logging.info('\nBuilding path/row SCENE_ID json')

    # Remove the existing file if necessary
    if os.path.isfile(pr_scene_json) and overwrite_flag:
        logging.debug('  Removing existing scene ID json')
        logging.debug('    {}'.format(pr_scene_json))
        os.remove(pr_scene_json)

    # If file exists, it will only be updated
    pr_scene_dict = defaultdict(list)
    if os.path.isfile(pr_scene_json):
        logging.debug('  Reading {}'.format(pr_scene_json))
        with open(pr_scene_json, 'r') as f:
            pr_scene_dict = json.load(f)

    # Full path/row list
    logging.debug('  Full path/row list')
    path_row_set = set()
    for path_rows in zone_pr_dict.values():
        path_row_set.update(path_rows)
    path_row_list = sorted(list(path_row_set))
    logging.debug('    {}'.format(', '.join(path_row_list)))

    # Initialize the Landsat collections
    # For now, don't honor the INI filter settings
    logging.debug('  Initialize the Landsat collections')
    # landsat_args = {
    #     k: v for section in ['INPUTS']
    #     for k, v in ini[section].items()
    #     if k in [
    #         'landsat4_flag', 'landsat5_flag',
    #         'landsat7_flag', 'landsat8_flag',
    #         # 'fmask_flag', 'acca_flag', 'fmask_source',
    #         # 'start_year', 'end_year',
    #         # 'start_month', 'end_month',
    #         # 'start_doy', 'end_doy'
    #         # 'scene_id_keep_list', 'scene_id_skip_list',
    #         # 'path_keep_list', 'row_keep_list',
    #         # 'adjust_method', 'mosaic_method'
    #     ]}
    landsat_args = {}
    landsat_args['landsat4_flag'] = True
    landsat_args['landsat5_flag'] = True
    landsat_args['landsat7_flag'] = True
    landsat_args['landsat8_flag'] = True
    landsat_args['mosaic_method'] = 'none'
    landsat = ee_common.Landsat(landsat_args)

    logging.debug('  Processing path/rows')
    for pr in path_row_list:
        logging.info('  {}'.format(pr))
        path, row = list(map(int, path_row_re.match(pr).groups()))
        landsat.path_keep_list = [path]
        landsat.row_keep_list = [row]
        landsat_coll = landsat.get_collection()

        # Get new scene ID list
        scene_id_list = [
            f['properties']['SCENE_ID']
            for f in utils.getinfo(landsat_coll)['features']]

        # Update existing scene ID list with new values
        # Sort SCENE_ID list based on date then path/row
        pr_scene_dict[pr] = sorted(
            list(set(pr_scene_dict[pr]) | set(scene_id_list)),
            key=lambda x: (x.split('_')[2], x.split('_')[1]))

    logging.debug('  Saving path/row SCENE_ID lists')
    logging.debug('    {}'.format(pr_scene_json))
    with open(pr_scene_json, 'w') as f:
        json.dump(pr_scene_dict, f, sort_keys=True)



    # # Get MOSAIC_ID lists for each path/row
    # logging.info('\nBuilding path/row MOSAIC_ID json')

    # # Remove existing files if necessary
    # if os.path.isfile(pr_mosaic_json) and overwrite_flag:
    #     logging.debug('  Removing existing zone mosaic ID json')
    #     logging.debug('    {}'.format(pr_mosaic_json))
    #     os.remove(pr_mosaic_json)

    # pr_mosaic_dict = defaultdict(list)
    # if os.path.isfile(pr_mosaic_json):
    #     logging.debug('  Reading {}'.format(pr_mosaic_json))
    #     with open(pr_mosaic_json, 'r') as f:
    #         pr_mosaic_dict = json.load(f)

    # # Get the different possible combinations of rows
    # # Use a comma separate string of path/rows as the key
    # path_row_combinations = set()
    # for pr in zone_pr_dict.values():
    #     # Add the mosaiced path/row combinations
    #     path_row_combinations.update([','.join(pr)])
    #     # Add each path/row separately also
    #     path_row_combinations.update(pr)
    # # pr_zone_dict = defaultdict(list)
    # # for zone_name, pr in zone_pr_dict.items():
    # #     pr_zone_dict[','.join(pr)].append(zone_name)

    # # Process each set of path/row combinations
    # # for path_rows in pr_zone_dict.keys():
    # for path_rows in path_row_combinations:
    #     logging.debug('  {}'.format(path_rows))
    #     # Full SCENE_ID list
    #     scene_id_list = []
    #     for pr in path_rows.split(','):
    #         scene_id_list.extend(pr_scene_dict[pr])

    #     # SCENE_IDs in the same path but different rows will be merged
    #     path_row_dict = defaultdict(list)
    #     for p, r in [[pr[1:4], pr[5:8]] for pr in path_rows.split(',')]:
    #         path_row_dict[p].append(r)
    #     mosaic_id_list = []
    #     for path, rows in path_row_dict.items():
    #         mosaic_id_list.extend([
    #             '{}XXX{}'.format(scene_id[:8], scene_id[11:])
    #             for scene_id in scene_id_list
    #             if scene_id[5:8] == path and scene_id[8:11]])

    #     # Sort MOSAIC_ID list based on date then path/row
    #     pr_mosaic_dict[path_rows] = sorted(
    #         list(set(pr_mosaic_dict[path_rows]) | set(mosaic_id_list)),
    #         key=lambda x: (x.split('_')[2], x.split('_')[1]))

    # logging.debug('  Saving path/row MOSAIC_ID lists')
    # logging.debug('    {}'.format(pr_mosaic_json))
    # with open(pr_mosaic_json, 'w') as f:
    #     json.dump(pr_mosaic_dict, f, sort_keys=True)




    # # Compute list of zones associated with each SCENE_ID
    # scene_zone_dict = defaultdict(list)
    # for zone_name, scene_id_list in zone_scene_dict.items():
    #     for scene_id in scene_id_list:
    #         mosaic_id = '{}XXX{}'.format(scene_id[:8], scene_id[11:])
    #         # scene_dt = datetime.datetime.strptime(scene_id[12:], '%Y%m%d')
    #         # Compute key from landsat, year, DOY, and path
    #         # Eventually we could add possible path values
    #         # mosaic_id = '{}_{}_{}_{}'.format(
    #         #     scene_id[:4], scene_dt.year, int(scene_dt.strftime('%j')),
    #         #     int(scene_id[5:8]))
    #         scene_zone_dict[mosaic_id].append(zone_name)

    # logging.debug('  Saving image zone lists')
    # logging.debug('    {}'.format(scene_zone_json))
    # with open(scene_zone_json, 'w') as f:
    #     json.dump(scene_zone_dict, f)





    # Compute path/row list from merged geometry
    # # Merge geometries
    # merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
    # for zone_ftr in zones['features']:
    #     zone_multipolygon = ogr.ForceToMultiPolygon(
    #         ogr.CreateGeometryFromJson(json.dumps(zone_ftr['geometry'])))
    #     for zone_polygon in zone_multipolygon:
    #         merge_geom.AddGeometry(zone_polygon)
    # merge_json = json.loads(merge_geom.ExportToJson())

    # # Build a path/row list from a merged geometry
    # # Build a merged ee.geometry
    # logging.debug('\nGetting path/row list (for merged geometry)')
    # merge_geom = ee.Geometry(
    #     geo_json=merge_json, opt_proj=zones_wkt,
    #     opt_geodesic=False)

    # # Load the WRS2 custom footprint collection
    # wrs2_coll = ee.FeatureCollection('users/cgmorton/wrs2_descending_conus_custom') \
    #     .filterBounds(merge_geom)

    # # Intersect the geometry and wrs2 collections
    # spatialFilter = ee.Filter.intersects(
    #     leftField='.geo', rightField='.geo', maxError=10)
    # join_coll = ee.Join.saveAll(matchesKey='scenes').apply(
    #     ee.FeatureCollection(merge_geom), wrs2_coll, spatialFilter)
    # path_row_list = [
    #     f['properties']['path_row']
    #     for f in join_coll.first().get('scenes').getInfo()]
    # logging.debug('  Path/rows: {}'.format(', '.join(path_row_list)))

    # Explicitly filter the WRS2 collection to the target path/rows
    # wrs2_coll = wrs2_coll.filter(ee.Filter.inList('path_row', path_rows))
    # print([f['properties']['path_row'] for f in wrs2_coll.getInfo()['features']])


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine Zonal Statistics Initialize',
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
