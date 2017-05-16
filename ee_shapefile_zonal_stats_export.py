#--------------------------------
# Name:         ee_shapefile_zonal_stats_export.py
# Purpose:      Download zonal stats for shapefiles using Earth Engine
# Created       2017-05-15
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
import datetime
import json
import logging
import math
import os
import re
import requests
from subprocess import check_output
import sys
from time import sleep
from io import StringIO

import ee
import numpy as np
from osgeo import ogr
import pandas as pd

import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils
import ee_tools.wrs2 as wrs2


def ee_zonal_stats(ini_path=None, overwrite_flag=False):
    """Earth Engine Zonal Stats Export

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nEarth Engine Zonal Stats Export')

    landsat_tables_folder = 'landsat_tables'

    # Regular expression to pull out Landsat scene_id
    # landsat_re = re.compile('L[ETC][4578]\d{6}\d{4}\d{3}\D{3}\d{2}')
    # landsat_re = re.compile(
    #     'L[ETC][4578]\d{6}(?P<YEAR>\d{4})(?P<DOY>\d{3})\D{3}\d{2}')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='EXPORT')
    inputs.parse_section(ini, section='ZONAL_STATS')

    # Set all zone specific parameters into a dictionary
    zone = {}

    # These may eventually be set in the INI file
    # Currently FMASK_PCT and QA are added automatically when making
    #   landsat daily CSV
    landsat_daily_fields = [
        'ZONE_FID', 'ZONE_NAME', 'DATE', 'SCENE_ID', 'LANDSAT',
        'PATH', 'ROW', 'YEAR', 'MONTH', 'DAY', 'DOY', 'CLOUD_SCORE',
        'PIXEL_COUNT', 'PIXEL_TOTAL', 'FMASK_COUNT', 'FMASK_TOTAL',
        'TS', 'ALBEDO_SUR', 'NDVI_TOA', 'NDVI_SUR', 'EVI_SUR',
        'NDWI_GREEN_NIR_SUR', 'NDWI_GREEN_SWIR1_SUR', 'NDWI_NIR_SWIR1_SUR',
        # 'NDWI_GREEN_NIR_TOA', 'NDWI_GREEN_SWIR1_TOA', 'NDWI_NIR_SWIR1_TOA',
        # 'NDWI_SWIR1_GREEN_TOA', 'NDWI_SWIR1_GREEN_SUR',
        # 'NDWI_TOA', 'NDWI_SUR',
        'TC_BRIGHT', 'TC_GREEN', 'TC_WET']
    gridmet_daily_fields = [
        'ZONE_FID', 'ZONE_NAME', 'DATE',
        'YEAR', 'MONTH', 'DAY', 'DOY', 'WATER_YEAR', 'ETO', 'PPT']
    gridmet_monthly_fields = [
        'ZONE_FID', 'ZONE_NAME', 'DATE',
        'YEAR', 'MONTH', 'WATER_YEAR', 'ETO', 'PPT']
    pdsi_dekad_fields = [
        'ZONE_FID', 'ZONE_NAME', 'DATE',
        'YEAR', 'MONTH', 'DAY', 'DOY', 'PDSI']

    # Get ee features from shapefile
    zone_geom_list = gdc.shapefile_2_geom_list_func(
        ini['INPUTS']['zone_shp_path'], zone_field=ini['INPUTS']['zone_field'],
        reverse_flag=False)
    # zone_count = len(zone_geom_list)
    # output_fmt = '_{0:0%sd}.csv' % str(int(math.log10(zone_count)) + 1)

    # Check if the zone_names are unique
    # Eventually support merging common zone_names
    if len(set([z[1] for z in zone_geom_list])) != len(zone_geom_list):
        logging.error(
            '\nERROR: There appear to be duplicate zone ID/name values.'
            '\n  Currently, the values in "{}" must be unique.'
            '\n  Exiting.'.format(ini['INPUTS']['zone_field']))
        return False

    # Filter features by FID
    if ini['INPUTS']['fid_keep_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] in ini['INPUTS']['fid_keep_list']]
    if ini['INPUTS']['fid_skip_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] not in ini['INPUTS']['fid_skip_list']]

    # Intentionally don't apply scene_id skip/keep lists
    # Compute zonal stats for all available images
    logging.info('  Not applying scene_id keep or skip lists')
    if ini['INPUTS']['scene_id_keep_list']:
        ini['INPUTS']['scene_id_keep_list'] = []
    if ini['INPUTS']['scene_id_keep_list']:
        ini['INPUTS']['scene_id_skip_list'] = []

    # Need zone_shp_path projection to build EE geometries
    zone['osr'] = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zone['proj'] = gdc.osr_wkt(zone['osr'])
    # zone['proj'] = ee.Projection(zone['proj']).wkt().getInfo()
    # zone['proj'] = zone['proj'].replace('\n', '').replace(' ', '')
    # logging.debug('  Zone Projection: {}'.format(zone['proj']))

    # Check that shapefile has matching spatial reference
    if not gdc.matching_spatref(zone['osr'], ini['SPATIAL']['osr']):
        logging.warning('  Zone OSR:\n{}\n'.format(zone['osr']))
        logging.warning('  Output OSR:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.warning('  Zone Proj4:   {}'.format(
            zone['osr'].ExportToProj4()))
        logging.warning('  Output Proj4: {}'.format(
            ini['SPATIAL']['osr'].ExportToProj4()))
        logging.warning(
            '\nWARNING: \n'
            'The output and zone spatial references do not appear to match\n'
            'This will likely cause problems!')
        input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(
            zone['osr'].ExportToWkt()))
        logging.debug('  Output Projection:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.debug('  Output Cellsize: {}'.format(
            ini['SPATIAL']['cellsize']))

    # Initialize Earth Engine API key
    ee.Initialize()

    # Get list of path/row strings to centroid coordinates
    if ini['INPUTS']['path_row_list']:
        ini['INPUTS']['path_row_geom'] = [
            wrs2.path_row_centroids[pr]
            for pr in ini['INPUTS']['path_row_list']
            if pr in wrs2.path_row_centroids.keys()]
        ini['INPUTS']['path_row_geom'] = ee.Geometry.MultiPoint(
            ini['INPUTS']['path_row_geom'], 'EPSG:4326')
    else:
        ini['INPUTS']['path_row_geom'] = None

    # Get current running tasks before getting file lists
    tasks = utils.get_ee_tasks()

    # Get list of existing images/files
    if ini['EXPORT']['export_dest'] == 'CLOUD':
        logging.debug('\nGetting cloud storage file list')
        ini['EXPORT']['cloud_file_list'] = utils.get_bucket_files(
            ini['EXPORT']['project_name'], ini['EXPORT']['export_ws'])
        # logging.debug(ini['EXPORT']['cloud_file_list'])
    # if ini['EXPORT']['export_dest'] == 'GDRIVE':
    #     logging.debug('\nGetting Google drive file list')
    #     ini['EXPORT']['gdrive_file_list'] = [
    #         os.path.join(ini['EXPORT']['output_ws'], x)
    #         for x in os.listdir(ini['EXPORT']['output_ws'])]
    #     logging.debug(ini['EXPORT']['gdrive_file_list'])


    # Get end date of GRIDMET (if needed)
    # This could be moved to inside the INI function
    if ini['ZONAL_STATS']['gridmet_monthly_flag']:
        for i in range(1, 10):
            try:
                gridmet_end_dt = ee.Date(ee.Image(
                    ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                        .filterDate(
                            '{}-01-01'.format(ini['INPUTS']['end_year'] - 1),
                            '{}-01-01'.format(ini['INPUTS']['end_year'] + 1)) \
                        .limit(1, 'system:time_start', False) \
                        .first()).get('system:time_start')) \
                        .format('YYYY-MM-dd').getInfo()
            except Exception as e:
                logging.error('  Exception: {}, retry {}'.format(e, i))
                logging.debug('{}'.format(e))
                sleep(i ** 2)
        gridmet_end_dt = datetime.datetime.strptime(
            gridmet_end_dt, '%Y-%m-%d')
        logging.debug('    Last GRIDMET date: {}'.format(gridmet_end_dt))

    # Calculate zonal stats for each feature separately
    for zone_fid, zone_name, zone_json in zone_geom_list:
        zone['fid'] = zone_fid
        zone['name'] = zone_name.replace(' ', '_')
        zone['json'] = zone_json
        logging.info('ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))

        # Build EE geometry object for zonal stats
        zone['geom'] = ee.Geometry(
            geo_json=zone['json'], opt_proj=zone['proj'], opt_geodesic=False)
        # logging.debug('  Centroid: {}'.format(
        #     zone['geom'].centroid(100).getInfo()['coordinates']))

        # Use feature geometry to build extent, transform, and shape
        zone['extent'] = gdc.Extent(
            ogr.CreateGeometryFromJson(json.dumps(zone['json'])).GetEnvelope())
        # zone['extent'] = gdc.Extent(zone['geom'].GetEnvelope())
        zone['extent'] = zone['extent'].ogrenv_swap()
        zone['extent'] = zone['extent'].adjust_to_snap(
            'EXPAND', ini['SPATIAL']['snap_x'], ini['SPATIAL']['snap_y'],
            ini['SPATIAL']['cellsize'])
        zone['geo'] = zone['extent'].geo(ini['SPATIAL']['cellsize'])
        zone['transform'] = gdc.geo_2_ee_transform(zone['geo'])
        zone['shape'] = zone['extent'].shape(ini['SPATIAL']['cellsize'])
        logging.debug('  Zone Shape: {}'.format(zone['shape']))
        logging.debug('  Zone Transform: {}'.format(zone['transform']))
        logging.debug('  Zone Extent: {}'.format(zone['extent']))
        # logging.debug('  Zone Geom: {}'.format(zone['geom'].getInfo()))

        # Assume all pixels in all 14+2 images could be reduced
        zone['max_pixels'] = zone['shape'][0] * zone['shape'][1] * 16
        logging.debug('  Max Pixels: {}'.format(zone['max_pixels']))

        # Set output spatial reference
        # Eventually allow user to manually set these
        # output_crs = zone['proj']
        ini['EXPORT']['transform'] = zone['transform']
        logging.debug('  Output Projection: {}'.format(ini['SPATIAL']['crs']))
        logging.debug('  Output Transform: {}'.format(
            ini['EXPORT']['transform']))

        #
        zone['output_ws'] = os.path.join(
            ini['ZONAL_STATS']['output_ws'], zone['name'])
        zone['tables_ws'] = os.path.join(
            ini['ZONAL_STATS']['output_ws'], zone['name'],
            landsat_tables_folder)
        if not os.path.isdir(zone['tables_ws']):
            os.makedirs(zone['tables_ws'])

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
        logging.info('  Done')


def landsat_func(export_fields, ini, zone, tasks, overwrite_flag=False):
    """

    Args:
        export_fields ():
        ini (dict): Input file parameters
        zone (dict): Zone specific parameters
        tasks ():
        gridmet_end_dt (datetime):
        overwrite_flag (bool): if True, overwrite existing files
    """
    logging.info('  Landsat')

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
        logging.info('  {}  {}'.format(start_date, end_date))

        start_year = max(start_dt.date().year, ini['INPUTS']['start_year'])
        end_year = min(end_dt.date().year, ini['INPUTS']['end_year'])
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
        output_id = '{}_landsat{}_{}'.format(zone['name'], crs_str, year_str)
        if ini['EXPORT']['mosaic_method']:
            export_id += '_' + ini['EXPORT']['mosaic_method'].lower()
            output_id += '_' + ini['EXPORT']['mosaic_method'].lower()

        export_path = os.path.join(
            ini['EXPORT']['export_ws'], export_id + '.csv')
        output_path = os.path.join(zone['tables_ws'], output_id + '.csv')
        logging.debug('  Export: {}'.format(export_path))
        logging.debug('  Output: {}'.format(output_path))

        # There is an EE bug that appends "ee_export" to the end of CSV
        #   file names when exporting to cloud storage
        # Also, use the sharelink path for reading the csv directly
        export_cloud_name = export_id + 'ee_export.csv'
        export_cloud_path = os.path.join(
            ini['EXPORT']['export_ws'], export_cloud_name)
        export_cloud_url = 'https://storage.googleapis.com/{}/{}'.format(
            ini['EXPORT']['bucket_name'], export_cloud_name)

        if overwrite_flag:
            if export_id in tasks.keys():
                logging.debug('  Task already submitted, cancelling')
                for task in tasks[export_id]:
                    ee.data.cancelTask(task)
                del tasks[export_id]

            if (ini['EXPORT']['export_dest'] == 'GDRIVE' and
                    os.path.isfile(export_path)):
                logging.debug('  Export CSV already exists, removing')
                os.remove(export_path)
            elif (ini['EXPORT']['export_dest'] == 'CLOUD' and
                    export_cloud_name in ini['EXPORT']['file_list']):
                logging.debug('    Export image already exists')
                # # Files in cloud storage are easily overwritten
                # #   so it is unneccesary to manually remove them
                # # This would remove an existing file
                # check_output(['gsutil', 'rm', export_cloud_path])

            if os.path.isfile(output_path):
                logging.debug('  Output CSV already exists, removing')
                os.remove(output_path)

        # This should probably be moved into an else block
        #   to avoid lots of os.path.isfile calls when overwriting
        if export_id in tasks.keys():
            logging.debug('  Task already submitted, skipping')
            continue
        elif (ini['EXPORT']['export_dest'] == 'GDRIVE' and
                os.path.isfile(export_path)):
            logging.debug('  Export CSV already exists, moving')
            # Modify CSV while copying from Google Drive
            try:
                export_df = pd.read_csv(export_path)
                export_df = export_df[export_fields]
                export_df.sort_values(by=['DATE', 'ROW'], inplace=True)
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
            continue
        elif (ini['EXPORT']['export_dest'] == 'CLOUD' and
                export_cloud_name in ini['EXPORT']['cloud_file_list']):
            logging.debug('    Export file already exists, moving')
            logging.debug('    Reading {}'.format(export_cloud_url))
            try:
                export_request = requests.get(export_cloud_url).content
                export_df = pd.read_csv(
                    StringIO(export_request.decode('utf-8')))
                export_df = export_df[export_fields]
                export_df.sort_values(by=['DATE', 'ROW'], inplace=True)
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
                continue
            logging.debug('    Removing {}'.format(export_cloud_path))
            try:
                check_output(['gsutil', 'rm', export_cloud_path])
            except Exception as e:
                logging.error('Unhandled Exception')
                logging.error(str(e))
            continue
        elif os.path.isfile(output_path):
            logging.debug('  Output CSV already exists, skipping')
            continue

        landsat_args = {
            k: v for section in ['INPUTS', 'EXPORT']
            for k, v in ini[section].items()
            if k in [
                'fmask_flag', 'acca_flag', 'fmask_type',
                'start_year', 'end_year',
                'start_month', 'end_month', 'start_doy', 'end_doy',
                'scene_id_keep_list', 'scene_id_skip_list',
                'path_keep_list', 'row_keep_list', 'adjust_method']}
        landsat_args['zone_geom'] = zone['geom']
        landsat_args['start_date'] = start_date
        landsat_args['end_date'] = end_date
        if ini['INPUTS']['path_row_geom']:
            landsat_args['path_row_geom'] = ini['INPUTS']['path_row_geom']

        landsat_coll = ee_common.get_landsat_images(
            ini['INPUTS']['landsat4_flag'], ini['INPUTS']['landsat5_flag'],
            ini['INPUTS']['landsat7_flag'], ini['INPUTS']['landsat8_flag'],
            ini['EXPORT']['mosaic_method'], landsat_args)

        # Mosaic overlapping images
        if ini['EXPORT']['mosaic_method']:
            landsat_coll = ee_common.mosaic_landsat_images(
                landsat_coll, ini['EXPORT']['mosaic_method'])

        # Debug
        # print(ee.Image(landsat_coll.first()).getInfo())
        # input('ENTER')
        # if ee.Image(landsat_coll.first()).getInfo() is None:
        #     logging.info('    No images, skipping')
        #     continue

        # Calculate values and statistics
        # Build function in loop to set water year ETo/PPT values
        def landsat_zonal_stats_func(image):
            """"""
            scene_id = ee.String(image.get('SCENE_ID'))
            date = ee.Date(image.get('system:time_start'))
            doy = ee.Number(date.getRelative('day', 'year')).add(1)
            # Using zone['geom'] as the geomtry should make it
            #   unnecessary to clip also
            # Decide whether to keep old NDWI
            input_mean = ee.Image(image).select(
                [
                    'albedo_sur', 'cloud_score',
                    'evi_sur', 'ndvi_sur', 'ndvi_toa',
                    'ndwi_green_nir_sur', 'ndwi_green_swir1_sur', 'ndwi_nir_swir1_sur',
                    # 'ndwi_green_nir_toa', 'ndwi_green_swir1_toa', 'ndwi_nir_swir1_toa',
                    # 'ndwi_swir1_green_sur','ndwi_swir1_green_toa',
                    # 'ndwi_sur','ndwi_toa',
                    'row', 'tc_bright', 'tc_green', 'tc_wet', 'ts'
                ]) \
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=zone['geom'],
                    crs=ini['SPATIAL']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False, tileScale=2,
                    maxPixels=zone['max_pixels'])
            # Use surface temperature to determine masked pixels
            pixel_count = ee.Image(image) \
                .select(['ts']).gt(0).unmask().select([0], ['pixel']) \
                .reduceRegion(
                    reducer=ee.Reducer.sum().combine(
                        ee.Reducer.count(), '', True),
                    geometry=zone['geom'],
                    crs=ini['SPATIAL']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False, tileScale=2,
                    maxPixels=zone['max_pixels'])
            # Fmask 0 is clear and 1 is water
            fmask_count = ee.Image(image) \
                .select(['fmask']).gt(1).select([0], ['fmask']) \
                .reduceRegion(
                    reducer=ee.Reducer.sum().combine(
                        ee.Reducer.count(), '', True),
                    geometry=zone['geom'],
                    crs=ini['SPATIAL']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False, tileScale=2,
                    maxPixels=zone['max_pixels'])
            return ee.Feature(
                None,
                {
                    'ZONE_NAME': zone['name'],
                    'ZONE_FID': zone['fid'],
                    'SCENE_ID': scene_id.slice(0, 16),
                    'LANDSAT': scene_id.slice(0, 3),
                    'PATH': ee.Number(scene_id.slice(3, 6)),
                    'ROW': ee.Number(input_mean.get('row')),
                    # 'ROW': ee.Number(scene_id.slice(6, 9)),
                    'DATE': date.format('YYYY-MM-dd'),
                    'YEAR': date.get('year'),
                    'MONTH': date.get('month'),
                    'DAY': date.get('day'),
                    'DOY': doy,
                    'PIXEL_COUNT': pixel_count.get('pixel_sum'),
                    'PIXEL_TOTAL': pixel_count.get('pixel_count'),
                    'FMASK_COUNT': fmask_count.get('fmask_sum'),
                    'FMASK_TOTAL': fmask_count.get('fmask_count'),
                    'CLOUD_SCORE': input_mean.get('cloud_score'),
                    'ALBEDO_SUR': input_mean.get('albedo_sur'),
                    'EVI_SUR': input_mean.get('evi_sur'),
                    'NDVI_SUR': input_mean.get('ndvi_sur'),
                    'NDVI_TOA': input_mean.get('ndvi_toa'),
                    'NDWI_GREEN_NIR_SUR': input_mean.get('ndwi_green_nir_sur'),
                    # 'NDWI_GREEN_NIR_TOA': input_mean.get('ndwi_green_nir_toa'),
                    'NDWI_GREEN_SWIR1_SUR': input_mean.get('ndwi_green_swir1_sur'),
                    # 'NDWI_GREEN_SWIR1_TOA': input_mean.get('ndwi_green_swir1_toa'),
                    'NDWI_NIR_SWIR1_SUR': input_mean.get('ndwi_nir_swir1_sur'),
                    # 'NDWI_NIR_SWIR1_TOA': input_mean.get('ndwi_nir_swir1_toa'),
                    # 'NDWI_SWIR1_GREEN_SUR': input_mean.get('ndwi_swir1_green_sur'),
                    # 'NDWI_SWIR1_GREEN_TOA': input_mean.get('ndwi_swir1_green_toa'),
                    # 'NDWI_SUR': input_mean.get('ndwi_swir1_green_sur'),
                    # 'NDWI_TOA': input_mean.get('ndwi_swir1_green_toa'),
                    'TC_BRIGHT': input_mean.get('tc_bright'),
                    'TC_GREEN': input_mean.get('tc_green'),
                    'TC_WET': input_mean.get('tc_wet'),
                    'TS': input_mean.get('ts')
                })
        stats_coll = landsat_coll.map(landsat_zonal_stats_func)

        # # DEADBEEF - Test the function for a single image
        # stats_info = landsat_zonal_stats_func(
        #     ee.Image(landsat_coll.first())).getInfo()
        # for k, v in sorted(stats_info['properties'].items()):
        #     logging.info('{:24s}: {}'.format(k, v))
        # input('ENTER')
        # return False

        # # DEADBEEF - Print the stats info to the screen
        # stats_info = stats_coll.getInfo()
        # import pprint
        # pp = pprint.PrettyPrinter(indent=4)
        # for ftr in stats_info['features']:
        #     pp.pprint(ftr)
        # input('ENTER')
        # return False

        # task = ee.batch.Export.table.toDrive(
        #     collection=stats_coll,
        #     description=export_id,
        #     folder=ini['EXPORT']['export_folder'],
        #     fileNamePrefix=export_id,
        #     fileFormat='CSV')
        # task.start()
        # return False

        #
        logging.debug('  Building export task')
        if ini['EXPORT']['export_dest'] == 'GDRIVE':
            task = ee.batch.Export.table.toDrive(
                collection=stats_coll,
                description=export_id,
                folder=ini['EXPORT']['export_folder'],
                fileNamePrefix=export_id,
                fileFormat='CSV')
        elif ini['EXPORT']['export_dest'] == 'CLOUD':
            task = ee.batch.Export.table.toCloudStorage(
                collection=stats_coll,
                description=export_id,
                bucket=ini['EXPORT']['bucket_name'],
                fileNamePrefix='{}'.format(export_id.replace('-', '')),
                # fileNamePrefix=export_id,
                fileFormat='CSV')

        # Download the CSV to your Google Drive
        logging.debug('  Starting export task')
        for i in range(1, 10):
            try:
                task.start()
                break
            except Exception as e:
                logging.error('  Exception: {}, retry {}'.format(e, i))
                logging.debug('{}'.format(e))
                sleep(i ** 2)
        # logging.debug('  Status: {}'.format(task.status()))
        # logging.debug('  Active: {}'.format(task.active()))

    # Combine/merge annual files into a single CSV
    # This code shouldn't be run until the tables have exported
    # Because that is difficult to know, the script will "silently"
    #   skip past any file that doesn't exist.
    logging.debug('\n  Merging annual Landsat CSV files')
    output_df = None
    for year in range(iter_start_year, iter_end_year, iter_years):
        start_dt = datetime.datetime(year, 1, 1)
        end_dt = (
            datetime.datetime(year + iter_years, 1, 1) -
            datetime.timedelta(0, 1))
        logging.debug('  {}  {}'.format(
            start_dt.date().isoformat(), end_dt.date().isoformat()))

        start_year = max(start_dt.date().year, ini['INPUTS']['start_year'])
        end_year = min(end_dt.date().year, ini['INPUTS']['end_year'])
        if iter_years > 1:
            year_str = '{}_{}'.format(start_year, end_year)
        else:
            year_str = '{}'.format(year)

        if ini['EXPORT']['mosaic_method']:
            year_str += '_' + ini['EXPORT']['mosaic_method'].lower()

        # Include EPSG code in export and output names
        if 'EPSG' in ini['SPATIAL']['crs']:
            crs_str = '_' + ini['SPATIAL']['crs'].replace(':', '').lower()
        else:
            crs_str = ''

        # There is a special case where the user may computes zonal stats in
        #   separate WGS84 Zone XX projections to match up with the raw Landsat
        #   imagery, but then they want a single output file of the results.
        # Look for any files with the correct naming but a different 326XX
        #   EPSG code.
        for input_name in os.listdir(zone['tables_ws']):
            if ini['SPATIAL']['crs'].startswith('EPSG:326'):
                input_re = re.compile('{}_landsat_epsg326\d{}_{}.csv'.format(
                    zone['name'], '{2}', year_str))
            else:
                input_re = re.compile('{}_landsat{}_{}.csv'.format(
                    zone['name'], crs_str, year_str))
            if not input_re.match(input_name):
                continue

            logging.info('    {}'.format(input_name))
            input_path = os.path.join(zone['tables_ws'], input_name)
            try:
                input_df = pd.read_csv(input_path)
            except Exception as e:
                logging.debug('    Error reading: {}'.format(
                    os.path.basename(input_path)))
                # input('ENTER')
                continue

            # # Remove 0 pixel count rows
            # if 'PIXEL_COUNT' in list(input_df.columns.values):
            #     input_df = input_df[input_df['PIXEL_COUNT'] > 0]

            # Add QA/QC bands?
            input_df['FMASK_PCT'] = (
                input_df['FMASK_COUNT'].astype(np.float) /
                input_df['FMASK_TOTAL'])
            input_df['QA'] = 0

            try:
                output_df = output_df.append(input_df)
            except:
                output_df = input_df.copy()

    if output_df is not None and not output_df.empty:
        output_path = os.path.join(
            zone['output_ws'], '{}_landsat_daily.csv'.format(zone['name']))
        logging.debug('  {}'.format(output_path))
        output_df.sort_values(by=['DATE', 'ROW'], inplace=True)
        output_df.to_csv(
            output_path, index=False,
            columns=export_fields + ['FMASK_PCT', 'QA'])
    else:
        logging.info('  Empty output dataframe, the CSV files may not be ready')


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

    # Export GRIDMET zonal stats
    # Insert one additional year the beginning for water year totals
    end_date = min(
        datetime.datetime(ini['INPUTS']['end_year'] + 1, 1, 1).strftime('%Y-%m-%d'),
        datetime.datetime.today().strftime('%Y-%m-%d'))
    gridmet_coll = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
        .select(['pet', 'pr'], ['eto', 'ppt']) \
        .filterDate('{}-01-01'.format(ini['INPUTS']['start_year'] - 1), end_date)

    def negative_adjust(image):
        # GRIDMET PPT zero values are
        return image.where(image.lt(0), 0)
    gridmet_coll = gridmet_coll.map(negative_adjust)

    export_id = '{}_{}_gridmet_daily'.format(
        os.path.splitext(ini['INPUTS']['zone_filename'])[0],
        zone['name'].replace(' ', '_').lower())
    output_id = '{}_gridmet_daily'.format(zone['name'].replace(' ', '_'))

    export_path = os.path.join(
        ini['EXPORT']['export_ws'], export_id + '.csv')
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('  Export: {}'.format(export_path))
    logging.debug('  Output: {}'.format(output_path))

    # There is an EE bug that appends "ee_export" to the end of CSV
    #   file names when exporting to cloud storage
    # Also, use the sharelink path for reading the csv directly
    export_cloud_name = export_id + 'ee_export.csv'
    export_cloud_path = os.path.join(
        ini['EXPORT']['export_ws'], export_cloud_name)
    export_cloud_url = 'https://storage.googleapis.com/{}/{}'.format(
        ini['EXPORT']['bucket_name'], export_cloud_name)

    if overwrite_flag:
        if export_id in tasks.keys():
            logging.debug('  Task already submitted, cancelling')
            for task in tasks[export_id]:
                ee.data.cancelTask(task)
            del tasks[export_id]

        if (ini['EXPORT']['export_dest'] == 'GDRIVE' and
                os.path.isfile(export_path)):
            logging.debug('  Export CSV already exists, removing')
            os.remove(export_path)
        elif (ini['EXPORT']['export_dest'] == 'CLOUD' and
                export_cloud_name in ini['EXPORT']['file_list']):
            logging.debug('    Export image already exists')
            # # Files in cloud storage are easily overwritten
            # #   so it is unneccesary to manually remove them
            # # This would remove an existing file
            # check_output(['gsutil', 'rm', export_path])

        if os.path.isfile(output_path):
            logging.debug('  Output CSV already exists, removing')
            os.remove(output_path)

    # This should probably be moved into an else block
    #   to avoid lots of os.path.isfile calls when overwriting
    if export_id in tasks.keys():
        logging.debug('  Task already submitted, skipping')
        return True
    elif (ini['EXPORT']['export_dest'] == 'GDRIVE' and
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
    elif (ini['EXPORT']['export_dest'] == 'CLOUD' and
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
        logging.debug('  Output CSV already exists, skipping')
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
                maxPixels=zone['max_pixels'])
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

    logging.debug('  Building export task')
    if ini['EXPORT']['export_dest'] == 'GDRIVE':
        task = ee.batch.Export.table.toDrive(
            collection=stats_coll,
            description=export_id,
            folder=ini['EXPORT']['export_folder'],
            fileNamePrefix=export_id,
            fileFormat='CSV')
    elif ini['EXPORT']['export_dest'] == 'CLOUD':
        task = ee.batch.Export.table.toCloudStorage(
            collection=stats_coll,
            description=export_id,
            bucket=ini['EXPORT']['bucket_name'],
            fileNamePrefix='{}'.format(export_id.replace('-', '')),
            # fileNamePrefix=export_id,
            fileFormat='CSV')

    logging.debug('  Starting export task')
    for i in range(1, 10):
        try:
            task.start()
            break
        except Exception as e:
            logging.error('  Exception: {}, retry {}'.format(e, i))
            logging.debug('{}'.format(e))
            sleep(i ** 2)
    # logging.debug('  Status: {}'.format(task.status()))
    # logging.debug('  Active: {}'.format(task.active()))


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

    # Export GRIDMET zonal stats
    # Insert one additional year the beginning for water year totals
    # Compute monthly sums of GRIDMET
    def monthly_sum(start_dt):
        month_sum = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
            .select(['pet', 'pr']) \
            .filterDate(
                ee.Date(start_dt),
                ee.Date(start_dt).advance(1, 'month')) \
            .sum()
        return ee.Image(month_sum) \
            .select([0, 1], ['eto', 'ppt']) \
            .set('system:time_start', ee.Date(start_dt).millis())

    # # DEADBEEF - It is really slow to make this call inside the zone loop
    #   Moved the getInfo outside of the loop in the main function
    # # Get the last GRIDMET date from the collection
    # for i in range(1, 10):
    #     try:
    #         gridmet_end_dt = ee.Date(ee.Image(
    #             ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
    #                 .filterDate(
    #                     '{}-01-01'.format(ini['INPUTS']['end_year'] - 1),
    #                     '{}-01-01'.format(ini['INPUTS']['end_year'] + 1)) \
    #                 .limit(1, 'system:time_start', False) \
    #                 .first()).get('system:time_start')) \
    #                 .format('YYYY-MM-dd').getInfo()
    #     except Exception as e:
    #         logging.error('  Exception: {}, retry {}'.format(e, i))
    #         logging.debug('{}'.format(e))
    #         sleep(i ** 2)
    # gridmet_end_dt = datetime.datetime.strptime(
    #     gridmet_end_dt, '%Y-%m-%d')
    # logging.debug('    Last GRIDMET date: {}'.format(gridmet_end_dt))

    gridmet_ym_list = [
        [y, m]
        for y in range(
            ini['INPUTS']['start_year'] - 1, ini['INPUTS']['end_year'] + 1)
        for m in range(1, 13)
        if datetime.datetime(y, m, 1) <= gridmet_end_dt]
    gridmet_dt_list = ee.List([
        ee.Date.fromYMD(y, m, 1) for y, m in gridmet_ym_list])
    gridmet_coll = ee.ImageCollection.fromImages(
        gridmet_dt_list.map(monthly_sum))

    export_id = '{}_{}_gridmet_monthly'.format(
        os.path.splitext(ini['INPUTS']['zone_filename'])[0],
        zone['name'].replace(' ', '_').lower())
    output_id = '{}_gridmet_monthly'.format(
        zone['name'].replace(' ', '_'))

    export_path = os.path.join(
        ini['EXPORT']['export_ws'], export_id + '.csv')
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('  Export: {}'.format(export_path))
    logging.debug('  Output: {}'.format(output_path))

    # There is an EE bug that appends "ee_export" to the end of CSV
    #   file names when exporting to cloud storage
    # Also, use the sharelink path for reading the csv directly
    export_cloud_name = export_id + 'ee_export.csv'
    export_cloud_path = os.path.join(
        ini['EXPORT']['export_ws'], export_cloud_name)
    export_cloud_url = 'https://storage.googleapis.com/{}/{}'.format(
        ini['EXPORT']['bucket_name'], export_cloud_name)

    if overwrite_flag:
        if export_id in tasks.keys():
            logging.debug('  Task already submitted, cancelling')
            for task in tasks[export_id]:
                ee.data.cancelTask(task)
            del tasks[export_id]

        if (ini['EXPORT']['export_dest'] == 'GDRIVE' and
                os.path.isfile(export_path)):
            logging.debug('  Export CSV already exists, removing')
            os.remove(export_path)
        elif (ini['EXPORT']['export_dest'] == 'CLOUD' and
                export_cloud_name in ini['EXPORT']['file_list']):
            logging.debug('    Export image already exists')
            # # Files in cloud storage are easily overwritten
            # #   so it is unneccesary to manually remove them
            # # This would remove an existing file
            # check_output(['gsutil', 'rm', export_path])

        if os.path.isfile(output_path):
            logging.debug('  Output CSV already exists, removing')
            os.remove(output_path)

    # This should probably be moved into an else block
    #   to avoid lots of os.path.isfile calls when overwriting
    if export_id in tasks.keys():
        logging.debug('  Task already submitted, skipping')
        return True
    elif (ini['EXPORT']['export_dest'] == 'GDRIVE' and
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
    elif (ini['EXPORT']['export_dest'] == 'CLOUD' and
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
        logging.debug('  Output CSV already exists, skipping')
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
                maxPixels=zone['max_pixels'])
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

    logging.debug('  Building export task')
    if ini['EXPORT']['export_dest'] == 'GDRIVE':
        task = ee.batch.Export.table.toDrive(
            collection=stats_coll,
            description=export_id,
            folder=ini['EXPORT']['export_folder'],
            fileNamePrefix=export_id,
            fileFormat='CSV')
    elif ini['EXPORT']['export_dest'] == 'CLOUD':
        task = ee.batch.Export.table.toCloudStorage(
            collection=stats_coll,
            description=export_id,
            bucket=ini['EXPORT']['bucket_name'],
            fileNamePrefix='{}'.format(export_id.replace('-', '')),
            # fileNamePrefix=export_id,
            fileFormat='CSV')

    logging.debug('  Starting export task')
    for i in range(1, 10):
        try:
            task.start()
            break
        except Exception as e:
            logging.error('  Exception: {}, retry {}'.format(e, i))
            logging.debug('{}'.format(e))
            sleep(i ** 2)
    # logging.debug('  Status: {}'.format(task.status()))
    # logging.debug('  Active: {}'.format(task.active()))


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
        zone['name'].replace(' ', '_').lower())
    output_id = '{}_pdsi_dekad'.format(
        zone['name'].replace(' ', '_'))

    export_path = os.path.join(
        ini['EXPORT']['export_ws'], export_id + '.csv')
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('  Export: {}'.format(export_path))
    logging.debug('  Output: {}'.format(output_path))

    # There is an EE bug that appends "ee_export" to the end of CSV
    #   file names when exporting to cloud storage
    # Also, use the sharelink path for reading the csv directly
    export_cloud_name = export_id + 'ee_export.csv'
    export_cloud_path = os.path.join(
        ini['EXPORT']['export_ws'], export_cloud_name)
    export_cloud_url = 'https://storage.googleapis.com/{}/{}'.format(
        ini['EXPORT']['bucket_name'], export_cloud_name)

    if overwrite_flag:
        if export_id in tasks.keys():
            logging.debug('  Task already submitted, cancelling')
            for task in tasks[export_id]:
                ee.data.cancelTask(task)
            del tasks[export_id]

        if (ini['EXPORT']['export_dest'] == 'GDRIVE' and
                os.path.isfile(export_path)):
            logging.debug('  Export CSV already exists, removing')
            os.remove(export_path)
        elif (ini['EXPORT']['export_dest'] == 'CLOUD' and
                export_cloud_name in ini['EXPORT']['file_list']):
            logging.debug('    Export image already exists')
            # # Files in cloud storage are easily overwritten
            # #   so it is unneccesary to manually remove them
            # # This would remove an existing file
            # check_output(['gsutil', 'rm', export_path])

        if os.path.isfile(output_path):
            logging.debug('  Output CSV already exists, removing')
            os.remove(output_path)

    # This should probably be moved into an else block
    #   to avoid lots of os.path.isfile calls when overwriting
    if export_id in tasks.keys():
        logging.debug('  Task already submitted, skipping')
        return True
    elif (ini['EXPORT']['export_dest'] == 'GDRIVE' and
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
    elif (ini['EXPORT']['export_dest'] == 'CLOUD' and
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
        logging.debug('  Output CSV already exists, skipping')
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
                maxPixels=zone['max_pixels'])
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
    if ini['EXPORT']['export_dest'] == 'GDRIVE':
        task = ee.batch.Export.table.toDrive(
            collection=stats_coll,
            description=export_id,
            folder=ini['EXPORT']['export_folder'],
            fileNamePrefix=export_id,
            fileFormat='CSV')
    elif ini['EXPORT']['export_dest'] == 'CLOUD':
        task = ee.batch.Export.table.toCloudStorage(
            collection=stats_coll,
            description=export_id,
            bucket=ini['EXPORT']['bucket_name'],
            fileNamePrefix='{}'.format(export_id.replace('-', '')),
            # fileNamePrefix=export_id,
            fileFormat='CSV')

    logging.debug('  Starting export task')
    for i in range(1, 10):
        try:
            task.start()
            break
        except Exception as e:
            logging.error('  Exception: {}, retry {}'.format(e, i))
            # logging.debug('{}'.format(e))
            sleep(i ** 2)
    # logging.debug('  Status: {}'.format(task.status()))
    # logging.debug('  Active: {}'.format(task.active()))


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine Zonal Statistics',
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

    ee_zonal_stats(ini_path=args.ini, overwrite_flag=args.overwrite)
