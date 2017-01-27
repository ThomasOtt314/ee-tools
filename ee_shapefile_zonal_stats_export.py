#--------------------------------
# Name:         ee_shapefile_zonal_stats_export.py
# Purpose:      Download zonal stats for shapefiles using Earth Engine
# Author:       Charles Morton
# Created       2017-01-27
# Python:       2.7
#--------------------------------

import argparse
from collections import defaultdict
import datetime
import json
import logging
import math
import os
import sys

import ee
import numpy as np
from osgeo import ogr
import pandas as pd

import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.ini_common as ini_common
import ee_tools.python_common as python_common


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
    # ini = ini_common.ini_parse(ini_path, section='ZONAL_STATS')
    ini = ini_common.read(ini_path)
    ini_common.parse_section(ini, section='INPUTS')
    ini_common.parse_section(ini, section='EXPORT')
    ini_common.parse_section(ini, section='ZONAL_STATS')

    # Set all zone specific parameters into a dictionary
    zone = {}

    # These may eventually be set in the INI file
    landsat_daily_fields = [
        ini['INPUTS']['zone_field'].upper(), 'DATE', 'SCENE_ID', 'LANDSAT',
        'PATH', 'ROW', 'YEAR', 'MONTH', 'DAY', 'DOY',
        'PIXEL_COUNT', 'FMASK_COUNT', 'DATA_COUNT', 'CLOUD_SCORE',
        'TS', 'ALBEDO_SUR', 'NDVI_TOA', 'NDVI_SUR', 'EVI_SUR',
        'NDWI_GREEN_NIR_SUR', 'NDWI_GREEN_SWIR1_SUR', 'NDWI_NIR_SWIR1_SUR',
        # 'NDWI_GREEN_NIR_TOA', 'NDWI_GREEN_SWIR1_TOA', 'NDWI_NIR_SWIR1_TOA',
        # 'NDWI_SWIR1_GREEN_TOA', 'NDWI_SWIR1_GREEN_SUR',
        # 'NDWI_TOA', 'NDWI_SUR',
        'TC_BRIGHT', 'TC_GREEN', 'TC_WET']
    gridmet_daily_fields = [
        ini['INPUTS']['zone_field'].upper(), 'DATE',
        'YEAR', 'MONTH', 'DAY', 'DOY', 'WATER_YEAR', 'ETO', 'PPT']
    gridmet_monthly_fields = [
        ini['INPUTS']['zone_field'].upper(), 'DATE',
        'YEAR', 'MONTH', 'WATER_YEAR', 'ETO', 'PPT']
    pdsi_dekad_fields = [
        ini['INPUTS']['zone_field'].upper(), 'DATE',
        'YEAR', 'MONTH', 'DAY', 'DOY', 'PDSI']

    # Get ee features from shapefile
    zone_geom_list = gdc.shapefile_2_geom_list_func(
        ini['INPUTS']['zone_path'], zone_field=ini['INPUTS']['zone_field'],
        reverse_flag=False)
    # zone_count = len(zone_geom_list)
    # output_fmt = '_{0:0%sd}.csv' % str(int(math.log10(zone_count)) + 1)

    # Filter features by FID
    if ini['INPUTS']['fid_keep_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] in ini['INPUTS']['fid_keep_list']]
    if ini['INPUTS']['fid_skip_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] not in ini['INPUTS']['fid_skip_list']]

    # Need zone_path projection to build EE geometries
    zone['osr'] = gdc.feature_path_osr(ini['INPUTS']['zone_path'])
    zone['proj'] = gdc.osr_proj(zone['osr'])
    # zone['proj'] = ee.Projection(zone['proj']).wkt().getInfo()
    # zone['proj'] = zone['proj'].replace('\n', '').replace(' ', '')
    # logging.debug('  Zone Projection: {}'.format(zone['proj']))

    # Check that shapefile has matching spatial reference
    if not gdc.matching_spatref(zone['osr'], ini['EXPORT']['osr']):
        logging.warning('  Zone OSR:\n{}\n'.format(zone['osr']))
        logging.warning('  Output OSR:\n{}\n'.format(
            ini['EXPORT']['osr'].ExportToWkt()))
        logging.warning('  Zone Proj4:   {}'.format(
            zone['osr'].ExportToProj4()))
        logging.warning('  Output Proj4: {}'.format(
            ini['EXPORT']['osr'].ExportToProj4()))
        logging.warning(
            '\nWARNING: \n'
            'The output and zone spatial references do not appear to match\n'
            'This will likely cause problems!')
        raw_input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(
            zone['osr'].ExportToWkt()))
        logging.debug('  Output Projection:\n{}\n'.format(
            ini['EXPORT']['osr'].ExportToWkt()))
        logging.debug('  Output Cellsize: {}'.format(
            ini['EXPORT']['cellsize']))


    # Initialize Earth Engine API key
    ee.Initialize()

    # Get current running tasks
    logging.debug('\nRunning tasks')
    tasks = defaultdict(list)
    for t in ee.data.getTaskList():
        if t['state'] in ['RUNNING', 'READY']:
            logging.debug('  {}'.format(t['description']))
            tasks[t['description']].append(t['id'])
            # tasks[t['id']] = t['description']


    # Calculate zonal stats for each feature separately
    for fid, zone_str, zone_json in sorted(zone_geom_list):
        logging.info('ZONE: {} (FID: {})'.format(zone_str, fid))

        if (not ini['INPUTS']['zone_field'] or
                ini['INPUTS']['zone_field'].upper() == 'FID'):
            zone['name'] = 'fid_' + zone_str
        else:
            zone['name'] = zone_str.lower().replace(' ', '_')

        # Build EE geometry object for zonal stats
        zone['geom'] = ee.Geometry(
            geo_json=zone_json, opt_proj=zone['proj'], opt_geodesic=False)
        logging.debug('  Centroid: {}'.format(
            zone['geom'].centroid(100).getInfo()['coordinates']))

        # Use feature geometry to build extent, transform, and shape
        zone['extent'] = gdc.Extent(
            ogr.CreateGeometryFromJson(json.dumps(zone_json)).GetEnvelope())
        # zone['extent'] = gdc.Extent(zone['geom'].GetEnvelope())
        zone['extent'] = zone['extent'].ogrenv_swap()
        zone['extent'].adjust_to_snap(
            'EXPAND', ini['EXPORT']['snap_x'], ini['EXPORT']['snap_y'],
            ini['EXPORT']['cellsize'])
        zone['geo'] = zone['extent'].geo(ini['EXPORT']['cellsize'])
        zone['transform'] = gdc.geo_2_ee_transform(zone['geo'])
        zone['shape'] = zone['extent'].shape(ini['EXPORT']['cellsize'])
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
        logging.debug('  Output Projection: {}'.format(ini['EXPORT']['crs']))
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
                gridmet_monthly_fields, ini, zone, tasks, overwrite_flag)
        if ini['ZONAL_STATS']['pdsi_flag']:
            pdsi_func(pdsi_dekad_fields, ini, zone, tasks, overwrite_flag)


def landsat_func(export_fields, ini, zone, tasks, overwrite_flag=False):
    """

    Args:
        ini (dict): Input file parameters
        zone (dict): Zone specific parameters
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
    for year in xrange(iter_start_year, iter_end_year, iter_years):
        start_dt = datetime.datetime(year, 1, 1)
        end_dt = (
            datetime.datetime(year + iter_years, 1, 1) -
            datetime.timedelta(0, 1))
        start_date = start_dt.date().isoformat()
        end_date = end_dt.date().isoformat()
        logging.info("  {}  {}".format(start_date, end_date))

        start_year = max(start_dt.date().year, ini['INPUTS']['start_year'])
        end_year = min(end_dt.date().year, ini['INPUTS']['end_year'])
        if iter_years > 1:
            year_str = '{}_{}'.format(start_year, end_year)
        else:
            year_str = '{}'.format(year)
        # logging.debug("  {}  {}".format(start_year, end_year))

        # Export Landsat zonal stats
        export_id = '{}_{}_landsat_{}'.format(
            ini['INPUTS']['zone_filename'], zone['name'], year_str)
        # export_id = '{}_{}_landsat_{}'.format(
        #     os.path.splitext(ini['INPUTS']['zone_filename'])[0], zone_str, year_str)
        output_id = '{}_landsat_{}'.format(zone['name'], year_str)
        if ini['EXPORT']['mosaic_method']:
            export_id += '_' + ini['EXPORT']['mosaic_method'].lower()
            output_id += '_' + ini['EXPORT']['mosaic_method'].lower()
        export_path = os.path.join(
            ini['EXPORT']['export_ws'], export_id + '.csv')
        output_path = os.path.join(zone['tables_ws'], output_id + '.csv')
        temp_path = os.path.join(
            ini['EXPORT']['export_ws'],
            os.path.basename(ini['ZONAL_STATS']['output_ws']),
            export_id + '.csv')
        logging.debug('  Export: {}'.format(export_path))
        logging.debug('  Output: {}'.format(output_path))
        logging.debug('  Temp:   {}'.format(temp_path))

        if overwrite_flag:
            if export_id in tasks.keys():
                logging.debug('  Task already submitted, cancelling')
                for task in tasks[export_id]:
                    ee.data.cancelTask(task)
                del tasks[export_id]
            if os.path.isfile(export_path):
                logging.debug('  Export CSV already exists, removing')
                os.remove(export_path)
            if os.path.isfile(output_path):
                logging.debug('  Output CSV already exists, removing')
                os.remove(output_path)

        if os.path.isfile(export_path):
            logging.debug('  Export CSV already exists, moving')
            # Modify CSV while copying from Google Drive
            export_df = pd.read_csv(export_path)
            if ('DATA_COUNT' not in export_df.columns.values and
                    'PIXEL_COUNT' in export_df.columns.values and
                    'FMASK_COUNT' in export_df.columns.values):
                export_df['DATA_COUNT'] = (
                    export_df['PIXEL_COUNT'] - export_df['FMASK_COUNT'])
            export_df = export_df[export_fields]
            export_df.sort_values(by=['DATE', 'ROW'], inplace=True)

            # Change fid zone strings back to integer values
            if zone['name'].startswith('fid_'):
                export_df[ini['INPUTS']['zone_field']] = int(zone['name'][4:])
                export_df[ini['INPUTS']['zone_field']] = export_df[
                    ini['INPUTS']['zone_field']].astype(np.int)
            export_df.to_csv(
                output_path, index=False,
                columns=export_fields)

            # DEADBEEF - For now, also move to a temp Google Drive folder
            # shutil.move(export_path, temp_path)
            os.remove(export_path)
            continue
        elif os.path.isfile(output_path):
            logging.debug('  Output CSV already exists, skipping')
            continue
        elif export_id in tasks.keys():
            logging.debug('  Task already submitted, skipping')
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
        # raw_input('ENTER')
        # if ee.Image(landsat_coll.first()).getInfo() is None:
        #     logging.info('    No images, skipping')
        #     continue

        # Calculate values and statistics
        # Build function in loop to set water year ETo/PPT values
        def landsat_zonal_stats_func(image):
            """"""
            scene_id = ee.String(image.get('SCENE_ID'))
            date = ee.Date(image.get("system:time_start"))
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
                    'tc_bright', 'tc_green', 'tc_wet', 'ts'
                ]) \
                .reduceRegion(
                    ee.Reducer.mean(), geometry=zone['geom'],
                    crs=ini['EXPORT']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False, tileScale=2,
                    maxPixels=zone['max_pixels'])
            input_cnt = ee.Image(image) \
                .select(['ts'], ['data_count']) \
                .reduceRegion(
                    ee.Reducer.count(), geometry=zone['geom'],
                    crs=ini['EXPORT']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False, tileScale=2,
                    maxPixels=zone['max_pixels'])
            total_cnt = ee.Image(image) \
                .select(['fmask'], ['pixel_count']) \
                .reduceRegion(
                    ee.Reducer.count(), geometry=zone['geom'],
                    crs=ini['EXPORT']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False, tileScale=2,
                    maxPixels=zone['max_pixels'])
            # Fmask 0 is clear and 1 is water
            fmask_cnt = ee.Image(image) \
                .select(['fmask']).gt(1).select([0], ['fmask_count']) \
                .reduceRegion(
                    ee.Reducer.sum(), geometry=zone['geom'],
                    crs=ini['EXPORT']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False, tileScale=2,
                    maxPixels=zone['max_pixels'])
            return ee.Feature(
                None,
                {
                    ini['INPUTS']['zone_field'].upper(): zone['name'],
                    'SCENE_ID': scene_id.slice(0, 16),
                    'LANDSAT': scene_id.slice(0, 3),
                    'PATH': ee.Number(scene_id.slice(3, 6)),
                    'ROW': ee.Number(scene_id.slice(6, 9)),
                    'DATE': date.format("YYYY-MM-dd"),
                    'YEAR': date.get('year'),
                    'MONTH': date.get('month'),
                    'DAY': date.get('day'),
                    'DOY': doy,
                    'PIXEL_COUNT': total_cnt.get('pixel_count'),
                    'FMASK_COUNT': fmask_cnt.get('fmask_count'),
                    'DATA_COUNT': input_cnt.get('data_count'),
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
        # raw_input('ENTER')

        # # DEADBEEF - Print the stats info to the screen
        # stats_info = stats_coll.getInfo()
        # import pprint
        # pp = pprint.PrettyPrinter(indent=4)
        # for ftr in stats_info['features']:
        #     pp.pprint(ftr)
        # raw_input('ENTER')

        # task = ee.batch.Export.table.toDrive(
        #     stats_coll,
        #     description=export_id,
        #     folder=ini['EXPORT']['export_folder'],
        #     fileNamePrefix=export_id,
        #     fileFormat='CSV')
        # task.start()
        # return False

        # Download the CSV to your Google Drive
        logging.debug('  Starting export task')
        i = 0
        while i <= 2:
            try:
                task = ee.batch.Export.table.toDrive(
                    stats_coll,
                    description=export_id,
                    folder=ini['EXPORT']['export_folder'],
                    fileNamePrefix=export_id,
                    fileFormat='CSV')
                task.start()
                logging.debug('  Active: {}'.format(task.active()))
                # logging.debug('  Status: {}'.format(task.status()))
                break
            except Exception as e:
                logging.error('  EE Exception submitting task, retrying')
                logging.debug('{}'.format(e))
                i += 1


    # Combine/merge annual files into a single CSV
    # This code shouldn't be run until the tables have exported
    # Because that is difficult to know, the script will "silently"
    #   skip past any file that doesn't exist.
    logging.debug('\n  Merging annual Landsat CSV files')
    output_df = None
    for year in xrange(iter_start_year, iter_end_year, iter_years):
        start_dt = datetime.datetime(year, 1, 1)
        end_dt = (
            datetime.datetime(year + iter_years, 1, 1) -
            datetime.timedelta(0, 1))
        logging.debug("  {}  {}".format(
            start_dt.date().isoformat(), end_dt.date().isoformat()))

        start_year = max(start_dt.date().year, ini['INPUTS']['start_year'])
        end_year = min(end_dt.date().year, ini['INPUTS']['end_year'])
        if iter_years > 1:
            year_str = '{}_{}'.format(start_year, end_year)
        else:
            year_str = '{}'.format(year)

        if ini['EXPORT']['mosaic_method']:
            year_str += '_' + ini['EXPORT']['mosaic_method'].lower()

        # logging.debug('    {}'.format(year))
        input_path = os.path.join(
            zone['tables_ws'],
            '{}_landsat_{}.csv'.format(zone['name'], year_str))
        try:
            input_df = pd.read_csv(input_path)
        except Exception as e:
            logging.debug('    Error reading: {}'.format(
                os.path.basename(input_path)))
            # raw_input('ENTER')
            continue
        try:
            output_df = output_df.append(input_df)
        except:
            output_df = input_df.copy()
    # print(output_df)

    if output_df is not None and not output_df.empty:
        output_path = os.path.join(
            zone['output_ws'], '{}_landsat_daily.csv'.format(zone['name']))
        logging.debug('  {}'.format(output_path))
        output_df.sort_values(by=['DATE', 'ROW'], inplace=True)
        output_df.to_csv(
            output_path, index=False, columns=export_fields)


def gridmet_daily_func(export_fields, ini, zone, tasks, overwrite_flag=False):
    """

    Args:
        ini (dict): Input file parameters
        zone (dict): Zone specific parameters
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
        zone['name'].lower().replace(' ', '_'))
    output_id = '{}_gridmet_daily'.format(
        zone['name'].lower().replace(' ', '_'))
    export_path = os.path.join(ini['EXPORT']['export_ws'], export_id + '.csv')
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('  Export: {}'.format(export_path))
    logging.debug('  Output: {}'.format(output_path))

    if overwrite_flag:
        if export_id in tasks.keys():
            logging.debug('  Task already submitted, cancelling')
            for task in tasks[export_id]:
                ee.data.cancelTask(task)
            del tasks[export_id]
        if os.path.isfile(export_path):
            logging.debug('  Export CSV already exists, removing')
            os.remove(export_path)
        if os.path.isfile(output_path):
            logging.debug('  Output CSV already exists, removing')
            os.remove(output_path)

    if os.path.isfile(export_path):
        logging.debug('  Export CSV already exists, moving')
        # Modify CSV while copying from Google Drive
        export_df = pd.read_csv(export_path)
        export_df = export_df[export_fields]
        export_df.sort_values(by=['DATE'], inplace=True)
        if ini['INPUTS']['zone_field'] == 'FID':
            # If zone_field is FID, zone['name'] is set to "FID_\d"
            export_df[ini['INPUTS']['zone_field']] = int(zone['name'][4:])
            export_df[ini['INPUTS']['zone_field']] = export_df[
                ini['INPUTS']['zone_field']].astype(np.int)
        export_df.to_csv(
            output_path, index=False, columns=export_fields)
        os.remove(export_path)
        # shutil.move(export_path, output_path)
    elif os.path.isfile(output_path):
        logging.debug('  Output CSV already exists, skipping')
    elif export_id in tasks.keys():
        logging.debug('  Task already submitted, skipping')
    else:
        # Calculate values and statistics
        # Build function in loop to set water year ETo/PPT values
        def gridmet_zonal_stats_func(image):
            """"""
            date = ee.Date(image.get("system:time_start"))
            year = ee.Number(date.get('year'))
            month = ee.Number(date.get('month'))
            doy = ee.Number(date.getRelative('day', 'year')).add(1)
            wyear = ee.Number(ee.Date.fromYMD(
                year, month, 1).advance(3, 'month').get('year'))
            input_mean = ee.Image(image) \
                .reduceRegion(
                    ee.Reducer.mean(), geometry=zone['geom'],
                    crs=ini['EXPORT']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False, tileScale=1,
                    maxPixels=zone['max_pixels'])
            return ee.Feature(
                None,
                {
                    ini['INPUTS']['zone_field'].upper(): zone['name'],
                    'DATE': date.format("YYYY-MM-dd"),
                    'YEAR': year,
                    'MONTH': month,
                    'DAY': date.get('day'),
                    'DOY': doy,
                    'WATER_YEAR': wyear,
                    'ETO': input_mean.get('eto'),
                    'PPT': input_mean.get('ppt')
                })
        stats_coll = gridmet_coll.map(gridmet_zonal_stats_func)

        logging.debug('  Starting export task')
        try:
            task = ee.batch.Export.table.toDrive(
                stats_coll,
                description=export_id,
                folder=ini['EXPORT']['export_folder'],
                fileNamePrefix=export_id,
                fileFormat='CSV')
            task.start()
            logging.debug('  Active: {}'.format(task.active()))
            # logging.debug('  Status: {}'.format(task.status()))
        except Exception as e:
            logging.error('  EE Exception submitting task, skipping')
            logging.debug('  {}'.format(e))


def gridmet_monthly_func(export_fields, ini, zone, tasks,
                         overwrite_flag=False):
    """

    Args:
        ini (dict): Input file parameters
        zone (dict): Zone specific parameters
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

    # Get the last GRIDMET date from the collection
    gridmet_end_dt = ee.Date(ee.Image(
        ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
            .filterDate(
                '{}-01-01'.format(ini['INPUTS']['end_year'] - 1),
                '{}-12-31'.format(ini['INPUTS']['end_year'])) \
            .limit(1, 'system:time_start', False) \
            .first()).get('system:time_start')) \
            .format('YYYY-MM-dd').getInfo()
    gridmet_end_dt = datetime.datetime.strptime(
        gridmet_end_dt, '%Y-%m-%d')
    logging.debug('    Last GRIDMET date: {}'.format(gridmet_end_dt))

    gridmet_ym_list = [
        [y, m] for y in range(ini['INPUTS']['start_year'] - 1, ini['INPUTS']['end_year'] + 1)
        for m in range(1, 13)
        if datetime.datetime(y, m, 1) <= gridmet_end_dt]
    gridmet_dt_list = ee.List([
        ee.Date.fromYMD(y, m, 1) for y, m in gridmet_ym_list])
    gridmet_coll = ee.ImageCollection.fromImages(
        gridmet_dt_list.map(monthly_sum))

    export_id = '{}_{}_gridmet_monthly'.format(
        os.path.splitext(ini['INPUTS']['zone_filename'])[0],
        zone['name'].lower().replace(' ', '_'))
    output_id = '{}_gridmet_monthly'.format(
        zone['name'].lower().replace(' ', '_'))
    export_path = os.path.join(ini['EXPORT']['export_ws'], export_id + '.csv')
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('  Export: {}'.format(export_path))
    logging.debug('  Output: {}'.format(output_path))

    if overwrite_flag:
        if export_id in tasks.keys():
            logging.debug('  Task already submitted, cancelling')
            for task in tasks[export_id]:
                ee.data.cancelTask(task)
            del tasks[export_id]
        if os.path.isfile(export_path):
            logging.debug('  Export CSV already exists, removing')
            os.remove(export_path)
        if os.path.isfile(output_path):
            logging.debug('  Output CSV already exists, removing')
            os.remove(output_path)

    if os.path.isfile(export_path):
        logging.debug('  Export CSV already exists, moving')
        # Modify CSV while copying from Google Drive
        export_df = pd.read_csv(export_path)
        export_df = export_df[export_fields]
        export_df.sort_values(by=['DATE'], inplace=True)
        if ini['INPUTS']['zone_field'] == 'FID':
            # If zone_field is FID, zone['name'] is set to "FID_\d"
            export_df[ini['INPUTS']['zone_field']] = int(zone['name'][4:])
            export_df[ini['INPUTS']['zone_field']] = export_df[
                ini['INPUTS']['zone_field']].astype(np.int)
        export_df.to_csv(
            output_path, index=False, columns=export_fields)
        os.remove(export_path)
        # shutil.move(export_path, output_path)
    elif os.path.isfile(output_path):
        logging.debug('  Output CSV already exists, skipping')
    elif export_id in tasks.keys():
        logging.debug('  Task already submitted, skipping')
    else:
        # Calculate values and statistics
        # Build function in loop to set water year ETo/PPT values
        def gridmet_zonal_stats_func(image):
            """"""
            date = ee.Date(image.get("system:time_start"))
            year = ee.Number(date.get('year'))
            month = ee.Number(date.get('month'))
            wyear = ee.Number(ee.Date.fromYMD(
                year, month, 1).advance(3, 'month').get('year'))
            input_mean = ee.Image(image) \
                .reduceRegion(
                    ee.Reducer.mean(), geometry=zone['geom'],
                    crs=ini['EXPORT']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False, tileScale=1,
                    maxPixels=zone['max_pixels'])
            return ee.Feature(
                None,
                {
                    ini['INPUTS']['zone_field'].upper(): zone['name'],
                    'DATE': date.format("YYYY-MM"),
                    'YEAR': year,
                    'MONTH': month,
                    'WATER_YEAR': wyear,
                    'ETO': input_mean.get('eto'),
                    'PPT': input_mean.get('ppt')
                })
        stats_coll = gridmet_coll.map(gridmet_zonal_stats_func)

        logging.debug('  Starting export task')
        try:
            task = ee.batch.Export.table.toDrive(
                stats_coll,
                description=export_id,
                folder=ini['EXPORT']['export_folder'],
                fileNamePrefix=export_id,
                fileFormat='CSV')
            task.start()
            logging.debug('  Active: {}'.format(task.active()))
            # logging.debug('  Status: {}'.format(task.status()))
        except Exception as e:
            logging.error('  EE Exception submitting task, skipping')
            logging.debug('  {}'.format(e))


def pdsi_func(export_fields, ini, zone, tasks, overwrite_flag=False):
    """

    Args:
        ini (dict): Input file parameters
        zone (dict): Zone specific parameters
    """

    logging.info('  GRIDMET PDSI')

    pdsi_coll = ee.ImageCollection('IDAHO_EPSCOR/PDSI') \
        .select(['pdsi'], ['pdsi']) \
        .filterDate(
            '{}-01-01'.format(ini['INPUTS']['start_year']),
            '{}-01-01'.format(ini['INPUTS']['end_year'] + 1))
    export_id = '{}_{}_pdsi_dekad'.format(
        os.path.splitext(ini['INPUTS']['zone_filename'])[0],
        zone['name'].lower().replace(' ', '_'))
    output_id = '{}_pdsi_dekad'.format(
        zone['name'].lower().replace(' ', '_'))
    export_path = os.path.join(ini['EXPORT']['export_ws'], export_id + '.csv')
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('  Export: {}'.format(export_path))
    logging.debug('  Output: {}'.format(output_path))

    if overwrite_flag:
        if export_id in tasks.keys():
            logging.debug('  Task already submitted, cancelling')
            for task in tasks[export_id]:
                ee.data.cancelTask(task)
            del tasks[export_id]
        if os.path.isfile(export_path):
            logging.debug('  Export CSV already exists, removing')
            os.remove(export_path)
        if os.path.isfile(output_path):
            logging.debug('  Output CSV already exists, removing')
            os.remove(output_path)

    if os.path.isfile(export_path):
        logging.debug('  Export CSV already exists, moving')
        # Modify CSV while copying from Google Drive
        export_df = pd.read_csv(export_path)
        export_df = export_df[export_fields]
        export_df.sort_values(by=['DATE'], inplace=True)
        if ini['INPUTS']['zone_field'] == 'FID':
            # If zone_field is FID, zone['name'] is set to "FID_\d"
            export_df[ini['INPUTS']['zone_field']] = int(zone['name'][4:])
            export_df[ini['INPUTS']['zone_field']] = export_df[
                ini['INPUTS']['zone_field']].astype(np.int)
        export_df.to_csv(
            output_path, index=False, columns=export_fields)
        os.remove(export_path)
        # shutil.move(export_path, output_path)
    elif os.path.isfile(output_path):
        logging.debug('  Output CSV already exists, skipping')
    elif export_id in tasks.keys():
        logging.debug('  Task already submitted, skipping')
    else:
        # Calculate values and statistics
        # Build function in loop to set water year ETo/PPT values
        def pdsi_zonal_stats_func(image):
            """"""
            date = ee.Date(image.get("system:time_start"))
            doy = ee.Number(date.getRelative('day', 'year')).add(1)
            input_mean = ee.Image(image) \
                .reduceRegion(
                    ee.Reducer.mean(), geometry=zone['geom'],
                    crs=ini['EXPORT']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False, tileScale=1,
                    maxPixels=zone['max_pixels'])
            return ee.Feature(
                None,
                {
                    ini['INPUTS']['zone_field'].upper(): zone['name'],
                    'DATE': date.format("YYYY-MM-dd"),
                    'YEAR': date.get('year'),
                    'MONTH': date.get('month'),
                    'DAY': date.get('day'),
                    'DOY': doy,
                    'PDSI': input_mean.get('pdsi'),
                })
        stats_coll = pdsi_coll.map(pdsi_zonal_stats_func)

        logging.debug('  Starting export task')
        try:
            task = ee.batch.Export.table.toDrive(
                stats_coll,
                description=export_id,
                folder=ini['EXPORT']['export_folder'],
                fileNamePrefix=export_id,
                fileFormat='CSV')
            task.start()
            logging.debug('  Active: {}'.format(task.active()))
            # logging.debug('  Status: {}'.format(task.status()))
        except Exception as e:
            logging.error('  EE Exception submitting task, skipping')
            logging.debug('{}'.format(e))

    # # Get current running tasks
    # logging.debug('\nRunning tasks')
    # tasks = defaultdict(list)
    # for t in ee.data.getTaskList():
    #     if t['state'] in ['RUNNING', 'READY']:
    #         logging.debug('  {}'.format(t['description']))
    #         tasks[t['description']].append(t['id'])
    #         # tasks[t['id']] = t['description']


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine Zonal Statistics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=lambda x: python_common.valid_file(x),
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    args = parser.parse_args()

    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    else:
        args.ini = python_common.get_ini_path(os.getcwd())
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
