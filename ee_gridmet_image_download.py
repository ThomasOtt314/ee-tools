#--------------------------------
# Name:         ee_gridmet_image_download.py
# Purpose:      Earth Engine GRIDMET Image Download
# Author:       Charles Morton
# Created       2017-01-24
# Python:       2.7
#--------------------------------

import argparse
from collections import defaultdict
import datetime
import json
import logging
import os
import shutil
import sys

import arcpy
from dateutil.relativedelta import relativedelta
import ee
from osgeo import ogr

import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.ini_common as ini_common
import ee_tools.python_common as python_common


def ee_image_download(ini_path=None, overwrite_flag=False):
    """Earth Engine Annual Mean Image Download

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nEarth Engine GRIDMET Image Download')

    # Generate a single for the merged geometries
    merge_geom_flag = True

    start_year = 1984
    end_year = 2016

    gridmet_download_bands = {
        'pet': 'ETo',
        'pr': 'PPT'}

    # If false, script will export annual and water year total images
    gridmet_monthly_flag = False

    gridmet_flag = True
    pdsi_flag = False

    pdsi_date_list = [
        '0120', '0220', '0320', '0420', '0520', '0620',
        '0720', '0820', '0920', '1020', '1120', '1220']
    # pdsi_date_list = ['0920', '1220']
    # pdsi_date_list = []

    if gridmet_monthly_flag:
        gridmet_folder = 'gridmet_monthly'
    else:
        gridmet_folder = 'gridmet_annual'
    if not pdsi_date_list:
        pdsi_folder = 'pdsi_full'
    else:
        pdsi_folder = 'pdsi'

    # Read config file
    ini = ini_common.ini_parse(ini_path, mode='image')

    nodata_value = -9999

    # Manually set output spatial reference
    logging.info('\nHardcoding GRIDMET snap, cellsize and spatial reference')
    ini['output_x'], ini['output_y'] = -124.79299639209513, 49.41685579737572
    ini['output_cs'] = 0.041666001963701
    # ini['output_cs'] = [0.041666001963701, 0.041666001489718]
    # ini['output_x'], ini['output_y'] = -124.79166666666666666667, 25.04166666666666666667
    # ini['output_cs'] = 1. / 24
    ini['output_osr'] = gdc.epsg_osr(4326)
    # ini['output_osr'] = gdc.epsg_osr(4269)
    ini['output_crs'] = 'EPSG:4326'
    logging.debug('  Snap: {} {}'.format(ini['output_x'], ini['output_y']))
    logging.debug('  Cellsize: {}'.format(ini['output_cs']))
    logging.debug('  OSR: {}'.format(ini['output_osr']))

    # Get ee features from shapefile
    zone_geom_list = ee_common.shapefile_2_geom_list_func(
        ini['zone_path'], zone_field=ini['zone_field'], reverse_flag=False)

    # Merge geometries
    if merge_geom_flag:
        merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for zone in zone_geom_list:
            zone_multipolygon = ogr.ForceToMultiPolygon(
                ogr.CreateGeometryFromJson(json.dumps(zone[2])))
            for zone_polygon in zone_multipolygon:
                merge_geom.AddGeometry(zone_polygon)
        # merge_json = json.loads(merge_mp.ExportToJson())
        zone_geom_list = [
            [0, ini['zone_name'], json.loads(merge_geom.ExportToJson())]]
        ini['zone_field'] = ''

    # Need zone_path projection to build EE geometries
    zone_osr = gdc.feature_path_osr(zone_path)
    zone_proj = gdc.osr_proj(zone_osr)
    # zone_proj = ee.Projection(zone_proj).wkt().getInfo()
    # zone_proj = zone_proj.replace('\n', '').replace(' ', '')
    logging.debug('  Zone Projection: {}'.format(zone_proj))


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


    # Download images for each feature separately
    for fid, zone_str, zone_json in sorted(zone_geom_list):
        if ini['fid_keep_list'] and fid not in ini['fid_keep_list']:
            continue
        elif ini['fid_skip_list'] and fid in ini['fid_skip_list']:
            continue
        logging.info('ZONE: {} ({})'.format(zone_str, fid))

        if ini['zone_field'].upper() == 'FID':
            zone_str = 'fid_' + zone_str
        else:
            zone_str = zone_str.lower().replace(' ', '_')

        # Build EE geometry object for zonal stats
        zone_geom = ee.Geometry(zone_json, zone_proj, False)

        # Project the zone_geom to the GRIDMET projection
        # if zone_proj != output_proj:
        zone_geom = zone_geom.transform(output_crs, 0.0001)

        # Get the extent from the Earth Engine geometry object?
        zone_extent = zone_geom.bounds().getInfo()['coordinates'][0]
        zone_extent = gdc.Extent([
            min(zip(*zone_extent)[0]), min(zip(*zone_extent)[1]),
            max(zip(*zone_extent)[0]), max(zip(*zone_extent)[1])])
        # # Use GDAL and geometry json to build extent, transform, and shape
        # zone_extent = gdc.Extent(
        #     ogr.CreateGeometryFromJson(json.dumps(zone_json)).GetEnvelope())
        # # zone_extent = gdc.Extent(zone_geom.GetEnvelope())
        # zone_extent.ymin, zone_extent.xmax = zone_extent.xmax, zone_extent.ymin

        # Adjust extent to match raster
        zone_extent.adjust_to_snap(
            'EXPAND', ini['output_x'], ini['output_y'], ini['output_cs'])
        zone_geo = zone_extent.geo(ini['output_cs'])
        zone_transform = ee_common.geo_2_ee_transform(zone_geo)
        zone_shape = zone_extent.shape(ini['output_cs'])
        logging.debug('  Zone Shape: {}'.format(zone_shape))
        logging.debug('  Zone Transform: {}'.format(zone_transform))
        logging.debug('  Zone Extent: {}'.format(zone_extent))
        # logging.debug('  Geom: {}'.format(zone_geom.getInfo()))

        output_transform = '[' + ','.join(map(str, zone_transform)) + ']'
        output_shape = '[{1}x{0}]'.format(*zone_shape)
        logging.debug('  Output Projection: {}'.format(ini['output_crs']))
        logging.debug('  Output Transform: {}'.format(output_transform))
        logging.debug('  Output Shape: {}'.format(output_shape))

        zone_gridmet_ws = os.path.join(ini['output_ws'], zone_str, gridmet_folder)
        zone_pdsi_ws = os.path.join(ini['output_ws'], zone_str, pdsi_folder)
        if not os.path.isdir(zone_gridmet_ws):
            os.makedirs(zone_gridmet_ws)
        if not os.path.isdir(zone_pdsi_ws):
            os.makedirs(zone_pdsi_ws)

        # GRIDMET PPT & ETo
        if gridmet_flag:
            # Process each image in the collection by date
            export_list = []
            for year in xrange(ini['start_year'], ini['end_year'] + 1):
                for b_key, b_name in sorted(gridmet_download_bands.items()):
                    if gridmet_monthly_flag:
                        # Monthly
                        for start_month in xrange(1, 13):
                            start_dt = datetime.datetime(year, start_month, 1)
                            end_dt = start_dt + relativedelta(months=1) - datetime.timedelta(0, 1)
                            export_list.append([
                                start_dt, end_dt,
                                '{:04d}{:02d}'.format(year, start_month),
                                b_key, b_name])
                    else:
                        # Calendar year
                        export_list.append([
                            datetime.datetime(year, 1, 1),
                            datetime.datetime(year + 1, 1, 1),
                            '{:04d}'.format(year), b_key, b_name])
                        # Water year
                        export_list.append([
                            datetime.datetime(year - 1, 10, 1),
                            datetime.datetime(year, 10, 1) - datetime.timedelta(0, 1),
                            '{:04d}wy'.format(year), b_key, b_name])

            for start_dt, end_dt, date_str, b_key, b_name in export_list:
                logging.info("{} {}".format(date_str, b_name))
                if end_dt > datetime.datetime.today():
                    logging.info('  End date after current date, skipping')
                    continue

                # Rename to match naming style from getDownloadURL
                #     image_name.band.tif
                export_id = '{}_{}_gridmet_{}'.format(
                    zone_name, date_str, b_name.lower())
                output_id = '{}_gridmet.{}'.format(date_str, b_name.lower())

                export_path = os.path.join(ini['export_ws'], export_id + '.tif')
                output_path = os.path.join(
                    zone_gridmet_ws, output_id + '.tif')
                logging.debug('  Export: {}'.format(export_path))
                logging.debug('  Output: {}'.format(output_path))

                if overwrite_flag:
                    if export_id in tasks.keys():
                        logging.debug('  Task already submitted, cancelling')
                        for task in tasks[export_id]:
                            ee.data.cancelTask(task)
                        del tasks[export_id]
                    if os.path.isfile(export_path):
                        logging.debug('  Export image already exists, removing')
                        python_common.remove_file(export_path)
                        # os.remove(export_path)
                    if os.path.isfile(output_path):
                        logging.debug('  Output image already exists, removing')
                        python_common.remove_file(output_path)
                        # os.remove(output_path)
                else:
                    if os.path.isfile(export_path):
                        logging.debug('  Export image already exists, moving')
                        shutil.move(export_path, output_path)
                        gdc.raster_path_set_nodata(output_path, nodata_value)
                        arcpy.CalculateStatistics_management(output_path)
                        # gdc.raster_statistics(output_path)
                        continue
                    elif os.path.isfile(output_path):
                        logging.debug('  Output image already exists, skipping')
                        continue
                    elif export_id in tasks.keys():
                        logging.debug('  Task already submitted, skipping')
                        continue

                # GRIDMET collection is available in EarthEngine
                gridmet_coll = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')\
                    .filterDate(start_dt, end_dt) \
                    .select([b_key])
                gridmet_image = ee.Image(gridmet_coll.sum())

                logging.debug('  Starting download task')
                task = ee.batch.Export.image.toDrive(
                    gridmet_image,
                    description=export_id,
                    folder=ini['export_folder'],
                    fileNamePrefix=export_id,
                    dimensions=output_shape,
                    crs=ini['output_crs'],
                    crsTransform=output_transform)
                try:
                    task.start()
                except:
                    logging.error(
                        '  Unhandled error starting download task, skipping')
                    continue
                # logging.debug(task.status())


        # GRIDMET PDSI
        if pdsi_flag:
            # Process each image in the collection by date
            export_list = []
            b_name = 'pdsi'
            for year in xrange(ini['start_year'], ini['end_year'] + 1):
                # Dekad
                for start_month in xrange(1, 13):
                    for start_day, end_day in zip([1, 10, 20], [10, 20, 30]):
                        if start_month == 12 and start_day == 20:
                            # Go to the first day of the next year (and month)
                            start_dt = datetime.datetime(year, start_month, start_day)
                            end_dt = datetime.datetime(year + 1, 1, 1)
                        elif start_month < 12 and start_day == 20:
                            # Go to the first day of the next month
                            start_dt = datetime.datetime(year, start_month, start_day)
                            end_dt = datetime.datetime(year, start_month + 1, 1)
                        else:
                            start_dt = datetime.datetime(year, start_month, start_day)
                            end_dt = datetime.datetime(year, start_month, end_day)
                        end_dt = end_dt - datetime.timedelta(0, 1)
                        export_list.append([
                            start_dt, end_dt,
                            '{:04d}{:02d}{:02d}'.format(year, start_month, start_day),
                            b_name])

            # Filter list to only keep last dekad of October and December
            if pdsi_date_list:
                export_list = [
                    [start_dt, end_dt, date_str, b_name]
                    for start_dt, end_dt, date_str, b_name in export_list
                    if start_dt.strftime('%m%d') in pdsi_date_list]

            for start_dt, end_dt, date_str, b_name in export_list:
                logging.info("{} {}".format(date_str, b_name))

                # Rename to match naming style from getDownloadURL
                #     image_name.band.tif
                export_id = '{}_{}_{}'.format(
                    os.path.splitext(zone_filename)[0].lower(),
                    date_str, b_name.lower())
                output_id = '{}_{}'.format(date_str, b_name.lower())

                export_path = os.path.join(ini['export_ws'], export_id + '.tif')
                output_path = os.path.join(
                    zone_pdsi_ws, output_id + '.tif')
                logging.debug('  Export: {}'.format(export_path))
                logging.debug('  Output: {}'.format(output_path))

                if overwrite_flag:
                    if export_id in tasks.keys():
                        logging.debug('  Task already submitted, cancelling')
                        for task in tasks[export_id]:
                            ee.data.cancelTask(task)
                        del tasks[export_id]
                    if os.path.isfile(export_path):
                        logging.debug('  Export image already exists, removing')
                        python_common.remove_file(export_path)
                        # os.remove(export_path)
                    if os.path.isfile(output_path):
                        logging.debug('  Output image already exists, removing')
                        python_common.remove_file(output_path)
                        # os.remove(output_path)
                else:
                    if os.path.isfile(export_path):
                        logging.debug('  Export image already exists, moving')
                        shutil.move(export_path, output_path)
                        gdc.raster_path_set_nodata(output_path, nodata_value)
                        arcpy.CalculateStatistics_management(output_path)
                        # gdc.raster_statistics(output_path)
                        continue
                    elif os.path.isfile(output_path):
                        logging.debug('  Output image already exists, skipping')
                        continue
                    elif export_id in tasks.keys():
                        logging.debug('  Task already submitted, skipping')
                        continue

                # PDSI collection is available in EarthEngine
                # Index the PDSI image directly
                pdsi_image = ee.Image('IDAHO_EPSCOR/PDSI/{}'.format(
                    start_dt.strftime('%Y%m%d')))
                # pdsi_coll = ee.ImageCollection('IDAHO_EPSCOR/PDSI')\
                #     .filterDate(start_dt, end_dt) \
                #     .select(['pdsi'])
                # pdsi_image = ee.Image(pdsi_coll.mean())

                logging.debug('  Starting download task')
                task = ee.batch.Export.image.toDrive(
                    pdsi_image,
                    description=export_id,
                    folder=ini['export_folder'],
                    fileNamePrefix=export_id,
                    dimensions=output_shape,
                    crs=ini['output_crs'],
                    crsTransform=output_transform)
                try:
                    task.start()
                except Exception as e:
                    logging.error(
                        '  Unhandled error starting download task, skipping')
                    continue
                # logging.debug(task.status())

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
        description='Earth Engine GRIDMET Image Download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    args = parser.parse_args()
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

    ee_image_download(ini_path=args.ini, overwrite_flag=args.overwrite)
