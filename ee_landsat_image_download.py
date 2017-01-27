#--------------------------------
# Name:         ee_landsat_image_download.py
# Purpose:      Earth Engine Landsat Image Download
# Author:       Charles Morton
# Created       2017-01-27
# Python:       2.7
#--------------------------------

import argparse
from collections import defaultdict
import datetime
import json
import logging
import os
import subprocess
import sys

# import arcpy
import ee
import numpy as np
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
    logging.info('\nEarth Engine Landsat Image Download')
    images_folder = 'landsat'

    if overwrite_flag:
        logging.warning(
            '\nAre you sure you want to overwrite existing images?')
        raw_input('Press ENTER to continue')

    # Regular expression to pull out Landsat scene_id
    # landsat_re = re.compile(
    #     'L[ETC][4578]\d{6}(?P<YEAR>\d{4})(?P<DOY>\d{3})\D{3}\d{2}')

    # Read config file
    # ini = ini_common.ini_parse(ini_path, section='IMAGE')
    ini = ini_common.read(ini_path)
    ini_common.parse_section(ini, section='INPUTS')
    ini_common.parse_section(ini, section='EXPORT')
    ini_common.parse_section(ini, section='IMAGES')

    nodata_value = -9999

    # Float32/Float64
    float_output_type = 'Float32'
    float_nodata_value = np.finfo(np.float32).min
    # Byte/Int16/UInt16/UInt32/Int32
    int_output_type = 'Byte'
    int_nodata_value = 255
    int_bands = ['cloud_score', 'fmask']

    # # Use ArcPy to compute the raster statistics
    # arcpy.CheckOutExtension('Spatial')
    # arcpy.env.overwriteOutput = True
    # arcpy.env.pyramid = 'PYRAMIDS 0'
    # arcpy.env.compression = 'LZW'

    # Get ee features from shapefile
    zone_geom_list = gdc.shapefile_2_geom_list_func(
        ini['INPUTS']['zone_path'], zone_field=ini['INPUTS']['zone_field'],
        reverse_flag=False)

    # Filter features by FID before merging geometries
    if ini['INPUTS']['fid_keep_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] in ini['INPUTS']['fid_keep_list']]
    if ini['INPUTS']['fid_skip_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] not in ini['INPUTS']['fid_skip_list']]

    # Merge geometries
    if ini['IMAGES']['merge_geom_flag']:
        merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for zone in zone_geom_list:
            zone_multipolygon = ogr.ForceToMultiPolygon(
                ogr.CreateGeometryFromJson(json.dumps(zone[2])))
            for zone_polygon in zone_multipolygon:
                merge_geom.AddGeometry(zone_polygon)
        # merge_json = json.loads(merge_mp.ExportToJson())
        zone_geom_list = [
            [0, ini['INPUTS']['zone_filename'], json.loads(merge_geom.ExportToJson())]]
        ini['INPUTS']['zone_field'] = ''

    # Need zone_path projection to build EE geometries
    zone_osr = gdc.feature_path_osr(ini['INPUTS']['zone_path'])
    zone_proj = gdc.osr_proj(zone_osr)
    # zone_proj = ee.Projection(zone_proj).wkt().getInfo()
    # zone_proj = zone_proj.replace('\n', '').replace(' ', '')
    # logging.debug('  Zone Projection: {}'.format(zone_proj))

    # Check that shapefile has matching spatial reference
    if not gdc.matching_spatref(zone_osr, ini['EXPORT']['osr']):
        logging.warning('  Zone OSR:\n{}\n'.format(zone_osr))
        logging.warning('  Output OSR:\n{}\n'.format(
            ini['EXPORT']['osr']))
        logging.warning('  Zone Proj4:   {}'.format(zone_osr.ExportToProj4()))
        logging.warning('  Output Proj4: {}'.format(
            ini['EXPORT']['osr'].ExportToProj4()))
        logging.warning(
            '\nWARNING: \n'
            'The output and zone spatial references do not appear to match\n'
            'This will likely cause problems!')
        raw_input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(
            zone_osr.ExportToWkt()))
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


    # Download images for each feature separately
    for zone_fid, zone_name, zone_json in zone_geom_list:
        zone_name = zone_name.replace(' ', '_')
        logging.info('ZONE: {} (FID: {})'.format(zone_name, zone_fid))

        # Build EE geometry object for zonal stats
        zone_geom = ee.Geometry(
            geo_json=zone_json, opt_proj=zone_proj, opt_geodesic=False)
        logging.debug('  Centroid: {}'.format(
            zone_geom.centroid(100).getInfo()['coordinates']))

        # Use feature geometry to build extent, transform, and shape
        zone_extent = gdc.Extent(
            ogr.CreateGeometryFromJson(json.dumps(zone_json)).GetEnvelope())
        # zone_extent = gdc.Extent(zone_geom.GetEnvelope())
        zone_extent.ymin, zone_extent.xmax = zone_extent.xmax, zone_extent.ymin
        zone_extent.adjust_to_snap(
            'EXPAND', ini['EXPORT']['snap_x'], ini['EXPORT']['snap_y'],
            ini['EXPORT']['cellsize'])
        zone_geo = zone_extent.geo(ini['EXPORT']['cellsize'])
        zone_transform = gdc.geo_2_ee_transform(zone_geo)
        zone_shape = zone_extent.shape(ini['EXPORT']['cellsize'])
        logging.debug('  Zone Shape: {}'.format(zone_shape))
        logging.debug('  Zone Transform: {}'.format(zone_transform))
        logging.debug('  Zone Extent: {}'.format(zone_extent))
        # logging.debug('  Zone Geom: {}'.format(zone_geom.getInfo()))

        output_transform = '[' + ','.join(map(str, zone_transform)) + ']'
        output_shape = '{1}x{0}'.format(*zone_shape)

        zone_images_ws = os.path.join(
            ini['IMAGES']['output_ws'], zone_name, images_folder)
        if not os.path.isdir(zone_images_ws):
            os.makedirs(zone_images_ws)

        # Keyword arguments for ee_common.get_landsat_collection() and
        #   ee_common.get_landsat_image()
        landsat_args = {
            k: v for section in ['INPUTS', 'EXPORT']
            for k, v in ini[section].items()
            if k in [
                'fmask_flag', 'acca_flag', 'fmask_type',
                'start_year', 'end_year',
                'start_month', 'end_month', 'start_doy', 'end_doy',
                'scene_id_keep_list', 'scene_id_skip_list',
                'path_keep_list', 'row_keep_list', 'adjust_method']}
        # landsat_args['start_date'] = start_date
        # landsat_args['end_date'] = end_date
        landsat_args['zone_geom'] = zone_geom

        # Move to EE common
        def get_collection_ids(image):
            return ee.Feature(None, {'id': image.get('system:index')})

        # Get list of available Landsat images
        scene_id_list = []
        if ini['INPUTS']['landsat4_flag']:
            logging.debug('  Getting Landsat 4 scene_id list')
            scene_id_list.extend([
                f['properties']['id']
                for f in ee_common.get_landsat_collection('LT4', **landsat_args)\
                    .map(get_collection_ids).getInfo()['features']])
        if ini['INPUTS']['landsat5_flag']:
            logging.debug('  Getting Landsat 5 scene_id list')
            scene_id_list.extend([
                f['properties']['id']
                for f in ee_common.get_landsat_collection('LT5', **landsat_args)\
                    .map(get_collection_ids).getInfo()['features']])
        if ini['INPUTS']['landsat7_flag']:
            logging.debug('  Getting Landsat 7 scene_id list')
            scene_id_list.extend([
                f['properties']['id']
                for f in ee_common.get_landsat_collection('LE7', **landsat_args)\
                    .map(get_collection_ids).getInfo()['features']])
        if ini['INPUTS']['landsat8_flag']:
            logging.debug('  Getting Landsat 8 scene_id list')
            scene_id_list.extend([
                f['properties']['id']
                for f in ee_common.get_landsat_collection('LC8', **landsat_args)\
                    .map(get_collection_ids).getInfo()['features']])

        # Get list of unique image "dates"
        # Keep scene_id components as string for set operation
        # If not mosaicing images, include path/row in set
        #   otherwise set to None
        if not ini['EXPORT']['mosaic_method']:
            scene_id_list = set([
                (image_id[9:13], image_id[13:16], image_id[0:3],
                    image_id[3:6], image_id[6:9])
                for image_id in scene_id_list])
        else:
            scene_id_list = set([
                (image_id[9:13], image_id[13:16], image_id[0:3], None, None)
                for image_id in scene_id_list])
        logging.debug('  Scene Count: {}\n'.format(len(scene_id_list)))

        # Process each image in the collection by date
        # Leave scene_id components as strings
        for year, doy, landsat, path, row in sorted(scene_id_list):
            scene_dt = datetime.datetime.strptime(
                '{}_{}'.format(year, doy), '%Y_%j')
            month = scene_dt.month
            day = scene_dt.day

            # If not mosaicing images, include path/row in name
            if not ini['EXPORT']['mosaic_method']:
                landsat_str = '{}{}{}'.format(landsat, path, row)
            else:
                landsat_str = '{}'.format(landsat)
            logging.info('{} {}-{:02d}-{:02d} (DOY {})'.format(
                landsat.upper(), year, month, day, doy))

            zone_year_ws = os.path.join(zone_images_ws, year)
            if not os.path.isdir(zone_year_ws):
                os.makedirs(zone_year_ws)

            # Get the prepped Landsat image by ID
            landsat_image = ee.Image(ee_common.get_landsat_image(
                landsat, year, doy, path, row, ini['EXPORT']['mosaic_method'],
                landsat_args))

            # Clip using the feature geometry
            if ini['IMAGES']['clip_landsat_flag']:
                landsat_image = landsat_image.clip(zone_geom)
            else:
                landsat_image = landsat_image.clip(ee.Geometry.Rectangle(
                    list(zone_extent), ini['EXPORT']['crs'], False))

            # DEADBEEF - Display a single image
            # ee_common.show_thumbnail(landsat_image.visualize(
            #     bands=['fmask', 'fmask', 'fmask'], min=0, max=4))
            # ee_common.show_thumbnail(landsat_image.visualize(
            #     bands=['toa_red', 'toa_green', 'toa_blue'],
            #     min=0.05, max=0.35, gamma=1.4))
            # return True

            # Set the masked values to a nodata value
            # so that the TIF can have a nodata value other than 0 set
            landsat_image = landsat_image.unmask(nodata_value, False)

            for band in ini['IMAGES']['download_bands']:
                logging.debug('  Band: {}'.format(band))

                # Rename to match naming style from getDownloadURL
                #     image_name.band.tif
                export_id = '{}_{}{:02d}{:02d}_{}_{}_{}'.format(
                    ini['INPUTS']['zone_filename'], year, month, day, doy,
                    landsat_str.lower(), band.lower())
                output_id = '{}{:02d}{:02d}_{}_{}.{}'.format(
                    year, month, day, doy, landsat_str.lower(), band)

                export_path = os.path.join(
                    ini['EXPORT']['export_ws'], export_id + '.tif')
                output_path = os.path.join(
                    zone_year_ws, output_id + '.tif')
                logging.debug('  Export: {}'.format(export_path))
                logging.debug('  Output: {}'.format(output_path))

                if overwrite_flag:
                    if export_id in tasks.keys():
                        logging.debug('  Task already submitted, cancelling')
                        for task in tasks[export_id]:
                            ee.data.cancelTask(task)
                        del tasks[export_id]
                    if os.path.isfile(export_path):
                        logging.debug(
                            '  Export image already exists, removing')
                        python_common.remove_file(export_path)
                        # os.remove(export_path)
                    if os.path.isfile(output_path):
                        logging.debug(
                            '  Output image already exists, removing')
                        python_common.remove_file(output_path)
                        # os.remove(output_path)
                else:
                    if os.path.isfile(export_path):
                        logging.debug('  Export image already exists, moving')
                        if band in int_bands:
                            subprocess.call([
                                'gdalwarp',
                                '-ot', int_output_type, '-overwrite',
                                '-of', 'GTiff', '-co', 'COMPRESS=LZW',
                                '-srcnodata', str(nodata_value),
                                '-dstnodata', str(int_nodata_value),
                                export_path, output_path])
                        else:
                            subprocess.call([
                                'gdalwarp',
                                '-ot', float_output_type, '-overwrite',
                                '-of', 'GTiff', '-co', 'COMPRESS=LZW',
                                '-srcnodata', str(nodata_value),
                                '-dstnodata', '{:f}'.format(float_nodata_value),
                                export_path, output_path])
                        with open(os.devnull, 'w') as devnull:
                            subprocess.call(
                                ['gdalinfo', '-stats', output_path],
                                stdout=devnull)
                        subprocess.call(['gdalmanage', 'delete', export_path])
                        continue
                    elif os.path.isfile(output_path):
                        logging.debug(
                            '  Output image already exists, skipping')
                        continue
                    elif export_id in tasks.keys():
                        logging.debug(
                            '  Task already submitted, skipping')
                        continue

                # Should composites include Ts?
                if band == 'refl_toa':
                    band_list = [
                        'toa_blue', 'toa_green', 'toa_red',
                        'toa_nir', 'toa_swir1', 'toa_swir2']
                    # band_list = ['toa_red', 'toa_green', 'toa_blue']
                elif band == 'refl_sur':
                    band_list = [
                        'sur_blue', 'sur_green', 'sur_red',
                        'sur_nir', 'sur_swir1', 'sur_swir2']
                    # band_list = ['sur_red', 'sur_green', 'sur_blue']
                elif band == 'tasseled_cap':
                    band_list = ['tc_bright', 'tc_green', 'tc_wet']
                else:
                    band_list = [band]
                band_image = landsat_image.select(band_list)

                # CGM 2016-09-26 - Don't apply any cloud masks to images
                # # Apply cloud mask before exporting
                # if fmask_flag and band not in ['refl_sur', 'cloud', 'fmask']:
                #     fmask = ee.Image(landsat_image.select(['fmask']))
                #     cloud_mask = fmask.eq(2).Or(fmask.eq(3)).Or(fmask.eq(4)).Not()
                #     band_image = band_image.updateMask(cloud_mask)

                logging.debug('  Starting download task')
                task = ee.batch.Export.image.toDrive(
                    band_image,
                    description=export_id,
                    folder=ini['EXPORT']['export_folder'],
                    fileNamePrefix=export_id,
                    dimensions=output_shape,
                    crs=ini['EXPORT']['crs'],
                    crsTransform=output_transform)
                try:
                    task.start()
                except Exception as e:
                    logging.error(
                        '  Unhandled error starting task, skipping\n'
                        '  {}'.format(str(e)))
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
        description='Earth Engine Landsat Image Download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=lambda x: python_common.valid_file(x),
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

    ee_image_download(ini_path=args.ini, overwrite_flag=args.overwrite)
