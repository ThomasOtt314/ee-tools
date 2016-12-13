#--------------------------------
# Name:         ee_landsat_image_download.py
# Purpose:      Earth Engine Landsat Image Download
# Author:       Charles Morton
# Created       2016-12-14
# Python:       2.7
#--------------------------------

import argparse
from collections import defaultdict
import ConfigParser
import datetime
import json
import logging
import os
import subprocess
import sys

import arcpy
import ee
import numpy as np
from osgeo import ogr

import ee_common
import gdal_common as gdc
import python_common


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

    # Generate a single for the merged geometries
    merge_geom_flag = True

    # Add to INI eventually, choices are 'fmask' or 'cfmask'
    fmask_type = 'fmask'

    if overwrite_flag:
        logging.warning(
            '\nAre you sure you want to overwrite existing images?')
        raw_input('Press ENTER to continue')

    # Regular expression to pull out Landsat scene_id
    # landsat_re = re.compile(
    #     'L[ETC][4578]\d{6}(?P<YEAR>\d{4})(?P<DOY>\d{3})\D{3}\d{2}')

    # Open config file
    config = ConfigParser.ConfigParser()
    try:
        config.readfp(open(ini_path))
    except:
        logging.error(('\nERROR: Input file could not be read, ' +
                       'is not an input file, or does not exist\n' +
                       'ERROR: ini_path = {}\n').format(ini_path))
        sys.exit()
    logging.debug('\nReading Input File')

    image_download_bands = config.get(
        'INPUTS', 'image_download_bands').split(',')
    image_download_bands = map(
        lambda x: x.strip().lower(), image_download_bands)
    logging.info('\nOutput Bands:')
    for band in image_download_bands:
        logging.info('  {}'.format(band))
    nodata_value = -9999

    # Float32/Float64
    float_output_type = 'Float32'
    float_nodata_value = np.finfo(np.float32).min
    # Byte/Int16/UInt16/UInt32/Int32
    int_output_type = 'Byte'
    int_nodata_value = 255
    int_bands = ['cloud_score', 'fmask']

    # Read in config file
    zone_input_ws = config.get('INPUTS', 'zone_input_ws')
    zone_filename = config.get('INPUTS', 'zone_filename')
    zone_field = config.get('INPUTS', 'zone_field')

    # Google Drive export folder
    gdrive_ws = config.get('INPUTS', 'gdrive_ws')
    export_folder = config.get('INPUTS', 'export_folder')
    export_ws = os.path.join(gdrive_ws, export_folder)

    # Build and check file paths
    zone_path = os.path.join(zone_input_ws, zone_filename)
    zone_name = os.path.splitext(zone_filename)[0].lower()
    if not os.path.isdir(export_ws):
        os.makedirs(export_ws)
    if not os.path.isdir(zone_input_ws):
        logging.error(
            '\nERROR: The zone workspace does not exist, exiting\n  {}'.format(
                zone_input_ws))
        sys.exit()
    elif not os.path.isfile(zone_path):
        logging.error(
            '\nERROR: The zone shapefile does not exist, exiting\n  {}'.format(
                zone_path))
        sys.exit()

    # Final output folder
    try:
        output_ws = config.get('INPUTS', 'images_ws')
        if not os.path.isdir(output_ws):
            os.makedirs(output_ws)
    except:
        output_ws = os.getcwd()
        logging.debug('  Defaulting output workspace to {}'.format(output_ws))

    # For now, hardcode snap, cellsize and spatial reference
    logging.info('\nHardcoding Landsat snap point, cellsize, and projection')
    output_x, output_y = 15, 15
    output_cs = 30
    # Download images as WGS 84 UTM Zone 11N
    output_crs = 'EPSG:32611'
    output_osr = gdc.epsg_osr(32611)
    logging.debug('  Snap: {} {}'.format(output_x, output_y))
    logging.debug('  Cellsize: {}'.format(output_cs))
    logging.debug('  OSR: {}'.format(output_osr))

    # Start/end year
    try:
        start_year = int(config.get('INPUTS', 'start_year'))
    except:
        start_year = None
    try:
        end_year = int(config.get('INPUTS', 'end_year'))
    except:
        end_year = None
    if start_year and end_year and end_year < start_year:
        logging.error(
            '\nERROR: End year must be >= start year')
        sys.exit()
    default_end_year = datetime.datetime.today().year + 1
    if (start_year and start_year not in range(1984, default_end_year) or
            end_year and end_year not in range(1984, default_end_year)):
        logging.error(
            '\nERROR: Year must be an integer from 1984-2015')
        sys.exit()

    # Start/end month
    try:
        start_month = int(config.get('INPUTS', 'start_month'))
    except:
        start_month = None
    try:
        end_month = int(config.get('INPUTS', 'end_month'))
    except:
        end_month = None
    if start_month and start_month not in range(1, 13):
        logging.error(
            '\nERROR: Start month must be an integer from 1-12')
        sys.exit()
    elif end_month and end_month not in range(1, 13):
        logging.error(
            '\nERROR: End month must be an integer from 1-12')
        sys.exit()

    # Start/end DOY
    try:
        start_doy = int(config.get('INPUTS', 'start_doy'))
    except:
        start_doy = None
    try:
        end_doy = int(config.get('INPUTS', 'end_doy'))
    except:
        end_doy = None
    if end_doy and end_doy > 273:
        logging.error(
            '\nERROR: End DOY has to be in the same water year as start DOY')
        sys.exit()
    if start_doy and start_doy not in range(1, 367):
        logging.error(
            '\nERROR: Start DOY must be an integer from 1-366')
        sys.exit()
    elif end_doy and end_doy not in range(1, 367):
        logging.error(
            '\nERROR: End DOY must be an integer from 1-366')
        sys.exit()
    # if end_doy < start_doy:
    #     logging.error(
    #         '\nERROR: End DOY must be >= start DOY')
    #     sys.exit()

    # Control which Landsat images are used
    try:
        landsat5_flag = config.getboolean('INPUTS', 'landsat5_flag')
    except:
        landsat5_flag = False
    try:
        landsat4_flag = config.getboolean('INPUTS', 'landsat4_flag')
    except:
        landsat4_flag = False
    try:
        landsat7_flag = config.getboolean('INPUTS', 'landsat7_flag')
    except:
        landsat7_flag = False
    try:
        landsat8_flag = config.getboolean('INPUTS', 'landsat8_flag')
    except:
        landsat8_flag = False

    # Cloudmasking
    # Force fmask and acca flags false to avoid applying cloud masks
    logging.info('  Not applying fmask or acca cloud masking')
    acca_flag = False
    fmask_flag = False
    # try:
    #     acca_flag = config.getboolean('INPUTS', 'acca_flag')
    # except:
    #     acca_flag = False
    # try:
    #     fmask_flag = config.getboolean('INPUTS', 'fmask_flag')
    # except:
    #     fmask_flag = False

    # Fmask source type
    try:
        fmask_type = config.get('INPUTS', 'fmask_type').lower()
    except:
        fmask_type = 'fmask'
        logging.debug(
            '  Defaulting Fmask source type to {}'.format(fmask_type))
    if fmask_type not in ['fmask', 'cfmask']:
        logging.error(
            ('\nERROR: Invalid Fmask source type: {}, must be "fmask" ' +
             'or "cfmask"').format(fmask_type))
        sys.exit()

    # Intentionally don't apply scene_id skip/keep lists
    # Compute zonal stats for all available images
    # Filter by scene_id when making summary tables
    logging.info('  Not applying scene_id keep or skip lists')
    scene_id_keep_list = []
    scene_id_skip_list = []

    # # Only process specific Landsat scenes
    # try:
    #     scene_id_keep_path = config.get('INPUTS', 'scene_id_keep_path')
    #     with open(scene_id_keep_path) as input_f:
    #         scene_id_keep_list = input_f.readlines()
    #     scene_id_keep_list = [x.strip()[:16] for x in scene_id_keep_list]
    # except IOError:
    #     logging.error('\nFileIO Error: {}'.format(scene_id_keep_path))
    #     sys.exit()
    # except:
    #     scene_id_keep_list = []

    # # Skip specific landsat scenes
    # try:
    #     scene_id_skip_path = config.get('INPUTS', 'scene_id_skip_path')
    #     with open(scene_id_skip_path) as input_f:
    #         scene_id_skip_list = input_f.readlines()
    #     scene_id_skip_list = [x.strip()[:16] for x in scene_id_skip_list]
    # except IOError:
    #     logging.error('\nFileIO Error: {}'.format(scene_id_skip_path))
    #     sys.exit()
    # except:
    #     scene_id_skip_list = []

    # Only process certain Landsat path/rows
    try:
        path_keep_list = list(python_common.parse_int_set(
            config.get('INPUTS', 'path_keep_list')))
    except:
        path_keep_list = []
    try:
        row_keep_list = list(python_common.parse_int_set(
            config.get('INPUTS', 'row_keep_list')))
    except:
        row_keep_list = []

    # Skip or keep certain FID
    try:
        fid_skip_list = list(python_common.parse_int_set(
            config.get('INPUTS', 'fid_skip_list')))
    except:
        fid_skip_list = []
    try:
        fid_keep_list = list(python_common.parse_int_set(
            config.get('INPUTS', 'fid_keep_list')))
    except:
        fid_keep_list = []


    # Use ArcPy to compute the raster statistics
    arcpy.CheckOutExtension('Spatial')
    arcpy.env.overwriteOutput = True
    arcpy.env.pyramid = 'PYRAMIDS 0'
    arcpy.env.compression = "LZW"

    # Initialize Earth Engine API key
    ee.Initialize()

    # Get ee features from shapefile
    zone_geom_list = ee_common.shapefile_2_geom_list_func(
        zone_path, zone_field=zone_field, reverse_flag=False)

    # Merge geometries
    if merge_geom_flag:
        merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for zone in zone_geom_list:
            zone_multipolygon = ogr.ForceToMultiPolygon(
                ogr.CreateGeometryFromJson(json.dumps(zone[2])))
            for zone_polygon in zone_multipolygon:
                merge_geom.AddGeometry(zone_polygon)
        # merge_json = json.loads(merge_mp.ExportToJson())
        zone_geom_list = [[0, zone_name, json.loads(merge_geom.ExportToJson())]]
        zone_field = None

    # Need zone_path projection to build EE geometries
    zone_osr = gdc.feature_path_osr(zone_path)
    zone_proj = gdc.osr_proj(zone_osr)
    # zone_proj = ee.Projection(zone_proj).wkt().getInfo()
    # zone_proj = zone_proj.replace('\n', '').replace(' ', '')
    logging.debug('  Zone Projection: {}'.format(zone_proj))

    # Check that shapefile has matching spatial reference
    if not gdc.matching_spatref(zone_osr, output_osr):
        logging.warning('  Zone OSR:\n{}\n'.format(zone_osr))
        logging.warning('  Output OSR:\n{}\n'.format(output_osr))
        logging.warning('  Zone Proj4:   {}'.format(zone_osr.ExportToProj4()))
        logging.warning('  Output Proj4: {}'.format(output_osr.ExportToProj4()))
        logging.warning(
            '\nWARNING: \n' +
            'The output and zone spatial references do not appear to match\n' +
            'This will likely cause problems!')
        raw_input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(zone_osr))
        logging.debug('  Output Projection:\n{}\n'.format(output_osr))
        logging.debug('  Output Cellsize: {}'.format(output_cs))


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
        if fid_keep_list and fid not in fid_keep_list:
            continue
        elif fid_skip_list and fid in fid_skip_list:
            continue
        logging.info('\nZONE: {} (FID: {})'.format(zone_str, fid))

        if not zone_field or zone_field.upper() == 'FID':
            zone_str = 'fid_' + zone_str
        else:
            zone_str = zone_str.lower().replace(' ', '_')

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
            'EXPAND', output_x, output_y, output_cs)
        zone_geo = gdc.extent_geo(zone_extent, output_cs)
        zone_transform = ee_common.geo_2_ee_transform(zone_geo)
        zone_shape = zone_extent.shape(output_cs)
        logging.debug('  Zone Shape: {}'.format(zone_shape))
        logging.debug('  Zone Transform: {}'.format(zone_transform))
        logging.debug('  Zone Extent: {}'.format(zone_extent))
        # logging.debug('  Geom: {}'.format(zone_geom.getInfo()))

        output_transform = '[' + ','.join(map(str, zone_transform)) + ']'
        output_shape = '{1}x{0}'.format(*zone_shape)

        zone_output_ws = os.path.join(output_ws, zone_str)
        zone_images_ws = os.path.join(zone_output_ws, images_folder)
        if not os.path.isdir(zone_images_ws):
            os.makedirs(zone_images_ws)

        # Keyword arguments for ee_common.get_landsat_collection() and
        #   ee_common.get_landsat_image()
        args = {
            'fmask_flag': fmask_flag, 'acca_flag': acca_flag,
            'fmask_type': fmask_type, 'zone_geom': zone_geom,
            'start_date': None, 'end_date': None,
            'start_year': start_year, 'end_year': end_year,
            'start_month': start_month, 'end_month': end_month,
            'start_doy': start_doy, 'end_doy': end_doy,
            'scene_id_keep_list': scene_id_keep_list,
            'scene_id_skip_list': scene_id_skip_list,
            'path_keep_list': path_keep_list, 'row_keep_list': row_keep_list}

        # Move to EE common
        def get_collection_ids(image):
            return ee.Feature(None, {'id': image.get('system:index')})

        # Get list of available Landsat images
        scene_id_list = []
        if landsat4_flag:
            logging.debug('  Getting Landsat 4 scene_id list')
            scene_id_list.extend([
                f['properties']['id']
                for f in ee_common.get_landsat_collection('LT4', **args).map(
                    get_collection_ids).getInfo()['features']])
        if landsat5_flag:
            logging.debug('  Getting Landsat 5 scene_id list')
            scene_id_list.extend([
                f['properties']['id']
                for f in ee_common.get_landsat_collection('LT5', **args).map(
                    get_collection_ids).getInfo()['features']])
        if landsat7_flag:
            logging.debug('  Getting Landsat 7 scene_id list')
            scene_id_list.extend([
                f['properties']['id']
                for f in ee_common.get_landsat_collection('LE7', **args).map(
                    get_collection_ids).getInfo()['features']])
        if landsat8_flag:
            logging.debug('  Getting Landsat 8 scene_id list')
            scene_id_list.extend([
                f['properties']['id']
                for f in ee_common.get_landsat_collection('LC8', **args).map(
                    get_collection_ids).getInfo()['features']])

        # Get list of unique image "dates"
        scene_id_list = set([
            (image_id[9:13], image_id[13:16], image_id[:3])
            for image_id in scene_id_list])
        logging.debug('  Scene Count: {}\n'.format(len(scene_id_list)))

        # Process each image in the collection by date
        # for image_id in scene_id_list:
        #     logging.info("{}".format(image_id))
        for year, doy, landsat in sorted(scene_id_list):
            scene_dt = datetime.datetime.strptime(
                '{}_{}'.format(year, doy), '%Y_%j')
            month = scene_dt.month
            day = scene_dt.day
            logging.info("{} {}-{:02d}-{:02d} (DOY {})".format(
                landsat, year, month, day, doy))
            zone_year_ws = os.path.join(zone_images_ws, year)
            if not os.path.isdir(zone_year_ws):
                os.makedirs(zone_year_ws)

            # Get the prepped Landsat image by ID
            landsat_image = ee_common.get_landsat_image(
                landsat, year, doy, args)

            # Clip using the feature geometry
            landsat_image = ee.Image(landsat_image).clip(zone_geom)

            # Set the masked values to a nodata value
            # so that the TIF can have a nodata value other than 0 set
            landsat_image = landsat_image.unmask(nodata_value, False)

            for band in image_download_bands:
                logging.debug('  Band: {}'.format(band))

                # Rename to match naming style from getDownloadURL
                #     image_name.band.tif
                export_id = '{}_{}{:02d}{:02d}_{}_{}_{}'.format(
                    zone_name, year, month, day, doy, landsat.lower(),
                    band.lower())
                output_id = '{}{:02d}{:02d}_{}_{}.{}'.format(
                    year, month, day, doy, landsat.lower(), band)

                export_path = os.path.join(export_ws, export_id + '.tif')
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
                    folder=export_folder,
                    fileNamePrefix=export_id,
                    dimensions=output_shape,
                    crs=output_crs,
                    crsTransform=output_transform)
                try:
                    task.start()
                except Exception as e:
                    logging.error('  Unhandled error starting task, skipping')
                    logging.error(str(e))
                    continue
                # logging.debug(task.status())
            # break

    # # Get current running tasks
    # logging.debug('\nRunning tasks')
    # tasks = defaultdict(list)
    # for t in ee.data.getTaskList():
    #     if t['state'] in ['RUNNING', 'READY']:
    #         logging.debug('  {}'.format(t['description']))
    #         tasks[t['description']].append(t['id'])
    #         # tasks[t['id']] = t['description']


def get_ini_path(workspace):
    import Tkinter, tkFileDialog
    root = Tkinter.Tk()
    ini_path = tkFileDialog.askopenfilename(
        initialdir=workspace, parent=root, filetypes=[('INI files', '.ini')],
        title='Select the target INI file')
    root.destroy()
    return ini_path


def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine Landsat Image Download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=lambda x: is_valid_file(parser, x),
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
        args.ini = get_ini_path(os.getcwd())
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
