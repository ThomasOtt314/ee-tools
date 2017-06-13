#--------------------------------
# Name:         ee_beamer_annual_mean_download.py
# Purpose:      Compute and download Beamer ETg images using Earth Engine
# Author:       Charles Morton
# Created       2017-06-13
# Python:       2.7
#--------------------------------

import argparse
from collections import defaultdict
import datetime as dt
from dateutil import rrule, relativedelta
import json
import logging
import os
# import re
import sys
from time import sleep
import urllib
import zipfile

import ee
import numpy as np
from osgeo import gdal, ogr

# This is an awful way of getting the parent folder into the path
# We really should package this up as a module with a setup.py
# This way the ee_tools folders would be in the
#   PYTHONPATH env. variable
ee_tools_path = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(ee_tools_path, 'ee_tools'))
sys.path.insert(0, ee_tools_path)
import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils
import ee_tools.wrs2 as wrs2

# ArcPy must be imported after OGR
import arcpy


def ee_beamer_et(ini_path=None, overwrite_flag=False):
    """Earth Engine Beamer ET Image Download

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nEarth Engine Beamer Annual Mean ETg Image Download')

    median_band_list = ['etg_mean', 'etg_lci', 'etg_uci', 'etg_lpi', 'etg_upi']
    nodata_value = -9999
    zips_folder = 'zips'
    annuals_folder = 'annuals'

    # Regular expression to pull out Landsat scene_id
    # landsat_re = re.compile('L[ETC]0[4578]_\d{3}XXX_\d{4}\d{2}\d{2}')
    # landsat_re = re.compile(
    #     '(?P<LANDSAT>L[ETC]0[4578])_(?P<PATH>\d{3})(?P<ROW>(\d{3})|(XXX))_'
    #     '(?P<DATE>\d{8})')
    #     # '(?P<YEAR>\d{4})(?P<MONTH>\d{2})(?P<DAY>\d{2}))')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='BEAMER')

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

    # Merge geometries
    if ini['INPUTS']['merge_geom_flag']:
        merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for zone in zone_geom_list:
            zone_multipolygon = ogr.ForceToMultiPolygon(
                ogr.CreateGeometryFromJson(json.dumps(zone[2])))
            for zone_polygon in zone_multipolygon:
                merge_geom.AddGeometry(zone_polygon)
        # merge_json = json.loads(merge_mp.ExportToJson())
        zone_geom_list = [[
            0, ini['INPUTS']['zone_filename'],
            json.loads(merge_geom.ExportToJson())]]
        ini['INPUTS']['zone_field'] = ''

    # Set all zone specific parameters into a dictionary
    zone = {}

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

    # Use ArcPy to compute the median
    arcpy.CheckOutExtension('Spatial')
    arcpy.env.overwriteOutput = True

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

    # Read in ETo and PPT data from file
    if (ini['BEAMER']['eto_source'] == 'file' or
            ini['BEAMER']['ppt_source'] == 'file'):
        data_array = np.genfromtxt(
            ini['BEAMER']['data_path'], delimiter=',', names=True, dtype=None)
        data_fields = data_array.dtype.names
        logging.debug('  CSV fields: {}'.format(', '.join(data_fields)))
        # DEADBEEF - Compare fields names assuming all upper case
        data_fields = [f.upper() for f in data_fields]
        eto_dict = defaultdict(dict)
        ppt_dict = defaultdict(dict)
        for row in data_array:
            z = str(row[data_fields.index(ini['BEAMER']['data_zone_field'])])
            y = int(row[data_fields.index(ini['BEAMER']['data_year_field'])])
            if ini['BEAMER']['eto_source'] == 'file':
                # DEADBEEF - Compare fields names assuming all upper case
                eto_dict[z][y] = row[data_fields.index(
                    ini['BEAMER']['data_eto_field'].upper())]
            if ini['BEAMER']['ppt_source'] == 'file':
                # DEADBEEF - Compare fields names assuming all upper case
                ppt_dict[z][y] = row[data_fields.index(
                    ini['BEAMER']['data_ppt_field'].upper())]

    # Get filtered/merged/prepped Landsat collection
    landsat_args = {
        k: v for section in ['INPUTS']
        for k, v in ini[section].items()
        if k in [
            'landsat4_flag', 'landsat5_flag', 'landsat7_flag',
            'landsat8_flag', 'fmask_flag', 'acca_flag', 'fmask_source',
            'start_year', 'end_year',
            'start_month', 'end_month', 'start_doy', 'end_doy',
            'scene_id_keep_list', 'scene_id_skip_list',
            'path_keep_list', 'row_keep_list',
            'adjust_method', 'mosaic_method', 'path_row_geom']}
    # Currently only using TOA collections and comput Tasumi at-surface
    #   reflectance is supported
    landsat_args['refl_type'] = 'toa'
    landsat_args['products'] = ['evi_sur']
    landsat = ee_common.Landsat(landsat_args)


    # Download images for each feature separately
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
        zone['max_pixels'] = zone['shape'][0] * zone['shape'][1]
        logging.debug('  Max Pixels: {}'.format(zone['max_pixels']))

        # Set output spatial reference
        # Eventually allow user to manually set these
        # output_crs = zone['proj']
        logging.debug('  Image Projection: {}'.format(ini['SPATIAL']['crs']))

        output_transform = '[' + ','.join(map(str, zone['transform'])) + ']'
        output_shape = '{1}x{0}'.format(*zone['shape'])
        logging.debug('  Image Transform: {}'.format(output_transform))
        logging.debug('  Image Shape: {}'.format(output_shape))

        zone_output_ws = os.path.join(ini['BEAMER']['output_ws'], zone_name)
        zone_zips_ws = os.path.join(zone_output_ws, zips_folder)
        zone_annuals_ws = os.path.join(zone_output_ws, annuals_folder)
        if not os.path.isdir(zone_zips_ws):
            os.makedirs(zone_zips_ws)
        if not os.path.isdir(zone_annuals_ws):
            os.makedirs(zone_annuals_ws)

        # Process date range by year
        interval_cnt = 1
        start_dt = dt.datetime(ini['INPUTS']['start_year'], 1, 1)
        end_dt = dt.datetime(
            ini['INPUTS']['end_year'] + 1, 1, 1) - dt.timedelta(0, 1)
        for i, iter_start_dt in enumerate(rrule.rrule(
                rrule.YEARLY, interval=interval_cnt,
                dtstart=start_dt, until=end_dt)):
            iter_end_dt = (
                iter_start_dt +
                relativedelta.relativedelta(years=interval_cnt) -
                dt.timedelta(0, 1))
            if ((ini['INPUTS']['start_month'] and
                    iter_end_dt.month < ini['INPUTS']['start_month']) or
                (ini['INPUTS']['end_month'] and
                    iter_start_dt.month > ini['INPUTS']['end_month'])):
                logging.debug('  {}  {}  skipping'.format(
                    iter_start_dt.date(), iter_end_dt.date()))
                continue
            elif ((ini['INPUTS']['start_doy'] and
                    int(iter_end_dt.strftime('%j')) < ini['INPUTS']['start_doy']) or
                  (ini['INPUTS']['end_doy'] and
                    int(iter_start_dt.strftime('%j')) > ini['INPUTS']['end_doy'])):
                logging.debug('  {}  {}  skipping'.format(
                    iter_start_dt.date(), iter_end_dt.date()))
                continue
            else:
                logging.info('{}  {}'.format(
                    iter_start_dt.date(), iter_end_dt.date()))
            year = iter_start_dt.year

            image_id = 'etg_{}_{}'.format(
                zone_name.lower().replace(' ', '_'), year)
            zip_path = os.path.join(
                zone_zips_ws, image_id + '.zip')
            # median_path = os.path.join(
            #     zone_output_ws, image_id + '.img')
            logging.debug('  Zip: {}'.format(zip_path))

            if os.path.isfile(zip_path) and overwrite_flag:
                logging.debug('    Output already exists, removing zip')
                os.remove(zip_path)
            elif os.path.isfile(zip_path) and not overwrite_flag:
                # Check that existing ZIP files can be opened
                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        pass
                except:
                    logging.warning('    Zip file error, removing'.format(i))
                    os.remove(zip_path)

            # Filter the GRIDMET collection
            wy_start_date = '{}-10-01'.format(year - 1)
            wy_end_date = '{}-10-01'.format(year)
            logging.debug('  WY: {} {}'.format(wy_start_date, wy_end_date))
            gridmet_coll = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .filterDate(wy_start_date, wy_end_date)

            # # PRISM collection was uploaded as an asset
            # if ini['BEAMER']['ppt_source'] == 'prism':
            #     def prism_time_start(input_image):
            #         """Set time_start property on PRISM water year PPT collection"""
            #         # Assume year is the 4th item separated by "_"
            #         water_year = ee.String(input_image.get('system:index')).split('_').get(3)
            #         date_start = ee.Date(ee.String(water_year).cat('-10-01'))
            #         return input_image.select([0], ['PPT']).set({
            #             'system:time_start': date_start.millis()
            #         })
            #     prism_coll = ee.ImageCollection('users/cgmorton/prism_800m_ppt_wy')
            #     prism_coll = prism_coll.map(prism_time_start) \
            #         .filterDate(wy_start_date, wy_end_date)

            # Get water year PPT from file
            # Convert all input data to mm to match GRIDMET data
            if ini['BEAMER']['ppt_source'] == 'file':
                wy_ppt_input = ppt_dict[zone_name][year]
                if ini['BEAMER']['data_ppt_units'] == 'mm':
                    pass
                elif ini['BEAMER']['data_ppt_units'] == 'in':
                    wy_ppt_input *= 25.4
                elif ini['BEAMER']['data_ppt_units'] == 'ft':
                    wy_ppt_input *= (25.4 * 12)
            elif ini['BEAMER']['ppt_source'] == 'gridmet':
                # GET GRIDMET value at centroid of geometry
                wy_ppt_input = float(ee_get_info(ee.ImageCollection(
                    gridmet_coll.map(ee_common.gridmet_ppt_func).sum()).getRegion(
                        zone['geom'].centroid(1), 500))[1][4])
                # Calculate GRIDMET zonal mean of geometry
                # wy_ppt_input = float(ee.ImageCollection(
                #     gridmet_coll.map(gridmet_ppt_func)).reduceRegion(
                #         reducer=ee.Reducer.sum(),
                #         geometry=zone['geom'],
                #         crs=ini['SPATIAL']['crs'],
                #         crsTransform=zone['transform'],
                #         bestEffort=False,
                #         tileScale=1).getInfo()['PPT']
            # elif ini['BEAMER']['ppt_source'] == 'prism':
            #     # Calculate PRISM zonal mean of geometry
            #     wy_ppt_input = float(ee.ImageCollection(
            #         prism_coll.map(ee_common.prism_ppt_func)).sum().reduceRegion(
            #             reducer=ee.Reducer.mean(),
            #             geometry=zone['geom'],
            #             crs=ini['SPATIAL']['crs'],
            #             crsTransform=zone['transform'],
            #             bestEffort=False,
            #             tileScale=1).getInfo()['PPT'])

            # Get water year ETo read from file
            # Convert all input data to mm for Beamer Method
            if ini['BEAMER']['eto_source'] == 'file':
                wy_eto_input = eto_dict[zone_name][year]
                if ini['BEAMER']['data_eto_units'] == 'mm':
                    pass
                elif ini['BEAMER']['data_eto_units'] == 'in':
                    wy_eto_input *= 25.4
                elif ini['BEAMER']['data_eto_units'] == 'ft':
                    wy_eto_input *= (25.4 * 12)
            # This assumes GRIMET data is in millimeters
            elif ini['BEAMER']['eto_source'] == 'gridmet':
                wy_eto_input = float(ee_get_info(ee.ImageCollection(
                    gridmet_coll.map(ee_common.gridmet_eto_func).sum()).getRegion(
                        zone['geom'].centroid(1), 500))[1][4])
                # wy_eto_input = float(ee.ImageCollection(
                #     gridmet_coll.map(gridmet_eto_func)).reduceRegion(
                #         reducer=ee.Reducer.sum(),
                #         geometry=zone['geom'],
                #         crs=ini['SPATIAL']['crs'],
                #         crsTransform=zone['transform'],
                #         bestEffort=False,
                #         tileScale=1).getInfo()
            logging.debug('  Input ETO: {} mm  PPT: {} mm'.format(
                wy_eto_input, wy_ppt_input))

            # Scale ETo & PPT
            wy_eto_input *= ini['BEAMER']['eto_factor']
            wy_ppt_input *= ini['BEAMER']['ppt_factor']

            # Convert output units from mm
            wy_ppt_output = wy_ppt_input
            wy_eto_output = wy_eto_input
            if ini['BEAMER']['output_ppt_units'] == 'mm':
                pass
            elif ini['BEAMER']['output_ppt_units'] == 'in':
                wy_ppt_output /= 25.4
            elif ini['BEAMER']['output_ppt_units'] == 'ft':
                wy_ppt_output /= (25.4 * 12)
            if ini['BEAMER']['output_eto_units'] == 'mm':
                pass
            elif ini['BEAMER']['output_eto_units'] == 'in':
                wy_eto_output /= 25.4
            elif ini['BEAMER']['output_eto_units'] == 'ft':
                wy_eto_output /= (25.4 * 12)
            logging.debug('  Output ETO: {} {} PPT: {} {}'.format(
                wy_eto_output, ini['BEAMER']['output_eto_units'],
                wy_ppt_output, ini['BEAMER']['output_ppt_units']))

            # Initialize the Landsat object for target zone and iteration
            landsat.zone_geom = zone['geom']
            landsat.start_date = iter_start_dt.strftime('%Y-%m-%d')
            landsat.end_date = iter_end_dt.strftime('%Y-%m-%d')
            landsat_coll = landsat.get_collection()
            # print([f['properties']['SCENE_ID'] for f in landsat_coll.getInfo()['features']])
            # raw_input('ENTER')

            # Add water year ETo and PPT values to each image
            def eto_ppt_func(img):
                """"""
                return ee.Image(img).setMulti({
                    'wy_eto': wy_eto_output,
                    'wy_ppt': wy_ppt_output
                })
            landsat_coll = ee.ImageCollection(landsat_coll.map(eto_ppt_func))

            # Build each collection separately then merge
            etg_coll = ee.ImageCollection(landsat_coll.map(landsat_etg_func))
            # print([float(x[4]) for x in etg_coll.getRegion(zone['geom'].centroid(1), 1).getInfo()[1:]])
            # print([float(x[4]) for x in ee.ImageCollection(ee.Image(etg_coll.mean())).getRegion(zone['geom'].centroid(1), 1).getInfo()[1:]])
            # raw_input('ENTER')

            # Clip using the feature geometry
            # Set the masked values to a nodata value
            # so that the TIF can have a nodata value other than 0 set
            etg_image = ee.Image(etg_coll.mean()) \
                .clip(zone['geom']) \
                .unmask(nodata_value, False)

            if not os.path.isfile(zip_path):
                # Get the download URL
                logging.debug('  Requesting')
                zip_url = None
                for i in range(1, 10):
                    try:
                        zip_url = etg_image.getDownloadURL({
                            'name': image_id,
                            'crs': ini['SPATIAL']['crs'],
                            'crs_transform': output_transform,
                            'dimensions': output_shape
                        })
                    except Exception as e:
                        logging.info('  Resending query')
                        logging.debug('  {}'.format(e))
                        sleep(i ** 2)
                        zip_url = None
                    if zip_url:
                        break

                # Try downloading a few times
                logging.info('  Downloading')
                for i in range(1, 10):
                    try:
                        urllib.urlretrieve(zip_url, zip_path)
                        break
                    except Exception as e:
                        logging.info('  Resending query')
                        logging.debug('  {}'.format(e))
                        sleep(i ** 2)
                        os.remove(zip_path)

            # Try extracting the files
            try:
                logging.info('  Extracting')
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(zone_annuals_ws)
            except Exception as e:
                logging.warning('    Error: could not extract'.format(i))
                logging.debug('  {}'.format(e))
                try:
                    os.remove(zip_path)
                except Exception as e:
                    pass

            # Set nodata value
            for item in os.listdir(zone_annuals_ws):
                if item.startswith(image_id) and item.endswith('.tif'):
                    raster_path_set_nodata(
                        os.path.join(zone_annuals_ws, item), nodata_value)
                    raster_statistics(
                        os.path.join(zone_annuals_ws, item))

        logging.info('\nComputing median of annual means')
        for band in median_band_list:
            logging.info('  {}'.format(band))
            image_band_list = [
                os.path.join(zone_annuals_ws, item)
                for item in os.listdir(zone_annuals_ws)
                if item.endswith('.{}.tif'.format(band.lower()))]
            # for image_path in image_band_list:
            #     raster_path_set_nodata(image_path, nodata_value)

            # Use ArcPy to compute the median
            median_path = os.path.join(
                zone_output_ws, 'etg_{}_median.{}.tif'.format(
                    zone_name.lower().replace(' ', '_'), band.lower()))
            logging.debug('  {}'.format(median_path))
            median_obj = arcpy.sa.CellStatistics(
                image_band_list, 'MEDIAN', 'DATA')
            median_obj.save(median_path)
            median_obj = None

            raster_statistics(median_path)

            # # Remove inputs after computing median
            # for image_path in image_band_list:
            #     arcpy.Delete_management(image_path)


def ee_get_info(ee_obj):
    for i in range(1, 10):
        try:
            return ee_obj.getInfo()
            break
        except Exception as e:
            logging.info('  Resending query')
            logging.debug('  {}'.format(e))
            sleep(i ** 2)
    return None


def landsat_etg_func(img):
    """Compute Beamer ET*/ET/ETg

    Properties:
        wy_eto
        wy_ppt

    """
    # ET*
    evi_sur = ee.Image(img).select(['evi_sur'])
    etstar_mean = etstar_func(evi_sur, 'mean').rename(['etstar_mean'])
    etstar_lpi = etstar_func(evi_sur, 'lpi').rename(['etstar_lpi'])
    etstar_upi = etstar_func(evi_sur, 'upi').rename(['etstar_upi'])
    etstar_lci = etstar_func(evi_sur, 'lci').rename(['etstar_lci'])
    etstar_uci = etstar_func(evi_sur, 'uci').rename(['etstar_uci'])

    # For each Landsat scene, I need to calculate water year PPT and ETo sums
    ppt = ee.Image.constant(ee.Number(img.get('wy_ppt')))
    eto = ee.Image.constant(ee.Number(img.get('wy_eto')))

    # ETg
    etg_mean = etg_func(etstar_mean, eto, ppt).rename(['etg_mean'])
    etg_lpi = etg_func(etstar_lpi, eto, ppt).rename(['etg_lpi'])
    etg_upi = etg_func(etstar_upi, eto, ppt).rename(['etg_upi'])
    etg_lci = etg_func(etstar_lci, eto, ppt).rename(['etg_lci'])
    etg_uci = etg_func(etstar_uci, eto, ppt).rename(['etg_uci'])

    # ET
    # et_mean = ee_common.et_func(etg_mean, gridmet_ppt)
    # et_lpi = ee_common.et_func(etg_lpi, gridmet_ppt)
    # et_upi = ee_common.et_func(etg_upi, gridmet_ppt)
    # et_lci = ee_common.et_func(etg_lci, gridmet_ppt)
    # et_uci = ee_common.et_func(etg_uci, gridmet_ppt)

    return ee.Image([etg_mean, etg_lpi, etg_upi, etg_lci, etg_uci]) \
        .rename(['etg_mean', 'etg_lpi', 'etg_upi', 'etg_lci', 'etg_uci']) \
        .copyProperties(img, ['system:index', 'system:time_start'])


def etstar_func(evi, etstar_type='mean'):
    """Compute Beamer ET* from EVI (assuming at-surface reflectance)"""
    def etstar(evi, c0, c1, c2):
        """Beamer ET*"""
        return ee.Image(evi) \
            .expression(
                'c0 + c1 * evi + c2 * (evi ** 2)',
                {'evi': evi, 'c0': c0, 'c1': c1, 'c2': c2}) \
            .max(0)
    if etstar_type == 'mean':
        return etstar(evi, -0.1955, 2.9042, -1.5916)
    elif etstar_type == 'lpi':
        return etstar(evi, -0.2871, 2.9192, -1.6263)
    elif etstar_type == 'upi':
        return etstar(evi, -0.1039, 2.8893, -1.5569)
    elif etstar_type == 'lci':
        return etstar(evi, -0.2142, 2.9175, -1.6554)
    elif etstar_type == 'uci':
        return etstar(evi, -0.1768, 2.8910, -1.5278)


def etg_func(etstar, eto, ppt):
    """Compute groundwater ET (ETg) (ET* x (ETo - PPT))"""
    return etstar.multiply(eto.subtract(ppt))


def raster_statistics(input_raster):
    """"""
    def band_statistics(input_band):
        """"""
        # stats = input_band.ComputeStatistics(False)
        stats = input_band.GetStatistics(0, 1)
        # input_band.SetStatistics(*stats)
        input_band.GetHistogram(stats[0], stats[1])
        # return stats

    output_raster_ds = gdal.Open(input_raster, 1)
    for band_i in range(int(output_raster_ds.RasterCount)):
        try:
            band = output_raster_ds.GetRasterBand(band_i + 1)
            band_statistics(band)
        except RuntimeError:
            logging.debug('  {} - band {} - all cells nodata'.format(
                input_raster, band_i + 1))
            continue
    output_raster_ds = None


def raster_path_set_nodata(raster_path, input_nodata):
    """Set raster nodata value for all bands"""
    raster_ds = gdal.Open(raster_path, 1)
    raster_ds_set_nodata(raster_ds, input_nodata)
    del raster_ds


def raster_ds_set_nodata(raster_ds, input_nodata):
    """Set raster dataset nodata value for all bands"""
    band_cnt = raster_ds.RasterCount
    for band_i in range(band_cnt):
        band = raster_ds.GetRasterBand(band_i + 1)
        band.SetNoDataValue(input_nodata)


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine Beamer Annual Mean ETg Image Download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True, type=utils.arg_valid_file,
        help='Input file', metavar='file')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{0:<20s} {1}'
    logging.info(log_f.format('Start Time:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))
    logging.info('')

    ee_beamer_et(ini_path=args.ini, overwrite_flag=args.overwrite)
