#--------------------------------
# Name:         ee_beamer_zonal_stats.py
# Purpose:      Beamer ET using Earth Engine
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
import pprint
import re
import sys
from time import sleep

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
sys.path.insert(0, os.path.join(ee_tools_path, 'ee_tools'))
sys.path.insert(0, ee_tools_path)
import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils
import ee_tools.wrs2 as wrs2


def ee_beamer_et(ini_path=None, overwrite_flag=True):
    """Earth Engine Beamer ET Zonal Stats

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nEarth Engine Beamer ET Zonal Stats')

    # Eventually get from INI (like ini['BEAMER']['landsat_products'])
    landsat_products = [
        'ndvi_toa', 'ndwi_toa', 'albedo_sur', 'ts', 'evi_sur'
    ]

    # Regular expression to pull out Landsat scene_id
    # If RE has capturing groups, findall call below will fail to extract ID
    landsat_re = re.compile('L[ETC]0[4578]_\d{3}XXX_\d{4}\d{2}\d{2}')
    # landsat_re = re.compile('L[ETC][4578]\d{3}XXX\d{4}\d{3}')
    # landsat_re = re.compile('L[ETC][4578]\d{3}\d{3}\d{4}\d{3}\D{3}\d{2}')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='BEAMER')

    # First row  of csv is header
    header_list = [
        'ZONE_FID', 'ZONE_NAME', 'DATE', 'SCENE_ID', 'LANDSAT', 'PATH', 'ROW',
        'YEAR', 'MONTH', 'DAY', 'DOY',
        'PIXEL_COUNT', 'PIXEL_TOTAL', 'FMASK_COUNT', 'FMASK_TOTAL',
        'LOW_ETG_COUNT', 'CLOUD_SCORE',
        'NDVI_TOA', 'NDWI_TOA', 'ALBEDO_SUR', 'TS',
        'EVI_SUR', 'ETSTAR_MEAN', 'ETG_MEAN',
        'ETG_LPI', 'ETG_UPI', 'ETG_LCI', 'ETG_UCI', 'WY_ETO', 'WY_PPT']
    int_fields = [
        'ZONE_FID', 'PATH', 'ROW', 'YEAR', 'MONTH', 'DAY', 'DOY',
        'PIXEL_COUNT', 'PIXEL_TOTAL', 'FMASK_COUNT', 'FMASK_TOTAL',
        'LOW_ETG_COUNT']
    float_fields = list(
        set(header_list) - set(int_fields) -
        set(['ZONE_NAME', 'DATE', 'SCENE_ID', 'LANDSAT']))

    # Remove the existing CSV
    output_path = os.path.join(
        ini['BEAMER']['output_ws'], ini['BEAMER']['output_name'])
    if overwrite_flag and os.path.isfile(output_path):
        os.remove(output_path)
    # Create an empty CSV
    if not os.path.isfile(output_path):
        data_df = pd.DataFrame(columns=header_list)
        data_df[int_fields] = data_df[int_fields].astype(np.int64)
        data_df[float_fields] = data_df[float_fields].astype(np.float32)
        data_df.to_csv(output_path, index=False)

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
            y = row[data_fields.index(ini['BEAMER']['data_year_field'])]
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
    landsat_args['products'] = landsat_products
    # Currently only using TOA collections and comput Tasumi at-surface
    #   reflectance is supported
    landsat_args['refl_type'] = 'toa'
    landsat = ee_common.Landsat(landsat_args)

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
        zone['max_pixels'] = zone['shape'][0] * zone['shape'][1]
        logging.debug('  Max Pixels: {}'.format(zone['max_pixels']))

        # Set output spatial reference
        # Eventually allow user to manually set these
        # output_crs = zone['proj']
        # ini['INPUTS']['transform'] = zone['transform']
        logging.debug('  Output Projection: {}'.format(ini['SPATIAL']['crs']))
        logging.debug('  Output Transform: {}'.format(zone['transform']))


        # Process date range by year
        start_dt = dt.datetime(ini['INPUTS']['start_year'], 1, 1)
        end_dt = dt.datetime(
            ini['INPUTS']['end_year'] + 1, 1, 1) - dt.timedelta(0, 1)
        iter_months = ini['BEAMER']['month_step']
        for i, iter_start_dt in enumerate(rrule.rrule(
                # rrule.YEARLY, interval=interval_cnt,
                rrule.MONTHLY, interval=iter_months,
                dtstart=start_dt, until=end_dt)):
            iter_end_dt = (
                iter_start_dt +
                # relativedelta.relativedelta(years=interval_cnt) -
                relativedelta.relativedelta(months=iter_months) -
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
                logging.info('  {}  {}'.format(
                    iter_start_dt.date(), iter_end_dt.date()))
            year = iter_start_dt.year

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
            #         wy = ee.String(
            #             input_image.get('system:index')).split('_').get(3)
            #         date_start = ee.Date(ee.String(wy).cat('-10-01'))
            #         return input_image.select([0], ['PPT']).setMulti({
            #             'system:time_start': date_start.advance(-1, 'year').millis()
            #         })
            #     prism_coll = ee.ImageCollection('users/cgmorton/prism_800m_ppt_wy')
            #     prism_coll = ee.ImageCollection(prism_coll.map(prism_time_start)) \
            #         .filterDate(wy_start_dt, wy_end_dt)
            #     # prism_coll = ee.ImageCollection(
            #     #     ee_common.MapsEngineAssets.prism_ppt_wy).filterDate(
            #     #         wy_start_dt, wy_end_dt)

            # Get water year PPT for centroid of zone or read from file
            # Convert all input data to mm to match GRIDMET data
            if ini['BEAMER']['ppt_source'] == 'file':
                wy_ppt_input = ppt_dict[zone][year]
                if ini['BEAMER']['data_ppt_units'] == 'mm':
                    pass
                elif ini['BEAMER']['data_ppt_units'] == 'in':
                    wy_ppt_input *= 25.4
                elif ini['BEAMER']['data_ppt_units'] == 'ft':
                    wy_ppt_input *= (25.4 * 12)
            elif ini['BEAMER']['ppt_source'] == 'gridmet':
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
            #     wy_ppt_input = float(ee_get_info(ee.ImageCollection(
            #         prism_coll.map(ee_common.prism_ppt_func)).sum().reduceRegion(
            #             reducer=ee.Reducer.mean(),
            #             geometry=zone['geom'],
            #             crs=ini['SPATIAL']['crs'],
            #             crsTransform=zone['transform'],
            #             bestEffort=False,
            #             tileScale=1))['PPT'])

            # Get water year ETo for centroid of zone or read from file
            # Convert all input data to mm for Beamer Method
            if ini['BEAMER']['eto_source'] == 'FILE':
                wy_eto_input = eto_dict[zone][year]
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
            logging.debug('  Input  ETO: {} mm  PPT: {} mm'.format(
                wy_eto_input, wy_ppt_input))

            # Scale ETo & PPT
            wy_eto_input *= ini['BEAMER']['eto_factor']
            wy_ppt_input *= ini['BEAMER']['ppt_factor']

            # Convert output units from mm
            wy_eto_output = wy_eto_input
            wy_ppt_output = wy_ppt_input
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
            logging.debug('  Output ETO: {} {}  PPT: {} {}'.format(
                wy_eto_output, ini['BEAMER']['output_eto_units'],
                wy_ppt_output, ini['BEAMER']['output_ppt_units']))

            # Initialize the Landsat object
            landsat.zone_geom = zone['geom']
            landsat.start_date = iter_start_dt.strftime('%Y-%m-%d')
            landsat.end_date = iter_end_dt.strftime('%Y-%m-%d')
            landsat_coll = landsat.get_collection()
            if ee.Image(landsat_coll.first()).getInfo() is None:
                logging.info('    No images, skipping')
                continue

            # # Print the collection SCENE_ID list
            # logging.debug('{}'.format(', '.join([
            #     f['properties']['SCENE_ID']
            #     for f in landsat_coll.getInfo()['features']])))
            # raw_input('ENTER')

            # Add water year ETo and PPT values to each image
            def eto_ppt_func(img):
                """"""
                return ee.Image(img).setMulti({
                    'wy_eto': wy_eto_input,
                    'wy_ppt': wy_ppt_input
                })
            landsat_coll = ee.ImageCollection(landsat_coll.map(eto_ppt_func))

            # Compute ETg
            image_coll = ee.ImageCollection(landsat_coll.map(landsat_etg_func))

            # # Get the output image URL
            # output_url = ee.Image(landsat_coll.first()) \
            #     .select(['red', 'green', 'blue']) \
            #     .visualize(min=[0, 0, 0], max=[0.4, 0.4, 0.4]) \
            #     .getThumbUrl({'format': 'png', 'size': '600'})
            # # This would load the image in your browser
            # import webbrowser
            # webbrowser.open(output_url)
            # # webbrowser.read(output_url)

            # # Show the output image
            # window = tk.Tk()
            # output_file = Image.open(io.BytesIO(urllib.urlopen(output_url).read()))
            # output_photo = ImageTk.PhotoImage(output_file)
            # label = tk.Label(window, image=output_photo)
            # label.pack()
            # window.mainloop()

            # Compute zonal stats of polygon
            def beamer_zonal_stats_func(input_image):
                """"""
                bands = len(landsat_args['products']) + 3

                # .clip(zone['geom']) \
                input_mean = input_image \
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=zone['geom'],
                        crs=ini['SPATIAL']['crs'],
                        crsTransform=zone['transform'],
                        bestEffort=False,
                        tileScale=1,
                        maxPixels=zone['max_pixels'] * bands)

                fmask_img = input_image.select(['fmask'])
                input_count = fmask_img.gt(1) \
                    .addBands(fmask_img.gte(0).unmask()) \
                    .rename(['fmask', 'pixel']) \
                    .reduceRegion(
                        reducer=ee.Reducer.sum().combine(
                            ee.Reducer.count(), '', True),
                        geometry=zone['geom'],
                        crs=ini['SPATIAL']['crs'],
                        crsTransform=zone['transform'],
                        bestEffort=False,
                        tileScale=1,
                        maxPixels=zone['max_pixels'] * 3)

                # DEADBEEF
                low_etg_cnt = input_image \
                    .select(['etg_mean'], ['low_etg_count']) \
                    .lte(ini['BEAMER']['low_etg_threshold']) \
                    .reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=zone['geom'],
                        crs=ini['SPATIAL']['crs'],
                        crsTransform=zone['transform'],
                        bestEffort=False,
                        tileScale=1,
                        maxPixels=zone['max_pixels'] * 2)

                # Save as image properties
                return ee.Feature(
                    None,
                    {
                        'scene_id': ee.String(input_image.get('SCENE_ID')),
                        'time': input_image.get('system:time_start'),
                        'row': input_mean.get('row'),
                        'pixel_count': input_count.get('pixel_sum'),
                        'pixel_total': input_count.get('pixel_count'),
                        'fmask_count': input_count.get('fmask_sum'),
                        'fmask_total': input_count.get('fmask_count'),
                        'cloud_score': input_mean.get('cloud_score'),
                        'low_etg_count': low_etg_cnt.get('low_etg_count'),
                        'ndvi_toa': input_mean.get('ndvi_toa'),
                        'ndwi_toa': input_mean.get('ndwi_toa'),
                        'albedo_sur': input_mean.get('albedo_sur'),
                        'ts': input_mean.get('ts'),
                        'evi_sur': input_mean.get('evi_sur'),
                        'etstar_mean': input_mean.get('etstar_mean'),
                        'etg_mean': input_mean.get('etg_mean'),
                        'etg_lpi': input_mean.get('etg_lpi'),
                        'etg_upi': input_mean.get('etg_upi'),
                        'etg_lci': input_mean.get('etg_lci'),
                        'etg_uci': input_mean.get('etg_uci')
                    })

            # Calculate values and statistics
            stats_coll = ee.ImageCollection(
                image_coll.map(beamer_zonal_stats_func))

            # # DEADBEEF - Test the function for a single image
            # stats_info = beamer_zonal_stats_func(
            #     ee.Image(image_coll.first())).getInfo()
            # print(stats_info)
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
            # raw_input('ENTER')
            # # return False

            # Get the values from EE
            stats_desc = ee_get_info(stats_coll)
            if stats_desc is None:
                logging.error('  Timeout error, skipping')
                continue

            # Save data for writing
            row_list = []
            for ftr in stats_desc['features']:
                try:
                    count = int(ftr['properties']['pixel_count'])
                except (KeyError, TypeError) as e:
                    # logging.debug('  Exception: {}'.format(e))
                    continue
                if count == 0:
                    logging.info('  COUNT: 0, skipping')
                    continue

                # First get scene ID and time
                try:
                    scene_id = landsat_re.findall(
                        ftr['properties']['scene_id'])[0]
                    scene_time = dt.datetime.utcfromtimestamp(
                        float(ftr['properties']['time']) / 1000)
                except:
                    pp = pprint.PrettyPrinter(indent=4)
                    pp.pprint(ftr)
                    raw_input('ENTER')

                # Extract and save other properties
                try:
                    row_list.append({
                        'ZONE_FID': zone_fid,
                        'ZONE_NAME': zone_name,
                        'SCENE_ID': scene_id,
                        'LANDSAT': scene_id[0:4],
                        'PATH': int(scene_id[5:8]),
                        'ROW': int(ftr['properties']['row']),
                        # 'ROW': int(scene_id[8:11]),
                        'DATE': scene_time.date().isoformat(),
                        'YEAR': int(scene_time.year),
                        'MONTH': int(scene_time.month),
                        'DAY': int(scene_time.day),
                        'DOY': int(scene_time.strftime('%j')),
                        'PIXEL_COUNT': int(ftr['properties']['pixel_count']),
                        'PIXEL_TOTAL': int(ftr['properties']['pixel_total']),
                        'FMASK_COUNT': int(ftr['properties']['fmask_count']),
                        'FMASK_TOTAL': int(ftr['properties']['fmask_total']),
                        'CLOUD_SCORE': float(ftr['properties']['cloud_score']),
                        'LOW_ETG_COUNT': int(ftr['properties']['low_etg_count']),
                        'NDVI_TOA': float(ftr['properties']['ndvi_toa']),
                        'NDWI_TOA': float(ftr['properties']['ndwi_toa']),
                        'ALBEDO_SUR': float(ftr['properties']['albedo_sur']),
                        'TS': float(ftr['properties']['ts']),
                        'EVI_SUR': float(ftr['properties']['evi_sur']),
                        'ETSTAR_MEAN': float(ftr['properties']['etstar_mean']),
                        'ETG_MEAN': float(ftr['properties']['etg_mean']),
                        'ETG_LPI': float(ftr['properties']['etg_lpi']),
                        'ETG_UPI': float(ftr['properties']['etg_upi']),
                        'ETG_LCI': float(ftr['properties']['etg_lci']),
                        'ETG_UCI': float(ftr['properties']['etg_uci']),
                        'WY_ETO': wy_eto_output,
                        'WY_PPT': wy_ppt_output})
                except (KeyError, TypeError) as e:
                    logging.info(
                        '  ERROR: {}\n  SCENE_ID: {}\n  '
                        '  There may not be an SR image to join to\n'
                        '  {}'.format(
                            e, scene_id, ftr['properties']))
                    # raw_input('ENTER')

            # Save all values to the dataframe (and export)
            if row_list:
                logging.debug('  Appending')
                data_df = data_df.append(row_list, ignore_index=True)
                logging.debug('  Saving')
                data_df[int_fields] = data_df[int_fields].astype(np.int64)
                data_df[float_fields] = data_df[float_fields].astype(np.float32)
                data_df = data_df.reindex_axis(header_list, axis=1)
                data_df.sort_values(
                    ['ZONE_FID', 'DATE', 'ROW'], ascending=True, inplace=True)
                # data_df.sort(
                #     ['ZONE_NAME', 'DATE'], ascending=[True, True], inplace=True)
                # logging.debug(data_df.dtypes)
                data_df.to_csv(output_path, index=False)
            del row_list


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
    evi_sur = img.select(['evi_sur'])
    etstar_mean = etstar_func(evi_sur, 'mean').rename(['etstar_mean'])
    etstar_lpi = etstar_func(evi_sur, 'lpi').rename(['etstar_lpi'])
    etstar_upi = etstar_func(evi_sur, 'upi').rename(['etstar_upi'])
    etstar_lci = etstar_func(evi_sur, 'lci').rename(['etstar_lci'])
    etstar_uci = etstar_func(evi_sur, 'uci').rename(['etstar_uci'])

    # For each Landsat scene, I need to calculate water year PPT and ETo sums
    ppt = ee.Image.constant(ee.Number(img.get('wy_ppt')))
    eto = ee.Image.constant(ee.Number(img.get('wy_eto')))
    # ppt = ee.ImageCollection.fromImages(
    #     refl_toa.get('gridmet_match')).map(gridmet_ppt_func).sum();
    # eto = ee.ImageCollection.fromImages(
    #     refl_toa.get('gridmet_match')).map(gridmet_eto_func).sum();

    # ETg
    etg_mean = etg_func(etstar_mean, eto, ppt).rename(['etg_mean'])
    etg_lpi = etg_func(etstar_lpi, eto, ppt).rename(['etg_lpi'])
    etg_upi = etg_func(etstar_upi, eto, ppt).rename(['etg_upi'])
    etg_lci = etg_func(etstar_lci, eto, ppt).rename(['etg_lci'])
    etg_uci = etg_func(etstar_uci, eto, ppt).rename(['etg_uci'])

    # ET
    # et_mean = ee_common.et_func(etg_mean, ppt)
    # et_lpi = ee_common.et_func(etg_lpi, ppt)
    # et_upi = ee_common.et_func(etg_upi, ppt)
    # et_lci = ee_common.et_func(etg_lci, ppt)
    # et_uci = ee_common.et_func(etg_uci, ppt)

    return ee.Image([
            # Send cloud and row images through for now
            img.select(['cloud_score']), img.select(['fmask']),
            img.select(['row']),
            img.select(['ndvi_toa']), img.select(['ndwi_toa']),
            # img.select(['ndwi_swir1_green_toa']),
            img.select(['albedo_sur']), img.select(['ts']),
            img.select(['evi_sur']), etstar_mean, etg_mean,
            etg_lpi, etg_upi, etg_lci, etg_uci]) \
        .copyProperties(img, ['SCENE_ID', 'system:time_start'])


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


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine Beamer ET Zonal Stats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True, type=utils.arg_valid_file,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    # parser.add_argument(
    #     '-o', '--overwrite', default=False, action='store_true',
    #     help='Force overwrite of existing files')
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

    ee_beamer_et(ini_path=args.ini)
    # ee_beamer_et(ini_path=args.ini, overwrite_flag=args.overwrite)