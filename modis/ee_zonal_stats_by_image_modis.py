#--------------------------------
# Name:         ee_zonal_stats_by_image_modis.py
# Purpose:      Download zonal stats by image using Earth Engine
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
from collections import defaultdict
import datetime
from itertools import groupby
import json
import logging
import os
import pprint
import re
from subprocess import check_output
import sys

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
    logging.info('\nEarth Engine zonal statistics by image')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='EXPORT')
    inputs.parse_section(ini, section='ZONAL_STATS')

    if ini['EXPORT']['export_dest'] != 'getinfo':
        logging.critical('\nERROR: Only GetInfo exports are currently supported\n')
        sys.exit()

    # # Zonal stats init file paths
    # zone_geojson_path = os.path.join(
    #     ini['ZONAL_STATS']['output_ws'],
    #     os.path.basename(ini['INPUTS']['zone_shp_path']).replace(
    #         '.shp', '.geojson'))

    # These may eventually be set in the INI file
    modis_fields = [
        'ZONE_NAME', 'ZONE_FID', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY']
    #     'AREA', 'PIXEL_SIZE', 'PIXEL_COUNT', 'PIXEL_TOTAL',
    #     'CLOUD_COUNT', 'CLOUD_TOTAL', 'CLOUD_PCT', 'QA']

    # Convert the shapefile to geojson
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
            ini['ZONAL_STATS']['zone_geojson'], ini['INPUTS']['zone_shp_path']])

    # # Get ee features from shapefile
    # zone_geom_list = gdc.shapefile_2_geom_list_func(
    #     ini['INPUTS']['zone_shp_path'], zone_field=ini['INPUTS']['zone_field'],
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
        str(z['properties'][ini['INPUTS']['zone_field']]).replace(' ', '_').lower()
        for z in zones_geojson['features']]
    if len(set(zone_names)) != len(zones_geojson['features']):
        logging.error(
            '\nERROR: There appear to be duplicate zone ID/name values.'
            '\n  Currently, the values in "{}" must be unique.'
            '\n  Exiting.'.format(ini['INPUTS']['zone_field']))
        return False

    # Get projection from shapefile to build EE geometries
    # GeoJSON technically should always be EPSG:4326 so don't assume
    #  coordinates system property will be set
    zones_osr = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zones_wkt = gdc.osr_wkt(zones_osr)

    # Check that shapefile has matching spatial reference
    if not gdc.matching_spatref(zones_osr, ini['SPATIAL']['osr']):
        logging.warning('  Zone OSR:\n{}\n'.format(zones_osr))
        logging.warning('  Output OSR:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.warning('  Zone Proj4:   {}'.format(
            zones_osr.ExportToProj4()))
        logging.warning('  Output Proj4: {}'.format(
            ini['SPATIAL']['osr'].ExportToProj4()))
        logging.warning(
            '\nWARNING: \n'
            'The output and zone spatial references do not appear to match\n'
            'This will likely cause problems!')
        input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(zones_wkt))
        logging.debug('  Output Projection:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.debug('  Output Cellsize: {}'.format(
            ini['SPATIAL']['cellsize']))

    # Initialize Earth Engine API key
    logging.info('\nInitializing Earth Engine')
    ee.Initialize()
    utils.ee_getinfo(ee.Number(1))

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

    # Calculate zonal stats for each image separately
    logging.debug('\nComputing zonal stats')
    if ini['ZONAL_STATS']['modis_daily_flag']:
        modis_daily_func(
            modis_fields, ini, zones_geojson, zones_wkt, overwrite_flag)


def modis_daily_func(export_fields, ini, zones_geojson, zones_wkt,
                     overwrite_flag=False):
    """

    Parameters
    ----------
    export_fields : list
    ini : dict
        Input file parameters.
    zones_geojson : dict
        Zones GeoJSON.
    zones_wkt : str
        Zones spatial reference Well Known Text.
    overwrite_flag : bool, optional
        If True, overwrite existing values (the default is False).
        Don't remove/replace the CSV file directly.

    """

    logging.info('\nMODIS')

    # DEADBEEF - For now, hardcode transform to a standard MODIS image
    # ini['EXPORT']['transform'] = [
    #     231.656358264, 0.0, -20015109.354, 0.0, -231.656358264, 10007554.677]
    # ini['EXPORT']['transform'] = '[{}]'.format(','.join(
    #     map(str, (500.0, 0.0, 0.0, 0.0, -500.0, 0.0))))
    # logging.debug('    Output Transform: {}'.format(
    #     ini['EXPORT']['transform']))

    # Only keep daily MODIS products
    modis_products = [
        p for p in ini['ZONAL_STATS']['modis_products']
        if p.split('_')[-1].upper() in ee_tools.modis.daily_collections]
    modis_fields = [f.upper() for f in modis_products]

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
    # # print(modis.get_daily_collection().aggregate_histogram('DATE').getInfo())
    # input('ENTER')

    def csv_writer(output_df, output_path, output_fields):
        """Write the dataframe to CSV with custom formatting"""
        csv_df = output_df.copy()

        # Convert float fields to objects, set NaN to None
        for field in csv_df.columns.values:
            if field.upper() not in modis_fields:
                continue
            csv_df[field] = csv_df[field].astype(object)
            null_mask = csv_df[field].isnull()
            csv_df.loc[null_mask, field] = None
            csv_df.loc[~null_mask, field] = csv_df.loc[~null_mask, field].map(
                lambda x: '{0:10.6f}'.format(x).strip())

        # Set field types
        for field in ['ZONE_FID', 'YEAR', 'MONTH', 'DAY', 'DOY']:
            csv_df[field] = csv_df[field].astype(int)
        # if csv_df['ZONE_NAME'].dtype == np.float64:
        #     csv_df['ZONE_NAME'] = csv_df['ZONE_NAME'].astype(int).astype(str)

        csv_df.reset_index(drop=False, inplace=True)
        csv_df.sort_values(by=['DATE'], inplace=True)
        csv_df.to_csv(output_path, index=False,
                      columns=output_fields)

    # Master list of zone dictionaries (name, fid, csv and ee.Geometry)
    zones = []
    for z_ftr in zones_geojson['features']:
        zone_name = str(
            z_ftr['properties'][ini['INPUTS']['zone_field']]).replace(' ', '_')

        # zone_geom = ogr.CreateGeometryFromJson(json.dumps(zone['json']))
        # # zone_area = zone_geom.GetArea()
        # if zone_geom.GetGeometryName() in ['POINT', 'MULTIPOINT']:
        #     # Compute area as cellsize * number of points
        #     point_count = 0
        #     for i in range(0, zone_geom.GetGeometryCount()):
        #         point_count += zone_geom.GetGeometryRef(i).GetPointCount()
        #     zone['area'] = ini['SPATIAL']['cellsize'] * point_count
        # else:
        #     # Adjusting area up to nearest multiple of cellsize to account for
        #     #   polygons that were modified to avoid interior holes
        #     zone_area = ini['SPATIAL']['cellsize'] * math.ceil(
        #         zone_geom.GetArea() / ini['SPATIAL']['cellsize'])

        # I have to build an EE geometry in order to set a non-WGS84 projection
        zones.append({
            'name': zone_name,
            'fid': int(z_ftr['id']),
            # 'area': zone_geom.GetArea(),
            'csv': os.path.join(
                ini['ZONAL_STATS']['output_ws'], zone_name,
                '{}_modis_daily.csv'.format(zone_name)),
            'ee_geom': ee.Geometry(
                geo_json=z_ftr['geometry'], opt_proj=zones_wkt,
                opt_geodesic=False)
        })

    # Read in all output CSV files
    logging.debug('\n  Reading CSV files')
    zone_df_list = []
    for zone in zones:
        logging.debug(
            '  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))

        # Build output folder if necessary
        if not os.path.isdir(os.path.dirname(zone['csv'])):
            os.makedirs(os.path.dirname(zone['csv']))

        # Make copy of export field list in order to retain existing columns
        # DEADBEEF - This won't work correctly in the zone loop
        output_fields = export_fields[:]

        # Read existing output table if possible
        logging.debug('    Reading CSV')
        logging.debug('    {}'.format(zone['csv']))
        try:
            zone_df = pd.read_csv(zone['csv'], parse_dates=['DATE'])
            zone_df['DATE'] = zone_df['DATE'].dt.strftime('%Y-%m-%d')
            # Move any existing columns not in export_fields to end of CSV
            # output_fields.extend([
            #     f for f in zone_df.columns.values if f not in export_fields])
            # zone_df = zone_df.reindex(columns=output_fields)
            # zone_df.sort_values(by=['DATE', 'ROW'], inplace=True)
            zone_df_list.append(zone_df)
        except IOError:
            logging.debug('    Output path doesn\'t exist, skipping')
        except AttributeError:
            logging.debug('    Output CSV appears to be empty')
        except Exception as e:
            logging.exception(
                '    ERROR: Unhandled Exception\n    {}'.format(e))
            input('ENTER')

    # Combine separate zone dataframes
    try:
        output_df = pd.concat(zone_df_list)
    except ValueError:
        logging.debug(
            '    Output path(s) doesn\'t exist, building empty dataframe')
        output_df = pd.DataFrame(columns=export_fields)
    except Exception as e:
        logging.exception(
            '    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')
    del zone_df_list

    output_df.ZONE_NAME = output_df.ZONE_NAME.astype(str)
    output_df.set_index(['ZONE_NAME', 'DATE'], inplace=True, drop=True)

    # Get list of possible dates based on INI
    export_dates = set(
        date_str for date_str in utils.date_range(
            '{}-01-01'.format(ini['INPUTS']['start_year']),
            '{}-12-31'.format(ini['INPUTS']['end_year'])))
    # logging.debug('  Export Dates: {}'.format(
    #     ', '.join(sorted(export_dates))))

    # For overwrite, drop all expected entries from existing output DF
    if overwrite_flag:
        output_df = output_df[
            ~output_df.index.get_level_values('DATE').isin(list(export_dates))]

    # zone_df below is intentionally being made as a subset copy of output_df
    # This raises a warning though
    pd.options.mode.chained_assignment = None  # default='warn'

    # Add empty entries separately for each zone
    logging.debug('\n  Identifying dates/zones with missing data')
    for zone in zones:
        logging.debug('  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))

        # Subset the data frame by zone name
        try:
            zone_df = output_df.iloc[
                output_df.index.get_level_values('ZONE_NAME') == zone['name']]
            zone_df.reset_index(inplace=True)
            zone_df.set_index(['DATE'], inplace=True, drop=True)
        except Exception as e:
            # This seems to happen with output_df is empty (overwrite_flag=True)
            logging.debug('    Exception: {}'.format(e))
            zone_df = pd.DataFrame()
            input('ENTER')

        # Get list of existing dates in the CSV
        if not zone_df.empty:
            zone_df_dates = set(zone_df.index.values)
        else:
            zone_df_dates = set()
        logging.debug('    Dates in zone CSV: {}'.format(
            ', '.join(zone_df_dates)))

        # List of DATES that are entirely missing
        # This may include scenes that don't intersect the zone
        missing_all_dates = export_dates - zone_df_dates
        logging.debug('    Dates not in CSV (missing all values): {}'.format(
            ', '.join(sorted(missing_all_dates))))
        if not missing_all_dates:
            continue

        # Create empty entries for all scenes that are missing
        # Identify whether the missing scenes intersect the zone or not
        # Set pixels counts for non-intersecting DATES to 0
        # Set pixel counts for intersecting DATES to NaN
        #   This will allow the zonal stats to set actual pixel count value
        logging.debug('    Appending all empty DATES')
        missing_df = pd.DataFrame(
            index=sorted(missing_all_dates), columns=output_df.columns)
        missing_df.index.name = 'DATE'
        missing_df['ZONE_NAME'] = zone['name']
        missing_df['ZONE_FID'] = zone['fid']
        missing_df['YEAR'] = pd.to_datetime(missing_df.index).year
        missing_df['MONTH'] = pd.to_datetime(missing_df.index).month
        missing_df['DAY'] = pd.to_datetime(missing_df.index).day
        missing_df['DOY'] = pd.to_datetime(missing_df.index).dayofyear.astype(int)
        # missing_df['AREA'] = zone['area']
        # missing_df['QA'] = np.nan
        # missing_df['PIXEL_SIZE'] = landsat.cellsize
        # missing_df['PIXEL_COUNT'] = np.nan
        # missing_df['PIXEL_TOTAL'] = np.nan
        # missing_df['FMASK_COUNT'] = np.nan
        # missing_df['FMASK_TOTAL'] = np.nan
        # missing_df['FMASK_PCT'] = np.nan
        # missing_df['CLOUD_SCORE'] = np.nan

        # Remove the overlapping missing entries
        # Then append the new missing entries to the zone CSV
        # if zone_df.index.intersection(missing_df.index).any():
        try:
            zone_df.drop(
                zone_df.index.intersection(missing_df.index),
                inplace=True)
        except ValueError:
            pass
        zone_df = zone_df.append(missing_df, sort=False)
        logging.debug('    Writing to CSV')
        csv_writer(zone_df, zone['csv'], export_fields)

        # Update the master dataframe
        zone_df.reset_index(inplace=True)
        zone_df.set_index(
            ['ZONE_NAME', 'DATE'], inplace=True, drop=True)
        try:
            output_df.drop(
                output_df.index.get_level_values('ZONE_NAME')==zone['name'],
                inplace=True)
        except Exception as e:
            # These seem to happen with the zone is not in the output_df
            logging.debug('    Exception: {}'.format(e))
            pass
        output_df = output_df.append(zone_df, sort=False)

    # Putting the warning back to the default balue
    pd.options.mode.chained_assignment = 'warn'

    # Identify SCENE_IDs that are missing any data
    # Filter based on product and SCENE_ID lists
    missing_date_mask = output_df.index.get_level_values('DATE') \
        .isin(export_dates)
    # missing_id_mask = (
    #     (output_df['PIXEL_COUNT'] != 0) &
    #     output_df.index.get_level_values('SCENE_ID').isin(export_ids))
    products = [f.upper() for f in ini['ZONAL_STATS']['landsat_products']]
    missing_df = output_df.loc[missing_date_mask, products].isnull()

    # List of DATES and products with some missing data
    missing_dates = set(missing_df[missing_df.any(axis=1)]
                          .index.get_level_values('DATE').values)

    # Skip processing if all dates already exist in the CSV
    if not missing_dates and not overwrite_flag:
        logging.info('\n  All scenes present, returning')
        return True
    else:
        logging.debug('\n  Scenes missing values: {}'.format(
            ', '.join(sorted(missing_dates))))

    def export_update(data_df):
        """Set/modify ancillary field values in the export CSV dataframe"""
        # First, remove any extra rows that were added for exporting
        # The DEADBEEF entry is added because the export structure is based on
        #   the first feature in the collection, so fields with nodata will
        #   be excluded
        data_df.drop(
            data_df[data_df['DATE'] == 'DEADBEEF'].index,
            inplace=True)

        # Add additional fields to the export data frame
        if not data_df.empty:
            data_df['DATE'] = pd.to_datetime(data_df['DATE'].str, format='%Y%m%d')
            data_df['YEAR'] = data_df['DATE'].dt.year
            data_df['MONTH'] = data_df['DATE'].dt.month
            data_df['DAY'] = data_df['DATE'].dt.day
            data_df['DOY'] = data_df['DATE'].dt.dayofyear.astype(int)
            data_df['DATE'] = data_df['DATE'].dt.strftime('%Y-%m-%d')
            # data_df['AREA'] = data_df['PIXEL_COUNT'] * modis.cellsize ** 2
            # data_df['PIXEL_SIZE'] = modis.cellsize

            # fmask_mask = data_df['CLOUD_TOTAL'] > 0
            # if fmask_mask.any():
            #     data_df.loc[fmask_mask, 'CLOUD_PCT'] = 100.0 * (
            #         data_df.loc[fmask_mask, 'CLOUD_COUNT'] /
            #         data_df.loc[fmask_mask, 'CLOUD_TOTAL'])
            # data_df['QA'] = 0

        # Remove unused export fields
        if 'system:index' in data_df.columns.values:
            del data_df['system:index']
        if '.geo' in data_df.columns.values:
            del data_df['.geo']

        data_df.set_index(
            ['ZONE_NAME', 'DATE'], inplace=True, drop=True)
        return data_df

    # Write values to file after each year
    # Group export dates by year (and path/row also?)
    export_dates_iter = [
        [year, list(dates)]
        for year, dates in groupby(sorted(export_dates), lambda x: x[:4])]
    for export_year, export_dates in export_dates_iter:
        logging.debug('\n  Iter year: {}'.format(export_year))

        for export_date in sorted(export_dates):
            export_dt = datetime.datetime.strptime(export_date, '%Y-%m-%d')
            next_date = (export_dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            logging.info('  {}'.format(export_date))

            # DEADBEEF - There has to be a way to do this selection in one line
            date_df = output_df.iloc[
                output_df.index.get_level_values('DATE') == export_date]
            # scene_df = scene_df[
            #     (scene_df['PIXEL_COUNT'] > 0) |
            #     (scene_df['PIXEL_COUNT'].isnull())]
            if date_df.empty:
                logging.info('\n  No missing data, skipping')
                input('ENTER')
                continue
            date_df.reset_index(inplace=True)
            date_df.set_index(['ZONE_NAME'], inplace=True, drop=True)

            # # Update/limit products list if necessary
            # export_products = set(
            #     f.lower()
            #     for f in scene_df.isnull().any(axis=0).index.values
            #     if f.lower() in modis_products)
            # logging.debug('  Products missing any values: {}'.format(
            #     ', '.join(sorted(export_products))))
            # modis.products = list(export_products)

            # Identify zones with any missing data
            export_zones = set(date_df[date_df.any(axis=1)].index)
            logging.debug('  Zones with missing data: {}'.format(
                ', '.join(sorted(export_zones))))
            #
            # Build collection of all features to test for each SCENE_ID
            zone_coll = ee.FeatureCollection([
                ee.Feature(
                    zone['ee_geom'],
                    {'ZONE_NAME': zone['name'], 'ZONE_FID': zone['fid']})
                for zone in zones if zone['name'] in export_zones])

            # Collection should only have one image
            modis.start_date = export_date
            modis.end_date = next_date

            for product in modis_products:
                logging.info('  {}'.format(product))
                modis_coll = modis.get_daily_collection(product)

                # DEBUG - Test that the MODIS collection is getting built
                # print(modis_coll.aggregate_histogram('DATE').getInfo())
                # input('ENTER')
                # print('Bands: {}'.format(
                #     [x['id'] for x in ee.Image(modis_coll.first()).getInfo()['bands']]))
                # print('Date: {}'.format(
                #     ee.Image(modis_coll.first()).getInfo()['properties']['DATE']))
                # input('ENTER')
                # if ee.Image(modis_coll.first()).getInfo() is None:
                #     logging.info('    No images, skipping')
                #     continue

                # Map over features dfor one image
                image = ee.Image(modis_coll.first())

                def zonal_stats_func(ftr):
                    """"""
                    date = ee.Date(image.get('system:time_start'))
                    # doy = ee.Number(date.getRelative('day', 'year')).add(1)
                    # bands = len(modis.products) + 1

                    # Using the feature/geometry should make it unnecessary to clip
                    input_mean = ee.Image(image) \
                        .select([product.upper()]) \
                        .reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=ftr.geometry(),
                            crs=ini['SPATIAL']['crs'],
                            crsTransform=ini['EXPORT']['transform'],
                            bestEffort=False,
                            tileScale=1)
                            # maxPixels=zone['max_pixels'] * bands)

                    # # Count unmasked cloud pixels to get pixel count
                    # # Count cloud > 1 to get cloud count (0 is clear and 1 is water)
                    # cloud_img = ee.Image(image).select(['cloud'])
                    # input_count = ee.Image([
                    #         cloud_img.gte(0).unmask().rename(['pixel']),
                    #         cloud_img.gt(1).rename(['fmask'])]) \
                    #     .reduceRegion(
                    #         reducer=ee.Reducer.sum().combine(
                    #             ee.Reducer.count(), '', True),
                    #         geometry=ftr.geometry(),
                    #         crs=ini['SPATIAL']['crs'],
                    #         crsTransform=ini['EXPORT']['transform'],
                    #         bestEffort=False,
                    #         tileScale=1)
                    #         # maxPixels=zone['max_pixels'] * 3)

                    # Standard output
                    zs_dict = {
                        'ZONE_NAME': ee.String(ftr.get('ZONE_NAME')),
                        # 'ZONE_FID': ee.Number(ftr.get('ZONE_FID')),
                        'DATE': date.format('YYYY-MM-dd'),
                        # 'YEAR': date.get('year'),
                        # 'MONTH': date.get('month'),
                        # 'DAY': date.get('day'),
                        # 'DOY': doy,
                        # 'AREA': input_count.get('pixel_count') * (modis.cellsize ** 2)
                        # 'PIXEL_SIZE': modis.cellsize,
                        # 'PIXEL_COUNT': input_count.get('pixel_sum'),
                        # 'PIXEL_TOTAL': input_count.get('pixel_count'),
                        # 'CLOUD_COUNT': input_count.get('cloud_sum'),
                        # 'CLOUD_TOTAL': input_count.get('cloud_count'),
                        # 'CLOUD_PCT': ee.Number(input_count.get('cloud_sum')) \
                        #     .divide(ee.Number(input_count.get('cloud_count'))) \
                        #     .multiply(100),
                        # 'QA': ee.Number(0)
                        product.upper(): input_mean.get(product.upper()),
                    }

                    # # Product specific output
                    # if modis.products:
                    #     zs_dict.update({
                    #         p.upper(): input_mean.get(p.lower())
                    #         for p in modis.products
                    #     })

                    return ee.Feature(None, zs_dict)

                stats_coll = zone_coll.map(zonal_stats_func, False)

                # Add a dummy entry to the stats collection
                # This is added because the export structure is based on the first
                #   entry in the collection, so fields with nodata will be excluded
                format_dict = {
                    'ZONE_NAME': 'DEADBEEF',
                    # 'ZONE_FID': -9999,
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

                if ini['EXPORT']['export_dest'] == 'getinfo':
                    logging.debug('    Requesting data')
                    export_df = pd.DataFrame([
                        ftr['properties']
                        for ftr in utils.ee_getinfo(stats_coll)['features']])
                    print(export_df)
                    input('ENTER')
                    export_df = export_update(export_df)

                    # Save data to main dataframe
                    if not export_df.empty:
                        logging.debug('    Processing data')
                        if overwrite_flag:
                            # Update happens inplace automatically
                            output_df.update(export_df)
                            # output_df = output_df.append(export_df, sort=False)
                        else:
                            # Combine first doesn't have an inplace parameter
                            output_df = output_df.combine_first(export_df)

        # # Save updated CSV
        # # if output_df is not None and not output_df.empty:
        # if not output_df.empty:
        #     logging.info('\n  Writing zone CSVs')
        #     for zone in zones:
        #         logging.debug(
        #             '  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))
        #         # logging.debug('    {}'.format(zone_output_path))
        #         zone_df = output_df.iloc[
        #             output_df.index.get_level_values('ZONE_NAME')==zone['name']]
        #         csv_writer(zone_df, zone['csv'], export_fields)
        # else:
        #     logging.info('\n  Empty output dataframe')


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine zonal statistics by image',
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
