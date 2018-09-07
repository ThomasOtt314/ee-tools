#--------------------------------
# Name:         ee_zonal_stats_by_zone_modis.py
# Purpose:      Download zonal stats by zone using Earth Engine
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
# from collections import defaultdict
import datetime
# from io import StringIO
import json
import logging
import math
import os
import pprint
# import re
# import requests
from subprocess import check_output
import sys

import ee
# import numpy as np
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
    logging.info('\nEarth Engine zonal statistics by zone')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='EXPORT')
    inputs.parse_section(ini, section='ZONAL_STATS')

    # These may eventually be set in the INI file
    modis_daily_fields = [
        'ZONE_NAME', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY']
    #     'AREA', 'PIXEL_SIZE', 'PIXEL_COUNT', 'PIXEL_TOTAL',
    #     'CLOUD_COUNT', 'CLOUD_TOTAL', 'CLOUD_PCT', 'QA']

    modis_daily_fields.extend(
        p.upper()
        for p in ini['ZONAL_STATS']['modis_products']
        if p.split('_')[-1].upper() in ee_tools.modis.collections_daily)

    # Convert the shapefile to geojson
    # if not os.path.isfile(ini['ZONAL_STATS']['zone_geojson']):
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
            ini['ZONAL_STATS']['zone_geojson'],
            ini['INPUTS']['zone_shp_path']])

    # # Get ee features from shapefile
    # zone_geom_list = gdc.shapefile_2_geom_list_func(
    #     ini['INPUTS']['zone_shp_path'],
    #     zone_field=ini['INPUTS']['zone_field'],
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
        str(z['properties'][ini['INPUTS']['zone_field']]) \
            .replace(' ', '_').lower()
        for z in zones_geojson['features']]
    if len(set(zone_names)) != len(zones_geojson['features']):
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
    utils.ee_request(ee.Number(1).getInfo())

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


    # Calculate zonal stats for each feature separately
    logging.info('')
    for zone_ftr in zones_geojson['features']:
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
        #     zone['geom'].centroid(1).getInfo()['coordinates']))

        # Use feature geometry to build extent, transform, and shape
        zone_geom = ogr.CreateGeometryFromJson(json.dumps(zone['json']))
        if zone_geom.GetGeometryName() in ['POINT', 'MULTIPOINT']:
            # Compute area as cellsize * number of points
            point_count = 0
            for i in range(0, zone_geom.GetGeometryCount()):
                point_count += zone_geom.GetGeometryRef(i).GetPointCount()
            zone['area'] = ini['SPATIAL']['cellsize'] * point_count
        else:
            zone['area'] = zone_geom.GetArea()
            # Adjusting area up to nearest multiple of cellsize to account for
            #   polygons that were modified to avoid interior holes
            # zone['area'] = ini['SPATIAL']['cellsize'] * math.ceil(
            #     zone_geom.GetArea() / ini['SPATIAL']['cellsize'])

        # zone['area'] = zone_geom.GetArea()
        zone['extent'] = gdc.Extent(zone_geom.GetEnvelope())
        # zone['extent'] = gdc.Extent(zone['geom'].GetEnvelope())
        zone['extent'] = zone['extent'].ogrenv_swap()
        zone['extent'] = zone['extent'].adjust_to_snap(
            'EXPAND', ini['SPATIAL']['snap_x'], ini['SPATIAL']['snap_y'],
            ini['SPATIAL']['cellsize'])
        zone['geo'] = zone['extent'].geo(ini['SPATIAL']['cellsize'])
        zone['transform'] = gdc.geo_2_ee_transform(zone['geo'])
        # zone['transform'] = '[' + ','.join(map(str, zone['transform'])) + ']'
        zone['shape'] = zone['extent'].shape(ini['SPATIAL']['cellsize'])
        logging.debug('    Zone Shape: {}'.format(zone['shape']))
        logging.debug('    Zone Transform: {}'.format(zone['transform']))
        logging.debug('    Zone Extent: {}'.format(zone['extent']))
        # logging.debug('    Zone Geom: {}'.format(zone['geom'].getInfo()))

        # Assume all pixels in all images could be reduced
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

        if ini['ZONAL_STATS']['modis_daily_flag']:
            modis_daily_func(modis_daily_fields, ini, zone, overwrite_flag)


def modis_daily_func(export_fields, ini, zone, overwrite_flag=False):
    """

    Function will attempt to generate export tasks only for missing DATES
    Also try to limit the products to only those with missing data

    Parameters
    ----------
    export_fields : list
    ini : dict
        Input file parameters.
    zone : dict
        Zone specific parameters.
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).
        Don't remove/replace the CSV file directly.

    """
    logging.info('  MODIS')

    # DEADBEEF - For now, hardcode transform to a standard MODIS image
    # ini['EXPORT']['transform'] = [
    #     231.656358264, 0.0, -20015109.354, 0.0, -231.656358264, 10007554.677]
    # logging.debug('    Output Transform: {}'.format(
    #     ini['EXPORT']['transform']))

    # Only keep daily MODIS products
    modis_products = [
        p for p in ini['ZONAL_STATS']['modis_products'][:]
        if p.split('_')[-1].upper() in ee_tools.modis.collections_daily]
    modis_fields = [p.upper() for p in modis_products]

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

    def csv_writer(output_df, output_path, output_fields):
        """Write the dataframe to CSV with custom formatting"""
        csv_df = output_df.copy()

        # Convert float fields to objects, set NaN to None
        float_fields = modis_fields[:]
        for field in csv_df.columns.values:
            if field.upper() not in float_fields:
                continue
            csv_df[field] = csv_df[field].astype(object)
            null_mask = csv_df[field].isnull()
            csv_df.loc[null_mask, field] = None
            csv_df.loc[~null_mask, field] = csv_df.loc[~null_mask, field].map(
                lambda x: '{0:10.6f}'.format(x).strip())
            # csv_df.loc[~null_mask, [field]] = csv_df.loc[~null_mask, [field]].apply(
            #     lambda x: '{0:10.6f}'.format(x[0]).strip(), axis=1)

        # Set field types
        for field in ['YEAR', 'MONTH', 'DAY', 'DOY']:
            csv_df[field] = csv_df[field].astype(int)
        # if csv_df['ZONE_NAME'].dtype == np.float64:
        #     csv_df['ZONE_NAME'] = csv_df['ZONE_NAME'].astype(int).astype(str)

        # csv_df.reset_index(drop=False, inplace=True)
        csv_df.sort_values(by=['DATE'], inplace=True)
        csv_df.to_csv(output_path, index=False, columns=output_fields)

    def clean_export_df(data_df):
        """Set/modify ancillary field values in the export CSV dataframe"""
        # Remove any extra rows that were added for exporting
        # The DEADBEEF entry is added because the export structure is based on
        #   the first feature in the collection, so fields with nodata will
        #   be excluded
        data_df.drop(data_df[data_df['DATE'] == 'DEADBEEF'].index,
                     inplace=True)

        # Remove unused export fields
        if 'system:index' in data_df.columns.values:
            del data_df['system:index']
        if '.geo' in data_df.columns.values:
            del data_df['.geo']

        # data_df.set_index('DATE', inplace=True, drop=True)

        return data_df

    # Make copy of export field list in order to retain existing columns
    output_fields = export_fields[:]

    # Read existing output table if possible
    # Otherwise build an empty dataframe
    # Only add new empty fields when reading in (don't remove any existing)
    output_id = output_id = '{}_modis_daily'.format(zone['name'])
    output_path = os.path.join(zone['output_ws'], output_id + '.csv')
    logging.debug('    {}'.format(output_path))
    try:
        output_df = pd.read_csv(output_path, parse_dates=['DATE'])
        output_df['DATE'] = output_df['DATE'].dt.strftime('%Y-%m-%d')
        output_df['ZONE_NAME'] = output_df['ZONE_NAME'].astype(str)

        # Move any existing columns not in export_fields to end of CSV
        output_fields.extend([
            f for f in output_df.columns.values if f not in export_fields])
        output_df = output_df.reindex(columns=output_fields)
        output_df.sort_values(by=['DATE'], inplace=True)
    except IOError:
        logging.debug(
            '    Output path doesn\'t exist, building empty dataframe')
        output_df = pd.DataFrame(columns=export_fields)
    except Exception as e:
        logging.exception('    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')

    # # # Use the DATE as the index
    # # output_df.set_index('DATE', inplace=True, drop=True)
    # # output_df.index.name = 'DATE'
    # # output_df['YEAR'] = pd.to_datetime(output_df.index).year
    # # output_df['MONTH'] = pd.to_datetime(output_df.index).month
    # # output_df['DAY'] = pd.to_datetime(output_df.index).day
    # # output_df['DOY'] = pd.to_datetime(output_df.index).dayofyear.astype(int)
    #
    # # Don't use DATE as the index
    # output_df['DATE'] = pd.to_datetime(output_df['DATE'], format='%Y-%m-%d')
    # output_df['YEAR'] = output_df['DATE'].dt.year
    # output_df['MONTH'] = output_df['DATE'].dt.month
    # output_df['DAY'] = output_df['DATE'].dt.day
    # output_df['DOY'] = output_df['DATE'].dt.dayofyear.astype(int)
    # output_df['DATE'] = output_df['DATE'].dt.strftime('%Y-%m-%d')

    # Since MODIS daily data should generally have global data for each day,
    # build date list from INI parameters instead of from collection dates
    # Querying the collection is probably be better for 8 and 16 day products
    export_dates = [
        date_dt for date_dt in utils.date_range(
            '{}-01-01'.format(ini['INPUTS']['start_year']),
            '{}-12-31'.format(ini['INPUTS']['end_year']))
        if date_dt < datetime.datetime.today()]
    if ini['INPUTS']['start_month'] and ini['INPUTS']['start_month'] > 1:
        export_dates = [d for d in export_dates
                        if d.month >= ini['INPUTS']['start_month']]
    if ini['INPUTS']['end_month'] and ini['INPUTS']['end_month'] < 12:
        export_dates = [d for d in export_dates
                        if d.month <= ini['INPUTS']['end_month']]
    if ini['INPUTS']['start_doy'] and ini['INPUTS']['start_doy'] > 1:
        export_dates = [d for d in export_dates
                        if d.month >= ini['INPUTS']['start_doy']]
    if ini['INPUTS']['end_doy'] and ini['INPUTS']['end_doy'] > 1:
        export_dates = [d for d in export_dates
                        if d.month <= ini['INPUTS']['end_doy']]
    export_dates = set(d.strftime('%Y-%m-%d') for d in export_dates)
    logging.debug('  Export Dates: {}'.format(
        ', '.join(sorted(export_dates))))

    # For overwrite, drop all expected entries from existing output DF
    if overwrite_flag:
        output_df = output_df[~output_df['DATE'].isin(list(export_dates))]

    # List of DATES that are entirely missing
    missing_all_dates = export_dates - set(output_df['DATE'].values)
    logging.debug('  DATES missing all values: {}'.format(
        ', '.join(sorted(missing_all_dates))))

    if missing_all_dates:
        logging.debug('  Appending empty DATES')

        # # Use DATE as index
        # missing_df = pd.DataFrame(index=sorted(missing_all_dates),
        #                           columns=output_df.columns)
        # missing_df['ZONE_NAME'] = zone['name']
        # missing_df.index.name = 'DATE'
        # missing_df['YEAR'] = pd.to_datetime(missing_df.index).year
        # missing_df['MONTH'] = pd.to_datetime(missing_df.index).month
        # missing_df['DAY'] = pd.to_datetime(missing_df.index).day
        # missing_df['DOY'] = pd.to_datetime(missing_df.index).dayofyear.astype(int)

        # Don't use DATE as index
        missing_df = pd.DataFrame(index=range(len(missing_all_dates)),
                                  columns=output_df.columns)
        missing_df['DATE'] = sorted(missing_all_dates)
        missing_df['ZONE_NAME'] = zone['name']
        missing_df['DATE'] = pd.to_datetime(missing_df['DATE'],
                                            format='%Y-%m-%d')
        missing_df['YEAR'] = missing_df['DATE'].dt.year
        missing_df['MONTH'] = missing_df['DATE'].dt.month
        missing_df['DAY'] = missing_df['DATE'].dt.day
        missing_df['DOY'] = missing_df['DATE'].dt.dayofyear.astype(int)
        missing_df['DATE'] = missing_df['DATE'].dt.strftime('%Y-%m-%d')

        # missing_df['AREA'] = zone['area']
        # missing_df['QA'] = np.nan
        # missing_df['QA'] = 0
        # missing_df['PIXEL_SIZE'] = modis.cellsize
        # missing_df['PIXEL_COUNT'] = 0
        # missing_df['PIXEL_TOTAL'] = 0
        # missing_df['CLOUD_COUNT'] = 0
        # missing_df['CLOUD_TOTAL'] = 0
        # missing_df['CLOUD_PCT'] = np.nan
        # missing_df[f] = missing_df[f].astype(int)

        # Remove the overlapping missing entries
        # Then append the new missing entries
        # if (not output_df.empty and not missing_df.empty and
        #         output_df['DATE'].intersection(missing_df['DATE']).any()):
        try:
            output_df.drop(output_df['DATE'].intersection(missing_df['DATE']),
                           inplace=True)
        except Exception as e:
            pass

        if not missing_df.empty:
            logging.debug('    Writing to CSV')
            output_df = output_df.append(missing_df, sort=False)
            csv_writer(output_df, output_path, output_fields)
        del missing_df

    for product in modis_products:
        product = product.upper()
        logging.info('  Product: {}'.format(product))

        # # The export dates could be computed separately for each product
        # # by getting the DATE list for the product collection.
        # modis_coll = modis.get_daily_collection(product)
        # export_dates = utils.ee_getinfo(modis_coll.aggregate_histogram('DATE'))

        # Identify DATES that are missing some data for the product
        missing_df = output_df.loc[
            output_df['DATE'].isin(export_dates), ['DATE', product]]
        missing_mask = missing_df[product].isnull()
        if not any(missing_mask):
            continue
        missing_dates = set(missing_df.loc[missing_mask, 'DATE'].values)
        logging.debug('    DATES missing product values: {}'.format(
            ', '.join(sorted(missing_dates))))

        # Don't use the date_keep_list if all the values are missing
        # Build the date list in the MODIS system:index format
        if not all(missing_mask):
            modis.date_keep_list = sorted([
                datetime.datetime.strptime(d, '%Y-%m-%d').strftime('%Y_%m_%d')
                for d in missing_dates])

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

            # Filter by iteration date in addition to input date parameters
            modis.start_date = start_date
            modis.end_date = end_date
            # modis.start_year = start_year
            # modis.end_year = end_year

            # Skip year range if DATE keep list is set and doesn't match
            if (modis.date_keep_list and not any(
                    set(int(x[0:4]) for x in modis.date_keep_list) &
                    set(range(start_year, end_year + 1)))):
                logging.debug('    {}  {}'.format(start_date, end_date))
                logging.debug('      No matching DATES for year range')
                continue
            else:
                logging.info('    {}  {}'.format(start_date, end_date))

            # Calculate values and statistics
            # Build function in loop to set water year ETo/PPT values
            def zonal_stats_func(image):
                """"""
                date = ee.Date(image.get('system:time_start'))
                bands = 1

                # Using zone['geom'] as the geometry should make it
                #   unnecessary to clip also
                input_mean = ee.Image(image) \
                    .select([product]) \
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=zone['geom'],
                        crs=ini['SPATIAL']['crs'],
                        crsTransform=ini['EXPORT']['transform'],
                        bestEffort=False,
                        tileScale=1)

                zs_dict = {
                    'ZONE_NAME': str(zone['name']),
                    'DATE': date.format('YYYY-MM-dd'),
                    product: input_mean.get(product),
                }

                return ee.Feature(None, zs_dict)

            modis_coll = modis.get_daily_collection(product)
            stats_coll = modis_coll.map(zonal_stats_func, False)

            # # DEBUG - Test the function for a single image
            # stats_info = zonal_stats_func(
            #     ee.Image(modis_coll.first())).getInfo()
            # pp.pprint(stats_info['properties'])
            # input('ENTER')
            #
            # # DEBUG - Print the stats info to the screen
            # stats_info = stats_coll.getInfo()
            # for ftr in stats_info['features']:
            #     pp.pprint(ftr)
            # input('ENTER')
            # # return False

            # Add a dummy entry to the stats collection
            format_dict = {
                'ZONE_NAME': 'DEADBEEF',
                'DATE': 'DEADBEEF',
                product: -9999,
            }
            stats_coll = ee.FeatureCollection(ee.Feature(None, format_dict)) \
                .merge(stats_coll)

            # # DEBUG - Print the stats info to the screen
            # stats_info = stats_coll.getInfo()
            # for ftr in stats_info['features']:
            #     pp.pprint(ftr)
            # input('ENTER')

            logging.debug('    Requesting data')
            export_info = utils.ee_getinfo(stats_coll)['features']
            export_df = pd.DataFrame([f['properties'] for f in export_info])
            export_df = clean_export_df(export_df)

            # Save data to main dataframe
            # Combine first needs DATE as the index to work
            if not export_df.empty:
                logging.debug('    Processing data')
                export_df.set_index(['DATE'], inplace=True, drop=True)
                output_df.set_index(['DATE'], inplace=True, drop=True)
                if overwrite_flag:
                    # Update happens inplace automatically
                    output_df.update(export_df)
                else:
                    # Combine first doesn't have an inplace parameter
                    output_df = output_df.combine_first(export_df)
                output_df.reset_index(inplace=True, drop=False)


    # Save updated CSV
    if output_df is not None and not output_df.empty:
        logging.info('  Writing CSV')
        csv_writer(output_df, output_path, output_fields)
    else:
        logging.info(
            '  Empty output dataframe\n'
            '  The exported CSV files may not be ready')


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
