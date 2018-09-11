#--------------------------------
# Name:         ee_zonal_stats_by_image_modis.py
# Purpose:      Download zonal stats by image using Earth Engine
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
import datetime
from itertools import groupby
import json
import logging
import os
import pprint
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
    modis_fields = ['ZONE_NAME', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY']

    # Add the products to the field lists
    modis_daily_fields = list(modis_fields) + [
        p.upper() for p in ini['ZONAL_STATS']['modis_products']
        if p.upper() in ee_tools.modis.PRODUCTS_DAILY]
    modis_8day_fields = list(modis_fields) + [
        p.upper() for p in ini['ZONAL_STATS']['modis_products']
        if p.upper() in ee_tools.modis.PRODUCTS_8DAY]
    modis_16day_fields = list(modis_fields) + [
        p.upper() for p in ini['ZONAL_STATS']['modis_products']
        if p.upper() in ee_tools.modis.PRODUCTS_16DAY]

    # Need a better way of ensuring zenith is always a product
    # Maybe in the inputs.py?
    if 'ZENITH_MOD09GA' not in modis_daily_fields:
        modis_daily_fields.append('ZENITH_MOD09GA')
    if 'ZENITH_MYD09GA' not in modis_daily_fields:
        modis_daily_fields.append('ZENITH_MYD09GA')

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
    zone_osr = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zone_wkt = gdc.osr_wkt(zone_osr)
    logging.debug('  Zone Projection:\n{}\n'.format(zone_wkt))
    logging.debug('  Output Projection:\n{}\n'.format(
        ini['SPATIAL']['osr'].ExportToWkt()))

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
            modis_daily_fields, ini, zones_geojson, zone_wkt, overwrite_flag)
    if ini['ZONAL_STATS']['modis_8day_flag']:
        modis_8day_func(
            modis_8day_fields, ini, zones_geojson, zone_wkt, overwrite_flag)
    if ini['ZONAL_STATS']['modis_16day_flag']:
        modis_16day_func(
            modis_16day_fields, ini, zones_geojson, zone_wkt, overwrite_flag)

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

    logging.info('\nMODIS Daily')

    # Build the transform from the INI values
    # ini['EXPORT']['transform'] = [
    #     ini['SPATIAL']['cellsize'], 0.0, ini['SPATIAL']['snap_x'],
    #     0.0, -ini['SPATIAL']['cellsize'], ini['SPATIAL']['snap_y']]
    # Hardcode transform to a standard MODIS image
    # ini['EXPORT']['transform'] = [
    #     231.656358264, 0.0, -20015109.354, 0.0, -231.656358264, 10007554.677]
    # logging.debug('    Output Transform: {}'.format(
    #     ini['EXPORT']['transform']))

    # Since MODIS daily data should generally have global data for each day,
    # build date list from INI parameters instead of from collection dates
    # Querying the collection is probably be better for 8 and 16 day products
    export_dates = [
        date_dt for date_dt in utils.date_range(
            '{}-01-01'.format(ini['INPUTS']['start_year']),
            '{}-12-31'.format(ini['INPUTS']['end_year']))
        if ((date_dt < datetime.datetime.today()) and
            (date_dt >= datetime.datetime(2000, 2, 24)))]
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
    # logging.debug('  Export Dates: {}'.format(
    #     ', '.join(sorted(export_dates))))

    # Only keep daily MODIS products
    modis_products = [
        p.upper() for p in ini['ZONAL_STATS']['modis_products']
        if p.upper() in ee_tools.modis.PRODUCTS_DAILY]

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
        for field in csv_df.columns.values:
            if field.upper() not in modis_products:
                continue
            elif field.upper() not in csv_df.columns.values:
                continue
            csv_df[field] = csv_df[field].astype(object)
            null_mask = csv_df[field].isnull()
            csv_df.loc[null_mask, field] = None
            csv_df.loc[~null_mask, field] = csv_df.loc[~null_mask, field].map(
                lambda x: '{0:10.6f}'.format(x).strip())

        # Set field types
        for field in ['YEAR', 'MONTH', 'DAY', 'DOY']:
            if field.upper() not in csv_df.columns.values:
                continue
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

        return data_df

    # Master list of zone dictionaries (name, fid, csv and ee.Geometry)
    zones = []
    for z_ftr in zones_geojson['features']:
        zone_name = str(
            z_ftr['properties'][ini['INPUTS']['zone_field']]).replace(' ', '_')

        # # Build the geometry object in order to compute the area
        # zone_geom = ogr.CreateGeometryFromJson(json.dumps(z_ftr['geometry']))
        # if zone_geom.GetGeometryName() in ['POINT', 'MULTIPOINT']:
        #     # Compute area as cellsize * number of points
        #     point_count = 0
        #     for i in range(0, zone_geom.GetGeometryCount()):
        #         point_count += zone_geom.GetGeometryRef(i).GetPointCount()
        #     zone_area = ini['SPATIAL']['cellsize'] * point_count
        # else:
        #     zone_area = zone_geom.GetArea()
        #     # Adjusting area up to nearest multiple of cellsize to account for
        #     #   polygons that were modified to avoid interior holes
        #     # zone_area = ini['SPATIAL']['cellsize'] * math.ceil(
        #     #     zone_geom.GetArea() / ini['SPATIAL']['cellsize'])

        # I have to build an EE geometry in order to set a non-WGS84 projection
        zones.append({
            'name': zone_name,
            'fid': int(z_ftr['id']),
            # 'area': zone_area,
            'csv': os.path.join(
                ini['ZONAL_STATS']['output_ws'], zone_name,
                '{}_modis_daily.csv'.format(zone_name)),
            'ee_geom': ee.Geometry(
                geo_json=z_ftr['geometry'], opt_proj=zones_wkt,
                opt_geodesic=False)
        })

    # Read in all output CSV files
    logging.info('\nReading CSV files & ')
    logging.info('Identifying dates/zones with missing data')
    zone_dataframes = {}
    for zone in zones:
        logging.debug('  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))

        # Build output folder if necessary
        if not os.path.isdir(os.path.dirname(zone['csv'])):
            os.makedirs(os.path.dirname(zone['csv']))

        # Make copy of export field list in order to retain existing columns
        # DEADBEEF - This won't work correctly in the zone loop
        # output_fields = export_fields + modis_fields

        # Read existing output table if possible
        # logging.debug('    Reading CSV')
        # logging.debug('    {}'.format(zone['csv']))
        try:
            zone_df = pd.read_csv(zone['csv'], parse_dates=['DATE'])
            zone_df['DATE'] = zone_df['DATE'].dt.strftime('%Y-%m-%d')
            zone_df['ZONE_NAME'] = zone_df['ZONE_NAME'].astype(str)

            # Move any existing columns not in export_fields to end of CSV
            # output_fields.extend([
            #     f for f in zone_df.columns.values if f not in export_fields])
            # zone_df = zone_df.reindex(columns=output_fields)
            # zone_df.sort_values(by=['DATE', 'ROW'], inplace=True)
            # zone_dataframes[zone['name']] = zone_df
        except IOError:
            logging.debug('    Output CSV doesn\'t exist, building')
            zone_df = pd.DataFrame(columns=export_fields)
        except AttributeError:
            logging.debug('    Output CSV appears to be empty, re-building')
            zone_df = pd.DataFrame(columns=export_fields)
        except Exception as e:
            logging.exception(
                '    ERROR: Unhandled Exception\n    {}'.format(e))
            input('ENTER')

        # For overwrite, drop all expected entries from existing output DF
        if overwrite_flag and not zone_df.empty:
            zone_df = zone_df[~zone_df['DATE'].isin(list(export_dates))]
            # zone_df = zone_df[
            #     ~zone_df.index.get_level_values('DATE').isin(
            #         list(export_dates))]

        # Get list of existing dates in the CSV
        if not zone_df.empty:
            zone_df_dates = set(zone_df['DATE'].values)
        else:
            zone_df_dates = set()
        # logging.debug('    Dates in zone CSV: {}'.format(
        #     ', '.join(sorted(zone_df_dates))))

        # List of DATES that are entirely missing
        # This may include scenes that don't intersect the zone
        missing_all_dates = export_dates - zone_df_dates
        # logging.debug('    Dates not in CSV (missing all values): {}'.format(
        #     ', '.join(sorted(missing_all_dates))))
        if not missing_all_dates:
            zone_dataframes[zone['name']] = zone_df
            continue

        # Create empty entries for all scenes that are missing
        # logging.debug('    Building missing dates dataframe')
        missing_df = pd.DataFrame(index=range(len(missing_all_dates)),
                                  columns=zone_df.columns)
        missing_df['DATE'] = sorted(missing_all_dates)
        missing_df['ZONE_NAME'] = zone['name']
        missing_df['ZONE_NAME'] = missing_df['ZONE_NAME'].astype(str)
        missing_df['DATE'] = pd.to_datetime(missing_df['DATE'],
                                            format='%Y-%m-%d')
        missing_df['YEAR'] = missing_df['DATE'].dt.year
        missing_df['MONTH'] = missing_df['DATE'].dt.month
        missing_df['DAY'] = missing_df['DATE'].dt.day
        missing_df['DOY'] = missing_df['DATE'].dt.dayofyear.astype(int)
        missing_df['DATE'] = missing_df['DATE'].dt.strftime('%Y-%m-%d')

        missing_df['ZENITH_MOD09GA'] = np.nan
        missing_df['ZENITH_MYD09GA'] = np.nan

        # Remove the overlapping missing entries
        # Then append the new missing entries to the zone CSV
        # logging.debug('    Removing from zone dataframe')
        try:
            zone_df.drop(zone_df['DATE'].intersection(missing_df['DATE']),
                         inplace=True)
        except:
            pass

        # logging.debug('    Appending to zone dataframe')
        zone_df = zone_df.append(missing_df, sort=False)
        zone_df.sort_values(by=['DATE'], inplace=True)

        # logging.debug('    Writing to CSV')
        csv_writer(zone_df, zone['csv'], export_fields)

        # logging.debug('    Saving to dataframes list')
        zone_dataframes[zone['name']] = zone_df
        del missing_df, zone_df

    if not zone_dataframes:
        logging.info('\nNo zone dataframes, exiting')
        return False

    # Combine separate zone dataframes
    logging.debug('\n  Building master dataframe')
    output_df = pd.concat(list(zone_dataframes.values()))
    output_df['ZONE_NAME'] = output_df['ZONE_NAME'].astype(str)
    output_df.set_index(['DATE', 'ZONE_NAME'], inplace=True, drop=True)
    # logging.debug(output_df.head())
    # logging.debug('  Output DF size: {}'.format(len(output_df)))
    del zone_dataframes

    # Process the products in reverse order so zenith is computed first
    for product in sorted(modis_products, reverse=True):
        logging.info('\nProduct: {}'.format(product))

        # Get a list of available dates
        modis.start_year = ini['INPUTS']['start_year']
        modis.end_year = ini['INPUTS']['end_year']
        modis.start_date = None
        modis.end_date = None
        product_dates = set(utils.ee_getinfo(
            modis.get_daily_collection(product).aggregate_histogram('DATE')))
        logging.debug('  DATES available for product values: {}'.format(
            ', '.join(sorted(product_dates))))

        # Set transform as a function of the product
        transform = [
            ee_tools.modis.CELLSIZE[product], 0.0, ini['SPATIAL']['snap_x'],
            0.0, -ee_tools.modis.CELLSIZE[product], ini['SPATIAL']['snap_y']]
        logging.debug('  Output Transform: {}'.format(transform))

        # Set the zenith field based on the product MOD/MYD
        if 'MOD' in product.upper():
            zenith_field = 'ZENITH_MOD09GA'
        else:
            zenith_field = 'ZENITH_MYD09GA'

        # Identify DATES that are missing some data for the product
        # Use the zenith field to track dates that can't have data
        if zenith_field == product:
            missing_df = output_df.loc[
                (output_df.index.get_level_values('DATE').isin(export_dates) &
                 output_df[product].isnull()),
                [product]]
        else:
            missing_df = output_df.loc[
                (output_df.index.get_level_values('DATE').isin(export_dates) &
                 output_df[product].isnull() &
                 ~output_df[zenith_field].isnull()),
                [product]]
        if missing_df.empty:
            continue

        missing_dates = set(missing_df.index.get_level_values('DATE').values)
        missing_dates = missing_dates & product_dates
        # logging.debug('  DATES missing product values: {}'.format(
        #     ', '.join(sorted(missing_dates))))

        # # Don't use the date_keep_list if all the values are missing
        # # Build the date list in the MODIS system:index format
        # if not all(missing_mask):
        #     modis.date_keep_list = sorted([
        #         datetime.datetime.strptime(d, '%Y-%m-%d').strftime('%Y_%m_%d')
        #         for d in missing_dates])

        # Write values to file after each year
        # Group export dates by year (and path/row also?)
        export_dates_iter = [
            [year, list(dates)]
            for year, dates in groupby(sorted(missing_dates), lambda x: x[:4])]

        for export_year, export_year_dates in export_dates_iter:
            logging.debug('\n  Iter year: {}'.format(export_year))

            for export_date in sorted(export_year_dates):
                export_dt = datetime.datetime.strptime(export_date, '%Y-%m-%d')
                next_date = (export_dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                logging.info('  {}'.format(export_date))

                export_zones = list(output_df.loc[
                    (output_df.index.get_level_values('DATE')==export_date) &
                    (output_df[product].isnull()),
                    [product]].index.get_level_values('ZONE_NAME').values)
                if not export_zones:
                    logging.info('\n  No zones with missing data, skipping')
                    # This shouldn't happen since the date would be skipped
                    input('ENTER')
                    continue
                else:
                    logging.debug('  Zones with missing data: {}'.format(
                        ', '.join(sorted(export_zones))))

                # # Identify zones with any missing data
                # export_zones = set(date_df[date_df.any(axis=1)]\
                #     .index.get_level_values('ZONE_NAME').values)
                # logging.debug('  Zones with missing data: {}'.format(
                #     ', '.join(sorted(export_zones))))
                #
                # Build collection of all features to test for each SCENE_ID
                zone_coll = ee.FeatureCollection([
                    ee.Feature(zone['ee_geom'], {'ZONE_NAME': zone['name']})
                    for zone in zones
                    if zone['name'] in export_zones])

                # Collection should only have one image
                modis.start_date = export_date
                modis.end_date = next_date
                modis.start_year = None
                modis.end_year = None

                # Map over features for one image
                image = ee.Image(modis.get_daily_collection(product).first())

                def zonal_mean_func(ftr):
                    input_mean = ee.Image(image).select([product])\
                        .reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=ftr.geometry(),
                            crs=ini['SPATIAL']['crs'],
                            crsTransform=transform,
                            bestEffort=False)
                    return ee.Feature(None, {
                        'ZONE_NAME': ee.String(ftr.get('ZONE_NAME')),
                        'DATE': ee.Date(image.get('system:time_start'))\
                            .format('YYYY-MM-dd'),
                        product: input_mean.get(product)})

                stats_coll = zone_coll.map(zonal_mean_func, False)

                # Add a dummy entry to the stats collection
                # This is added because the export structure is based on the first
                #   entry in the collection, so fields with nodata will be excluded
                format_dict = {
                    'ZONE_NAME': 'DEADBEEF',
                    'DATE': 'DEADBEEF',
                    product: -9999,
                }
                stats_coll = ee.FeatureCollection(ee.Feature(None, format_dict))\
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

                # Set NaN zenith values to 90
                if 'ZENITH' in product:
                    export_df.fillna(90, inplace=True)

                # Save data to main dataframe
                if not export_df.empty:
                    logging.debug('    Processing data')
                    export_df.set_index(['DATE', 'ZONE_NAME'], inplace=True,
                                        drop=True)
                    if overwrite_flag:
                        # Update happens inplace automatically
                        output_df.update(export_df)
                    else:
                        # Combine first doesn't have an inplace parameter
                        output_df = output_df.combine_first(export_df)

            # Save updated CSV
            # if output_df is not None and not output_df.empty:
            if not output_df.empty:
                logging.info('\n  Writing zone CSVs')
                for zone in zones:
                    logging.debug(
                        '  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))
                    # logging.debug('    {}'.format(zone_output_path))
                    zone_df = output_df.loc[
                        output_df.index.get_level_values('ZONE_NAME')==zone['name'], :]
                    if not zone_df.empty:
                        zone_df.reset_index(inplace=True, drop=False)
                        csv_writer(zone_df, zone['csv'], export_fields)
            else:
                logging.info('\n  Empty output dataframe')


def modis_8day_func(export_fields, ini, zones_geojson, zones_wkt,
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

    logging.info('\nMODIS 8-Day')


def modis_16day_func(export_fields, ini, zones_geojson, zones_wkt,
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

    logging.info('\nMODIS 16-Day')


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
