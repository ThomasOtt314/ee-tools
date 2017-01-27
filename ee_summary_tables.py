#--------------------------------
# Name:         ee_summary_tables.py
# Purpose:      Generate summary tables
# Created       2017-01-27
# Python:       2.7
#--------------------------------

import argparse
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd
from pandas import ExcelWriter

import ee_tools.gdal_common as gdc
import ee_tools.ini_common as ini_common
import ee_tools.python_common as python_common


def main(ini_path=None, overwrite_flag=True):
    """Generate summary tables

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing tables
    """

    logging.info('\nGenerate summary tables')

    # Read config file
    # ini = ini_common.ini_parse(ini_path, section='TABLES')
    ini = ini_common.read(ini_path)
    ini_common.parse_section(ini, section='INPUTS')
    ini_common.parse_section(ini, section='SUMMARY')
    ini_common.parse_section(ini, section='TABLES')

    landsat_annual_fields = [
        'ZONE_FID', 'ZONE_NAME', 'YEAR', 'SCENE_COUNT',
        'PIXEL_COUNT', 'FMASK_COUNT', 'DATA_COUNT', 'CLOUD_SCORE',
        'TS', 'ALBEDO_SUR', 'NDVI_TOA', 'NDVI_SUR', 'EVI_SUR',
        'NDWI_GREEN_NIR_SUR', 'NDWI_GREEN_SWIR1_SUR', 'NDWI_NIR_SWIR1_SUR',
        # 'NDWI_GREEN_NIR_TOA', 'NDWI_GREEN_SWIR1_TOA', 'NDWI_NIR_SWIR1_TOA',
        # 'NDWI_SWIR1_GREEN_TOA', 'NDWI_SWIR1_GREEN_SUR',
        # 'NDWI_TOA', 'NDWI_SUR',
        'TC_BRIGHT', 'TC_GREEN', 'TC_WET']

    # Build and check file paths
    output_path = os.path.join(
        ini['SUMMARY']['output_ws'], ini['TABLES']['output_filename'])

    year_list = range(ini['INPUTS']['start_year'], ini['INPUTS']['end_year'] + 1)
    month_list = list(python_common.wrapped_range(
        ini['INPUTS']['start_month'], ini['INPUTS']['end_month'], 1, 12))
    doy_list = list(python_common.wrapped_range(
        ini['INPUTS']['start_doy'], ini['INPUTS']['end_doy'], 1, 366))

    # GRIDMET month range (default to water year)
    gridmet_start_month = ini['SUMMARY']['gridmet_start_month']
    gridmet_end_month = ini['SUMMARY']['gridmet_end_month']
    gridmet_months = list(python_common.month_range(
        gridmet_start_month, gridmet_end_month))
    logging.info('\nGridmet months: {}'.format(
        ', '.join(map(str, gridmet_months))))

    # Add merged row XXX to keep list
    ini['INPUTS']['row_keep_list'].append('XXX')

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


    logging.info('\nProcessing zones')
    output_df = None
    zone_list = []
    for zone_fid, zone_name, zone_json in zone_geom_list:
        zone_name = zone_name.replace(' ', '_')
        logging.info('ZONE: {} (FID: {})'.format(zone_name, zone_fid))
        zone_list.append(zone_name)

        zone_stats_ws = os.path.join(ini['SUMMARY']['output_ws'], zone_name)
        if not os.path.isdir(zone_stats_ws):
            logging.debug('Folder {} does not exist, skipping'.format(
                zone_stats_ws))
            continue

        landsat_daily_path = os.path.join(
            zone_stats_ws, '{}_landsat_daily.csv'.format(zone_name))
        gridmet_daily_path = os.path.join(
            zone_stats_ws, '{}_gridmet_daily.csv'.format(zone_name))
        gridmet_monthly_path = os.path.join(
            zone_stats_ws, '{}_gridmet_monthly.csv'.format(zone_name))
        if not os.path.isfile(landsat_daily_path):
            logging.error('  Landsat daily CSV does not exist, skipping zone')
            continue
        elif (not os.path.isfile(gridmet_daily_path) and
              not os.path.isfile(gridmet_monthly_path)):
            logging.error(
                '  GRIDMET daily or monthly CSV does not exist, skipping zone')
            continue

        logging.debug('  Reading Landsat CSV')
        landsat_df = pd.read_csv(landsat_daily_path)

        logging.debug('  Filtering Landsat dataframe')
        landsat_df = landsat_df[landsat_df['PIXEL_COUNT'] > 0]

        # This assumes that there are L5/L8 images in the dataframe
        if not landsat_df.empty:
            max_pixel_count = max(landsat_df['PIXEL_COUNT'])
        else:
            max_pixel_count = 0

        if year_list:
            landsat_df = landsat_df[landsat_df['YEAR'].isin(year_list)]
        if month_list:
            landsat_df = landsat_df[landsat_df['MONTH'].isin(month_list)]
        if doy_list:
            landsat_df = landsat_df[landsat_df['DOY'].isin(doy_list)]

        if ini['INPUTS']['path_keep_list']:
            landsat_df = landsat_df[
                landsat_df['PATH'].isin(ini['INPUTS']['path_keep_list'])]
        if ini['INPUTS']['row_keep_list']:
            landsat_df = landsat_df[
                landsat_df['ROW'].isin(ini['INPUTS']['row_keep_list'])]

        # Assume the default is for these to be True and only filter if False
        if not ini['INPUTS']['landsat4_flag']:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LT4']
        if not ini['INPUTS']['landsat5_flag']:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LT5']
        if not ini['INPUTS']['landsat7_flag']:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LE7']
        if not ini['INPUTS']['landsat8_flag']:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LC8']
        if ini['INPUTS']['scene_id_keep_list']:
            landsat_df = landsat_df[landsat_df['SCENE_ID'].isin(
                ini['INPUTS']['scene_id_keep_list'])]
        if ini['INPUTS']['scene_id_skip_list']:
            landsat_df = landsat_df[np.logical_not(landsat_df['SCENE_ID'].isin(
                ini['INPUTS']['scene_id_skip_list']))]

        # First filter by average cloud score
        if ini['SUMMARY']['max_cloud_score'] < 100 and not landsat_df.empty:
            logging.debug('    Maximum cloud score: {0}'.format(
                ini['SUMMARY']['max_cloud_score']))
            landsat_df = landsat_df[
                landsat_df['CLOUD_SCORE'] <= ini['SUMMARY']['max_cloud_score']]

        # Filter by Fmask percentage
        if ini['SUMMARY']['max_fmask_pct'] < 100 and not landsat_df.empty:
            landsat_df['FMASK_PCT'] = 100 * (
                landsat_df['FMASK_COUNT'] / landsat_df['PIXEL_COUNT'])
            logging.debug('    Max Fmask threshold: {}'.format(
                ini['SUMMARY']['max_fmask_pct']))
            landsat_df = landsat_df[
                landsat_df['FMASK_PCT'] <= ini['SUMMARY']['max_fmask_pct']]

        # Filter low count SLC-off images
        if ini['SUMMARY']['min_slc_off_pct'] > 0 and not landsat_df.empty:
            logging.debug('    Mininum SLC-off threshold: {}%'.format(
                ini['SUMMARY']['min_slc_off_pct']))
            logging.debug('    Maximum pixel count: {}'.format(
                max_pixel_count))
            slc_off_mask = (
                (landsat_df['LANDSAT'] == 'LE7') &
                ((landsat_df['YEAR'] >= 2004) |
                 ((landsat_df['YEAR'] == 2003) & (landsat_df['DOY'] > 151))))
            slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / max_pixel_count)
            landsat_df = landsat_df[
                ((slc_off_pct >= ini['SUMMARY']['min_slc_off_pct']) & slc_off_mask) |
                (~slc_off_mask)]

        if landsat_df.empty:
            logging.error(
                '  Empty Landsat dataframe after filtering, skipping zone')
            zone_list.remove(zone_name)
            continue

        logging.debug('  Computing Landsat annual summaries')
        landsat_df = landsat_df\
            .groupby(['ZONE_FID', 'ZONE_NAME', 'YEAR'])\
            .agg({
                'PIXEL_COUNT': {
                    'PIXEL_COUNT': 'mean',
                    'SCENE_COUNT': 'count'},
                'FMASK_COUNT': {'FMASK_COUNT': 'mean'},
                'DATA_COUNT': {'DATA_COUNT': 'mean'},
                'CLOUD_SCORE': {'CLOUD_SCORE': 'mean'},
                'ALBEDO_SUR': {'ALBEDO_SUR': 'mean'},
                'EVI_SUR': {'EVI_SUR': 'mean'},
                'NDVI_SUR': {'NDVI_SUR': 'mean'},
                'NDVI_TOA': {'NDVI_TOA': 'mean'},
                'NDWI_GREEN_NIR_SUR': {'NDWI_GREEN_NIR_SUR': 'mean'},
                'NDWI_GREEN_SWIR1_SUR': {'NDWI_GREEN_SWIR1_SUR': 'mean'},
                'NDWI_NIR_SWIR1_SUR': {'NDWI_NIR_SWIR1_SUR': 'mean'},
                # 'NDWI_GREEN_NIR_TOA': {'NDWI_GREEN_NIR_TOA': 'mean'},
                # 'NDWI_GREEN_SWIR1_TOA': {'NDWI_GREEN_SWIR1_TOA': 'mean'},
                # 'NDWI_NIR_SWIR1_TOA': {'NDWI_NIR_SWIR1_TOA': 'mean'},
                # 'NDWI_SWIR1_GREEN_SUR': {'NDWI_SWIR1_GREEN_SUR': 'mean'},
                # 'NDWI_SWIR1_GREEN_TOA': {'NDWI_SWIR1_GREEN_TOA': 'mean'},
                # 'NDWI_SUR': {'NDWI_SUR': 'mean'},
                # 'NDWI_TOA': {'NDWI_TOA': 'mean'},
                'TC_BRIGHT': {'TC_BRIGHT': 'mean'},
                'TC_GREEN': {'TC_GREEN': 'mean'},
                'TC_WET': {'TC_WET': 'mean'},
                'TS': {'TS': 'mean'}
            })
        landsat_df.columns = landsat_df.columns.droplevel(0)
        landsat_df.reset_index(inplace=True)
        landsat_df = landsat_df[landsat_annual_fields]
        landsat_df['PIXEL_COUNT'] = landsat_df['PIXEL_COUNT'].astype(np.int)
        landsat_df['SCENE_COUNT'] = landsat_df['SCENE_COUNT'].astype(np.int)
        landsat_df['FMASK_COUNT'] = landsat_df['FMASK_COUNT'].astype(np.int)
        landsat_df['DATA_COUNT'] = landsat_df['DATA_COUNT'].astype(np.int)
        landsat_df.sort_values(by='YEAR', inplace=True)

        if os.path.isfile(gridmet_monthly_path):
            logging.debug('  Reading montly GRIDMET CSV')
            gridmet_df = pd.read_csv(gridmet_monthly_path)
        elif os.path.isfile(gridmet_daily_path):
            logging.debug('  Reading daily GRIDMET CSV')
            gridmet_df = pd.read_csv(gridmet_daily_path)

        logging.debug('  Computing GRIDMET summaries')
        # Summarize GRIDMET for target months year
        if (gridmet_start_month in [10, 11, 12] and
                gridmet_end_month in [10, 11, 12]):
            month_mask = (
                (gridmet_df['MONTH'] >= gridmet_start_month) &
                (gridmet_df['MONTH'] <= gridmet_end_month))
            gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR'] + 1
        elif (gridmet_start_month in [10, 11, 12] and
              gridmet_end_month not in [10, 11, 12]):
            month_mask = gridmet_df['MONTH'] >= gridmet_start_month
            gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR'] + 1
            month_mask = gridmet_df['MONTH'] <= gridmet_end_month
            gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR']
        else:
            month_mask = (
                (gridmet_df['MONTH'] >= gridmet_start_month) &
                (gridmet_df['MONTH'] <= gridmet_end_month))
            gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR']
        gridmet_df['GROUP_YEAR'] = gridmet_df['GROUP_YEAR'].astype(int)

        if year_list:
            gridmet_df = gridmet_df[gridmet_df['GROUP_YEAR'].isin(year_list)]
            if gridmet_df.empty:
                logging.error(
                    '    Empty GRIDMET dataframe after filtering by year')
                continue

        # Group GRIDMET data by user specified range (default is water year)
        gridmet_df = gridmet_df\
            .groupby(['ZONE_FID', 'ZONE_NAME', 'GROUP_YEAR'])\
            .agg({'ETO': {'ETO': 'sum'}, 'PPT': {'PPT': 'sum'}})
        gridmet_df.columns = gridmet_df.columns.droplevel(0)
        gridmet_df.reset_index(inplace=True)
        gridmet_df.rename(columns={'GROUP_YEAR': 'YEAR'}, inplace=True)
        gridmet_df.sort_values(by='YEAR', inplace=True)

        # Merge Landsat and GRIDMET collections
        zone_df = landsat_df.merge(
            gridmet_df, on=['ZONE_FID', 'ZONE_NAME', 'YEAR'])
        # zone_df = landsat_df.merge(gridmet_df, on=['ZONE_FID', 'YEAR'])

        if output_df is None:
            output_df = zone_df.copy()
        else:
            output_df = output_df.append(zone_df)

        del landsat_df, gridmet_df, zone_df

    if output_df is not None and not output_df.empty:
        logging.info('\nWriting summary tables to Excel')
        excel_f = ExcelWriter(output_path)
        logging.debug('  {}'.format(output_path))
        for zone_name in zone_list:
            logging.debug('  {}'.format(zone_name))
            zone_df = output_df[output_df['ZONE_NAME'] == zone_name]
            zone_df.to_excel(
                excel_f, sheet_name=zone_name, index=False, float_format='%.4f')
            del zone_df
        excel_f.save()
    else:
        logging.info('  Empty output dataframe, not writing summary tables')


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate summary tables',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=lambda x: python_common.valid_file(x),
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    # parser.add_argument(
    #     '-o', '--overwrite', default=False, action='store_true',
    #     help='Force overwrite of existing files')
    args = parser.parse_args()

    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    else:
        args.ini = python_common.get_ini_path(os.getcwd())
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{0}'.format('#' * 80))
    log_f = '{0:<20s} {1}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini)
    # main(ini_path=args.ini, overwrite_flag=args.overwrite)
