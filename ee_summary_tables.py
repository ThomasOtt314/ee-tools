#--------------------------------
# Name:         ee_summary_tables.py
# Purpose:      Generate summary tables
# Created       2016-10-17
# Python:       2.7
#--------------------------------

import argparse
import ConfigParser
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd
from pandas import ExcelWriter

import common


def main(ini_path=None, overwrite_flag=True):
    """Generate summary tables

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing tables
    """

    logging.info('\nGenerate summary tables')

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

    # Read in config file
    zone_input_ws = config.get('INPUTS', 'zone_input_ws')
    zone_filename = config.get('INPUTS', 'zone_filename')
    zone_field = config.get('INPUTS', 'zone_field')

    landsat_annual_fields = [
        zone_field, 'YEAR', 'SCENE_COUNT',
        'PIXEL_COUNT', 'FMASK_COUNT', 'DATA_COUNT', 'CLOUD_SCORE',
        'TS', 'ALBEDO_SUR', 'NDVI_TOA', 'NDVI_SUR', 'EVI_SUR',
        'NDWI_GREEN_NIR_SUR', 'NDWI_GREEN_SWIR1_SUR', 'NDWI_NIR_SWIR1_SUR',
        # 'NDWI_GREEN_NIR_TOA', 'NDWI_GREEN_SWIR1_TOA', 'NDWI_NIR_SWIR1_TOA',
        # 'NDWI_SWIR1_GREEN_TOA', 'NDWI_SWIR1_GREEN_SUR',
        # 'NDWI_TOA', 'NDWI_SUR',
        'TC_BRIGHT', 'TC_GREEN', 'TC_WET']

    # Build and check file paths
    zone_path = os.path.join(zone_input_ws, zone_filename)
    if not os.path.isfile(zone_path):
        logging.error(
            '\nERROR: The zone shapefile does not exist\n  {}'.format(
                zone_path))
        sys.exit()

    # Read in config file
    try:
        output_ws = config.get('INPUTS', 'output_ws')
    except:
        output_ws = os.getcwd()
        logging.debug('Defaulting output workspace to {}'.format(output_ws))
    if not os.path.isdir(output_ws):
        logging.error(
            '\nERROR: The output folder does not exist\n  {}'.format(
                output_ws))
        sys.exit()

    output_filename = config.get('INPUTS', 'output_filename')
    output_path = os.path.join(output_ws, output_filename)

    # Start/end year
    try:
        start_year = int(config.get('INPUTS', 'start_year'))
    except:
        start_year = 1984
        logging.debug('Defaulting start year to {}'.format(start_year))
    try:
        end_year = int(config.get('INPUTS', 'end_year'))
    except:
        end_year = datetime.datetime.today().year
        logging.debug('Defaulting end year to {}'.format(end_year))
    if start_year and end_year and end_year < start_year:
        logging.error('\nERROR: End year must be >= start year')
        sys.exit()
    default_end_year = datetime.datetime.today().year + 1
    if (start_year and start_year not in range(1984, default_end_year) or
            end_year and end_year not in range(1984, default_end_year)):
        logging.error('\nERROR: Year must be an integer from 1984-2016')
        sys.exit()
    year_list = range(start_year, end_year + 1)

    # Start/end month
    try:
        start_month = int(config.get('INPUTS', 'start_month'))
    except:
        logging.debug('Defaulting start_month = None')
        start_month = None
    try:
        end_month = int(config.get('INPUTS', 'end_month'))
    except:
        logging.debug('Defaulting end_month = None')
        end_month = None
    if start_month and start_month not in range(1, 13):
        logging.error('\nERROR: Start month must be an integer from 1-12')
        sys.exit()
    elif end_month and end_month not in range(1, 13):
        logging.error('\nERROR: End month must be an integer from 1-12')
        sys.exit()
    month_list = common.wrapped_range(start_month, end_month, 1, 12)

    # Start/end DOY
    try:
        start_doy = int(config.get('INPUTS', 'start_doy'))
    except:
        logging.debug('Defaulting start_doy = None')
        start_doy = None
    try:
        end_doy = int(config.get('INPUTS', 'end_doy'))
    except:
        logging.debug('Defaulting end_doy = None')
        end_doy = None
    if start_doy and start_doy not in range(1, 367):
        logging.error('\nERROR: Start DOY must be an integer from 1-366')
        sys.exit()
    elif end_doy and end_doy not in range(1, 367):
        logging.error('\nERROR: End DOY must be an integer from 1-366')
        sys.exit()
    doy_list = common.wrapped_range(start_doy, end_doy, 1, 366)

    # Control which Landsat images are used
    # Default to True so that all Landsats are used if not set
    try:
        landsat5_flag = config.getboolean('INPUTS', 'landsat5_flag')
    except:
        logging.debug('Defaulting landsat5_flag = True')
        landsat5_flag = True
    try:
        landsat4_flag = config.getboolean('INPUTS', 'landsat4_flag')
    except:
        logging.debug('Defaulting landsat4_flag = True')
        landsat4_flag = True
    try:
        landsat7_flag = config.getboolean('INPUTS', 'landsat7_flag')
    except:
        logging.debug('Defaulting landsat7_flag = True')
        landsat7_flag = True
    try:
        landsat8_flag = config.getboolean('INPUTS', 'landsat8_flag')
    except:
        logging.debug('Defaulting landsat8_flag = True')
        landsat8_flag = True

    # Remove scenes with cloud score above target percentage
    try:
        max_cloud_score = config.getfloat('INPUTS', 'max_cloud_score')
    except:
        logging.debug(
            'Not filtering by pixel count percent\n  (max_cloud_score = 100)')
        max_cloud_score = 100
    if max_cloud_score < 0 or max_cloud_score > 100:
        logging.error('\nERROR: max_cloud_score must be in the range 0-100')
        sys.exit()
    if max_cloud_score > 0 and max_cloud_score < 1:
        logging.error(
            '\nWARNING: max_cloud_score must be a percent (0-100)' +
            '\n  The value entered appears to be a decimal in the range 0-1')
        raw_input('  Press ENTER to continue')

    # Remove scenes with Fmask counts above the target percentage
    try:
        max_fmask_pct = config.getfloat('INPUTS', 'max_fmask_pct')
    except:
        logging.debug(
            'Not filtering by pixel count percent\n  (max_fmask_pct = 100)')
        max_fmask_pct = 100
    if max_fmask_pct < 0 or max_fmask_pct > 100:
        logging.error('\nERROR: max_fmask_pct must be in the range 0-100')
        sys.exit()
    if max_fmask_pct > 0 and max_fmask_pct < 1:
        logging.error(
            '\nWARNING: max_fmask_pct must be a percent (0-100)' +
            '\n  The value entered appears to be a decimal in the range 0-1')
        raw_input('  Press ENTER to continue')

    # Remove SLC-off scenes with pixel counts below the target percentage
    try:
        min_slc_off_pct = config.getfloat(
            'INPUTS', 'min_slc_off_pct')
    except:
        logging.debug(
            'Not filtering SLC-off images by Fmask percent' +
            '  (min_slc_off_pct = 0)')
        min_slc_off_pct = 0
    if min_slc_off_pct < 0 or min_slc_off_pct > 100:
        logging.error(
            '\nERROR: min_slc_off_pct must be in the range 0-100')
        sys.exit()
    if min_slc_off_pct > 0 and min_slc_off_pct < 1:
        logging.error(
            '\nWARNING: min_slc_off_pct must be a percent (0-100)' +
            '\n  The value entered appears to be a decimal in the range 0-1')
        raw_input('  Press ENTER to continue')

    # Only process specific Landsat scenes
    try:
        scene_id_keep_path = config.get('INPUTS', 'scene_id_keep_path')
        with open(scene_id_keep_path) as input_f:
            scene_id_keep_list = input_f.readlines()
        scene_id_keep_list = [x.strip()[:16] for x in scene_id_keep_list]
    except IOError:
        logging.error('\nFileIO Error: {}'.format(scene_id_keep_path))
        sys.exit()
    except:
        logging.debug('Defaulting scene_id_keep_list = []')
        scene_id_keep_list = []

    # Skip specific landsat scenes
    try:
        scene_id_skip_path = config.get('INPUTS', 'scene_id_skip_path')
        with open(scene_id_skip_path) as input_f:
            scene_id_skip_list = input_f.readlines()
        scene_id_skip_list = [x.strip()[:16] for x in scene_id_skip_list]
    except IOError:
        logging.error('\nFileIO Error: {}'.format(scene_id_skip_path))
        sys.exit()
    except:
        logging.debug('Defaulting scene_id_skip_list = []')
        scene_id_skip_list = []

    # Only process certain Landsat path/rows
    try:
        path_keep_list = list(
            common.parse_int_set(config.get('INPUTS', 'path_keep_list')))
    except:
        logging.debug('Defaulting path_keep_list = []')
        path_keep_list = []
    try:
        row_keep_list = list(
            common.parse_int_set(config.get('INPUTS', 'row_keep_list')))
    except:
        logging.debug('Defaulting row_keep_list = []')
        row_keep_list = []

    # Skip or keep certain FID
    try:
        fid_skip_list = list(
            common.parse_int_set(config.get('INPUTS', 'fid_skip_list')))
    except:
        logging.debug('Defaulting fid_skip_list = []')
        fid_skip_list = []
    try:
        fid_keep_list = list(
            common.parse_int_set(config.get('INPUTS', 'fid_keep_list')))
    except:
        logging.debug('Defaulting fid_keep_list = []')
        fid_keep_list = []

    # GRIDMET month range (default to water year)
    try:
        gridmet_start_month = int(config.get('INPUTS', 'gridmet_start_month'))
    except:
        logging.debug('Defaulting gridmet_start_month = None')
        gridmet_start_month = None
    try:
        gridmet_end_month = int(config.get('INPUTS', 'gridmet_end_month'))
    except:
        logging.debug('Defaulting gridmet_end_month = None')
        gridmet_end_month = None
    if gridmet_start_month and gridmet_start_month not in range(1, 13):
        logging.error(
            '\nERROR: GRIDMET start month must be an integer from 1-12')
        sys.exit()
    elif gridmet_end_month and gridmet_end_month not in range(1, 13):
        logging.error(
            '\nERROR: GRIDMET end month must be an integer from 1-12')
        sys.exit()
    if gridmet_start_month is None and gridmet_end_month is None:
        gridmet_start_month = 10
        gridmet_end_month = 9
    gridmet_months = list(
        common.month_range(gridmet_start_month, gridmet_end_month))
    logging.info('\nGridmet months: {}'.format(
        ', '.join(map(str, gridmet_months))))

    # Get ee features from shapefile
    zone_geom_list = common.shapefile_2_geom_list_func(
        zone_path, zone_field)

    logging.info('\nProcessing zones')
    output_df = None
    zone_list = []
    for fid, zone_str, zone_json in sorted(zone_geom_list):
        if fid_keep_list and fid not in fid_keep_list:
            continue
        elif fid_skip_list and fid in fid_skip_list:
            continue
        logging.info('ZONE: {} (FID: {})'.format(zone_str, fid))

        if not zone_field or zone_field.upper() == 'FID':
            zone_str = 'fid_' + zone_str
        else:
            zone_str = zone_str.lower().replace(' ', '_')
        zone_list.append(zone_str)

        zone_output_ws = os.path.join(output_ws, zone_str)
        if not os.path.isdir(zone_output_ws):
            logging.debug('Folder {} does not exist, skipping'.format(
                zone_output_ws))
            continue

        landsat_daily_path = os.path.join(
            zone_output_ws, '{}_landsat_daily.csv'.format(zone_str))
        gridmet_daily_path = os.path.join(
            zone_output_ws, '{}_gridmet_daily.csv'.format(zone_str))
        gridmet_monthly_path = os.path.join(
            zone_output_ws, '{}_gridmet_monthly.csv'.format(zone_str))
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
        if path_keep_list:
            landsat_df = landsat_df[landsat_df['PATH'].isin(path_keep_list)]
        if row_keep_list:
            landsat_df = landsat_df[landsat_df['ROW'].isin(row_keep_list)]
        # Assume the default is for these to be True and only filter if False
        if not landsat4_flag:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LT4']
        if not landsat5_flag:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LT5']
        if not landsat7_flag:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LE7']
        if not landsat8_flag:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LC8']
        if scene_id_keep_list:
            landsat_df = landsat_df[
                landsat_df['SCENE_ID'].isin(scene_id_keep_list)]
        if scene_id_skip_list:
            landsat_df = landsat_df[np.logical_not(
                landsat_df['SCENE_ID'].isin(scene_id_skip_list))]

        # First filter by average cloud score
        if max_cloud_score < 100 and not landsat_df.empty:
            logging.debug(
                '    Maximum cloud score: {0}'.format(max_cloud_score))
            landsat_df = landsat_df[
                landsat_df['CLOUD_SCORE'] <= max_cloud_score]
        # Filter by Fmask percentage
        if max_fmask_pct < 100 and not landsat_df.empty:
            landsat_df['FMASK_PCT'] = 100 * (
                landsat_df['FMASK_COUNT'] / landsat_df['PIXEL_COUNT'])
            logging.debug(
                '    Max Fmask threshold: {}'.format(max_fmask_pct))
            landsat_df = landsat_df[
                landsat_df['FMASK_PCT'] <= max_fmask_pct]
        # Filter low count SLC-off images
        if min_slc_off_pct > 0 and not landsat_df.empty:
            logging.debug('    Mininum SLC-off threshold: {}%'.format(
                max_fmask_pct))
            logging.debug('    Maximum pixel count: {}'.format(
                max_pixel_count))
            slc_off_mask = (
                (landsat_df['LANDSAT'] == 'LE7') &
                ((landsat_df['YEAR'] >= 2004) |
                 ((landsat_df['YEAR'] == 2003) & (landsat_df['DOY'] > 151))))
            slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / max_pixel_count)
            landsat_df = landsat_df[
                ((slc_off_pct >= min_slc_off_pct) & slc_off_mask) |
                (~slc_off_mask)]

        if landsat_df.empty:
            logging.error(
                '  Empty Landsat dataframe after filtering, skipping zone')
            zone_list.remove(zone_str)
            continue

        logging.debug('  Computing Landsat annual summaries')
        landsat_df = landsat_df.groupby([zone_field, 'YEAR']).agg({
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
        gridmet_df = gridmet_df.groupby([zone_field, 'GROUP_YEAR']).agg({
            'ETO': {'ETO': 'sum'}, 'PPT': {'PPT': 'sum'}})
        gridmet_df.columns = gridmet_df.columns.droplevel(0)
        gridmet_df.reset_index(inplace=True)
        gridmet_df.rename(columns={'GROUP_YEAR': 'YEAR'}, inplace=True)
        gridmet_df.sort_values(by='YEAR', inplace=True)

        # Merge Landsat and GRIDMET collections
        zone_df = landsat_df.merge(gridmet_df, on=[zone_field, 'YEAR'])

        if output_df is None:
            output_df = zone_df.copy()
        else:
            output_df = output_df.append(zone_df)

        del landsat_df, gridmet_df, zone_df

    if output_df is not None and not output_df.empty:
        logging.info('\nWriting summary tables to Excel')
        excel_f = ExcelWriter(output_path)
        logging.debug('  {}'.format(output_path))
        for zone_str in zone_list:
            logging.debug('  {}'.format(zone_str))
            if zone_str.startswith('fid_'):
                zone_df = output_df[output_df[zone_field] == int(zone_str[4:])]
            else:
                zone_df = output_df[output_df[zone_field] == zone_str]
            zone_df.to_excel(
                excel_f, sheet_name=zone_str, index=False, float_format='%.4f')
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
        '-i', '--ini', type=lambda x: common.is_valid_file(parser, x),
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    # parser.add_argument(
    #     '-o', '--overwrite', default=False, action='store_true',
    #     help='Force overwrite of existing files')
    args = parser.parse_args()

    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    else:
        args.ini = common.get_ini_path(os.getcwd())
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
