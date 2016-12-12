#--------------------------------
# Name:         ee_summary_figures.py
# Purpose:      Generate summary figures
# Created       2016-10-17
# Python:       2.7
#--------------------------------

import argparse
import ConfigParser
import datetime
import logging
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy import stats

import common


def main(ini_path=None, overwrite_flag=True, show_flag=False):
    """Generate summary figures

    Args:
        ini_path (str): file path of the control file
        overwrite_flag (bool): if True, overwrite existing figures
        show_flag (bool): if True, show figures as they are being built
    """

    logging.info('\nGenerate summary figures')

    # Band options
    band_list = [
        'albedo_sur', 'cloud_score', 'data_count', 'eto',
        'evi_sur', 'fmask_count', 'ndvi_sur', 'ndvi_toa',
        'ndwi_green_nir_sur', 'ndwi_green_nir_toa',
        'ndwi_green_swir1_sur', 'ndwi_green_swir1_toa',
        'ndwi_nir_swir1_sur', 'ndwi_nir_swir1_toa',
        'ndwi_swir1_green_sur', 'ndwi_swir1_green_toa',
        # 'ndwi_sur', 'ndwi_toa',
        'pixel_count', 'ppt', 'tc_bright', 'tc_green', 'tc_wet', 'ts']
    band_name = {
        'albedo_sur': 'Albedo',
        'cloud_score': 'Cloud Score',
        'data_count': 'Data Count',
        'eto': 'ETo',
        'evi_sur': 'EVI',
        'fmask_count': 'Fmask Count',
        'ndvi_sur': 'NDVI',
        'ndvi_toa': 'NDVI (TOA)',
        'ndwi_green_nir_sur': 'NDWI (Green, NIR)',
        'ndwi_green_nir_toa': 'NDWI (Green, NIR) (TOA)',
        'ndwi_green_swir1_sur': 'NDWI (Green, SWIR1)',
        'ndwi_green_swir1_toa': 'NDWI (Green, SWIR1) (TOA)',
        'ndwi_nir_swir1_sur': 'NDWI (NIR, SWIR1)',
        'ndwi_nir_swir1_toa': 'NDWI (NIR, SWIR1) (TOA)',
        'ndwi_swir1_green_sur': 'NDWI (SWIR1, Green)',
        'ndwi_swir1_green_toa': 'NDWI (SWIR1, Green) (TOA)',
        # 'ndwi_sur': 'NDWI (SWIR1, GREEN)',
        # 'ndwi_toa': 'NDWI (SWIR1, GREEN) (TOA)',
        'pixel_count': 'Pixel Count',
        'ppt': 'PPT',
        'tc_bright': 'Brightness',
        'tc_green': 'Greeness',
        'tc_wet': 'Wetness',
        'ts': 'Ts'
    }
    band_unit = {
        'albedo_sur': 'dimensionless',
        'cloud_score': 'dimensionless',
        'data_count': 'dimensionless',
        'evi_sur': 'dimensionless',
        'eto': 'mm',
        'fmask_count': 'dimensionless',
        'ndvi_sur': 'dimensionless',
        'ndvi_toa': 'dimensionless',
        'ndwi_green_nir_sur': 'dimensionless',
        'ndwi_green_nir_toa': 'dimensionless',
        'ndwi_green_swir1_sur': 'dimensionless',
        'ndwi_green_swir1_toa': 'dimensionless',
        'ndwi_nir_swir1_sur': 'dimensionless',
        'ndwi_nir_swir1_toa': 'dimensionless',
        'ndwi_swir1_green_sur': 'dimensionless',
        'ndwi_swir1_green_toa': 'dimensionless',
        # 'ndwi_sur': 'dimensionless',
        # 'ndwi_toa': 'dimensionless',
        'pixel_count': 'dimensionless',
        'ppt': 'mm',
        'tc_bright': 'dimensionless',
        'tc_green': 'dimensionless',
        'tc_wet': 'dimensionless',
        'ts': 'K',
    }
    band_color = {
        'albedo_sur': '#CF4457',
        'cloud_score': '0.5',
        'data_count': '0.5',
        'eto': '#348ABD',
        'fmask_count': '0.5',
        'evi_sur': '#FFA500',
        'ndvi_sur': '#A60628',
        'ndvi_toa': '#A60628',
        'ndwi_green_nir_sur': '#4eae4b',
        'ndwi_green_nir_toa': '#4eae4b',
        'ndwi_green_swir1_sur': '#4eae4b',
        'ndwi_green_swir1_toa': '#4eae4b',
        'ndwi_nir_swir1_sur': '#4eae4b',
        'ndwi_nir_swir1_toa': '#4eae4b',
        'ndwi_swir1_green_sur': '#4eae4b',
        'ndwi_swir1_green_toa': '#4eae4b',
        # 'ndwi_sur': '#4eae4b',
        # 'ndwi_toa': '#4eae4b',
        'pixel_count': '0.5',
        'ppt': '0.5',
        'tc_bright': '#E24A33',
        'tc_green': '#E24A33',
        'tc_wet': '#E24A33',
        'ts': '#188487'
    }

    # A couple of color palettes to sample from
    # import seaborn as sns
    # print(sns.color_palette("hls", 20).as_hex())
    # print(sns.color_palette("husl", 20).as_hex())
    # print(sns.color_palette("hsv", 20).as_hex())
    # print(sns.color_palette("Set1", 20).as_hex())
    # print(sns.color_palette("Set2", 20).as_hex())


    # Hardcoded plot options
    figures_folder = 'figures'
    fig_type = 'large'

    plot_dict = dict()

    # Plot PPT as a 'bar' or 'line' for now
    # plot_dict['scatter_best_fit'] = True

    # Center y-labels in figure window (instead of centering on ticks/axes)
    plot_dict['center_ylabel'] = False

    # Axes percentages must be 0-1
    plot_dict['timeseries_band_ax_pct'] = [0.3, 0.92]
    plot_dict['timeseries_ppt_ax_pct'] = [0.0, 0.35]
    plot_dict['complement_band_ax_pct'] = [0.0, 0.5]
    plot_dict['complement_eto_ax_pct'] = [0.4, 1.0]

    if fig_type.lower() == 'large':
        plot_dict['title_fs'] = 12
        plot_dict['xtick_fs'] = 10
        plot_dict['ytick_fs'] = 10
        plot_dict['xlabel_fs'] = 10
        plot_dict['ylabel_fs'] = 10
        plot_dict['legend_fs'] = 10
        plot_dict['ts_ms'] = 3
        plot_dict['comp_ms'] = 4
        plot_dict['timeseries_ax'] = [0.12, 0.13, 0.78, 0.81]
        plot_dict['scatter_ax'] = [0.12, 0.10, 0.82, 0.84]
        plot_dict['complement_ax'] = [0.12, 0.10, 0.78, 0.84]
        plot_dict['fig_size'] = (6.0, 5.0)
    elif fig_type.lower() == 'small':
        plot_dict['title_fs'] = 10
        plot_dict['xtick_fs'] = 8
        plot_dict['ytick_fs'] = 8
        plot_dict['xlabel_fs'] = 8
        plot_dict['ylabel_fs'] = 8
        plot_dict['legend_fs'] = 8
        plot_dict['ts_ms'] = 1.5
        plot_dict['comp_ms'] = 2
        plot_dict['timeseries_ax'] = [0.18, 0.21, 0.67, 0.70]
        plot_dict['scatter_ax'] = [0.18, 0.21, 0.67, 0.70]
        plot_dict['complement_ax'] = [0.18, 0.16, 0.67, 0.75]
        plot_dict['fig_size'] = (3.0, 2.5)
    plot_dict['fig_dpi'] = 300
    plot_dict['show'] = show_flag
    plot_dict['overwrite'] = overwrite_flag


    # CSV parameters
    # Zone field will be inserted after it is read in from INI file
    landsat_annual_fields = [
        'YEAR', 'SCENE_COUNT',
        'PIXEL_COUNT', 'FMASK_COUNT', 'DATA_COUNT', 'CLOUD_SCORE',
        'TS', 'ALBEDO_SUR', 'NDVI_TOA', 'NDVI_SUR', 'EVI_SUR',
        'NDWI_GREEN_NIR_SUR', 'NDWI_GREEN_SWIR1_SUR', 'NDWI_NIR_SWIR1_SUR',
        # 'NDWI_GREEN_NIR_TOA', 'NDWI_GREEN_SWIR1_TOA', 'NDWI_NIR_SWIR1_TOA',
        # 'NDWI_SWIR1_GREEN_TOA', 'NDWI_SWIR1_GREEN_SUR',
        # 'NDWI_TOA', 'NDWI_SUR',
        'TC_BRIGHT', 'TC_GREEN', 'TC_WET']


    # Open config file
    config = ConfigParser.ConfigParser()
    try:
        config.readfp(open(ini_path))
    except:
        logging.error('\nERROR: Input file could not be read, ' +
                      'is not an input file, or does not exist\n' +
                      'ERROR: ini_path = {}\n').format(ini_path)
        sys.exit()
    logging.info('\nReading Input File')

    # Get and check config file parameters
    # Get figure bands from INI file
    try:
        timeseries_bands = config.get(
            'INPUTS', 'timeseries_bands').split(',')
        timeseries_bands = map(
            lambda x: x.strip().lower(), timeseries_bands)
    except:
        logging.warning('  timeseries_bands option not set in INI')
        timeseries_bands = []
    try:
        scatter_bands = config.get(
            'INPUTS', 'scatter_bands').split(',')
        scatter_bands = [
            map(lambda x: x.strip().lower(), b.split(':'))
            for b in scatter_bands]
    except:
        logging.warning('  scatter_bands option not set in INI')
        scatter_bands = []
    try:
        complementary_bands = config.get(
            'INPUTS', 'complementary_bands').split(',')
        complementary_bands = map(
            lambda x: x.strip().lower(), complementary_bands)
    except:
        logging.warning('  complementary_bands option not set in INI')
        complementary_bands = []

    # Check figure bands
    if timeseries_bands:
        logging.info('Timeseries Bands:')
        for band in timeseries_bands:
            if band not in band_list:
                logging.info(
                    '  Invalid timeseries band: {}, exiting'.format(band))
                return False
            logging.info('  {}'.format(band))
    if scatter_bands:
        logging.info('Scatter Bands (x:y):')
        for band_x, band_y in scatter_bands:
            if band_x not in band_list:
                logging.info(
                    '  Invalid scatter band: {}, exiting'.format(band_x))
                return False
            elif band_y not in band_list:
                logging.info(
                    '  Invalid band: {}, exiting'.format(band_y))
                return False
            logging.info('  {}:{}'.format(band_x, band_y))
    if complementary_bands:
        logging.info('Complementary Bands:')
        for band in complementary_bands:
            if band not in band_list:
                logging.info(
                    '  Invalid complementary band: {}, exiting'.format(band))
                return False
            logging.info('  {}'.format(band))

    # Plot options
    # Plot PPT as a line or a bar chart
    try:
        plot_dict['ppt_plot_type'] = config.get(
            'INPUTS', 'ppt_plot_type').strip().upper()
    except:
        plot_dict['ppt_plot_type'] = 'LINE'
        logging.debug('Defaulting ppt_plot_type to LINE')
    if plot_dict['ppt_plot_type'] not in ['LINE', 'BAR']:
        logging.error('\nERROR: ppt_plot_type must be "LINE" or "BAR"')
        sys.exit()

    # Add a regression line and equation/R^2 to the scatter plot
    try:
        plot_dict['scatter_best_fit'] = config.getboolean(
            'INPUTS', 'best_fit_flag')
    except:
        plot_dict['scatter_best_fit'] = False
        logging.debug('Defaulting best_fit_flag to False')

    #
    zone_input_ws = config.get('INPUTS', 'zone_input_ws')
    zone_filename = config.get('INPUTS', 'zone_filename')
    zone_field = config.get('INPUTS', 'zone_field')
    landsat_annual_fields.insert(0, zone_field)

    # Build and check file paths
    zone_path = os.path.join(zone_input_ws, zone_filename)
    if not os.path.isfile(zone_path):
        logging.error(
            '\nERROR: The zone shapefile does not exist\n  {}'.format(
                zone_path))
        sys.exit()

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

    figures_ws = os.path.join(output_ws, figures_folder)
    if not os.path.isdir(figures_ws):
        os.makedirs(figures_ws)

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
        logging.error(
            '\nERROR: Year must be an integer from 1984-2016')
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
        logging.error('\nERROR: cloud_score must be in the range 0-100')
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
        # zone_list.append(zone_str)

        zone_output_ws = os.path.join(output_ws, zone_str)
        # zone_figures_ws = output_ws
        # zone_figures_ws = os.path.join(output_ws, zone_str)
        if not os.path.isdir(zone_output_ws):
            logging.debug('  Folder {} does not exist, skipping'.format(
                zone_output_ws))
            continue
        # elif not os.path.isdir(zone_figures_ws):
        #     os.makedirs(zone_figures_ws)

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
        # Assume the default is for these to be True and only filter if False
        if not landsat4_flag:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LT4']
        if not landsat5_flag:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LT5']
        if not landsat7_flag:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LE7']
        if not landsat8_flag:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LC8']
        if path_keep_list:
            landsat_df = landsat_df[landsat_df['PATH'].isin(path_keep_list)]
        if row_keep_list:
            landsat_df = landsat_df[landsat_df['ROW'].isin(row_keep_list)]
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
            'TS': {'TS': 'mean'},
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
        gridmet_group_df = gridmet_df.groupby([zone_field, 'GROUP_YEAR']).agg({
            'ETO': {'ETO': 'sum'}, 'PPT': {'PPT': 'sum'}})
        gridmet_group_df.columns = gridmet_group_df.columns.droplevel(0)
        gridmet_group_df.reset_index(inplace=True)
        gridmet_group_df.rename(columns={'GROUP_YEAR': 'YEAR'}, inplace=True)
        gridmet_group_df.sort_values(by='YEAR', inplace=True)

        # # Group GRIDMET data by month
        # gridmet_month_df = gridmet_df.groupby(
        #     [zone_field, 'GROUP_YEAR', 'MONTH']).agg({
        #         'ETO': {'ETO': 'sum'}, 'PPT': {'PPT': 'sum'}})
        # gridmet_month_df.columns = gridmet_month_df.columns.droplevel(0)
        # gridmet_month_df.reset_index(inplace=True)
        # gridmet_month_df.rename(columns={'GROUP_YEAR': 'YEAR'}, inplace=True)
        # # gridmet_month_df.sort_values(by=['YEAR', 'MONTH'], inplace=True)
        # gridmet_month_df.reset_index(inplace=True)
        # # Rename monthly PPT columns
        # gridmet_month_df['MONTH'] = 'PPT_M' + gridmet_month_df['MONTH'].astype(str)
        # # Pivot rows up to separate columns
        # gridmet_month_df = gridmet_month_df.pivot_table(
        #     'PPT', [zone_field, 'YEAR'], 'MONTH')
        # gridmet_month_df.reset_index(inplace=True)
        # columns = [zone_field, 'YEAR'] + ['PPT_M{}'.format(m) for m in gridmet_months]
        # gridmet_month_df = gridmet_month_df[columns]
        # del gridmet_month_df.index.name

        # Merge Landsat and GRIDMET collections
        zone_df = landsat_df.merge(gridmet_group_df, on=[zone_field, 'YEAR'])
        # zone_df = zone_df.merge(gridmet_month_df, on=[zone_field, 'YEAR'])
        if zone_df is None or zone_df.empty:
            logging.info('  Empty zone dataframe, not generating figures')
            continue

        # Adjust year range based on data availability?
        # start_year = min(zone_df['YEAR']),
        # end_year = max(zone_df['YEAR'])

        logging.debug('  Generating figures')
        for band in timeseries_bands:
            timeseries_plot(
                band, zone_df, zone_str, figures_ws, start_year, end_year,
                band_name, band_unit, band_color, plot_dict)

        for band_x, band_y in scatter_bands:
            scatter_plot(
                band_x, band_y, zone_df, zone_str, figures_ws,
                band_name, band_unit, band_color, plot_dict)

        for band in complementary_bands:
            complementary_plot(
                band, zone_df, zone_str, figures_ws,
                band_name, band_unit, band_color, plot_dict)

        del landsat_df, gridmet_df, zone_df


def timeseries_plot(band, zone_df, zone_str, figures_ws,
                    start_year, end_year,
                    band_name, band_unit, band_color, plot_dict):
    """"""
    ppt_band = 'ppt'
    logging.debug('    Timeseries: {} & {}'.format(
        band_name[band], band_name[ppt_band]))
    figure_path = os.path.join(
        figures_ws,
        '{}_timeseries_{}_&_ppt.png'.format(
            zone_str.lower(), band.lower(), ppt_band))

    fig = plt.figure(figsize=plot_dict['fig_size'])
    fig_ax = plot_dict['timeseries_ax']

    # Position the adjusted axes
    # Draw PPT first so that band lines are on top of PPT bars
    ppt_ax = fig_ax[:]
    ppt_ax[1] = fig_ax[1] + plot_dict['timeseries_ppt_ax_pct'][0] * fig_ax[3]
    ppt_ax[3] = fig_ax[3] * (
        plot_dict['timeseries_ppt_ax_pct'][1] -
        plot_dict['timeseries_ppt_ax_pct'][0])
    ax2 = fig.add_axes(ppt_ax)
    band_ax = fig_ax[:]
    band_ax[1] = fig_ax[1] + plot_dict['timeseries_band_ax_pct'][0] * fig_ax[3]
    band_ax[3] = fig_ax[3] * (
        plot_dict['timeseries_band_ax_pct'][1] -
        plot_dict['timeseries_band_ax_pct'][0])
    ax1 = fig.add_axes(band_ax)

    ax0 = fig.add_axes(fig_ax)
    ax0.set_title('{}'.format(zone_str), size=plot_dict['title_fs'], y=1.01)

    ax2.set_xlabel('Year', fontsize=plot_dict['xlabel_fs'])
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')
    ax1.set_xlim([start_year - 1, end_year + 1])
    ax2.set_xlim([start_year - 1, end_year + 1])
    ax1.yaxis.set_label_position("left")
    ax2.yaxis.set_label_position("right")
    if plot_dict['center_ylabel']:
        fig.text(
            0.02, 0.5, '{} [{}]'.format(band_name[band], band_unit[band]),
            fontsize=plot_dict['ylabel_fs'],
            ha='center', va='center', rotation='vertical')
        fig.text(
            0.98, 0.5,
            '{} [{}]'.format(band_name[ppt_band], band_unit[ppt_band]),
            fontsize=plot_dict['ylabel_fs'],
            ha='center', va='center', rotation='vertical')
    else:
        ax1.set_ylabel(
            '{} [{}]'.format(band_name[band], band_unit[band]),
            fontsize=plot_dict['ylabel_fs'])
        ax2.set_ylabel(
            '{} [{}]'.format(band_name['ppt'], band_unit['ppt']),
            fontsize=plot_dict['ylabel_fs'])
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax1.tick_params(axis='y', labelsize=plot_dict['ytick_fs'])
    ax2.tick_params(axis='y', labelsize=plot_dict['ytick_fs'])
    ax2.tick_params(axis='x', labelsize=plot_dict['xtick_fs'])
    ax2.tick_params(axis='x', which='both', top='off')
    ax0.axes.get_xaxis().set_ticks([])
    ax0.axes.get_yaxis().set_ticks([])
    ax1.axes.get_xaxis().set_ticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax0.patch.set_visible(False)
    ax1.patch.set_visible(False)
    ax2.patch.set_visible(False)

    # Plot the band data
    ax1.plot(
        zone_df['YEAR'].values, zone_df[band.upper()].values,
        marker='o', ms=plot_dict['ts_ms'], c=band_color[band.lower()],
        label=band_name[band])

    # Plot precipitation first (so that it is in back)
    if plot_dict['ppt_plot_type'] == 'BAR':
        ax2.bar(
            zone_df['YEAR'].values - 0.5, zone_df['PPT'].values,
            width=1, color='0.6', edgecolor='0.5', label=band_name[ppt_band])
        # ax2.set_ylim = [min(zone_df['PPT'].values), max(zone_df['PPT'].values)]
    elif plot_dict['ppt_plot_type'] == 'LINE':
        ax2.plot(
            zone_df['YEAR'].values, zone_df['PPT'].values,
            marker='x', c='0.4', ms=plot_dict['ts_ms'], lw=0.7,
            label=band_name[ppt_band])

    # Legend
    h2, l2 = ax2.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()
    ax0.legend(
        h1 + h2, l1 + l2, loc='upper right', frameon=False,
        fontsize=plot_dict['legend_fs'], numpoints=1)

    if plot_dict['overwrite'] or not os.path.isfile(figure_path):
        plt.savefig(figure_path, dpi=plot_dict['fig_dpi'])
    if plot_dict['show']:
        plt.show()
    plt.close()
    del fig, ax0, ax1, ax2
    return True


def scatter_plot(band_x, band_y, zone_df, zone_str, figures_ws,
                 band_name, band_unit, band_color, plot_dict):
    """"""
    logging.debug('    Scatter: {} vs {}'.format(
        band_name[band_x], band_name[band_y]))
    figure_path = os.path.join(
        figures_ws,
        '{}_scatter_{}_vs_{}.png'.format(
            zone_str.lower(), band_x.lower(), band_y.lower()))

    fig = plt.figure(figsize=plot_dict['fig_size'])
    fig_ax = plot_dict['scatter_ax']
    ax0 = fig.add_axes(fig_ax)
    ax0.set_title('{}'.format(zone_str), size=plot_dict['title_fs'], y=1.01)
    ax0.set_xlabel(
        '{}'.format(band_name[band_x]), fontsize=plot_dict['xlabel_fs'])
    ax0.set_ylabel(
        '{}'.format(band_name[band_y]), fontsize=plot_dict['ylabel_fs'])

    # Regression line
    if plot_dict['scatter_best_fit']:
        m, b, r, p, std_err = stats.linregress(
            zone_df[band_x.upper()].values, zone_df[band_y.upper()].values)
        # m, b = np.polyfit(
        #     zone_df[band_x.upper()].values, zone_df[band_y.upper()].values, 1)
        x = np.array(
            [min(zone_df[band_x.upper()]), max(zone_df[band_x.upper()])])
        ax0.plot(x, m * x + b, '-', c='0.1')
        plt.figtext(0.68, 0.17, ('$y = {0:0.4f}x+{1:0.3f}$'.format(m, b)))
        plt.figtext(0.68, 0.13, ('$R^2\! = {0:0.4f}$'.format(r ** 2)))

    ax0.plot(
        zone_df[band_x.upper()].values, zone_df[band_y.upper()].values,
        linestyle='', marker='o', c='0.5', ms=plot_dict['comp_ms'])

    if plot_dict['overwrite'] or not os.path.isfile(figure_path):
        plt.savefig(figure_path, dpi=plot_dict['fig_dpi'])
    if plot_dict['show']:
        plt.show()
    plt.close()
    del fig, ax0

    return True


def complementary_plot(band, zone_df, zone_str, figures_ws,
                       band_name, band_unit, band_color, plot_dict):
    """"""
    logging.debug('    Complementary: {}'.format(band_name[band]))
    figure_path = os.path.join(
        figures_ws,
        '{}_complementary_{}.png'.format(zone_str.lower(), band.lower()))

    fig = plt.figure(figsize=plot_dict['fig_size'])
    fig_ax = plot_dict['complement_ax']
    ax0 = fig.add_axes(fig_ax)
    ax0.set_title('{}'.format(zone_str), size=plot_dict['title_fs'], y=1.01)

    # Position the adjusted axes
    eto_ax = fig_ax[:]
    eto_ax[1] = fig_ax[1] + plot_dict['complement_eto_ax_pct'][0] * fig_ax[3]
    eto_ax[3] = fig_ax[3] * (
        plot_dict['complement_eto_ax_pct'][1] -
        plot_dict['complement_eto_ax_pct'][0])
    ax1 = fig.add_axes(eto_ax)
    band_ax = fig_ax[:]
    band_ax[1] = fig_ax[1] + plot_dict['complement_band_ax_pct'][0] * fig_ax[3]
    band_ax[3] = fig_ax[3] * (
        plot_dict['complement_band_ax_pct'][1] -
        plot_dict['complement_band_ax_pct'][0])
    ax2 = fig.add_axes(band_ax)

    ax2.set_xlabel(
        '{}'.format(band_name['ppt']), fontsize=plot_dict['xlabel_fs'])
    ax1.yaxis.set_label_position("left")
    ax2.yaxis.set_label_position("right")
    if plot_dict['center_ylabel']:
        fig.text(
            0.02, 0.5, '{} [{}]'.format(band_name['eto'], band_unit['eto']),
            fontsize=plot_dict['ylabel_fs'],
            ha='center', va='center', rotation='vertical')
        fig.text(
            0.98, 0.5, '{} [{}]'.format(band_name[band], band_unit[band]),
            fontsize=plot_dict['ylabel_fs'],
            ha='center', va='center', rotation='vertical')
    else:
        ax1.set_ylabel(
            '{} [{}]'.format(band_name['eto'], band_unit['eto']),
            fontsize=plot_dict['ylabel_fs'])
        ax2.set_ylabel(
            '{} [{}]'.format(band_name[band], band_unit[band]),
            fontsize=plot_dict['ylabel_fs'])
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax1.tick_params(axis='y', labelsize=plot_dict['ytick_fs'])
    ax2.tick_params(axis='y', labelsize=plot_dict['ytick_fs'])
    ax2.tick_params(axis='x', labelsize=plot_dict['xtick_fs'])
    ax2.tick_params(axis='x', which='both', top='off')
    ax0.axes.get_xaxis().set_ticks([])
    ax0.axes.get_yaxis().set_ticks([])
    ax1.axes.get_xaxis().set_ticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax1.patch.set_visible(False)
    ax2.patch.set_visible(False)

    ax1.set_xlim([min(zone_df['PPT']) - 10, max(zone_df['PPT']) + 10])
    ax2.set_xlim([min(zone_df['PPT']) - 10, max(zone_df['PPT']) + 10])

    ax1.plot(
        zone_df['PPT'].values, zone_df['ETO'].values, label=band_name['eto'],
        linestyle='', marker='^', c=band_color['eto'], ms=plot_dict['comp_ms'])
    ax2.plot(
        zone_df['PPT'].values, zone_df[band.upper()].values,
        linestyle='', marker='o', ms=plot_dict['comp_ms'],
        label=band_name[band], c=band_color[band.lower()])

    # Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax0.legend(
        h1 + h2, l1 + l2, loc='upper right', frameon=False,
        fontsize=plot_dict['legend_fs'], numpoints=1)

    if plot_dict['overwrite'] or not os.path.isfile(figure_path):
        plt.savefig(figure_path, dpi=plot_dict['fig_dpi'])
    if plot_dict['show']:
        plt.show()
    plt.close()
    del fig, ax0, ax1, ax2
    return True


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate summary figures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '--show', default=False, action='store_true', help='Show plots')
    # parser.add_argument(
    #     '-o', '--overwrite', default=False, action='store_true',
    #     help='Force overwrite of existing files')
    args = parser.parse_args()
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

    main(ini_path=args.ini, show_flag=args.show)
    # main(ini_path=args.ini, overwrite_flag=args.overwrite)