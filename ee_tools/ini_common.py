#--------------------------------
# Name:         ini_common.py
# Purpose:      Common INI reading/parsing functions
# Author:       Charles Morton
# Created       2017-04-13
# Python:       2.7
#--------------------------------

# import ConfigParser
# from collections import defaultdict
import datetime
import logging
import os
import sys

import configparser
# from backports import configparser

import ee_tools.gdal_common as gdc
import ee_tools.python_common as python_common


def read(ini_path):
    logging.debug('\nReading Input File')
    # Open config file
    config = configparser.ConfigParser()
    try:
        config.read(ini_path)
    except:
        logging.error('\nERROR: Input file could not be read, '
                      'is not an input file, or does not exist\n'
                      'ERROR: ini_path = {}\n'.format(ini_path))
        sys.exit()

    # Force conversion of unicode to strings
    ini = dict()
    for section in config.keys():
        ini[str(section)] = {}
        for k, v in config[section].items():
            ini[str(section)][str(k)] = v
    return ini


def parse_section(ini, section):
    logging.debug('  Checking {} section'.format(section))
    if section not in ini.keys():
        logging.error(
            '\nERROR: Input file does not have an {} section'.format(section))
        sys.exit()

    if section == 'INPUTS':
        parse_inputs(ini)
    elif section == 'EXPORT':
        parse_export(ini)
    elif section == 'IMAGES':
        parse_images(ini)
    elif section == 'ZONAL_STATS':
        parse_zonal_stats(ini)
    elif section == 'SUMMARY':
        parse_summary(ini)
    elif section == 'TABLES':
        parse_tables(ini)
    elif section == 'FIGURES':
        parse_figures(ini)


def get_param(ini, section, input_name, output_name, get_type,
              default='MANDATORY'):
    """Get INI parameters by type and set default values

    Args:
        ini (dict): Nested dictionary of INI file keys/values
        section (str): Section name
        input_name (str): Parameter name in INI file
        output_name (str): Parameter name in code
        get_type (): Python type
        default (): Default value to use if parameter was not set.
            Defaults to "MANDATORY".
            "MANDATORY" will cause script to exit if key does not exist.
    """

    try:
        if get_type is bool:
            ini[section][output_name] = (
                ini[section][input_name].lower() == "true")
            # ini[section][output_name] = distutils.util.strtobool(
            #     ini[section][input_name])
            # ini[section][output_name] = ini.getboolean(section, input_name)
            # ini[section][output_name] = ini[section].getboolean(input_name)
        elif get_type is int:
            ini[section][output_name] = int(ini[section][input_name])
        elif get_type is float:
            ini[section][output_name] = float(ini[section][input_name])
        elif get_type is list:
            ini[section][output_name] = str(ini[section][input_name])
        else:
            ini[section][output_name] = str(ini[section][input_name])
            # Convert 'None' (strings) to None
            if ini[section][output_name].lower() == 'none':
                ini[section][output_name] = None
    except (KeyError, configparser.NoOptionError):
        if default == 'MANDATORY':
            logging.error(
                '\nERROR: {} was not set in the INI, exiting\n'.format(
                    input_name))
            sys.exit()
        else:
            ini[section][output_name] = default
            logging.debug('  Setting {} = {}'.format(
                input_name, ini[section][output_name]))
    except ValueError:
        logging.error('\nERROR: Invalid value for "{}"'.format(
            input_name))
        sys.exit()
    except Exception as e:
        logging.error('\nERROR: Unhandled error\n  {}'.format(e))
        sys.exit()

    # If the parameter is renamed, remove the old name/parameter
    if input_name != output_name:
        del ini[section][input_name]


def parse_inputs(ini, section='INPUTS'):
    # MANDATORY PARAMETERS
    # section, input_name, output_name, description, get_type
    param_list = [
        ['zone_workspace', 'zone_input_ws', str],
        ['zone_filename', 'zone_filename', str],
        ['zone_field', 'zone_field', str]
    ]
    for input_name, output_name, get_type in param_list:
        get_param(ini, section, input_name, output_name, get_type)

    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        # Control which Landsat images are used
        ['landsat4_flag', 'landsat4_flag', bool, False],
        ['landsat5_flag', 'landsat5_flag', bool, False],
        ['landsat7_flag', 'landsat7_flag', bool, False],
        ['landsat8_flag', 'landsat8_flag', bool, False],
        # Date filtering
        ['start_year', 'start_year', int, None],
        ['end_year', 'end_year', int, None],
        ['start_month', 'start_month', int, None],
        ['end_month', 'end_month', int, None],
        ['start_doy', 'start_doy', int, None],
        ['end_doy', 'end_doy', int, None],
        # Scene ID filtering
        ['scene_id_keep_path', 'scene_id_keep_path', str, ''],
        ['scene_id_skip_path', 'scene_id_skip_path', str, ''],
        # Path/row filtering
        ['path_keep_list', 'path_keep_list', list, []],
        ['row_keep_list', 'row_keep_list', list, []],
        ['path_row_list', 'path_row_list', list, []],
        # FID filtering
        ['fid_skip_list', 'fid_skip_list', list, []],
        ['fid_keep_list', 'fid_keep_list', list, []]
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # Build and check file paths
    ini[section]['zone_path'] = os.path.join(
        ini[section]['zone_input_ws'], ini[section]['zone_filename'])
    # Strip off file extension from filename after building path
    ini[section]['zone_filename'] = os.path.splitext(
        ini[section]['zone_filename'])[0]
    if not os.path.isdir(ini[section]['zone_input_ws']):
        logging.error(
            '\nERROR: The zone workspace does not exist, '
            'exiting\n  {}'.format(ini[section]['zone_input_ws']))
        sys.exit()
    elif not os.path.isfile(ini[section]['zone_path']):
        logging.error(
            '\nERROR: The zone shapefile does not exist, '
            'exiting\n  {}'.format(ini[section]['zone_path']))
        sys.exit()

    # Start/end year
    if (ini[section]['start_year'] and ini[section]['end_year'] and
            ini[section]['end_year'] < ini[section]['start_year']):
        logging.error(
            '\nERROR: End year must be >= start year')
        sys.exit()
    default_end_year = datetime.datetime.today().year + 1
    if ((ini[section]['start_year'] and
            ini[section]['start_year'] not in range(1984, default_end_year)) or
        (ini[section]['end_year'] and
            ini[section]['end_year'] not in range(1984, default_end_year))):
        logging.error('\nERROR: Year must be an integer from 1984-{}'.format(
            default_end_year - 1))
        sys.exit()

    # Start/end month
    if (ini[section]['start_month'] and
            ini[section]['start_month'] not in range(1, 13)):
        logging.error(
            '\nERROR: Start month must be an integer from 1-12')
        sys.exit()
    elif (ini[section]['end_month'] and
            ini[section]['end_month'] not in range(1, 13)):
        logging.error('\nERROR: End month must be an integer from 1-12')
        sys.exit()

    # Start/end DOY
    if ini[section]['end_doy'] and ini[section]['end_doy'] > 273:
        logging.error(
            '\nERROR: End DOY has to be in the same water year as start DOY')
        sys.exit()
    if (ini[section]['start_doy'] and
            ini[section]['start_doy'] not in range(1, 367)):
        logging.error(
            '\nERROR: Start DOY must be an integer from 1-366')
        sys.exit()
    elif (ini[section]['end_doy'] and
            ini[section]['end_doy'] not in range(1, 367)):
        logging.error('\nERROR: End DOY must be an integer from 1-366')
        sys.exit()
    # if ini[section]['end_doy'] < ini[section]['start_doy']:
    #     logging.error('\nERROR: End DOY must be >= start DOY')
    #     sys.exit()

    if ini[section]['fid_keep_list']:
        ini[section]['fid_keep_list'] = sorted(list(
            python_common.parse_int_set(ini[section]['fid_keep_list'])))
    if ini[section]['fid_skip_list']:
        ini[section]['fid_skip_list'] = sorted(list(
            python_common.parse_int_set(ini[section]['fid_skip_list'])))

    # Convert path/row ranges to list
    if ini[section]['path_keep_list']:
        ini[section]['path_keep_list'] = sorted(list(
            python_common.parse_int_set(ini[section]['path_keep_list'])))
    if ini[section]['row_keep_list']:
        ini[section]['row_keep_list'] = sorted(list(
            python_common.parse_int_set(ini[section]['row_keep_list'])))
    if ini[section]['path_row_list']:
        ini[section]['path_row_list'] = sorted([
            pr.strip() for pr in ini[section]['path_row_list'].split(',')])

    # Intentionally don't apply scene_id skip/keep lists
    # Compute zonal stats for all available images
    # Filter by scene_id when making summary tables
    logging.info('  Not applying scene_id keep or skip lists')
    ini[section]['scene_id_keep_list'] = []
    ini[section]['scene_id_skip_list'] = []

    # # Only process specific Landsat scenes
    # try:
    #     with open(config['scene_id_keep_path']) as input_f:
    #         scene_id_keep_list = input_f.readlines()
    #     ini[section]['scene_id_keep_list'] = [
    #         x.strip()[:16] for x in scene_id_keep_list.split(',')]
    # except IOError:
    #     logging.error('\nFileIO Error: {}'.format(
    #         config['scene_id_keep_path']))
    #     sys.exit()
    # except:
    #     ini[section]['scene_id_keep_list'] = []

    # # Skip specific landsat scenes
    # try:
    #     with open(config['scene_id_skip_path']) as input_f:
    #         scene_id_skip_list = input_f.readlines()
    #     ini[section]['scene_id_skip_list'] = [
    #         x.strip()[:16] for x in scene_id_skip_list.split(',')]
    # except IOError:
    #     logging.error('\nFileIO Error: {}'.format(
    #         config['scene_id_skip_path']))
    #     sys.exit()
    # except:
    #     ini[section]['scene_id_skip_list'] = []


def parse_export(ini, section='EXPORT'):
    """"""
    # MANDATORY PARAMETERS
    # section, input_name, output_name, description, get_type
    param_list = [
        # Output spatial reference
        ['output_snap', 'snap', str],
        ['output_cs', 'cellsize', float],
        ['output_proj', 'crs', str],
        # Google Drive
        ['gdrive_workspace', 'gdrive_ws', str],
        # DEADBEEF - Moved to optional for now
        # ['export_folder', 'export_folder', str],
    ]
    for input_name, output_name, get_type in param_list:
        get_param(ini, section, input_name, output_name, get_type)

    # OPTIONAL PARAMETERS
    # section, input_name, output_name, description, get_type, default
    param_list = [
        # DEADBEEF - Moved to optional for now
        ['export_folder', 'export_folder', str, ''],
        # Cloud masking
        ['acca_flag', 'acca_flag', bool, False],
        ['fmask_flag', 'fmask_flag', bool, False],
        #
        ['fmask_type', 'fmask_type', str, None],
        ['mosaic_method', 'mosaic_method', str, 'mean'],
        ['adjust_method', 'adjust_method', str, None]
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # DEADBEEF
    if ini[section]['export_folder']:
        logging.info(
            '\n  There are currently issues writing to Google Drive folders'
            '  Setting "export_folder" = ""\n')
        ini[section]['export_folder'] = ''
        raw_input('ENTER')

    # Build and check file paths
    ini[section]['export_ws'] = os.path.join(
        ini[section]['gdrive_ws'], ini[section]['export_folder'])
    if not os.path.isdir(ini[section]['export_ws']):
        os.makedirs(ini[section]['export_ws'])

    # Convert snap points to list
    ini[section]['snap'] = [
        int(i) for i in ini[section]['snap'].split(',')
        if i.strip().isdigit()][:2]
    # Compute snap points separately
    ini[section]['snap_x'], ini[section]['snap_y'] = ini[section]['snap']

    # Compute OSR from EGSG code
    try:
        ini[section]['osr'] = gdc.epsg_osr(
            int(ini[section]['crs'].split(':')[1]))
    except:
        logging.error(
            '\nERROR: The output projection could not be converted to a '
            'spatial reference object\n  {}'.format(
                ini[section]['crs']))
        sys.exit()

    logging.debug('  Snap: {} {}'.format(
        ini[section]['snap_x'], ini[section]['snap_y']))
    logging.debug('  Cellsize: {}'.format(ini[section]['cellsize']))
    logging.debug('  CRS: {}'.format(ini[section]['crs']))
    # logging.debug('  OSR: {}\n'.format(
    #     ini[section]['osr'].ExportToWkt())

    # Fmask source type
    if ini[section]['fmask_flag'] and not ini[section]['fmask_type']:
        logging.error(
            '\nERROR: Fmask source type must be set if fmask_flag = True')
        sys.exit()
    if ini[section]['fmask_type']:
        ini[section]['fmask_type'] = ini[section]['fmask_type'].lower()
        if ini[section]['fmask_type'] not in ['fmask', 'cfmask']:
            logging.error(
                '\nERROR: Invalid Fmask source type: {}\n'
                '  Must be "fmask" or "cfmask"'.format(
                    ini[section]['fmask_type']))
            sys.exit()

    # Mosaic method
    if ini[section]['mosaic_method']:
        ini[section]['mosaic_method'] = ini[section]['mosaic_method'].lower()
        mosaic_method_list = ['mean', 'median', 'mosaic', 'min', 'max']
        if ini[section]['mosaic_method'] not in mosaic_method_list:
            logging.error(
                '\nERROR: Invalid mosaic method: {}\n''  Must be: {}'.format(
                    ini[section]['mosaic_method'],
                    ', '.join(mosaic_method_list)))
            sys.exit()

    # Adjust Landsat Red and NIR bands
    if ini[section]['adjust_method']:
        ini[section]['adjust_method'] = ini[section]['adjust_method'].upper()
        adjust_method_list = ['OLI_2_ETM', 'ETM_2_OLI']
        if ini[section]['adjust_method'] not in adjust_method_list:
            logging.error(
                '\nERROR: Invalid mosaic method: {}\n  Must be: {}'.format(
                    ini[section]['adjust_method'],
                    ', '.join(adjust_method_list)))
            sys.exit()


def parse_images(ini, section='IMAGES'):
    """"""
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['images_workspace', 'output_ws', str, os.getcwd()],
        ['download_bands', 'download_bands', str, ''],
        ['merge_geometries_flag', 'merge_geom_flag', bool, False],
        ['clip_landsat_flag', 'clip_landsat_flag', bool, True],
        ['image_buffer', 'image_buffer', int, 0]
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # Build output folder if necessary
    if not os.path.isdir(ini[section]['output_ws']):
        os.makedirs(ini[section]['output_ws'])

    # Image download bands
    ini[section]['download_bands'] = map(
        lambda x: x.strip().lower(), ini[section]['download_bands'].split(','))
    logging.info('\n  Output Bands:')
    for band in ini[section]['download_bands']:
        logging.info('    {}'.format(band))


def parse_zonal_stats(ini, section='ZONAL_STATS'):
    """"""

    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['stats_workspace', 'output_ws', str, os.getcwd()],
        ['landsat_flag', 'landsat_flag', bool, True],
        ['gridmet_daily_flag', 'gridmet_daily_flag', bool, False],
        ['gridmet_monthly_flag', 'gridmet_monthly_flag', bool, False],
        ['pdsi_flag', 'pdsi_flag', bool, False],
        ['year_step', 'year_step', int, 1]
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # Build output folder if necessary
    if not os.path.isdir(ini[section]['output_ws']):
        os.makedirs(ini[section]['output_ws'])

    if ini[section]['year_step'] < 1 or ini[section]['year_step'] > 40:
        logging.error('\nERROR: year_step must be an integer from 1-40')
        sys.exit()


def parse_summary(ini, section='SUMMARY'):
    """"""
    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['stats_workspace', 'output_ws', str, os.getcwd()],
        ['max_cloud_score', 'max_cloud_score', float, 70],
        ['max_fmask_pct', 'max_fmask_pct', float, 100],
        ['min_slc_off_pct', 'min_slc_off_pct', float, 50],
        ['gridmet_start_month', 'gridmet_start_month', int, 10],
        ['gridmet_end_month', 'gridmet_end_month', int, 9]
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # Remove scenes with cloud score above target percentage
    if (ini[section]['max_cloud_score'] < 0 or
            ini[section]['max_cloud_score'] > 100):
        logging.error('\nERROR: max_cloud_score must be in the range 0-100')
        sys.exit()
    if (ini[section]['max_cloud_score'] > 0 and
            ini[section]['max_cloud_score'] < 1):
        logging.error(
            '\nWARNING: max_cloud_score must be a percent (0-100)' +
            '\n  The value entered appears to be a decimal in the range 0-1')
        raw_input('  Press ENTER to continue')

    # Remove scenes with Fmask counts above the target percentage
    if (ini[section]['max_fmask_pct'] < 0 or
            ini[section]['max_fmask_pct'] > 100):
        logging.error('\nERROR: max_fmask_pct must be in the range 0-100')
        sys.exit()
    if (ini[section]['max_fmask_pct'] > 0 and
            ini[section]['max_fmask_pct'] < 1):
        logging.error(
            '\nWARNING: max_fmask_pct must be a percent (0-100)' +
            '\n  The value entered appears to be a decimal in the range 0-1')
        raw_input('  Press ENTER to continue')

    # Remove SLC-off scenes with pixel counts below the target percentage
    if (ini[section]['min_slc_off_pct'] < 0 or
            ini[section]['min_slc_off_pct'] > 100):
        logging.error(
            '\nERROR: min_slc_off_pct must be in the range 0-100')
        sys.exit()
    if (ini[section]['min_slc_off_pct'] > 0 and
            ini[section]['min_slc_off_pct'] < 1):
        logging.error(
            '\nWARNING: min_slc_off_pct must be a percent (0-100)' +
            '\n  The value entered appears to be a decimal in the range 0-1')
        raw_input('  Press ENTER to continue')

    # GRIDMET month range (default to water year)
    if (ini[section]['gridmet_start_month'] and
            ini[section]['gridmet_start_month'] not in range(1, 13)):
        logging.error(
            '\nERROR: GRIDMET start month must be an integer from 1-12')
        sys.exit()
    elif (ini[section]['gridmet_end_month'] and
            ini[section]['gridmet_end_month'] not in range(1, 13)):
        logging.error(
            '\nERROR: GRIDMET end month must be an integer from 1-12')
        sys.exit()
    if (ini[section]['gridmet_start_month'] is None and
            ini[section]['gridmet_end_month'] is None):
        ini[section]['gridmet_start_month'] = 10
        ini[section]['gridmet_end_month'] = 9


def parse_tables(ini, section='TABLES'):
    """"""
    # MANDATORY PARAMETERS
    # param_section, input_name, output_name, get_type
    param_list = [
        ['output_filename', 'output_filename', str]
    ]
    for input_name, output_name, get_type in param_list:
        get_param(ini, section, input_name, output_name, get_type)

    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['output_ws', 'output_ws', str, os.getcwd()]
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)


def parse_figures(ini, section='FIGURES'):
    """"""
    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['output_ws', 'output_ws', str, os.getcwd()],
        ['ppt_plot_type', 'ppt_plot_type', str, 'LINE'],
        ['best_fit_flag', 'scatter_best_fit', bool, False],
        ['timeseries_bands', 'timeseries_bands', str, 'ndvi_toa'],
        ['scatter_bands', 'scatter_bands', str, 'ppt:ndvi_sur, ppt:evi_sur'],
        ['complementary_bands', 'complementary_bands', str, 'evi_sur']
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    ini[section]['timeseries_bands'] = map(
        lambda x: x.strip().lower(),
        ini[section]['timeseries_bands'].split(','))

    ini[section]['scatter_bands'] = [
        map(lambda x: x.strip().lower(), b.split(':'))
        for b in ini[section]['scatter_bands'].split(',')]

    ini[section]['complementary_bands'] = map(
        lambda x: x.strip().lower(),
        ini[section]['complementary_bands'].split(','))

    if ini[section]['ppt_plot_type'].upper() not in ['LINE', 'BAR']:
        logging.error('\nERROR: ppt_plot_type must be "LINE" or "BAR"')
        sys.exit()
