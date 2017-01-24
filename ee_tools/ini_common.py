# import ConfigParser
import datetime
import logging
import os
import sys

import configparser

import ee_tools.gdal_common as gdc
import ee_tools.python_common as python_common


def ini_parse(ini_path, mode='zonal_stats'):
    """Read the input parameter file, set defaults, and check values

    Eventually make this a class and/or separate out the validation
        and the reading.

    Args:
        ini_path (str): Input INI file path
        mode (str): Enable different parameters depending on export type.
            Eventually handle this using INI sections.

    Returns:
        dict: 

    """

    # Open config file
    config = configparser.ConfigParser()
    try:
        config.read(ini_path)
    except:
        logging.error('\nERROR: Input file could not be read, '
                      'is not an input file, or does not exist\n'
                      'ERROR: ini_path = {}\n'.format(ini_path))
        sys.exit()
    logging.debug('\nReading Input File')

    # Fow now, assume all values are in one INPUTS section
    if 'INPUTS' not in config.sections():
        logging.error('\nERROR: Input file must have an "INPUTS" section'
                      '  Insert the following as the first line: [INPUTS]')
        sys.exit()

    # Read all inputs directly into a dictionary
    # options = config['INPUTS'].items()

    options = {}

    # Vary some options depending on mode
    # Eventually this should be handled with INI sections
    if mode.lower() == 'zonal_stats':
        output_ws_param = 'zs_output_ws'
    elif mode.lower() == 'image':
        output_ws_param = 'image_output_ws'
    else:
        output_ws_param = 'output_ws'


    # Get output spatial reference parameter separately for now
    try:
        options['output_snap'] = [
            int(i) for i in config['INPUTS'].get('output_snap').split(',')
            if i.strip().isdigit()][:2]
    except:
        options['output_snap'] = [15, 15]
    options['snap_x'], options['snap_y'] = options['output_snap']

    # Output cellsize
    try:
        options['output_cs'] = int(config['INPUTS'].get('output_cs'))
    except:
        options['output_cs'] = 30

    # Output EPSG code
    try:
        options['output_crs'] = config['INPUTS'].get('output_proj')
    except:
        options['output_crs'] = ''

    # Compute OSR from EGSG code
    if options['output_crs']:
        options['output_osr'] = gdc.epsg_osr(
            int(options['output_crs'].split(':')[1]))
    else:
        options['output_osr'] = None

    logging.debug('  Snap: {} {}'.format(options['snap_x'], options['snap_y']))
    logging.debug('  Cellsize: {}'.format(options['output_cs']))
    logging.debug('  CRS: {}'.format(options['output_crs']))
    # logging.debug('  OSR: {}\n'.format(options['output_osr'].ExportToWkt()))


    # Image download specific options
    # Eventually move down and set below
    try:
        image_download_bands = config.get(
            'IMAGES', 'image_download_bands').split(',')
        options['image_download_bands'] = map(
            lambda x: x.strip().lower(), image_download_bands)
        logging.info('\n  Output Bands:')
        for band in options['image_download_bands']:
            logging.info('    {}'.format(band))
    except:
        options['image_download_bands'] = []


    # Get mandatory parameters
    # section, input_name, output_name, description, get_type
    param_list = [
        # Read in zone shapefile information
        ['INPUTS', 'zone_input_ws', 'zone_input_ws', str],
        ['INPUTS', 'zone_filename', 'zone_filename', str],
        ['INPUTS', 'zone_field', 'zone_field', str],
        # Google Drive export folder
        ['INPUTS', 'gdrive_ws', 'gdrive_ws', str],
        ['INPUTS', 'export_folder', 'export_folder', str],
    ]
    for section, input_name, output_name, get_type in param_list:
        try:
            if get_type is bool:
                options[output_name] = config[section].getboolean(input_name)
            elif get_type is int:
                options[output_name] = config[section].getint(input_name)
            else:
                options[output_name] = config[section].get(input_name)
        except:
            logging.error(
                '\nERROR: {} was not set in the INI'
                ', exiting\n'.format(input_name))
            sys.exit()


    # Build and check file paths
    options['zone_path'] = os.path.join(
        options['zone_input_ws'], options['zone_filename'])
    options['zone_name'] = os.path.splitext(options['zone_filename'])[0].lower()
    options['export_ws'] = os.path.join(
        options['gdrive_ws'], options['export_folder'])
    if not os.path.isdir(options['export_ws']):
        os.makedirs(options['export_ws'])
    if not os.path.isdir(options['zone_input_ws']):
        logging.error(
            '\nERROR: The zone workspace does not exist, '
            'exiting\n  {}'.format(options['zone_input_ws']))
        sys.exit()
    elif not os.path.isfile(options['zone_path']):
        logging.error(
            '\nERROR: The zone shapefile does not exist, '
            'exiting\n  {}'.format(options['zone_path']))
        sys.exit()


    # Get parameters with default values
    # input_name, output_name, default, get_type
    param_list = [
        # Output folder
        ['INPUTS', output_ws_param, 'output_ws', os.getcwd(), str],
        # Zonal stats options
        ['INPUTS', 'zs_landsat_flag', 'zs_landsat_flag', True, bool],
        ['INPUTS', 'zs_gridmet_daily_flag', 'zs_gridmet_daily_flag', False, bool],
        ['INPUTS', 'zs_gridmet_monthly_flag', 'zs_gridmet_monthly_flag', False, bool],
        ['INPUTS', 'zs_pdsi_flag', 'zs_pdsi_flag', False, bool],
        # Control which Landsat images are used
        ['INPUTS', 'landsat4_flag', 'landsat4_flag', False, bool],
        ['INPUTS', 'landsat5_flag', 'landsat5_flag', False, bool],
        ['INPUTS', 'landsat7_flag', 'landsat7_flag', False, bool],
        ['INPUTS', 'landsat8_flag', 'landsat8_flag', False, bool],
        # Cloudmasking
        ['INPUTS', 'acca_flag', 'acca_flag', False, bool],
        ['INPUTS', 'fmask_flag', 'fmask_flag', False, bool],
        ['INPUTS', 'fmask_type', 'fmask_type', None, str],
        #
        ['INPUTS', 'mosaic_method', 'mosaic_method', 'mean', str],
        ['INPUTS', 'adjust_method', 'adjust_method', None, str],
        # Date filtering
        ['INPUTS', 'start_year', 'start_year', None, int],
        ['INPUTS', 'end_year', 'end_year', None, int],
        ['INPUTS', 'start_month', 'start_month', None, int],
        ['INPUTS', 'end_month', 'end_month', None, int],
        ['INPUTS', 'start_doy', 'start_doy', None, int],
        ['INPUTS', 'end_doy', 'end_doy', None, int],
        # Scene ID filtering
        ['INPUTS', 'scene_id_keep_path', 'scene_id_keep_path', '', str],
        ['INPUTS', 'scene_id_skip_path', 'scene_id_skip_path', '', str],
        # Path/row filtering
        ['INPUTS', 'path_keep_list', 'path_keep_list', [], list],
        ['INPUTS', 'row_keep_list', 'row_keep_list', [], list],
        # FID filtering
        ['INPUTS', 'fid_skip_list', 'fid_skip_list', [], list],
        ['INPUTS', 'fid_keep_list', 'fid_keep_list', [], list]
    ]
    for section, input_name, output_name, default, get_type in param_list:
        try:
            if get_type is bool:
                options[output_name] = config[section].getboolean(input_name)
            elif get_type is int:
                options[output_name] = config[section].getint(input_name)
            elif get_type is list:
                options[output_name] = list(python_common.parse_int_set(
                    config[section].get(input_name)))
            else:
                options[output_name] = config[section].get(input_name)
                # Convert 'None' (strings) to None
                if options[output_name].lower() == 'none':
                    options[output_name] = None
        except:
            options[output_name] = default
            logging.debug('  Defaulting {} = {}'.format(
                input_name, options[output_name]))

        # Convert 'None' strings to None
        # This could probably be handled
        if (type(options[output_name]) is str and 
                options[output_name].lower() == 'none'):
            options[output_name] = None


    # Build output folder if necessary
    if not os.path.isdir(options['output_ws']):
        os.makedirs(options['output_ws'])

    # Start/end year
    if (options['start_year'] and options['end_year'] and 
            options['end_year'] < options['start_year']):
        logging.error(
            '\nERROR: End year must be >= start year')
        sys.exit()
    default_end_year = datetime.datetime.today().year + 1
    if ((options['start_year'] and 
            options['start_year'] not in range(1984, default_end_year)) or
        (options['end_year'] and 
            options['end_year'] not in range(1984, default_end_year))):
        logging.error('\nERROR: Year must be an integer from 1984-{}'.format(
            default_end_year-1))
        sys.exit()

    # Start/end month
    if options['start_month'] and options['start_month'] not in range(1, 13):
        logging.error(
            '\nERROR: Start month must be an integer from 1-12')
        sys.exit()
    elif options['end_month'] and options['end_month'] not in range(1, 13):
        logging.error('\nERROR: End month must be an integer from 1-12')
        sys.exit()

    # Start/end DOY
    if options['end_doy'] and options['end_doy'] > 273:
        logging.error(
            '\nERROR: End DOY has to be in the same water year as start DOY')
        sys.exit()
    if options['start_doy'] and options['start_doy'] not in range(1, 367):
        logging.error(
            '\nERROR: Start DOY must be an integer from 1-366')
        sys.exit()
    elif options['end_doy'] and options['end_doy'] not in range(1, 367):
        logging.error('\nERROR: End DOY must be an integer from 1-366')
        sys.exit()
    # if options['end_doy'] < options['start_doy']:
    #     logging.error('\nERROR: End DOY must be >= start DOY')
    #     sys.exit()

    # Fmask source type
    if options['fmask_flag']:
        options['fmask_type'] = options['fmask_type'].lower()
        if options['fmask_type'] not in ['fmask', 'cfmask']:
            logging.error(
                '\nERROR: Invalid Fmask source type: {}\n'
                '  Must be "fmask" or "cfmask"'.format(options['fmask_type']))
            sys.exit()
    else:
        options['fmask_type'] = None

    # Mosaic method
    options['mosaic_method'] = options['mosaic_method'].lower()
    mosaic_method_list = ['mean', 'median', 'mosaic', 'min', 'max']
    if options['mosaic_method'] not in mosaic_method_list:
        logging.error(
            '\nERROR: Invalid mosaic method: {}\n  Must be: {}'.format(
                options['mosaic_method'], ', '.join(mosaic_method_list)))
        sys.exit()

    # Adjust Landsat Red and NIR bands
    if options['adjust_method']:
        options['adjust_method'] = options['adjust_method'].upper()
        adjust_method_list = ['OLI_2_ETM', 'ETM_2_OLI']
        if options['adjust_method'] not in mosaic_method_list:
            logging.error(
                '\nERROR: Invalid mosaic method: {}\n  Must be: {}'.format(
                    options['adjust_method'], ', '.join(adjust_method_list)))
            sys.exit()

    # Intentionally don't apply scene_id skip/keep lists
    # Compute zonal stats for all available images
    # Filter by scene_id when making summary tables
    logging.info('  Not applying scene_id keep or skip lists')
    options['scene_id_keep_list'] = []
    options['scene_id_skip_list'] = []

    # # Only process specific Landsat scenes
    # try:
    #     with open(config['scene_id_keep_path']) as input_f:
    #         scene_id_keep_list = input_f.readlines()
    #     options['scene_id_keep_list'] = [
    #         x.strip()[:16] for x in scene_id_keep_list]
    # except IOError:
    #     logging.error('\nFileIO Error: {}'.format(
    #         config['scene_id_keep_path']))
    #     sys.exit()
    # except:
    #     options['scene_id_keep_list'] = []

    # # Skip specific landsat scenes
    # try:
    #     with open(config['scene_id_skip_path']) as input_f:
    #         scene_id_skip_list = input_f.readlines()
    #     options['scene_id_skip_list'] = [
    #         x.strip()[:16] for x in scene_id_skip_list]
    # except IOError:
    #     logging.error('\nFileIO Error: {}'.format(
    #         config['scene_id_skip_path']))
    #     sys.exit()
    # except:
    #     options['scene_id_skip_list'] = []

    return options
