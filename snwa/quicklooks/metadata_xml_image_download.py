#--------------------------------
# Name:         metadata_xml_image_download.py
# Created       2016-08-23
# Python:       2.7
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import sys
import urllib


def main(xml_path, output_folder, landsat_folder=None, skip_list_path=None,
         overwrite_flag=False):
    """Download Landsat Quicklook images

    Args:
        xml_path (str): file path of Landsat bulk metadata XML
        output_folder (str): folder path
        landsat_folder (str): folder path of Landsat scenes.
            Script assumes scenes are organized in separate
            folders by path, row, and year
        skip_list_path (str): file path of Landsat skip list
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nDownload Landsat Quicklooks')
    cloud_folder_name = 'cloudy'

    start_month = 1
    end_month = 12

    year_list = list(xrange(1984, dt.datetime.now().year + 1))
    # year_list = []

    path_row_list = []
    path_list = []
    row_list = []

    browse_col = 'browseAvailable'
    url_col = 'browseURL'
    scene_col = 'sceneID'
    # sensor_col = 'sensor'
    date_col = 'acquisitionDate'
    cloud_cover_col = 'cloudCover'
    # cloud_full_col = 'cloudCoverFull'
    path_col = 'path'
    row_col = 'row'
    data_type_col = 'DATA_TYPE_L1'
    # available_col = 'L1_AVAILABLE'

    data_types = ['L1T']
    # data_types = ['L1T', 'L1GT']

    # Force all values to be integers
    try:
        path_row_list = path_row_list[:]
    except:
        path_row_list = []
    try:
        path_list = map(int, path_list)
    except:
        path_list = []
    try:
        row_list = map(int, row_list)
    except:
        row_list = []
    try:
        year_list = map(int, year_list)
    except:
        year_list = []

    # Error checking
    if not os.path.isfile(xml_path):
        logging.error('The XML file {0} doesn\'t exists'.format(xml_path))
        sys.exit()
    if skip_list_path and not os.path.isfile(skip_list_path):
        logging.error('The skip list file {0} doesn\'t exists'.format(
            skip_list_path))
        sys.exit()

    # Read in skip list
    skip_list = []
    if skip_list_path:
        with open(skip_list_path, 'r') as skip_f:
            skip_list = skip_f.readlines()
            skip_list = [item.strip()[:16] for item in skip_list]

    # Read in the Metadata XML file
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Each item is a "row" of data
    download_list = []
    for row in root:
        # Save all the attributes and text values in a dictionary
        row_dict = dict()
        for item in row:
            row_dict[item.tag.split('}')[1]] = item.text
        # Skip row if the dictionary is empty (first line)
        if not row_dict:
            continue

        scene_id = row_dict[scene_col]
        # sensor = row_dict[sensor_col].upper()
        path = int(row_dict[path_col])
        row = int(row_dict[row_col])
        path_row = 'p{0:02d}r{1:02d}'.format(path, row)
        image_dt = dt.datetime.strptime(row_dict[date_col], '%Y-%m-%d')
        year = image_dt.year

        # Quicklook image path
        image_folder = os.path.join(output_folder, path_row, str(year))
        image_name = '{0}_{1}.jpg'.format(
            dt.datetime.strftime(image_dt, '%Y_%j'), scene_id[:3])
        image_path = os.path.join(image_folder, image_name)

        # "Cloudy" quicklooks are moved to a separate folder
        cloud_path = os.path.join(image_folder, cloud_folder_name, image_name)

        # Remove exist
        if overwrite_flag:
            if os.path.isfile(image_path):
                # logging.debug('  {} - removing'.format(scene_id))
                os.remove(image_path)
            if os.path.isfile(cloud_path):
                # logging.debug('  {} - removing'.format(scene_id))
                os.remove(image_path)
        # Skip if file is already classified as cloud
        elif os.path.isfile(cloud_path):
            if os.path.isfile(image_path):
                os.remove(image_path)
            logging.debug('  {} - cloudy, skipping'.format(scene_id))
            continue

        # Skip scenes first by path/row
        if path_row_list and path_row not in path_row_list:
            logging.debug('  {} - path/row, skipping'.format(scene_id))
            continue
        elif path_list and path not in path_list:
            logging.debug('  {} - path, skipping'.format(scene_id))
            continue
        elif row_list and row not in row_list:
            logging.debug('  {} - row, skipping'.format(scene_id))
            continue
        elif year_list and year not in year_list:
            logging.debug('  {} - year, skipping'.format(scene_id))
            continue

        # Skip early/late months
        elif start_month and image_dt.month < start_month:
            logging.debug('  {} - start month, skipping'.format(scene_id))
            continue
        elif end_month and image_dt.month > end_month:
            logging.debug('  {} - end month, skipping'.format(scene_id))
            continue

        # Skip scenes that don't have a browse image
        elif (browse_col in row_dict.keys() and
              row_dict[browse_col].upper() == 'N'):
            logging.info('  {} - night time, skipping'.format(scene_id))
            continue

        # Only download quick looks for existing scenes
        if landsat_folder is not None:
            scene_path = os.path.join(
                landsat_folder, str(path), str(row), str(year),
                scene_id + '.tar.gz')
            if not os.path.join(scene_path):
                logging.debug('  {} - no tar.gz, skipping'.format(scene_id))
                continue

        # Skip Landsat 7 SLC OFF images
        # elif row_dict[sensor_col].upper() == 'LANDSAT_ETM_SLC_OFF':
        #    logging.debug('  {} - SLC_OFF'.format(scene_id))
        #    continue

        # # Try downloading fully cloudy scenes to cloud folder
        # if int(row_dict[cloud_cover_col]) >= 9:
        #    image_path = cloud_path[:]
        #    logging.info('  {} - cloud_cover >= 9, downloading to cloudy'.format(
        #         scene_id))

        # Try downloading non-L1T quicklooks to the cloud folder
        if (data_type_col not in row_dict.keys() or
                (data_type_col in row_dict.keys() and
                 row_dict[data_type_col].upper() not in data_types)):
            if os.path.isfile(image_path):
                os.remove(image_path)
            image_path = cloud_path[:]
            logging.info('  {} - not L1T, downloading to cloudy'.format(
                scene_id))

        # Try downloading scenes in skip list to cloudy folder
        if skip_list and scene_id[:16] in skip_list:
            if os.path.isfile(image_path):
                os.remove(image_path)
            image_path = cloud_path[:]
            logging.info('  {} - skip list, downloading to cloudy'.format(
                scene_id))

        # Check if file exists last
        if os.path.isfile(image_path):
            logging.debug('  {} - image exists, skipping'.format(scene_id))
            continue

        # Save download URL and save path
        download_list.append([image_path, row_dict[url_col]])

    # Download Landsat Look Images
    for image_path, image_url in sorted(download_list):
        image_folder = os.path.dirname(image_path)
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder)

        # Make cloudy image folder also
        if (os.path.basename(image_folder) != cloud_folder_name and
            not os.path.isdir(os.path.join(image_folder, cloud_folder_name))):
            os.makedirs(os.path.join(image_folder, cloud_folder_name))

        logging.info('{0}'.format(image_path))
        urllib.urlretrieve(image_url, image_path)


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description=('Download Landsat Quicklook images\n' +
                     'Beware that many values are hardcoded!'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-x', '--xml', type=lambda x: is_valid_file(parser, x),
        help='Landsat bulk metadata XML')
    parser.add_argument(
        '--output', default=sys.path[0], help='Output folder')
    parser.add_argument(
        '--landsat', default=None, help='Landsat tar.gz folder')
    parser.add_argument(
        '--skiplist', default=None, help='Skips files in skip list')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Include existing scenes in scene download list')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    if args.xml and os.path.isfile(os.path.abspath(args.xml)):
        args.xml = os.path.abspath(args.xml)
    else:
        args.xml = get_xml_path(os.getcwd())
    if os.path.isdir(os.path.abspath(args.output)):
        args.output = os.path.abspath(args.output)
    if args.landsat and os.path.isdir(os.path.abspath(args.landsat)):
        args.landsat = os.path.abspath(args.landsat)
    return args


def get_xml_path(workspace):
    import Tkinter, tkFileDialog
    root = Tkinter.Tk()
    ini_path = tkFileDialog.askopenfilename(
        initialdir=workspace, parent=root, filetypes=[('XML files', '.xml')],
        title='Select the target XML file')
    root.destroy()
    return ini_path


def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{0}'.format('#' * 80))
    logging.info('{0:<20s} {1}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{0:<20s} {1}'.format('Current Directory:', os.getcwd()))
    logging.info('{0:<20s} {1}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(xml_path=args.xml, output_folder=args.output,
         landsat_folder=args.landsat, skip_list_path=args.skiplist,
         overwrite_flag=args.overwrite)
