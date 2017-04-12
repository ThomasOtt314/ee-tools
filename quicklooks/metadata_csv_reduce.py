import argparse
import datetime as dt
import logging
import os

import pandas as pd


def main(csv_ws=os.getcwd()):
    """"""
    logging.info('\nFilter/reducing Landsat Metdata CSV files')

    csv_list = [
        'LANDSAT_8.csv', 'LANDSAT_ETM.csv', 'LANDSAT_ETM_SLC_OFF.csv',
        'LANDSAT_TM-1980-1989.csv', 'LANDSAT_TM-1990-1999.csv',
        'LANDSAT_TM-2000-2009.csv', 'LANDSAT_TM-2010-2012.csv']

    # Input fields
    browse_col = 'browseAvailable'
    url_col = 'browseURL'
    scene_col = 'sceneID'
    date_col = 'acquisitionDate'
    cloud_cover_col = 'cloudCover'
    path_col = 'path'
    row_col = 'row'
    data_type_col = 'DATA_TYPE_L1'

    sensor_col = 'sensor'
    cloud_full_col = 'cloudCoverFull'
    # available_col = 'L1_AVAILABLE'

    # Only load the following columns from the CSV
    use_cols = [
        browse_col, url_col, scene_col, date_col, cloud_cover_col,
        path_col, row_col, data_type_col, sensor_col, cloud_full_col,
        'sceneStartTime', 'sunElevation', 'sunAzimuth']
        # 'UTM_ZONE', 'IMAGE_QUALITY', available_col, 'satelliteNumber']

    for csv_name in csv_list:
        logging.info('{}'.format(csv_name))
        csv_path = os.path.join(csv_ws, csv_name)

        # Read in the CSV
        input_df = pd.read_csv(csv_path)

        # parse_dates=[date_col]
        # logging.debug('  {}'.format(', '.join(input_df.columns.values)))
        # logging.debug(input_df.head())
        logging.debug('  Scene count: {}'.format(len(input_df)))

        # Keep target columns
        input_df = input_df[use_cols]

        # Remove high latitute rows
        input_df = input_df[input_df[row_col] < 100]
        input_df = input_df[input_df[row_col] > 9]
        logging.debug('  Scene count: {}'.format(len(input_df)))

        input_df = input_df[input_df['sunElevation'] > 0]
        logging.debug('  Scene count: {}'.format(len(input_df)))

        # Save to CSV
        input_df.to_csv(csv_path)


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description=('Filter Landsat Metadata CSV files'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')

    main()
