#--------------------------------
# Name:         ee_summary_qaqc.py
# Purpose:      Generate summary tables
# Created       2017-04-26
# Python:       2.7
#--------------------------------

import argparse
import datetime
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.covariance import EllipticEnvelope

import ee_tools.gdal_common as gdc
import ee_tools.ini_common as ini_common
import ee_tools.python_common as python_common


def main(ini_path=None, overwrite_flag=True):
    """Generate summary QA/QC

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing tables
    """

    logging.info('\nGenerate summary QA/QC')

    # Read config file
    # ini = ini_common.ini_parse(ini_path, section='TABLES')
    ini = ini_common.read(ini_path)
    ini_common.parse_section(ini, section='INPUTS')
    ini_common.parse_section(ini, section='SUMMARY')
    ini_common.parse_section(ini, section='TABLES')

    landsat_daily_fields = [
        'ZONE_FID', 'ZONE_NAME', 'DATE', 'SCENE_ID', 'LANDSAT',
        'PATH', 'ROW', 'YEAR', 'MONTH', 'DAY', 'DOY', 'CLOUD_SCORE',
        'PIXEL_COUNT', 'PIXEL_TOTAL', 'FMASK_COUNT', 'FMASK_TOTAL',
        'TS', 'ALBEDO_SUR', 'NDVI_TOA', 'NDVI_SUR', 'EVI_SUR',
        'NDWI_GREEN_NIR_SUR', 'NDWI_GREEN_SWIR1_SUR', 'NDWI_NIR_SWIR1_SUR',
        # 'NDWI_GREEN_NIR_TOA', 'NDWI_GREEN_SWIR1_TOA', 'NDWI_NIR_SWIR1_TOA',
        # 'NDWI_SWIR1_GREEN_TOA', 'NDWI_SWIR1_GREEN_SUR',
        # 'NDWI_TOA', 'NDWI_SUR',
        'TC_BRIGHT', 'TC_GREEN', 'TC_WET']

    # year_list = range(ini['INPUTS']['start_year'], ini['INPUTS']['end_year'] + 1)
    # month_list = list(python_common.wrapped_range(
    #     ini['INPUTS']['start_month'], ini['INPUTS']['end_month'], 1, 12))
    # doy_list = list(python_common.wrapped_range(
    #     ini['INPUTS']['start_doy'], ini['INPUTS']['end_doy'], 1, 366))

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

    # Assume variables follow some sort of solar cycle
    cos_fit_func = lambda x, a, b, c: (
        a * np.cos((2 * np.pi / 365) * (x - 1) + b) + c)


    def summary_plots(landsat_df, plot_name, color_field='QA'):
        # Plot all QA 0 points
        fig, ax = plt.subplots(4, sharex=True, figsize=(8, 12))
        qa_mask = (landsat_df['QA'].values == 0)
        landsat_df[qa_mask].plot.scatter(
            x='DOY', y='NDVI_TOA', c=color_field, cmap='viridis_r',
            ax=ax[0], s=3, xlim=(1, 366))
        landsat_df[qa_mask].plot.scatter(
            x='DOY', y='NDWI_GREEN_SWIR1_SUR', c=color_field, cmap='viridis_r',
            ax=ax[1], s=3)
        landsat_df[qa_mask].plot.scatter(
            x='DOY', y='ALBEDO_SUR', c=color_field, cmap='viridis_r',
            ax=ax[2], s=3)
        landsat_df[qa_mask].plot.scatter(
            x='DOY', y='TS', c=color_field, cmap='viridis_r',
            ax=ax[3], s=3)
        fig.savefig(output_figure_fmt.format(plot_name))


    logging.info('\nProcessing zones')
    # output_df = None
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
        if not os.path.isfile(landsat_daily_path):
            logging.error('  Landsat daily CSV does not exist, skipping zone')
            continue

        qaqc_ws = os.path.join(zone_stats_ws, 'qaqc')
        if not os.path.isdir(qaqc_ws):
            os.makedirs(qaqc_ws)

        output_table_path = landsat_daily_path.replace('.csv', '_qaqc.csv')
        output_figure_fmt = os.path.join(
            qaqc_ws, '{}_{}.png'.format(zone_name, '{}'))

        logging.debug('  Reading Landsat CSV')
        landsat_df = pd.read_csv(landsat_daily_path)

        # logging.debug('  Filtering Landsat dataframe')
        # landsat_df = landsat_df[landsat_df['PIXEL_COUNT'] > 0]

        # Set initial QA band
        landsat_df['QA'] = 0
        landsat_df['OUTLIER_SCORE'] = np.nan

        summary_plots(landsat_df, 'original', color_field='QA')






        # Albedo threshold
        logging.debug('  Albedo Threshold Filtering')
        threshold_qa = 3
        threshold_max = 0.20
        x, y, qa = landsat_df[['DOY', 'ALBEDO_SUR', 'QA']].values.transpose()
        popt, pcov = optimize.curve_fit(
            cos_fit_func, x[qa < threshold_qa], y[qa < threshold_qa])

        # Set the QA flag for any value with a lower level QA
        threshold_mask = (
            (qa < threshold_qa) &
            (y > (cos_fit_func(x, *popt) + threshold_max)))
        landsat_df.loc[threshold_mask, 'QA'] = threshold_qa




        # Ts threshold
        logging.debug('  Ts Threshold Filtering')
        threshold_qa = 3
        threshold_min = 20
        x, y, qa = landsat_df[['DOY', 'TS', 'QA']].values.transpose()
        popt, pcov = optimize.curve_fit(
            cos_fit_func, x[qa < threshold_qa], y[qa < threshold_qa])

        # Set the QA flag for any value with a lower level QA
        threshold_mask = (
            (qa < threshold_qa) &
            (y < (cos_fit_func(x, *popt) - threshold_min)))
        landsat_df.loc[threshold_mask, 'QA'] = threshold_qa

        summary_plots(landsat_df, 'threshold', color_field='QA')




        # Sigma threshold filtering
        logging.debug('  Sigma Filtering')
        sigma = 3.0
        sigma_qa = 2

        # plot_vars = ['NDWI_GREEN_SWIR1_SUR', ]
        plot_vars = ['NDVI_TOA', 'ALBEDO_SUR', 'TS', 'NDWI_GREEN_SWIR1_SUR', ]
        for plot_var in plot_vars:

            # For now, start with the raw data
            x, y, qa = landsat_df[['DOY', plot_var, 'QA']].values.transpose()
            qa_mask = (qa < sigma_qa)

            # Fit a function to the data
            popt, pcov = optimize.curve_fit(
                cos_fit_func, x[qa_mask], y[qa_mask])

            # Detrend the data and compute the variance
            # ybar = np.mean(y[qa_mask] - cos_fit_func(x[qa_mask], *popt))
            yhat = np.std(y[qa_mask] - cos_fit_func(x[qa_mask], *popt))

            # Mask out the data
            landsat_df.loc[
                qa_mask & (y > (cos_fit_func(x, *popt) + sigma * yhat)), ['QA']] = sigma_qa
            landsat_df.loc[
                qa_mask & (y < (cos_fit_func(x, *popt) - sigma * yhat)), ['QA']] = sigma_qa

            # Compute a new best fit line
            x, y, qa = landsat_df[['DOY', plot_var, 'QA']].values.transpose()
            qa_mask = (qa < sigma_qa)
            popt, pcov = optimize.curve_fit(cos_fit_func, x[qa_mask], y[qa_mask])

        summary_plots(landsat_df, 'sigma', color_field='QA')




        # Cluster Filtering
        logging.debug('  Cluster Filtering')
        cluster_qa = 1
        plot_vars = ['NDVI_TOA', 'NDWI_GREEN_SWIR1_SUR', 'ALBEDO_SUR', 'TS']
        # plot_vars = ['NDVI_TOA', 'NDWI_GREEN_SWIR1_SUR', 'ALBEDO_SUR', 'TS', 'DOY']
        X = landsat_df[plot_vars].values
        doy = landsat_df['DOY'].values
        qa = landsat_df['QA'].values
        qa_mask = qa < cluster_qa

        # Manually normalize the data
        X[:, plot_vars.index('TS')] = (X[:, plot_vars.index('TS')] - 270.0) / (330.0 - 270.0)
        # X[:, plot_vars.index('DOY')] = -np.cos(2 * np.pi * (X[:, plot_vars.index('DOY')] - 1) / 365)

        # Detrend the data assuming a cosine function before computing outliers
        for plot_var in plot_vars:
            x = doy[qa_mask]
            y = X[:, plot_vars.index(plot_var)][qa_mask]
            popt, pcov = optimize.curve_fit(cos_fit_func, x, y)
            X[:, plot_vars.index(plot_var)][qa_mask] = y - cos_fit_func(x, *popt)

        # Y = sklearn.cluster.AgglomerativeClustering(
        #     n_clusters=4, linkage='average').fit_predict(X)
        clf = EllipticEnvelope(contamination=0.15)
        clf.fit(X[qa_mask])
        S = clf.decision_function(X[qa_mask])
        Y = clf.predict(X[qa_mask])
        # print(sorted(list(set(Y))))

        # Set QA to 1 if QA is 0 and the value is identified as an outlier
        landsat_df.loc[qa_mask, 'QA'][Y != 1] = cluster_qa
        landsat_df.loc[qa_mask, 'OUTLIER'] = S




        # Plot all QA 0 points
        summary_plots(landsat_df, 'cluster', color_field='OUTLIER')




        # Save the QA/QC'd data
        landsat_df.to_csv(output_table_path, index=False, columns=landsat_daily_fields)

        break





        # # This assumes that there are L5/L8 images in the dataframe
        # if not landsat_df.empty:
        #     max_pixel_count = max(landsat_df['PIXEL_COUNT'])
        # else:
        #     max_pixel_count = 0

    #     if year_list:
    #         landsat_df = landsat_df[landsat_df['YEAR'].isin(year_list)]
    #     if month_list:
    #         landsat_df = landsat_df[landsat_df['MONTH'].isin(month_list)]
    #     if doy_list:
    #         landsat_df = landsat_df[landsat_df['DOY'].isin(doy_list)]

    #     if ini['INPUTS']['path_keep_list']:
    #         landsat_df = landsat_df[
    #             landsat_df['PATH'].isin(ini['INPUTS']['path_keep_list'])]
    #     if (ini['INPUTS']['row_keep_list'] and
    #             ini['INPUTS']['row_keep_list'] != ['XXX']):
    #         landsat_df = landsat_df[
    #             landsat_df['ROW'].isin(ini['INPUTS']['row_keep_list'])]

    #     # Assume the default is for these to be True and only filter if False
    #     if not ini['INPUTS']['landsat4_flag']:
    #         landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LT4']
    #     if not ini['INPUTS']['landsat5_flag']:
    #         landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LT5']
    #     if not ini['INPUTS']['landsat7_flag']:
    #         landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LE7']
    #     if not ini['INPUTS']['landsat8_flag']:
    #         landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LC8']
    #     if ini['INPUTS']['scene_id_keep_list']:
    #         landsat_df = landsat_df[landsat_df['SCENE_ID'].isin(
    #             ini['INPUTS']['scene_id_keep_list'])]
    #     if ini['INPUTS']['scene_id_skip_list']:
    #         landsat_df = landsat_df[np.logical_not(landsat_df['SCENE_ID'].isin(
    #             ini['INPUTS']['scene_id_skip_list']))]

    #     # First filter by average cloud score
    #     if ini['SUMMARY']['max_cloud_score'] < 100 and not landsat_df.empty:
    #         logging.debug('    Maximum cloud score: {0}'.format(
    #             ini['SUMMARY']['max_cloud_score']))
    #         landsat_df = landsat_df[
    #             landsat_df['CLOUD_SCORE'] <= ini['SUMMARY']['max_cloud_score']]

    #     # Filter by Fmask percentage
    #     if ini['SUMMARY']['max_fmask_pct'] < 100 and not landsat_df.empty:
    #         landsat_df['FMASK_PCT'] = 100 * (
    #             landsat_df['FMASK_COUNT'] / landsat_df['FMASK_TOTAL'])
    #         logging.debug('    Max Fmask threshold: {}'.format(
    #             ini['SUMMARY']['max_fmask_pct']))
    #         landsat_df = landsat_df[
    #             landsat_df['FMASK_PCT'] <= ini['SUMMARY']['max_fmask_pct']]

    #     # Filter low count SLC-off images
    #     if ini['SUMMARY']['min_slc_off_pct'] > 0 and not landsat_df.empty:
    #         logging.debug('    Mininum SLC-off threshold: {}%'.format(
    #             ini['SUMMARY']['min_slc_off_pct']))
    #         # logging.debug('    Maximum pixel count: {}'.format(
    #         #     max_pixel_count))
    #         slc_off_mask = (
    #             (landsat_df['LANDSAT'] == 'LE7') &
    #             ((landsat_df['YEAR'] >= 2004) |
    #              ((landsat_df['YEAR'] == 2003) & (landsat_df['DOY'] > 151))))
    #         slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / landsat_df['PIXEL_TOTAL'])
    #         # slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / max_pixel_count)
    #         landsat_df = landsat_df[
    #             ((slc_off_pct >= ini['SUMMARY']['min_slc_off_pct']) & slc_off_mask) |
    #             (~slc_off_mask)]

    #     if landsat_df.empty:
    #         logging.error(
    #             '  Empty Landsat dataframe after filtering, skipping zone')
    #         zone_list.remove(zone_name)
    #         # raw_input('ENTER')
    #         continue

    #     logging.debug('  Computing Landsat annual summaries')
    #     landsat_df = landsat_df\
    #         .groupby(['ZONE_FID', 'ZONE_NAME', 'YEAR'])\
    #         .agg({
    #             'PIXEL_COUNT': {
    #                 'PIXEL_COUNT': 'mean',
    #                 'SCENE_COUNT': 'count'},
    #             'PIXEL_TOTAL': {'PIXEL_TOTAL': 'mean'},
    #             'FMASK_COUNT': {'FMASK_COUNT': 'mean'},
    #             'FMASK_TOTAL': {'FMASK_TOTAL': 'mean'},
    #             'CLOUD_SCORE': {'CLOUD_SCORE': 'mean'},
    #             'ALBEDO_SUR': {'ALBEDO_SUR': 'mean'},
    #             'EVI_SUR': {'EVI_SUR': 'mean'},
    #             'NDVI_SUR': {'NDVI_SUR': 'mean'},
    #             'NDVI_TOA': {'NDVI_TOA': 'mean'},
    #             'NDWI_GREEN_NIR_SUR': {'NDWI_GREEN_NIR_SUR': 'mean'},
    #             'NDWI_GREEN_SWIR1_SUR': {'NDWI_GREEN_SWIR1_SUR': 'mean'},
    #             'NDWI_NIR_SWIR1_SUR': {'NDWI_NIR_SWIR1_SUR': 'mean'},
    #             # 'NDWI_GREEN_NIR_TOA': {'NDWI_GREEN_NIR_TOA': 'mean'},
    #             # 'NDWI_GREEN_SWIR1_TOA': {'NDWI_GREEN_SWIR1_TOA': 'mean'},
    #             # 'NDWI_NIR_SWIR1_TOA': {'NDWI_NIR_SWIR1_TOA': 'mean'},
    #             # 'NDWI_SWIR1_GREEN_SUR': {'NDWI_SWIR1_GREEN_SUR': 'mean'},
    #             # 'NDWI_SWIR1_GREEN_TOA': {'NDWI_SWIR1_GREEN_TOA': 'mean'},
    #             # 'NDWI_SUR': {'NDWI_SUR': 'mean'},
    #             # 'NDWI_TOA': {'NDWI_TOA': 'mean'},
    #             'TC_BRIGHT': {'TC_BRIGHT': 'mean'},
    #             'TC_GREEN': {'TC_GREEN': 'mean'},
    #             'TC_WET': {'TC_WET': 'mean'},
    #             'TS': {'TS': 'mean'}
    #         })
    #     landsat_df.columns = landsat_df.columns.droplevel(0)
    #     landsat_df.reset_index(inplace=True)
    #     landsat_df = landsat_df[landsat_annual_fields]
    #     landsat_df['SCENE_COUNT'] = landsat_df['SCENE_COUNT'].astype(np.int)
    #     landsat_df['PIXEL_COUNT'] = landsat_df['PIXEL_COUNT'].astype(np.int)
    #     landsat_df['PIXEL_TOTAL'] = landsat_df['PIXEL_TOTAL'].astype(np.int)
    #     landsat_df['FMASK_COUNT'] = landsat_df['FMASK_COUNT'].astype(np.int)
    #     landsat_df['FMASK_TOTAL'] = landsat_df['FMASK_TOTAL'].astype(np.int)
    #     landsat_df.sort_values(by='YEAR', inplace=True)

    #     if os.path.isfile(gridmet_monthly_path):
    #         logging.debug('  Reading montly GRIDMET CSV')
    #         gridmet_df = pd.read_csv(gridmet_monthly_path)
    #     elif os.path.isfile(gridmet_daily_path):
    #         logging.debug('  Reading daily GRIDMET CSV')
    #         gridmet_df = pd.read_csv(gridmet_daily_path)

    #     logging.debug('  Computing GRIDMET summaries')
    #     # Summarize GRIDMET for target months year
    #     if (gridmet_start_month in [10, 11, 12] and
    #             gridmet_end_month in [10, 11, 12]):
    #         month_mask = (
    #             (gridmet_df['MONTH'] >= gridmet_start_month) &
    #             (gridmet_df['MONTH'] <= gridmet_end_month))
    #         gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR'] + 1
    #     elif (gridmet_start_month in [10, 11, 12] and
    #           gridmet_end_month not in [10, 11, 12]):
    #         month_mask = gridmet_df['MONTH'] >= gridmet_start_month
    #         gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR'] + 1
    #         month_mask = gridmet_df['MONTH'] <= gridmet_end_month
    #         gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR']
    #     else:
    #         month_mask = (
    #             (gridmet_df['MONTH'] >= gridmet_start_month) &
    #             (gridmet_df['MONTH'] <= gridmet_end_month))
    #         gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR']
    #     gridmet_df['GROUP_YEAR'] = gridmet_df['GROUP_YEAR'].astype(int)

    #     if year_list:
    #         gridmet_df = gridmet_df[gridmet_df['GROUP_YEAR'].isin(year_list)]
    #         if gridmet_df.empty:
    #             logging.error(
    #                 '    Empty GRIDMET dataframe after filtering by year')
    #             continue

    #     # Group GRIDMET data by user specified range (default is water year)
    #     gridmet_df = gridmet_df\
    #         .groupby(['ZONE_FID', 'ZONE_NAME', 'GROUP_YEAR'])\
    #         .agg({'ETO': {'ETO': 'sum'}, 'PPT': {'PPT': 'sum'}})
    #     gridmet_df.columns = gridmet_df.columns.droplevel(0)
    #     gridmet_df.reset_index(inplace=True)
    #     gridmet_df.rename(columns={'GROUP_YEAR': 'YEAR'}, inplace=True)
    #     gridmet_df.sort_values(by='YEAR', inplace=True)

    #     # Merge Landsat and GRIDMET collections
    #     zone_df = landsat_df.merge(
    #         gridmet_df, on=['ZONE_FID', 'ZONE_NAME', 'YEAR'])
    #     # zone_df = landsat_df.merge(gridmet_df, on=['ZONE_FID', 'YEAR'])

    #     if output_df is None:
    #         output_df = zone_df.copy()
    #     else:
    #         output_df = output_df.append(zone_df)

    #     del landsat_df, gridmet_df, zone_df

    # if output_df is not None and not output_df.empty:
    #     logging.info('\nWriting summary tables to Excel')
    #     excel_f = ExcelWriter(output_path)
    #     logging.debug('  {}'.format(output_path))
    #     for zone_name in zone_list:
    #         logging.debug('  {}'.format(zone_name))
    #         zone_df = output_df[output_df['ZONE_NAME'] == zone_name]
    #         zone_df.to_excel(
    #             excel_f, sheet_name=zone_name, index=False, float_format='%.4f')
    #         del zone_df
    #     excel_f.save()
    # else:
    #     logging.info('  Empty output dataframe, not writing summary tables')


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate summary QA/QC',
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
