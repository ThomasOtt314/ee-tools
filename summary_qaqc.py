#--------------------------------
# Name:         ee_summary_qaqc.py
# Purpose:      Generate summary tables
# Created       2017-05-15
# Python:       3.6
#--------------------------------

import argparse
import datetime
import logging
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.covariance import EllipticEnvelope

import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils


def main(ini_path=None, plot_flag=False, overwrite_flag=True):
    """Generate summary QA/QC

    Proposed QA values
    0 - Clear Scene
    1 -
    2 -
    3 -
    4 - CLOUD_SCORE == 100 or TS < 260 or SCENE_ID in skip list
    # 5 - Scene in skip list (should this be merged with 4?)

    Args:
        ini_path (str):
        plot_flag (bool): if True, generate QA/QC plots
        overwrite_flag (bool): if True, overwrite existing tables
    """

    logging.info('\nGenerate summary QA/QC')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SUMMARY')
    # inputs.parse_section(ini, section='TABLES')

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
    output_fields = landsat_daily_fields + ['FMASK_PCT', 'OUTLIER_SCORE', 'QA']

    # year_list = range(ini['INPUTS']['start_year'], ini['INPUTS']['end_year'] + 1)
    # month_list = list(utils.wrapped_range(
    #     ini['INPUTS']['start_month'], ini['INPUTS']['end_month'], 1, 12))
    # doy_list = list(utils.wrapped_range(
    #     ini['INPUTS']['start_doy'], ini['INPUTS']['end_doy'], 1, 366))

    # Add merged row XXX to keep list
    ini['INPUTS']['row_keep_list'].append('XXX')

    # Get ee features from shapefile
    zone_geom_list = gdc.shapefile_2_geom_list_func(
        ini['INPUTS']['zone_shp_path'], zone_field=ini['INPUTS']['zone_field'],
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

        qaqc_ws = os.path.join(zone_stats_ws, 'figures')
        if not os.path.isdir(qaqc_ws):
            os.makedirs(qaqc_ws)

        output_figure_fmt = os.path.join(
            qaqc_ws, '{}_{}.png'.format(zone_name, '{}'))

        logging.debug('  Reading Landsat CSV')
        landsat_df = pd.read_csv(landsat_daily_path)
        # landsat_df = pd.read_csv(
        #     landsat_daily_path, parse_dates=['DATE'], index_col='DATE')

        logging.debug('  Filtering Landsat dataframe')
        landsat_df = landsat_df[landsat_df['PIXEL_COUNT'] > 0]

        # Compute FMask percent
        landsat_df['FMASK_PCT'] = (
            landsat_df['FMASK_COUNT'].astype(np.float) /
            landsat_df['FMASK_TOTAL'])
        # Scale cloud score for plotting
        # landsat_df['CLOUD_SCORE'] = landsat_df['CLOUD_SCORE'] / 100.0

        # Add QA/QC bands if necessary
        # if 'QA' not in list(landsat_df.columns.values):
        landsat_df['QA'] = 0
        # if 'OUTLIER_SCORE' not in list(landsat_df.columns.values):
        landsat_df['OUTLIER_SCORE'] = np.nan

        # Set QA flag for scenes in skip list
        if ini['INPUTS']['scene_id_skip_list']:
            # Use primary ROW value for checking skip list SCENE_ID
            scene_id_df = pd.Series([
                s.replace('XXX', '{:03d}'.format(int(r)))
                for s, r in zip(landsat_df['SCENE_ID'], landsat_df['ROW'])])
            skip_mask = scene_id_df.isin(
                ini['INPUTS']['scene_id_skip_list']).values
            landsat_df.loc[skip_mask, 'QA'] = 4
            # Reset QA if scenes is not in skip list and flag is set
            # landsat_df.loc[(landsat_df['QA'] != 3) & (~skip_mask), 'QA'] = 0
            landsat_df.loc[~skip_mask, 'QA'] = 0
            del scene_id_df, skip_mask

        # Set initial QA band values
        landsat_df.loc[landsat_df['CLOUD_SCORE'] >= 95, 'QA'] = 3
        landsat_df.loc[landsat_df['TS'] < 260, 'QA'] = 3
        # landsat_df.loc[landsat_df['PIXEL_COUNT'] == 0, 'QA'] = 3
        # landsat_df.loc[pd.isnull(landsat_df['CLOUD_SCORE']), 'QA'] = 3

        # Build initial plots
        # summary_doy_plots(
        #     landsat_df, output_figure_fmt.format('a_original'),
        #     color_field='QA')

        # Initially filter by looking for outliers by DOY
        # Remove extreme albedo and Ts values
        max_threshold(landsat_df, field='ALBEDO_SUR', delta=0.20, qa_value=2)
        min_threshold(landsat_df, field='TS', delta=20, qa_value=2)
        # We might want to do two rounds of this
        # max_threshold(landsat_df, field='ALBEDO_SUR', delta=0.10, qa_value=2)
        # min_threshold(landsat_df, field='TS', delta=15, qa_value=2)
        # summary_doy_plots(
        #     landsat_df, output_figure_fmt.format('b_threshold'),
        #     color_field='QA')

        # # Detrend the data and remove values > 3 sigma
        # sigma_filtering(
        #     landsat_df, sigma_value=3, qa_value=2,
        #     fields=['ALBEDO_SUR', 'TS', 'NDWI_GREEN_SWIR1_SUR'])
        # summary_doy_plots(
        #     landsat_df, output_figure_fmt.format('c_sigma'),
        #     color_field='QA')

        # Identify outliers using scikit-learn EllipticEnvelope
        # Higher contamination value will remove more points
        outlier_filtering(
            landsat_df, qa_value=1, contamination=0.05,
            fields=['NDVI_TOA', 'ALBEDO_SUR', 'TS', 'NDWI_GREEN_SWIR1_SUR'])
        # outlier_filtering(
        #     landsat_df, qa_value=1, contamination=0.05,
        #     fields=['ALBEDO_SUR', 'TS', 'NDWI_GREEN_SWIR1_SUR'])

        # Plot all QA 0 points
        if plot_flag:
            summary_doy_plots(
                landsat_df[landsat_df['QA'] <= 2],
                output_figure_fmt.format('d_outlier'),
                color_field='QA')
            # summary_doy_plots(
            #     landsat_df[landsat_df['QA'] == 0],
            #     output_figure_fmt.format('e_final'), color_field='QA')
            # # summary_doy_plots(
            # #     landsat_df, output_figure_fmt.format('d_outlier_values'),
            # #     color_field='OUTLIER_SCORE')




        # # Generate annual plots
        # plot_var = 'NDVI_TOA'
        # # plot_vars = ['NDVI_TOA', 'NDWI_GREEN_SWIR1_SUR', 'ALBEDO_SUR', 'TS']
        # year_list = sorted(list(set(landsat_df['YEAR'].values)))
        # # print(year_list)

        # plot_df = landsat_df.loc[landsat_df['QA']==0, ['DOY', plot_var]]
        # x, y = plot_df.values.transpose()
        # x_new = np.arange(1, 367)

        # fit_func = lambda x, a, b, c: a * np.cos((2 * np.pi / 365) * (x - 1) + b) + c # Target function
        # popt, pcov = optimize.curve_fit(fit_func, x, y)

        # fig, ax = plt.subplots(figsize=(12, 8))
        # ax.set_xlim([1, 366])

        # popt_list = []
        # for year in year_list:
        #     plot_df = landsat_df.loc[
        #         (landsat_df['QA']==0) & (landsat_df['YEAR']==year),
        #         ['DOY', plot_var]]
        #     x, y = plot_df.values.transpose()
        #     x_new = np.arange(1, 367)

        #     year_popt, pcov = optimize.curve_fit(fit_func, x, y)
        #     popt_list.append(year_popt)
        #     # print(popt)

        #     ax.plot(x, y, marker='o', c='0.5', ms=1.5, lw=0, label=None)
        #     ax.plot(x_new, cos_fit_func(x_new, *year_popt), c='0.5', lw=1.0, label=None)
        #     # break

        # ax.plot(x_new, cos_fit_func(x_new, *popt), lw=3, color='black', label='Original')
        # # ax.plot(x_new, cos_fit_func(x_new, *np.median(np.array(popt_list), axis=0)), lw=3, color='red', label='Median')
        # plt.legend()
        # plt.show()




        # Save the QA/QC'd data
        landsat_df.to_csv(landsat_daily_path, index=False, columns=output_fields)


        # backup_path = landsat_daily_path.replace('.csv', '.csv.bak')
        # shutil.copy(landsat_daily_path, backup_path)
        # try:
        #     landsat_df.to_csv(
        #         landsat_daily_path, index=False, columns=landsat_daily_fields)
        #     os.remove(backup_path)
        # except:
        #     shutil.move(backup_path, landsat_daily_path)




        # # First filter by average cloud score
        # if ini['SUMMARY']['max_cloud_score'] < 100 and not landsat_df.empty:
        #     logging.debug('    Maximum cloud score: {0}'.format(
        #         ini['SUMMARY']['max_cloud_score']))
        #     landsat_df = landsat_df[
        #         landsat_df['CLOUD_SCORE'] <= ini['SUMMARY']['max_cloud_score']]

        # # Filter by Fmask percentage
        # if ini['SUMMARY']['max_fmask_pct'] < 100 and not landsat_df.empty:
        #     landsat_df['FMASK_PCT'] = 100 * (
        #         landsat_df['FMASK_COUNT'] / landsat_df['FMASK_TOTAL'])
        #     logging.debug('    Max Fmask threshold: {}'.format(
        #         ini['SUMMARY']['max_fmask_pct']))
        #     landsat_df = landsat_df[
        #         landsat_df['FMASK_PCT'] <= ini['SUMMARY']['max_fmask_pct']]

        # # Filter low count SLC-off images
        # if ini['SUMMARY']['min_slc_off_pct'] > 0 and not landsat_df.empty:
        #     logging.debug('    Mininum SLC-off threshold: {}%'.format(
        #         ini['SUMMARY']['min_slc_off_pct']))
        #     # logging.debug('    Maximum pixel count: {}'.format(
        #     #     max_pixel_count))
        #     slc_off_mask = (
        #         (landsat_df['LANDSAT'] == 'LE7') &
        #         ((landsat_df['YEAR'] >= 2004) |
        #          ((landsat_df['YEAR'] == 2003) & (landsat_df['DOY'] > 151))))
        #     slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / landsat_df['PIXEL_TOTAL'])
        #     # slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / max_pixel_count)
        #     landsat_df = landsat_df[
        #         ((slc_off_pct >= ini['SUMMARY']['min_slc_off_pct']) & slc_off_mask) |
        #         (~slc_off_mask)]

        del landsat_df


def cos_fit_func(x, a, b, c):
    """Assume variables follow some sort of solar cycle

    Normalize DOY (1-366) to the range [0,2*pi)
    """
    return a * np.cos((2 * np.pi * (x - 1.0) / 366) + b) + c
    # cos_fit_func = lambda x, a, b, c: (
    #     a * np.cos((2 * np.pi / 365) * (x - 1) + b) + c)


def summary_doy_plots(landsat_df, output_path, color_field=None,
                      plot_vars=[
                        'NDVI_TOA', 'ALBEDO_SUR', 'TS',
                        'NDWI_GREEN_SWIR1_SUR', 'CLOUD_SCORE',
                        'FMASK_PCT']):
    """Generate summary plots by DOY"""
    kargs = {'s': 4, 'figsize': (8, 12), 'xlim': (1, 366)}
    if color_field:
        # Make the figure wider to leave space for a legend
        kargs.update({'c': color_field, 'figsize': (9, 12)})
        if color_field.upper() == 'QA':
            kargs.update({
                'cmap': 'viridis', 'vmin': 0,
                'vmax': max(landsat_df['QA'])})
        elif color_field.upper() == 'OUTLIER_SCORE':
            kargs.update({
                'cmap': 'viridis_r', 'vmin': -2, 'vmax': 2})

    # Plot all QA 0 points
    fig, ax = plt.subplots(len(plot_vars), sharex=True)
    ax[0].xaxis.set_minor_locator(MultipleLocator(10))
    ax[0].set_xticklabels(map(str, range(0, 365, 10)))
    # qa_mask = (landsat_df['QA'].values == 0)
    for plot_i, plot_var in enumerate(plot_vars):
        ax[plot_i].patch.set_facecolor('0.96')
        landsat_df.plot.scatter(
            x='DOY', y=plot_var, ax=ax[plot_i], **kargs)

    # Trying to write the DOY values to the x axis ticks
    # ax[-1].set_xticks(range(0, 365, 50))
    # ax[-1].set_xticks(range(0, 365, 10), minor=True)
    # ax[-1].set_xticklabels(map(str, range(0, 365, 10)))
    # ax[-1].set_xlabel('DOY')

    fig.tight_layout()
    fig.savefig(output_path, fig_dpi=150)
    fig.clf()
    plt.close(fig)


def max_threshold(landsat_df, field, delta, qa_value):
    logging.debug('  Maximum {} Filtering'.format(field))
    x, y, qa = landsat_df[['DOY', field, 'QA']].values.transpose()
    if not np.any(qa < qa_value):
        return False
    popt, pcov = optimize.curve_fit(
        cos_fit_func, x[qa < qa_value], y[qa < qa_value])

    # Set the QA flag for any value with a lower level QA
    threshold_mask = (
        (qa < qa_value) &
        (y > (cos_fit_func(x, *popt) + delta)))
    if np.any(threshold_mask):
        landsat_df.loc[threshold_mask, 'QA'] = qa_value


def min_threshold(landsat_df, field, delta, qa_value):
    logging.debug('  Minimum {} Filtering'.format(field))
    x, y, qa = landsat_df[['DOY', field, 'QA']].values.transpose()
    if not np.any(qa < qa_value):
        return False
    popt, pcov = optimize.curve_fit(
        cos_fit_func, x[qa < qa_value], y[qa < qa_value])

    # Set the QA flag for any value with a lower level QA
    threshold_mask = (
        (qa < qa_value) &
        (y < (cos_fit_func(x, *popt) - delta)))
    if np.any(threshold_mask):
        landsat_df.loc[threshold_mask, 'QA'] = qa_value


def sigma_filtering(landsat_df, sigma_value, qa_value, fields):
    logging.debug('  Sigma Filtering')

    for filter_field in fields:
        # For now, start with the raw data
        x, y, qa = landsat_df[['DOY', filter_field, 'QA']].values.transpose()
        qa_mask = (qa < qa_value)

        # Fit a function to the data
        popt, pcov = optimize.curve_fit(
            cos_fit_func, x[qa_mask], y[qa_mask])

        # Detrend the data and compute the variance
        # ybar = np.mean(y[qa_mask] - cos_fit_func(x[qa_mask], *popt))
        yhat = np.std(y[qa_mask] - cos_fit_func(x[qa_mask], *popt))

        # Mask out the data
        mask = qa_mask & (y > (cos_fit_func(x, *popt) + sigma_value * yhat))
        landsat_df.loc[mask, ['QA']] = qa_value
        mask = qa_mask & (y > (cos_fit_func(x, *popt) + sigma_value * yhat))
        landsat_df.loc[mask, ['QA']] = qa_value


def outlier_filtering(landsat_df, qa_value, fields, contamination=0.15):
    logging.debug('  Cluster Filtering')
    X = np.copy(landsat_df[fields].values)
    doy = landsat_df['DOY'].values
    qa = landsat_df['QA'].values

    # Manually normalize the data
    X[:, fields.index('TS')] = (X[:, fields.index('TS')] - 270.0) / (330.0 - 270.0)
    # X[:, plot_vars.index('DOY')] = -np.cos(2 * np.pi * (X[:, fields.index('DOY')] - 1) / 365)

    # # Detrend each variable assuming a cosine function before computing outliers
    # # This is done to try to have the data be normally distributed
    # # The documentation suggested this would help EllipticEnvelope
    # cos_fields = ['TS', 'ALBEDO_SUR', 'NDWI_GREEN_SWIR1_SUR']
    # for filter_field in list(set(fields) & set(cos_fields)):
    #     logging.debug('    Detrending: {}'.format(filter_field))
    #     # Should QA=1 be included to develop the best fit line?
    #     # Let's assume no, incase outliers have already been identified
    #     popt, pcov = optimize.curve_fit(
    #         cos_fit_func, doy[qa < qa_value],
    #         X[:, fields.index(filter_field)][qa < qa_value])

    #     # Apply the correction to all values for now
    #     # X will be masked/sliced below
    #     X[:, fields.index(filter_field)] -= cos_fit_func(doy, *popt)

    clf = EllipticEnvelope(contamination=contamination)
    qa_mask = qa < qa_value
    clf.fit(X[qa_mask])
    S = clf.decision_function(X[qa_mask])
    Y = clf.predict(X[qa_mask])
    # Y = sklearn.cluster.AgglomerativeClustering(
    #     n_clusters=4, linkage='average').fit_predict(X)
    logging.debug('    Filtered: {}'.format(sum(Y != 1)))
    logging.debug('    Y values: {}'.format(sorted(list(set(Y)))))

    # Set QA to 1 if QA is 0 and the value is identified as an outlier
    qa_series = landsat_df.loc[qa_mask, 'QA'].copy(deep=True)
    qa_series[Y != 1] = qa_value
    landsat_df.loc[qa_mask, 'QA'] = qa_series
    # This isn't working for some reason
    # landsat_df.loc[qa_mask, 'QA'][Y != 1] = qa_value
    landsat_df.loc[qa_mask, 'OUTLIER_SCORE'] = S


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate summary QA/QC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=utils.arg_valid_file,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '--plots', default=False, action='store_true',
        help='Generate QA/QC plots')
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
        args.ini = utils.get_ini_path(os.getcwd())
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

    main(ini_path=args.ini, plot_flag=args.plots)
    # main(ini_path=args.ini, plot_flag=args.plots,
    #      overwrite_flag=args.overwrite)
