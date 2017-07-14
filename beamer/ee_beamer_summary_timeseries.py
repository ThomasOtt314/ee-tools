#--------------------------------
# Name:         ee_beamer_summary_timeseries.py
# Purpose:      Generate interactive timeseries figures
# Created       2017-06-20
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
import datetime
import logging
import os
import sys

from bokeh.io import output_file, save, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, Range1d, TapTool
from bokeh.models.glyphs import Circle
from bokeh.plotting import figure
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# This is an awful way of getting the parent folder into the path
# We really should package this up as a module with a setup.py
# This way the ee_tools folders would be in the
#   PYTHONPATH env. variable
ee_tools_path = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(ee_tools_path, 'ee_tools'))
sys.path.insert(0, ee_tools_path)
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils


def main(ini_path, show_flag=False, overwrite_flag=True):
    """Generate Bokeh figures

    Bokeh issues:
    Adjust y range based on non-muted data
        https://stackoverflow.com/questions/43620837/how-to-get-bokeh-to-dynamically-adjust-y-range-when-panning
    Linked interactive legends so that there is only one legend for the gridplot
    Maybe hide or mute QA values above max (instead of filtering them in advance)

    Args:
        ini_path (str):
        show_flag (bool): if True, show the figures in the browser.
            Default is False.
        overwrite_flag (bool): if True, overwrite existing tables.
            Default is True (for now)
    """
    logging.info('\nGenerate interactive timeseries figures')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SUMMARY')

    # Eventually read from INI
    plot_var_list = ['NDVI_TOA', 'ALBEDO_SUR', 'TS', 'NDWI_TOA', 'EVI_SUR']
    # plot_var_list = [
    #     'NDVI_TOA', 'ALBEDO_SUR', 'TS', 'NDWI_TOA',
    #     'CLOUD_SCORE', 'FMASK_PCT']

    year_list = range(
        ini['INPUTS']['start_year'], ini['INPUTS']['end_year'] + 1)
    month_list = list(utils.wrapped_range(
        ini['INPUTS']['start_month'], ini['INPUTS']['end_month'], 1, 12))
    doy_list = list(utils.wrapped_range(
        ini['INPUTS']['start_doy'], ini['INPUTS']['end_doy'], 1, 366))

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
    zone_list = []
    for zone_fid, zone_name, zone_json in zone_geom_list:
        zone_name = zone_name.replace(' ', '_')
        # zone['json'] = zone_json
        logging.info('ZONE: {} (FID: {})'.format(zone_name, zone_fid))
        zone_list.append(zone_name)

        zone_stats_ws = os.path.join(ini['SUMMARY']['output_ws'], zone_name)
        figures_ws = os.path.join(zone_stats_ws, 'figures')
        if not os.path.isdir(zone_stats_ws):
            logging.debug('Folder {} does not exist, skipping'.format(
                zone_stats_ws))
            continue
        if not os.path.isdir(figures_ws):
            os.makedirs(figures_ws)

        # Output file paths
        output_doy_path = os.path.join(
            ini['SUMMARY']['output_ws'],
            '{}_timeseries_doy.html'.format(zone_name))
        output_date_path = os.path.join(
            ini['SUMMARY']['output_ws'],
            '{}_timeseries_date.html'.format(zone_name))

        landsat_daily_path = os.path.join(
            ini['SUMMARY']['output_ws'], 'brady_v5.csv')
        # landsat_daily_path = os.path.join(
        #     zone_stats_ws, '{}_landsat_daily.csv'.format(zone_name))
        if not os.path.isfile(landsat_daily_path):
            logging.error('  Landsat daily CSV does not exist, skipping zone')
            continue

        logging.debug('  Reading Landsat CSV')
        landsat_df = pd.read_csv(
            landsat_daily_path, parse_dates=['DATE'], index_col='DATE')

        # Filter to the target zone
        landsat_df = landsat_df[landsat_df['ZONE_NAME'] == zone_name]

        # Check for QA field
        if 'QA' not in landsat_df.columns.values:
            # logging.warning(
            #     '  WARNING: QA field not present in CSV\n'
            #     '  To compute QA/QC values, please run "ee_summary_qaqc.py"\n'
            #     '  Script will continue with no QA/QC values')
            landsat_df['QA'] = 0
            # raw_input('ENTER')
            # logging.error(
            #     '\nPlease run the "ee_summary_qaqc.py" script '
            #     'to compute QA/QC values\n')
            # sys.exit()

        # Check that plot variables are present
        for plot_var in plot_var_list:
            if plot_var not in landsat_df.columns.values:
                logging.error(
                    '  The variable {} does not exist in the '
                    'dataframe'.format(plot_var))
                sys.exit()

        logging.debug('  Filtering Landsat dataframe')
        landsat_df = landsat_df[landsat_df['PIXEL_COUNT'] > 0]

        # # This assumes that there are L5/L8 images in the dataframe
        # if not landsat_df.empty:
        #     max_pixel_count = max(landsat_df['PIXEL_COUNT'])
        # else:
        #     max_pixel_count = 0

        if year_list:
            landsat_df = landsat_df[landsat_df['YEAR'].isin(year_list)]
        if month_list:
            landsat_df = landsat_df[landsat_df['MONTH'].isin(month_list)]
        if doy_list:
            landsat_df = landsat_df[landsat_df['DOY'].isin(doy_list)]

        if ini['INPUTS']['path_keep_list']:
            landsat_df = landsat_df[
                landsat_df['PATH'].isin(ini['INPUTS']['path_keep_list'])]
        if (ini['INPUTS']['row_keep_list'] and
                ini['INPUTS']['row_keep_list'] != ['XXX']):
            landsat_df = landsat_df[
                landsat_df['ROW'].isin(ini['INPUTS']['row_keep_list'])]

        # Assume the default is for these to be True and only filter if False
        if not ini['INPUTS']['landsat4_flag']:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LT04']
        if not ini['INPUTS']['landsat5_flag']:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LT05']
        if not ini['INPUTS']['landsat7_flag']:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LE07']
        if not ini['INPUTS']['landsat8_flag']:
            landsat_df = landsat_df[landsat_df['LANDSAT'] != 'LC08']

        if ini['INPUTS']['scene_id_keep_list']:
            # Replace XXX with primary ROW value for checking skip list SCENE_ID
            scene_id_df = pd.Series([
                s.replace('XXX', '{:03d}'.format(int(r)))
                for s, r in zip(landsat_df['SCENE_ID'], landsat_df['ROW'])])
            landsat_df = landsat_df[scene_id_df.isin(
                ini['INPUTS']['scene_id_keep_list']).values]
            # This won't work: SCENE_ID have XXX but scene_id_skip_list don't
            # landsat_df = landsat_df[landsat_df['SCENE_ID'].isin(
            #     ini['INPUTS']['scene_id_keep_list'])]
        if ini['INPUTS']['scene_id_skip_list']:
            # Replace XXX with primary ROW value for checking skip list SCENE_ID
            scene_id_df = pd.Series([
                s.replace('XXX', '{:03d}'.format(int(r)))
                for s, r in zip(landsat_df['SCENE_ID'], landsat_df['ROW'])])
            landsat_df = landsat_df[np.logical_not(scene_id_df.isin(
                ini['INPUTS']['scene_id_skip_list']).values)]
            # This won't work: SCENE_ID have XXX but scene_id_skip_list don't
            # landsat_df = landsat_df[np.logical_not(landsat_df['SCENE_ID'].isin(
            #     ini['INPUTS']['scene_id_skip_list']))]

        # Filter by QA/QC value
        if ini['SUMMARY']['max_qa'] >= 0 and not landsat_df.empty:
            logging.debug('    Maximum QA: {0}'.format(
                ini['SUMMARY']['max_qa']))
            landsat_df = landsat_df[landsat_df['QA'] <= ini['SUMMARY']['max_qa']]

        # First filter by average cloud score
        if ini['SUMMARY']['max_cloud_score'] < 100 and not landsat_df.empty:
            logging.debug('    Maximum cloud score: {0}'.format(
                ini['SUMMARY']['max_cloud_score']))
            landsat_df = landsat_df[
                landsat_df['CLOUD_SCORE'] <= ini['SUMMARY']['max_cloud_score']]

        # Filter by Fmask percentage
        if ini['SUMMARY']['max_fmask_pct'] < 100 and not landsat_df.empty:
            landsat_df['FMASK_PCT'] = 100 * (
                landsat_df['FMASK_COUNT'] / landsat_df['FMASK_TOTAL'])
            logging.debug('    Max Fmask threshold: {}'.format(
                ini['SUMMARY']['max_fmask_pct']))
            landsat_df = landsat_df[
                landsat_df['FMASK_PCT'] <= ini['SUMMARY']['max_fmask_pct']]

        # Filter low count SLC-off images
        if ini['SUMMARY']['min_slc_off_pct'] > 0 and not landsat_df.empty:
            logging.debug('    Mininum SLC-off threshold: {}%'.format(
                ini['SUMMARY']['min_slc_off_pct']))
            # logging.debug('    Maximum pixel count: {}'.format(
            #     max_pixel_count))
            slc_off_mask = (
                (landsat_df['LANDSAT'] == 'LE7') &
                ((landsat_df['YEAR'] >= 2004) |
                 ((landsat_df['YEAR'] == 2003) & (landsat_df['DOY'] > 151))))
            slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / landsat_df['PIXEL_TOTAL'])
            # slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / max_pixel_count)
            landsat_df = landsat_df[
                ((slc_off_pct >= ini['SUMMARY']['min_slc_off_pct']) & slc_off_mask) |
                (~slc_off_mask)]

        if landsat_df.empty:
            logging.error(
                '  Empty Landsat dataframe after filtering, skipping zone')
            continue

        # Compute colors for each QA value
        logging.debug('  Building column data source')
        qa_values = sorted(list(set(landsat_df['QA'].values)))
        colors = {
            qa: "#%02x%02x%02x" % (int(r), int(g), int(b))
            for qa, (r, g, b, _) in zip(
                qa_values,
                255 * cm.viridis(mpl.colors.Normalize()(qa_values)))
        }
        logging.debug('  QA values: {}'.format(
            ', '.join(map(str, qa_values))))

        # Unpack the data by QA type to support interactive legends
        qa_sources = dict()
        for qa_value in qa_values:
            qa_df = landsat_df[landsat_df['QA'] == qa_value]
            qa_data = {
                'INDEX': list(range(len(qa_df.index))),
                'DATE': qa_df.index,
                'TIME': qa_df.index.map(lambda x: x.strftime('%Y-%m-%d')),
                'DOY': qa_df['DOY'].values,
                'QA': qa_df['QA'].values,
                'COLOR': [colors[qa] for qa in qa_df['QA'].values]
            }
            for plot_var in plot_var_list:
                if plot_var in qa_df.columns.values:
                    qa_data.update({plot_var: qa_df[plot_var].values})
            qa_sources[qa_value] = ColumnDataSource(qa_data)

        # Selection
        selected_circle = Circle(
            fill_color='COLOR', line_color='COLOR')
        nonselected_circle = Circle(
            fill_color='#aaaaaa', line_color='#aaaaaa')


        # Plot the data by DOY
        logging.debug('  Building DOY timeseries figure')
        if os.path.isfile(output_doy_path):
            os.remove(output_doy_path)
        output_file(output_doy_path, title=zone_name)

        figure_args = dict(
            plot_width=750, plot_height=250, title=None,
            tools="xwheel_zoom,xpan,xbox_zoom,reset,box_select",
            # tools="xwheel_zoom,xpan,xbox_zoom,reset,tap",
            active_scroll="xwheel_zoom")
        plot_args = dict(
            size=4, alpha=0.9, color='COLOR')
        if ini['SUMMARY']['max_qa'] > 0:
            plot_args['legend'] = 'QA'

        figures = []
        for plot_i, plot_var in enumerate(plot_var_list):
            if plot_i == 0:
                f = figure(
                    # x_range=Range1d(1, 366, bounds=(1, 366)),
                    y_axis_label=plot_var, **figure_args)
            else:
                f = figure(
                    x_range=f.x_range, y_axis_label=plot_var, **figure_args)

            for qa, source in sorted(qa_sources.items()):
                r = f.circle('DOY', plot_var, source=source, **plot_args)
                r.selection_glyph = selected_circle
                r.nonselection_glyph = nonselected_circle
                r.muted_glyph = nonselected_circle
                # DEADBEEF - This will display high QA points as muted
                # if qa > ini['SUMMARY']['max_qa']:
                #     r.muted = True
                #     # r.visible = False

            f.add_tools(
                HoverTool(tooltips=[("DATE", "@TIME"), ("DOY", "@DOY")]))

            # if ini['SUMMARY']['max_qa'] > 0:
            f.legend.location = "top_left"
            f.legend.click_policy = "hide"
            # f.legend.click_policy = "mute"
            f.legend.orientation = "horizontal"

            figures.append(f)

        # Try to not allow more than 4 plots in a column
        p = gridplot(
            figures, ncols=len(plot_var_list) // 3,
            sizing_mode='stretch_both')

        if show_flag:
            show(p)
        save(p)


        # Plot the data by DATE
        logging.debug('  Building date timeseries figure')
        if os.path.isfile(output_date_path):
            os.remove(output_date_path)
        output_file(output_date_path, title=zone_name)

        figure_args = dict(
            plot_width=750, plot_height=250, title=None,
            tools="xwheel_zoom,xpan,xbox_zoom,reset,box_select",
            # tools="xwheel_zoom,xpan,xbox_zoom,reset,tap",
            active_scroll="xwheel_zoom",
            x_axis_type="datetime",)
        plot_args = dict(
            size=4, alpha=0.9, color='COLOR')
        if ini['SUMMARY']['max_qa'] > 0:
            plot_args['legend'] = 'QA'

        figures = []
        for plot_i, plot_var in enumerate(plot_var_list):
            if plot_i == 0:
                f = figure(
                    # x_range=Range1d(x_limit[0], x_limit[1], bounds=x_limit),
                    y_axis_label=plot_var, **figure_args)
            else:
                f = figure(
                    x_range=f.x_range, y_axis_label=plot_var, **figure_args)

            if plot_var == 'TS':
                f.y_range.bounds = (270, None)

            for qa, source in sorted(qa_sources.items()):
                r = f.circle('DATE', plot_var, source=source, **plot_args)
                r.selection_glyph = selected_circle
                r.nonselection_glyph = nonselected_circle
                r.muted_glyph = nonselected_circle
                # DEADBEEF - This will display high QA points as muted
                # if qa > ini['SUMMARY']['max_qa']:
                #     r.muted = True
                #     # r.visible = False
            f.add_tools(
                HoverTool(tooltips=[("DATE", "@TIME"), ("DOY", "@DOY")]))

            # if ini['SUMMARY']['max_qa'] > 0:
            f.legend.location = "top_left"
            f.legend.click_policy = "hide"
            # f.legend.click_policy = "mute"
            f.legend.orientation = "horizontal"

            figures.append(f)

        # Try to not allow more than 4 plots in a column
        p = gridplot(
            figures, ncols=len(plot_var_list) // 3,
            sizing_mode='stretch_both')

        if show_flag:
            show(p)
        save(p)

        # Don't automatically build all plots if show is True
        if show_flag:
            input('Press ENTER to continue')
        # break


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate interactive timeseries figures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=utils.arg_valid_file,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '--show', default=False, action='store_true',
        help='Show figures')
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
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, show_flag=args.show)
