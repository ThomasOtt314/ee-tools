#--------------------------------
# Name:         summary_timeseries.py
# Purpose:      Generate interactive timeseries figures
# Created       2017-05-02
# Python:       2.7
#--------------------------------

import argparse
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
# import numpy as np
import pandas as pd

import ee_tools.gdal_common as gdc
import ee_tools.ini_common as ini_common
import ee_tools.python_common as python_common


def main(ini_path=None, show_flag=False, overwrite_flag=True):
    """Generate Bokeh figures

    Args:
        ini_path (str):
        show_flag (bool): if True, show the figures in the browser.
            Default is False.
        overwrite_flag (bool): if True, overwrite existing tables.
            Default is True (for now)
    """
    logging.info('\nGenerate interactive timeseries figures')

    # Read config file
    ini = ini_common.read(ini_path)
    ini_common.parse_section(ini, section='INPUTS')
    ini_common.parse_section(ini, section='SUMMARY')

    plot_var_list = ['NDVI_TOA', 'ALBEDO_SUR', 'TS', 'NDWI_GREEN_SWIR1_SUR']

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
        if not os.path.isdir(zone_stats_ws):
            logging.debug('Folder {} does not exist, skipping'.format(
                zone_stats_ws))
            continue

        # Output file paths
        output_doy_path = os.path.join(
            zone_stats_ws, '{}_timeseries_doy.html'.format(zone_name))
        output_date_path = os.path.join(
            zone_stats_ws, '{}_timeseries_date.html'.format(zone_name))

        landsat_daily_path = os.path.join(
            zone_stats_ws, '{}_landsat_daily.csv'.format(zone_name))
        if not os.path.isfile(landsat_daily_path):
            logging.error('  Landsat daily CSV does not exist, skipping zone')
            continue

        logging.debug('  Reading Landsat CSV')
        landsat_df = pd.read_csv(
            landsat_daily_path, parse_dates=['DATE'], index_col='DATE')
        # landsat_df = pd.read_csv(landsat_daily_path)
        # landsat_df.DATE = pd.to_datetime(landsat_df['DATE'], format='%Y-%m-%d')
        # landsat_df.set_index(['DATE'], inplace=True)
        landsat_df['TS'] = landsat_df['TS'].astype(float)

        # Check for QA field
        if 'QA' not in landsat_df.columns.values:
            logging.error(
                '\nPlease run the "ee_summary_qaqc.py" script '
                'to compute QA/QC values\n')
            sys.exit()

        # Check that plot variables are present
        for plot_var in plot_var_list:
            if plot_var not in landsat_df.columns.values:
                logging.error(
                    '  The variable {} does not exist in the '
                    'dataframe'.format(plot_var))
                sys.exit()

        # Compute colors for each QA value
        qa_values = sorted(list(set(landsat_df['QA'].values)))
        colors = {
            qa: "#%02x%02x%02x" % (int(r), int(g), int(b))
            for qa, (r, g, b, _) in zip(
                qa_values,
                255 * cm.viridis(mpl.colors.Normalize()(qa_values)))
        }

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
            fill_alpha=1, fill_color='COLOR', line_color='COLOR')
        nonselected_circle = Circle(
            fill_alpha=0.8, fill_color='#aaaaaa', line_color='#aaaaaa')


        # Plot the data by DOY
        if os.path.isfile(output_doy_path):
            os.remove(output_doy_path)
        output_file(output_doy_path, title=zone_name)

        figure_args = dict(
            plot_width=750, plot_height=250, title=None,
            tools="xwheel_zoom,xpan,xbox_zoom,reset,box_select",
            # tools="xwheel_zoom,xpan,xbox_zoom,reset,tap",
            active_scroll="xwheel_zoom")
        plot_args = dict(size=4, alpha=0.9, color='COLOR', legend='QA')

        figures = []
        for plot_i, plot_var in enumerate(plot_var_list):
            if plot_i == 0:
                f = figure(
                    # x_range=Range1d(x_limit[0], x_limit[1], bounds=x_limit),
                    y_axis_label=plot_var, **figure_args)
            else:
                f = figure(
                    x_range=f.x_range, y_axis_label=plot_var, **figure_args)

            for qa, source in sorted(qa_sources.items()):
                r = f.circle('DOY', plot_var, source=source, **plot_args)
                r.selection_glyph = selected_circle
                r.nonselection_glyph = nonselected_circle
            f.add_tools(
                HoverTool(tooltips=[("DATE", "@TIME"), ("DOY", "@DOY")]))
            f.legend.location = "top_left"
            f.legend.click_policy = "hide"
            # f.legend.click_policy = "mute"
            f.legend.orientation = "horizontal"
            figures.append(f)

        p = gridplot(figures, ncols=1, sizing_mode='stretch_both')

        if show_flag:
            show(p)
        save(p)



        # Plot the data by DATE
        if os.path.isfile(output_date_path):
            os.remove(output_date_path)
        output_file(output_date_path, title=zone_name)

        figure_args = dict(
            plot_width=750, plot_height=250, title=None,
            tools="xwheel_zoom,xpan,xbox_zoom,reset,box_select",
            # tools="xwheel_zoom,xpan,xbox_zoom,reset,tap",
            active_scroll="xwheel_zoom",
            x_axis_type="datetime",)
        plot_args = dict(size=4, alpha=0.9, color='COLOR', legend='QA')
        # x_limit = (1, 366)

        figures = []
        for plot_i, plot_var in enumerate(plot_var_list):
            if plot_i == 0:
                f = figure(
                    # x_range=Range1d(x_limit[0], x_limit[1], bounds=x_limit),
                    y_axis_label=plot_var, **figure_args)
            else:
                f = figure(
                    x_range=f.x_range, y_axis_label=plot_var, **figure_args)

            for qa, source in sorted(qa_sources.items()):
                r = f.circle('DATE', plot_var, source=source, **plot_args)
                r.selection_glyph = selected_circle
                r.nonselection_glyph = nonselected_circle
            f.add_tools(
                HoverTool(tooltips=[("DATE", "@TIME"), ("DOY", "@DOY")]))
            f.legend.location = "top_left"
            f.legend.click_policy = "hide"
            # f.legend.click_policy = "mute"
            f.legend.orientation = "horizontal"
            figures.append(f)

        p = gridplot(figures, ncols=1, sizing_mode='stretch_both')

        if show_flag:
            show(p)
        save(p)

        # break


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate interactive timeseries figures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=lambda x: python_common.valid_file(x),
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

    main(ini_path=args.ini, show_flag=args.show)
    # main(ini_path=args.ini, show_flag=args.show,
    #      overwrite_flag=args.overwrite)
