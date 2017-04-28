#--------------------------------
# Name:         bokeh_qaqc.py
# Purpose:      Generate Bokeh figures
# Created       2017-04-28
# Python:       2.7
#--------------------------------

import argparse
import datetime
import logging
import os
import sys

from bokeh.io import output_file, save, show
from bokeh.layouts import column, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Range1d, TapTool
from bokeh.models.glyphs import Circle
# from bokeh.models.widgets import Slider
from bokeh.plotting import figure
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd

import ee_tools.gdal_common as gdc
import ee_tools.ini_common as ini_common
import ee_tools.python_common as python_common


def main(ini_path=None, overwrite_flag=False):
    """Generate Bokeh figures

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing tables
    """
    logging.info('\nGenerate Bokeh figures')

    # Read config file
    ini = ini_common.read(ini_path)
    ini_common.parse_section(ini, section='INPUTS')
    ini_common.parse_section(ini, section='SUMMARY')

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

        # # Compute FMask percent
        # landsat_df['FMASK_PCT'] = (
        #     landsat_df['FMASK_COUNT'].astype(np.float) /
        #     landsat_df['FMASK_TOTAL'])

        # # Apply additional basic QA/QC filtering
        # if ('QA' in list(landsat_df.columns.values) and
        #         set(landsat_df['QA']) != set([0])):
        #     logging.debug('  Filtering using QA flag (QA==0)')
        #     landsat_df = landsat_df[landsat_df['QA'] == 0]
        # else:
        #     # If QA flag was not set, apply some basic filtering
        #     logging.debug('  Not filtering by QA flag')
        #     landsat_df = landsat_df[landsat_df['PIXEL_COUNT'] > 0]
        #     landsat_df = landsat_df[landsat_df['CLOUD_SCORE'] < 100]
        #     landsat_df = landsat_df[landsat_df['TS'] > 260]

        # Map QA values to RGB colors
        colors = [
            "#%02x%02x%02x" % (int(r), int(g), int(b))
            for r, g, b, _ in 255 * cm.viridis(
                mpl.colors.Normalize()(landsat_df['QA'].values))
        ]

        #
        source = ColumnDataSource(data={
            'DOY': landsat_df['DOY'].values,
            'NDVI_TOA': landsat_df['NDVI_TOA'].values,
            'ALBEDO_SUR': landsat_df['ALBEDO_SUR'].values,
            'TS': landsat_df['TS'].values,
            'NDWI_SUR': landsat_df['NDWI_GREEN_SWIR1_SUR'].values,
            'COLOR': colors
        })


        # # show the tooltip
        # hover = HoverTool(tooltips=[("DOY", "$DOY")])
        # hover.mode = 'mouse'

        figure_args = dict(
            plot_width=750, plot_height=250, title=None,
            tools="xwheel_zoom,xpan,xbox_zoom,reset,box_select",
            active_scroll="xwheel_zoom")
        plot_args = dict(size=3, alpha=0.9, color='COLOR')
        x_limit = (1, 366)

        selected_circle = Circle(
            fill_alpha=1, fill_color='COLOR', line_color='COLOR')
        nonselected_circle = Circle(
            fill_alpha=0.8, fill_color='#aaaaaa', line_color='#aaaaaa')

        p1 = figure(
            x_range=Range1d(x_limit[0], x_limit[1], bounds=x_limit),
            y_axis_label='NDVI_TOA', **figure_args)
        r1 = p1.circle('DOY', 'NDVI_TOA', source=source, **plot_args)
        r1.selection_glyph = selected_circle
        r1.nonselection_glyph = nonselected_circle
        p1.add_tools(HoverTool(tooltips=[("DOY", "@DOY")]))

        p2 = figure(
            x_range=p1.x_range, y_axis_label='ALBEDO_SUR', **figure_args)
        r2 = p2.circle('DOY', 'ALBEDO_SUR', source=source, **plot_args)
        r2.selection_glyph = selected_circle
        r2.nonselection_glyph = nonselected_circle
        p2.add_tools(HoverTool(tooltips=[("DOY", "@DOY")]))

        p3 = figure(
            x_range=p1.x_range, y_axis_label='TS', **figure_args)
        r3 = p3.circle('DOY', 'TS', source=source, **plot_args)
        r3.selection_glyph = selected_circle
        r3.nonselection_glyph = nonselected_circle
        p3.add_tools(HoverTool(tooltips=[("DOY", "@DOY")]))

        p4 = figure(
            x_range=p1.x_range, y_axis_label='TS', x_axis_label='DOY',
            **figure_args)
        r4 = p4.circle('DOY', 'TS', source=source, **plot_args)
        r4.selection_glyph = selected_circle
        r4.nonselection_glyph = nonselected_circle
        p4.add_tools(HoverTool(tooltips=[("DOY", "@DOY")]))

        # slider = Slider(start=1, end=365, value=1, step=1, title="Slider")

        p = column(p1, p2, p3, p4)
        # taptool = p.select(type=TapTool)

        show(p)

        output_path = os.path.join(
            ini['SUMMARY']['output_ws'], '{}.html'.format(zone_name))
        f = output_file(output_path, title=zone_name)
        save(p)

        break


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate Bokeh figures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=lambda x: python_common.valid_file(x),
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
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

    main(ini_path=args.ini, overwrite_flag=args.overwrite)
