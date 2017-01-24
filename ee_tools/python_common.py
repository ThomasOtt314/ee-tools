#--------------------------------
# Name:         python_common.py
# Purpose:      Common Python support functions
# Created       2017-01-22
# Python:       2.7
#--------------------------------

import os
import json
import logging

from osgeo import ogr


def month_range(start, end):
    m = int(start)
    while True:
        yield(m)
        if m == end:
            break
        m += 1
        if m > 12:
            m = 1


def get_ini_path(workspace):
    import Tkinter, tkFileDialog
    root = Tkinter.Tk()
    ini_path = tkFileDialog.askopenfilename(
        initialdir=workspace, parent=root, filetypes=[('INI files', '.ini')],
        title='Select the target INI file')
    root.destroy()
    return ini_path


def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg


def parse_int_set(nputstr=""):
    """Return list of numbers given a string of ranges

    Originally in python_common.py

    http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-python.html
    """
    selection = set()
    invalid = set()
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    for x in range(first, last + 1):
                        selection.add(x)
            except:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    # print "Invalid set: " + str(invalid)
    return selection


def wrapped_range(a, b, x_min=1, x_max=12):
    """Return the values between a range b for a given start/end

    Originally in python_common.py

    >>> wrapped_range(1, 12, 1, 12))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    >>> wrapped_range(None, None, 1, 12))
    []
    >>> wrapped_range(None, 12, 1, 12))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    >>> wrapped_range(1, None, 1, 12))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    >>> wrapped_range(10, 9, 1, 12))
    [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> wrapped_range(3, 5, 1, 12))
    [3, 4, 5]
    >>> wrapped_range(10, 1, 1, 12))
    [10, 11, 12, 1]

    """
    output = []
    if a is None and b is None:
        return output
    if a is None:
        a = x_min
    if b is None:
        b = x_max
    x = a
    while True:
        output.append(x)
        if x == b:
            break
        x += 1
        if x > x_max:
            x = x_min
    return output


def shapefile_2_geom_list_func(input_path, zone_field=None,
                               reverse_flag=True, simplify_flag=False):
    """Return a list of feature geometries in the shapefile

    Also return the FID and value in zone_field
    FID value will be returned if zone_field is not set or does not exist

    """
    ftr_geom_list = []
    input_ds = ogr.Open(input_path)
    input_lyr = input_ds.GetLayer()
    input_ftr_defn = input_lyr.GetLayerDefn()
    # input_proj = input_lyr.GetSpatialRef().ExportToWkt()
    if zone_field in feature_lyr_fields(input_lyr):
        zone_field_i = input_ftr_defn.GetFieldIndex(zone_field)
    elif zone_field.upper() == 'FID':
        zone_field = None
        logging.info('Using FID as zone field')
    else:
        zone_field = None
        logging.warning('The zone field entered was not found, using FID')
    input_ftr = input_lyr.GetNextFeature()
    while input_ftr:
        input_fid = input_ftr.GetFID()
        if zone_field is not None:
            input_zone = str(input_ftr.GetField(zone_field_i))
        else:
            input_zone = str(input_fid)
        input_geom = input_ftr.GetGeometryRef()
        if simplify_flag:
            input_geom = input_geom.Simplify(1)
            reverse_flag = False
        # Convert feature to GeoJSON
        json_str = input_ftr.ExportToJson()
        json_obj = json.loads(json_str)
        # Reverse the point order from counter-clockwise to clockwise
        if reverse_flag and input_geom.GetGeometryName() == 'MULTIPOLYGON':
            for i in range(len(json_obj['geometry']['coordinates'])):
                for j in range(len(json_obj['geometry']['coordinates'][i])):
                    json_obj['geometry']['coordinates'][i][j] = list(reversed(
                        json_obj['geometry']['coordinates'][i][j]))
        elif reverse_flag and input_geom.GetGeometryName() == 'POLYGON':
            for i in range(len(json_obj['geometry']['coordinates'])):
                json_obj['geometry']['coordinates'][i] = list(reversed(
                    json_obj['geometry']['coordinates'][i]))
        # Save the JSON object in the list
        ftr_geom_list.append([input_fid, input_zone, json_obj['geometry']])
        input_geom = None
        input_ftr = input_lyr.GetNextFeature()
    input_ds = None
    return ftr_geom_list


def feature_path_fields(feature_path):
    """"""
    feature_ds = ogr.Open(feature_path)
    field_list = feature_ds_fields(feature_ds)
    feature_ds = None
    return field_list


def feature_ds_fields(feature_ds):
    """"""
    feature_lyr = feature_ds.GetLayer()
    return feature_lyr_fields(feature_lyr)


def feature_lyr_fields(feature_lyr):
    """"""
    feature_lyr_defn = feature_lyr.GetLayerDefn()
    return [
        feature_lyr_defn.GetFieldDefn(i).GetNameRef()
        for i in range(feature_lyr_defn.GetFieldCount())]
