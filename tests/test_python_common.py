# import os
# import json
# import logging

# from osgeo import ogr
import pytest

import ee_tools.python_common as python_common


# def test_month_range(start, end):
#     assert False


# def test_get_ini_path(workspace):
#     import Tkinter, tkFileDialog
#     assert False


# def test_is_valid_file(parser, arg):
#     assert False


# def test_parse_int_set(nputstr=""):
#     """Return list of numbers given a string of ranges

#     Originally in python_common.py

#     http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-python.html
#     """
#     assert False


# def test_remove_file(file_path):
#     """Remove a feature/raster and all of its anciallary files"""
#     assert False


@pytest.mark.parametrize(
    "a,b,x_min,x_max,expected",
    [
        [1, 12, 1, 12, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        [None, None, 1, 12, []],
        [None, 12, 1, 12, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        [1, None, 1, 12, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        [10, 9, 1, 12, [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        [3, 5, 1, 12, [3, 4, 5]],
        [10, 1, 1, 12, [10, 11, 12, 1]]
    ]
)
def test_wrapped_range(a, b, x_min, x_max, expected):
    """Return the values between a range b for a given start/end"""
    assert python_common.wrapped_range(a, b, x_min, x_max) == expected


# def test_shapefile_2_geom_list_func(input_path, zone_field=None,
#                                reverse_flag=True, simplify_flag=False):
#     """Return a list of feature geometries in the shapefile

#     Also return the FID and value in zone_field
#     FID value will be returned if zone_field is not set or does not exist

#     """
#     assert False


# def test_feature_path_fields(feature_path):
#     """"""
#     assert False


# def test_feature_ds_fields(feature_ds):
#     """"""
#     assert False


# def test_feature_lyr_fields(feature_lyr):
#     """"""
#     assert False
