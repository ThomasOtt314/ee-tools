# from collections import OrderedDict
# import copy
# import csv
# import itertools
# import logging
# import math
# import os
# import random
# import sys
# import warnings

# import numpy as np
# from osgeo import gdal, ogr, osr
import pytest

import ee_tools.gdal_common as gdc

# gdal.UseExceptions()


def test_extent_properties(extent=[0, 10, 100, 90], expected=[0, 10, 100, 90]):
    extent = gdc.Extent(extent)
    assert extent.xmin == expected[0]
    assert extent.ymin == expected[1]
    assert extent.xmax == expected[2]
    assert extent.ymax == expected[3]


def test_extent_rounding(extent=[0.111111, 0.111111, 0.888888, 0.888888],
                         ndigits=3, expected=[0.111, 0.111, 0.889, 0.889]):
    extent = gdc.Extent(extent, ndigits=ndigits)
    assert list(extent) == expected


def test_extent_str(extent=[0, 10, 100, 90], expected='0.0 10.0 100.0 90.0'):
    extent = gdc.Extent(extent)
    assert str(extent) == expected


def test_extent_list(extent=[0, 10, 100, 90], expected=[0, 10, 100, 90]):
    extent = gdc.Extent(extent)
    assert list(extent) == expected


# Add checks for other snap points, cellsizes, and negative extents
@pytest.mark.parametrize(
    "extent,method,snap_x,snap_y,cs,expected",
    [
        [[5, 14, 95, 86], 'ROUND', 0, 0, 10, [10, 10, 100, 90]],
        [[5, 15, 95, 85], 'EXPAND', 0, 0, 10, [0, 10, 100, 90]],
        [[5, 15, 95, 85], 'SHRINK', 0, 0, 10, [10, 20, 90, 80]]
    ]
)
def test_extent_adjust_to_snap(extent, method, snap_x, snap_y, cs, expected):
    extent = gdc.Extent(extent)
    # Extent is modified in place
    extent.adjust_to_snap(method, snap_x, snap_y, cs)
    assert list(extent) == expected


@pytest.mark.parametrize(
    "extent,distance,expected",
    [
        [[0, 10, 100, 90], 10, [-10, 0, 110, 100]],
        [[0, 10, 100, 90], -10, [10, 20, 90, 80]]
    ]
)
def test_extent_buffer_extent(extent, distance, expected):
    extent = gdc.Extent(extent)
    # Extent is modified in place
    extent.buffer_extent(distance)
    assert list(extent) == expected

# def test_extent_split_extent(self):
#     """List of extent terms (xmin, ymin, xmax, ymax)"""
#     assert False

# def test_extent_copy(self):
#     """Return a copy of the extent"""
#     assert False


def test_extent_corner_points(extent=[0, 10, 100, 90],
                              expected=[(0, 90), (100, 90), (100, 10), (0, 10)]):
    """Corner points in clockwise order starting with upper-left point"""
    extent = gdc.Extent(extent)
    assert extent.corner_points() == expected


# def test_extent_ul_lr_swap(self):
#     """Copy of extent object reordered as xmin, ymax, xmax, ymin

#     Some gdal utilities want the extent described using upper-left and
#     lower-right points.
#         gdal_translate -projwin ulx uly lrx lry
#         gdal_merge -ul_lr ulx uly lrx lry

#     """
#     assert False


def test_extent_ogrenv_swap(extent=[0, 10, 100, 90], expected=[0, 100, 10, 90]):
    """Copy of extent object reordered as xmin, xmax, ymin, ymax

    OGR feature (shapefile) extents are different than GDAL raster extents
    """
    extent = gdc.Extent(extent)
    assert list(extent.ogrenv_swap()) == expected


# def test_extent_origin(self):
#     """Origin (upper-left corner) of the extent"""
#     assert False

# def test_extent_center(self):
#     """Centroid of the extent"""
#     assert False


def test_extent_shape(extent=[0, 10, 100, 90], cs=10, expected=(8, 10)):
    """Return number of rows and columns of the extent

    Args:
        cs (int): cellsize
    Returns:
        tuple of raster rows and columns
    """
    extent = gdc.Extent(extent)
    assert extent.shape(cs=cs) == expected


def test_extent_geo(extent=[0, 10, 100, 90], cs=10,
                    expected=(0, 10, 0, 90, 0, -10)):
    """Geo-tranform of the extent"""
    extent = gdc.Extent(extent)
    assert extent.geo(cs=cs) == expected


# def test_extent_geometry(self):
#     """GDAL geometry object of the extent"""
#     assert False

# def test_extent_intersect_point(self, xy):
#     """"Test if Point XY intersects the extent"""
#     assert False





# def test_raster_driver(raster_path):
#     """Return the GDAL driver from a raster path

#     Currently supports ERDAS Imagine format, GeoTiff,
#     HDF-EOS (HDF4), BSQ/BIL/BIP, and memory drivers.

#     Args:
#         raster_path (str): filepath to a raster

#     Returns:
#         GDAL driver: GDAL raster driver

#     """
#     assert False


# def test_numpy_to_gdal_type(numpy_type):
#     """Return the GDAL raster data type based on the NumPy array data type

#     The following built in functions do roughly the same thing
#         NumericTypeCodeToGDALTypeCode
#         GDALTypeCodeToNumericTypeCode

#     Args:
#         numpy_type (:class:`np.dtype`): NumPy array type
#             (i.e. np.bool, np.float32, etc)

#     Returns:
#         g_type: GDAL `datatype <http://www.gdal.org/gdal_8h.html#a22e22ce0a55036a96f652765793fb7a4/>`
#         _equivalent to the input NumPy :class:`np.dtype`

#     """
#     assert False


# def test_numpy_type_nodata(numpy_type):
#     """Return the default nodata value based on the NumPy array data type

#     Args:
#         numpy_type (:class:`np.dtype`): numpy array type
#             (i.e. np.bool, np.float32, etc)

#     Returns:
#         nodata_value: Nodata value for GDAL which defaults to the
#             minimum value for the number type

#     """
#     assert False


# def test_gdal_to_numpy_type(gdal_type):
#     """Return the NumPy array data type based on a GDAL type

#     Args:
#         gdal_type (:class:`gdal.type`): GDAL data type

#     Returns:
#         numpy_type: NumPy datatype (:class:`np.dtype`)

#     """
#     assert False


# def test_matching_spatref(osr_a, osr_b):
#     """Test if two spatial reference objects match

#     Compare common components of PROJ4 strings

#     Args:
#         osr_a: OSR spatial reference object
#         osr_b: OSR spatial reference object

#     Returns:
#         Bool: True if OSR objects match. Otherwise, False.

#     """
#     assert False


# # def test_matching_spatref(osr_a, osr_b):
# #     """Test if two spatial reference objects match

# #     Args:
# #         osr_a: OSR spatial reference object
# #         osr_b: OSR spatial reference object

# #     Returns:
# #         Bool: True if OSR objects match. Otherwise, False.

# #     """
# #     assert False


# def test_osr_proj(input_osr):
#     """Return the projection WKT of a spatial reference object

#     Args:
#         input_osr (:class:`osr.SpatialReference`): the input OSR
#             spatial reference

#     Returns:
#         WKT: :class:`osr.SpatialReference` in WKT format

#     """
#     assert False


# def test_epsg_osr(input_epsg):
#     """Return the spatial reference object of an EPSG code

#     Args:
#         input_epsg (int): EPSG spatial reference code as integer

#     Returns:
#         osr.SpatialReference: :class:`osr.SpatialReference` object

#     """
#     assert False


# def test_proj4_osr(input_proj4):
#     """Return the spatial reference object of an PROJ4 code

#     Args:
#         input_proj4 (str): PROJ4 projection or coordinate system description

#     Returns:
#         osr.SpatialReference: :class:`osr.SpatialReference` object

#     """
#     assert False


# def test_feature_path_osr(feature_path):
#     """Return the spatial reference of a feature path

#     Args:
#         feature_path (str): file path to the OGR supported feature

#     Returns:
#         osr.SpatialReference: :class:`osr.SpatialReference` of the
#             input feature file path

#     """
#     assert False


# def test_feature_ds_osr(feature_ds):
#     """Return the spatial reference of an opened feature dataset

#     Args:
#         feature_ds (:class:`ogr.Datset`): Opened feature dataset
#             from which you desire the spatial reference

#     Returns:
#         osr.SpatialReference: :class:`osr.SpatialReference` of the input
#             OGR feature dataset

#     """
#     assert False


# def test_feature_lyr_osr(feature_lyr):
#     """Return the spatial reference of a feature layer

#     Args:
#         feature_lyr (:class:`ogr.Layer`): OGR feature layer from
#             which you desire the :class:`osr.SpatialReference`

#     Returns:
#         osr.SpatialReference: the :class:`osr.SpatialReference` object
#             of the input feature layer

#     """
#     assert False


# def test_feature_lyr_extent(feature_lyr):
#     """Return the extent of an opened feature layer

#     Args:
#         feature_lyr (:class:`ogr.Layer`): An OGR feature
#             layer

#     Returns:
#         gdal_common.extent: :class:`gdal_common.extent` of the
#             input feature layer

#     """
#     assert False


# def test_raster_ds_geo(raster_ds):
#     """Return the geo-transform of an opened raster dataset

#     Args:
#         raster_ds (:class:`gdal.Dataset`): An Opened gdal raster dataset

#     Returns:
#         tuple: :class:`gdal.Geotransform` of the input dataset

#     """
#     assert False


# def test_round_geo(geo, n=10):
#     """Round the values of a geotransform to n digits

#     Args:
#         geo (tuple): :class:`gdal.Geotransform` object
#         n (int): number of digits to round the
#             :class:`gdal.Geotransform` to

#     Returns:
#         tuple: :class:`gdal.Geotransform` rounded to n digits

#     """
#     assert False


# def test_raster_ds_extent(raster_ds):
#     """Return the extent of an opened raster dataset

#     Args:
#         raster_ds (:class:`gdal.Dataset`): An opened GDAL raster
#             dataset

#     Returns:
#         tuple: :class:`gdal_common.extent` of the input dataset

#     """
#     assert False


def test_geo_cellsize(geo=(0, 10, 0, 90, 0, -10), x_only=False,
                      expected=(10, -10)):
    """Return pixel width & pixel height of a geo-transform"""
    assert gdc.geo_cellsize(geo, x_only) == expected


def test_geo_cellsize_x_only(geo=(0, 10, 0, 90, 0, -10), x_only=True,
                             expected=10):
    """Return pixel width of a geo-transform"""
    assert gdc.geo_cellsize(geo, x_only) == expected


def test_geo_origin(geo=(0, 10, 0, 90, 0, -10), expected=(0, 90)):
    """Return upper-left corner of geo-transform"""
    assert gdc.geo_origin(geo) == expected


def test_geo_extent(geo=(0, 10, 0, 90, 0, -10), rows=8, cols=10,
                    expected=[0, 10, 100, 90]):
    """Return the extent from a geo-transform and array shape (rows, cols)"""
    assert list(gdc.geo_extent(geo, rows, cols)) == expected


# def test_raster_path_shape(raster_path):
#     """Return the number of rows and columns in a raster

#     Args:
#         raster_path (str): file path of the raster


#     Returns:
#         tuple of raster rows and columns
#     """
#     assert False


# def test_raster_ds_shape(raster_ds):
#     """Return the number of rows and columns in an opened raster dataset

#     Args:
#         raster_ds: opened raster dataset

#     Returns:
#         tuple of raster rows and columns
#     """
#     assert False


@pytest.mark.parametrize(
    "a,b,expected",
    [
        [[0, 0, 20, 20], [10, 10, 30, 30], True],
        [[0, 0, 20, 20], [30, 30, 50, 50], False]
    ]
)
def test_extents_overlap(a, b, expected):
    """Test if two extents overlap"""
    assert gdc.extents_overlap(gdc.Extent(a), gdc.Extent(b)) == expected


@pytest.mark.parametrize(
    "extent_list,expected",
    [
        [[[0, 0, 20, 20], [10, 10, 30, 30]], [0, 0, 30, 30]],
        [[[0, 0, 20, 20], [10, 10, 30, 30], [20, 20, 40, 40]], [0, 0, 40, 40]]
    ]
)
def test_union_extents(extent_list, expected):
    """Return the union of all input extents"""
    extent_list = [gdc.Extent(extent) for extent in extent_list]
    assert list(gdc.union_extents(extent_list)) == expected


@pytest.mark.parametrize(
    "extent_list,expected",
    [
        [[[0, 0, 20, 20], [10, 10, 30, 30]], [10, 10, 20, 20]],
        [[[0, 0, 20, 20], [10, 0, 30, 20], [0, 10, 20, 30]], [10, 10, 20, 20]]
    ]
)
def test_intersect_extents(extent_list, expected):
    """Return the intersection of all input extents"""
    extent_list = [gdc.Extent(extent) for extent in extent_list]
    assert list(gdc.intersect_extents(extent_list)) == expected


# def test_project_extent(input_extent, input_osr, output_osr, cellsize):
#     """Project extent to different spatial reference / coordinate system

#     Args:
#         input_extent (): the input gdal_common.extent to be reprojected
#         input_osr (): OSR spatial reference of the input extent
#         output_osr (): OSR spatial reference of the desired output
#         cellsize (): the cellsize used to calculate the new extent.
#             This cellsize is in the input spatial reference

#     Returns:
#         tuple: :class:`gdal_common.extent` in the desired projection
#     """
#     assert False


# def test_array_offset_geo(full_geo, x_offset, y_offset):
#     """Return sub_geo that is offset from full_geo

#     Args:
#         full_geo (): gdal.geotransform to create the offset geotransform
#         x_offset (): number of cells to move in x direction
#         y_offset (): number of cells to move in y direction

#     Returns:
#         gdal.Geotransform offset by the spefiied number of x/y cells
#     """
#     assert False


# def test_array_geo_offsets(full_geo, sub_geo, cs):
#     """Return x/y offset of a gdal.geotransform based on another gdal.geotransform

#     Args:
#         full_geo (): larger gdal.geotransform from which the offsets should be calculated
#         sub_geo (): smaller form
#         cs (int): cellsize

#     Returns:
#         x_offset: number of cells of the offset in the x direction
#         y_offset: number of cells of the offset in the y direction
#     """
#     assert False


# def test_raster_to_array(input_raster, band=1, mask_extent=None,
#                     fill_value=None, return_nodata=True):
#     """Return a NumPy array from a raster

#     Output array size will match the mask_extent if mask_extent is set

#     Args:
#         input_raster (str): Filepath to the raster for array creation
#         band (int): band to convert to array in the input raster
#         mask_extent: Mask defining desired portion of raster
#         fill_value (float): Value to Initialize empty array with
#         return_nodata (bool): If True, the function will return the no data value

#     Returns:
#         output_array: The array of the raster values
#         output_nodata: No data value of the raster file
#     """
#     assert False


# def test_raster_ds_to_array(input_raster_ds, band=1, mask_extent=None,
#                        fill_value=None, return_nodata=True):
#     """Return a NumPy array from an opened raster dataset

#     Output array size will match the mask_extent if mask_extent is set

#     Args:
#         input_raster_ds (): opened raster dataset as gdal raster
#         band (int): band number to read the array from
#         mask_extent (): subset extent of the raster if desired
#         fill_value (float): Value to Initialize empty array with
#         return_nodata (bool): If True, returns no data value with the array

#     Returns:
#         output_array: The array of the raster values
#         output_nodata: No data value of the raster file
#     """
#     assert False


# def test_project_array(input_array, resampling_type,
#                   input_osr, input_cs, input_extent,
#                   output_osr, output_cs, output_extent,
#                   output_nodata=None):
#     """Project a NumPy array to a new spatial reference

#     This function doesn't correctly handle masked arrays
#     Must pass output_extent & output_cs to get output raster shape
#     There is not enough information with just output_geo and output_cs

#     Args:
#         input_array (array: :class:`numpy.array`):
#         resampling_type ():
#         input_osr (:class:`osr.SpatialReference):
#         input_cs (int):
#         input_extent ():
#         output_osr (:class:`osr.SpatialReference):
#         output_cs (int):
#         output_extent ():
#         output_nodata (float):

#     Returns:
#         array: :class:`numpy.array`
#     """

#     assert False
