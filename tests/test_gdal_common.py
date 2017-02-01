import os

import numpy as np
from osgeo import gdal, osr
import pytest

import ee_tools.gdal_common as gdc


# Test Geo Arrays
geo_array_params = [
    {
        'extent': [2000, 1000, 2100, 1080], 'cellsize': 10, 'epsg': 32611,
        'dtype': np.uint8, 'v_min': 0, 'v_max': 100, 'nodata': 255
    }
    # {
    #     'extent': [2000, 1000, 2100, 1080], 'cellsize': 10, 'epsg': 3311,
    #     'dtype': np.float32, 'v_min': 0, 'v_max': 100, 'nodata': 255
    # }
]


class GeoArray:
    """"""
    def __init__(self, extent, cellsize, epsg, dtype, v_min, v_max, nodata):
        # Intentionally not making extent an extent object
        self.extent = extent
        self.cellsize = cellsize

        # Is it bad to have these being built the same as in gdal_common?
        # These could be computed using the Extent methods instead
        self.geo = (
            self.extent[0], self.cellsize, 0.,
            self.extent[3], 0., -self.cellsize)
        self.transform = (
            self.cellsize, 0, self.extent[0],
            0, -self.cellsize, self.extent[3])
        self.cols = int(round(abs(
            (self.extent[0] - self.extent[2]) / self.cellsize), 0))
        self.rows = int(round(abs(
            (self.extent[3] - self.extent[1]) / -self.cellsize), 0))
        self.shape = (self.rows, self.cols)
        self.origin = (self.extent[0], self.extent[3])
        self.center = (
            self.extent[0] + 0.5 * abs(self.extent[2] - self.extent[0]),
            self.extent[1] + 0.5 * abs(self.extent[3] - self.extent[1]))

        # Spatial Reference
        self.epsg = epsg
        self.osr = osr.SpatialReference()
        self.osr.ImportFromEPSG(self.epsg)
        self.proj4 = self.osr.ExportToProj4()
        self.wkt = self.osr.ExportToWkt()
        # self.osr = gdc.epsg_osr(epsg)
        # self.proj4 = gdc.osr_proj4(self.osr)
        # self.wkt = gdc.osr_wkt(self.osr)

        # Array/Raster Type
        # self.gtype = gtype
        self.dtype = dtype
        self.gtype = gdc.numpy_to_gdal_type(dtype)

        # Array
        self.array = np.random.uniform(low=v_min, high=v_max, size=self.shape)\
            .astype(self.dtype)
        self.nodata = nodata


@pytest.fixture(scope='module', params=geo_array_params)
def geo_array(request):
    return GeoArray(**request.param)


@pytest.fixture(scope='module')
def raster_ds(geo_array):
    """Build an in memory raster dataset"""
    print('Building in-memory raster dataset')

    driver = gdal.GetDriverByName('MEM')
    # driver = gdal.GetDriverByName('HFA')
    output_ds = driver.Create(
        '', geo_array.cols, geo_array.rows, 1, geo_array.gtype)
    output_ds.SetProjection(geo_array.wkt)
    output_ds.SetGeoTransform(geo_array.geo)
    output_band = output_ds.GetRasterBand(1)
    output_band.SetNoDataValue(geo_array.nodata)
    output_band.WriteArray(geo_array.array, 0, 0)
    yield output_ds

    print('Closing in-memory raster datasets')
    output_ds = None


def test_osr(epsg=32611):
    """Test building OSR from EPSG code to test GDAL_DATA environment variable

    This should fail if GDAL_DATA is not set or is invalid.
    """
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg)
    wkt = sr.ExportToWkt()
    print('GDAL_DATA = {}'.format(os.environ['GDAL_DATA']))
    print('EPSG: {}'.format(epsg))
    print('WKT: {}'.format(wkt))
    assert wkt


def test_extent_properties(geo_array):
    extent = geo_array.extent
    extent_obj = gdc.Extent(extent)
    assert extent_obj.xmin == extent[0]
    assert extent_obj.ymin == extent[1]
    assert extent_obj.xmax == extent[2]
    assert extent_obj.ymax == extent[3]


def test_extent_rounding(extent=[0.111111, 0.111111, 0.888888, 0.888888],
                         ndigits=3, expected=[0.111, 0.111, 0.889, 0.889]):
    assert list(gdc.Extent(extent, ndigits=ndigits)) == expected


def test_extent_str(geo_array):
    extent = geo_array.extent
    expected = ' '.join(['{:.1f}'.format(x) for x in extent])
    assert str(gdc.Extent(extent)) == expected


def test_extent_list(geo_array):
    extent = geo_array.extent
    assert list(gdc.Extent(extent)) == extent


def test_extent_adjust_to_snap_round(geo_array):
    extent_mod = geo_array.extent[:]
    # Adjust test extent out to the rounding limits
    extent_mod[0] = extent_mod[0] + 0.499 * geo_array.cellsize
    extent_mod[1] = extent_mod[1] - 0.5 * geo_array.cellsize
    extent_mod[2] = extent_mod[2] - 0.5 * geo_array.cellsize
    extent_mod[3] = extent_mod[3] + 0.499 * geo_array.cellsize
    extent_mod = list(gdc.Extent(extent_mod).adjust_to_snap(
        'ROUND', 0, 0, geo_array.cellsize))
    assert extent_mod == geo_array.extent


def test_extent_adjust_to_snap_expand(geo_array):
    extent_mod = geo_array.extent[:]
    # Shrink the test extent in almost a full cellsize
    extent_mod[0] = extent_mod[0] + 0.99 * geo_array.cellsize
    extent_mod[1] = extent_mod[1] + 0.5 * geo_array.cellsize
    extent_mod[2] = extent_mod[2] - 0.5 * geo_array.cellsize
    extent_mod[3] = extent_mod[3] - 0.99 * geo_array.cellsize
    extent_mod = list(gdc.Extent(extent_mod).adjust_to_snap(
        'EXPAND', 0, 0, geo_array.cellsize))
    assert extent_mod == geo_array.extent


def test_extent_adjust_to_snap_shrink(geo_array):
    extent_mod = geo_array.extent[:]
    # Expand the test extent out almost a full cellsize
    extent_mod[0] = extent_mod[0] - 0.99 * geo_array.cellsize
    extent_mod[1] = extent_mod[1] - 0.5 * geo_array.cellsize
    extent_mod[2] = extent_mod[2] + 0.5 * geo_array.cellsize
    extent_mod[3] = extent_mod[3] + 0.99 * geo_array.cellsize
    extent_mod = list(gdc.Extent(extent_mod).adjust_to_snap(
        'SHRINK', 0, 0, geo_array.cellsize))
    assert extent_mod == geo_array.extent


# @pytest.mark.parametrize(
#     "geo_array,distance",
#     [
#         [geo_array, 10],
#         [geo_array, -10]
#     ]
# )
# def test_extent_buffer(geo_array, distance):
#     expected = geo_array.extent[:]
#     expected[0] = expected[0] - distance
#     expected[1] = expected[1] - distance
#     expected[2] = expected[2] + distance
#     expected[3] = expected[3] + distance
#     assert list(gdc.Extent(geo_array.extent).buffer(distance)) == expected


# def test_extent_split(extent=test_extent):
#     """List of extent terms (xmin, ymin, xmax, ymax)"""
#     assert gdc.Extent(extent).split() == extent


def test_extent_copy(geo_array):
    """Return a copy of the extent"""
    orig_extent = gdc.Extent(geo_array.extent)
    copy_extent = orig_extent.copy()
    # Modify the original extent
    orig_extent = orig_extent.buffer(10)
    # Check that the copy hasn't changed
    assert list(copy_extent) == geo_array.extent


def test_extent_corner_points(geo_array):
    """Corner points in clockwise order starting with upper-left point"""
    expected = [
        (geo_array.extent[0], geo_array.extent[3]),
        (geo_array.extent[2], geo_array.extent[3]),
        (geo_array.extent[2], geo_array.extent[1]),
        (geo_array.extent[0], geo_array.extent[1])]
    assert gdc.Extent(geo_array.extent).corner_points() == expected


def test_extent_ul_lr_swap(geo_array):
    """Copy of extent object reordered as xmin, ymax, xmax, ymin

    Some gdal utilities want the extent described using upper-left and
    lower-right points.
        gdal_translate -projwin ulx uly lrx lry
        gdal_merge -ul_lr ulx uly lrx lry

    """
    expected = [
        geo_array.extent[0], geo_array.extent[3],
        geo_array.extent[2], geo_array.extent[1]]
    assert list(gdc.Extent(geo_array.extent).ul_lr_swap()) == expected


def test_extent_ogrenv_swap(geo_array):
    """Copy of extent object reordered as xmin, xmax, ymin, ymax

    OGR feature (shapefile) extents are different than GDAL raster extents
    """
    expected = [
        geo_array.extent[0], geo_array.extent[2],
        geo_array.extent[1], geo_array.extent[3]]
    assert list(gdc.Extent(geo_array.extent).ogrenv_swap()) == expected


def test_extent_origin(geo_array):
    """Origin (upper-left corner) of the extent"""
    assert gdc.Extent(geo_array.extent).origin() == geo_array.origin


def test_extent_center(geo_array):
    """Centroid of the extent"""
    assert gdc.Extent(geo_array.extent).center() == geo_array.center


def test_extent_shape(geo_array):
    """Return number of rows and columns of the extent

    Args:
        cs (int): cellsize
    Returns:
        tuple of raster rows and columns
    """
    extent = gdc.Extent(geo_array.extent)
    assert extent.shape(cs=geo_array.cellsize) == geo_array.shape


def test_extent_geo(geo_array):
    """Geo-tranform of the extent"""
    extent = gdc.Extent(geo_array.extent)
    assert extent.geo(cs=geo_array.cellsize) == geo_array.geo


def test_extent_geometry(geo_array):
    """Check GDAL geometry by checking if WKT matches"""
    extent_wkt = gdc.Extent(geo_array.extent).geometry().ExportToWkt()
    expected = [
        "{} {} 0".format(int(x), int(y))
        for x, y in gdc.Extent(geo_array.extent).corner_points()]
    # First point is repeated in geometry
    expected = "POLYGON (({}))".format(','.join(expected + [expected[0]]))
    assert extent_wkt == expected


def test_extent_intersect_point(geo_array):
    """"Test if Point XY intersects the extent"""
    extent = gdc.Extent(geo_array.extent)
    origin = geo_array.origin
    cs = geo_array.cellsize
    assert not extent.intersect_point([origin[0] - cs, origin[1] + cs])
    assert extent.intersect_point([origin[0], origin[1]])
    assert extent.intersect_point([origin[0] + cs, origin[1] - cs])
    # assert extent.intersect_point(xy) == expected
    # assert extent.intersect_point(xy) == expected


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
#     """Return the GDAL raster data type based on the NumPy array data type"""
#     assert False


# def test_numpy_type_nodata(numpy_type):
#     """Return the default nodata value based on the NumPy array data type"""
#     assert False


# def test_gdal_to_numpy_type(gdal_type):
#     """Return the NumPy array data type based on a GDAL type"""
#     assert False


# def test_matching_spatref(osr_a, osr_b):
#     """Test if two spatial reference objects match"""
#     assert False


# # def test_matching_spatref(osr_a, osr_b):
# #     """Test if two spatial reference objects match"""
# #     assert False


# def test_osr_proj(input_osr):
#     """Return the projection WKT of a spatial reference object"""
#     assert False


# def test_epsg_osr(input_epsg):
#     """Return the spatial reference object of an EPSG code"""
#     assert False


# def test_proj4_osr(input_proj4):
#     """Return the spatial reference object of an PROJ4 code"""
#     assert False


# def test_feature_path_osr(feature_path):
#     """Return the spatial reference of a feature path"""
#     assert False


# def test_feature_ds_osr(feature_ds):
#     """Return the spatial reference of an opened feature dataset"""
#     assert False


# def test_feature_lyr_osr(feature_lyr):
#     """Return the spatial reference of a feature layer"""
#     assert False


# def test_feature_lyr_extent(feature_lyr):
#     """Return the extent of an opened feature layer"""
#     assert False


def test_raster_ds_geo(raster_ds, geo_array):
    """Return the geo-transform of an opened raster dataset"""
    assert gdc.raster_ds_geo(raster_ds) == geo_array.geo


# def test_round_geo(geo, n=10):
#     """Round the values of a geotransform to n digits"""
#     assert False


def test_raster_ds_extent(raster_ds, geo_array):
    """Return the extent of an opened raster dataset"""
    assert list(gdc.raster_ds_extent(raster_ds)) == geo_array.extent


def test_geo_cellsize(geo_array):
    """Return pixel width & pixel height of a geo-transform"""
    cellsize = gdc.geo_cellsize(geo_array.geo, x_only=False)
    assert cellsize == (geo_array.cellsize, -geo_array.cellsize)


def test_geo_cellsize_x_only(geo_array):
    """Return pixel width of a geo-transform"""
    assert gdc.geo_cellsize(geo_array.geo, x_only=True) == geo_array.cellsize


def test_geo_origin(geo_array):
    """Return upper-left corner of geo-transform"""
    assert gdc.geo_origin(geo_array.geo) == geo_array.origin


def test_geo_extent(geo_array):
    """Return the extent from a geo-transform and array shape (rows, cols)"""
    extent = list(gdc.geo_extent(
        geo_array.geo, geo_array.rows, geo_array.cols))
    assert extent == geo_array.extent


# def test_raster_path_shape(raster_path):
#     """Return the number of rows and columns in a raster
#     """
#     assert False


def test_raster_ds_shape(raster_ds, geo_array):
    """Return the number of rows and columns in an opened raster dataset"""
    assert gdc.raster_ds_shape(raster_ds) == geo_array.shape


# def test_raster_path_set_nodata(raster_path, input_nodata):
#     """Set raster nodata value for all bands"""
#     assert False


# def test_raster_ds_set_nodata(raster_ds, input_nodata):
#     """Set raster dataset nodata value for all bands"""
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


# def test_array_offset_geo(test_geo, x_offset=test_cs, y_offset=test_cs,
#                           expected):
#     """Return sub_geo that is offset from full_geo"""
#     assert gdc.array_offset_geo(test_geo, 15, )


# def test_array_geo_offsets(full_geo, sub_geo, cs):
#     """Return x/y offset of a gdal.geotransform based on another gdal.geotransform"""
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


def test_raster_ds_to_array(raster_ds, geo_array):
    """Test reading NumPy array from raster

    Output array size will match the mask_extent if mask_extent is set

    Args:
        input_raster_ds (): opened raster dataset as gdal raster
        band (int): band number to read the array from
        mask_extent (): subset extent of the raster if desired
        fill_value (float): Value to Initialize empty array with
        return_nodata (bool): If True, returns no data value with the array
    """
    assert np.array_equal(
        gdc.raster_ds_to_array(raster_ds, return_nodata=False),
        geo_array.array)


def test_raster_ds_to_array_with_nodata(raster_ds, geo_array):
    """Test reading raster array and nodata value"""
    raster_array, raster_nodata = gdc.raster_ds_to_array(
        raster_ds, return_nodata=True)
    assert np.array_equal(raster_array, geo_array.array)
    assert raster_nodata == geo_array.nodata


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


# def test_shapefile_2_geom_list_func(input_path, zone_field=None,
#                                reverse_flag=False, simplify_flag=False):
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


def test_json_reverse_polygon(geo_array):
    """Reverse the point order from counter-clockwise to clockwise"""
    pts = map(list, gdc.Extent(geo_array.extent).corner_points())
    json_geom = {'type': 'Polygon', 'coordinates': [pts]}
    expected = {'type': 'Polygon', 'coordinates': [pts[::-1]]}
    assert gdc.json_reverse_func(json_geom) == expected


def test_json_reverse_multipolygon(geo_array):
    """Reverse the point order from counter-clockwise to clockwise"""
    pts = map(list, gdc.Extent(geo_array.extent).corner_points())
    json_geom = {'type': 'MultiPolygon', 'coordinates': [[pts]]}
    expected = {'type': 'MultiPolygon', 'coordinates': [[pts[::-1]]]}
    assert gdc.json_reverse_func(json_geom) == expected


def test_json_strip_z_polygon(geo_array):
    """Strip Z value from coordinates"""
    pts = map(list, gdc.Extent(geo_array.extent).corner_points())
    json_geom = {'type': 'Polygon', 'coordinates': [[p + [1.0] for p in pts]]}
    expected = {'type': 'Polygon', 'coordinates': [pts]}
    assert gdc.json_strip_z_func(json_geom) == expected


def test_json_strip_z_multipolygon(geo_array):
    """Strip Z value from coordinates"""
    pts = map(list, gdc.Extent(geo_array.extent).corner_points())
    json_geom = {
        'type': 'MultiPolygon',
        'coordinates': [[[p + [1.0] for p in pts]]]}
    expected = {'type': 'MultiPolygon', 'coordinates': [[pts]]}
    assert gdc.json_strip_z_func(json_geom) == expected


def test_geo_2_ee_transform(geo_array):
    """ EE crs transforms are different than GDAL geo transforms

    EE: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]
    GDAL: [xTranslation, xScale, xShearing, yTranslation, yShearing, yScale]
    """
    assert gdc.geo_2_ee_transform(geo_array.geo) == geo_array.transform


def test_ee_transform_2_geo(geo_array):
    """ EE crs transforms are different than GDAL geo transforms

    EE: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]
    GDAL: [xTranslation, xScale, xShearing, yTranslation, yShearing, yScale]
    """
    assert gdc.ee_transform_2_geo(geo_array.transform) == geo_array.geo
