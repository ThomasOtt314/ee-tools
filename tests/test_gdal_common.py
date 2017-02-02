import os

import numpy as np
from osgeo import gdal, osr
import pytest

import ee_tools.gdal_common as gdc


# Test parameters
grid_params = [
    {'extent': [2000, 1000, 2100, 1080], 'cellsize': 10, 'epsg': 32611}
    # {'extent': [0.111111, 0.111111, 0.888888, 0.888888], 'cellsize':0.1, 'epsg': 4326}
    # {'extent': [-2000, -1000, -1900, -920], 'cellsize': 10, 'epsg': 3310}
]

# # v_nodata is only used when constructing the test array
# array_params = [
#     {'dtype': np.float32, 'v_min': 240, 'v_max': 255, 'v_nodata': 255}
# ]

# v_nodata is only used when constructing the test array
# If nodata is Default, use the default nodata value based on the dtype
# If nodata is None, don't write the nodata value to the band (TIF images)
raster_params = [
    {
        'filename': '', 'nodata': 'DEFAULT',
        'dtype': np.uint8, 'v_min': 0, 'v_max': 10},
    {
        'filename': '', 'nodata': 'DEFAULT',
        'dtype': np.float32, 'v_min': 0, 'v_max': 10},
    {
        'filename': '', 'nodata': 'DEFAULT',
        'dtype': np.float32, 'v_min': 0, 'v_max': 10, 'v_nodata': 0},
    # Test default_nodata_value = 255
    {
        'filename': '', 'nodata': None,
        'dtype': np.float32, 'v_min': 240, 'v_max': 255},
    # {
    #     'filename': 'test.img', 'nodata': 'DEFAULT',
    #     'dtype': np.float32, 'v_min': 0, 'v_max': 100},
    # {
    #     'filename': 'test.tif', 'nodata': 'DEFAULT',
    #     'dtype': np.float32, 'v_min': 0, 'v_max': 100},
]


class Grid(object):
    """"""
    def __init__(self, extent, cellsize, epsg):
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


# class GeoArray(Grid):
#     """Generate a GeoArray with random values in range"""
#     def __init__(self, grid, dtype, v_min, v_max, v_nodata):
#         # Copy properties from Grid
#         # How do I get these values automatically (or inherit them)?
#         self.extent = grid.extent
#         self.geo = grid.geo
#         self.shape = grid.shape
#         self.cols = grid.cols
#         self.rows = grid.rows
#         self.wkt = grid.wkt

#         # Array/Raster Type
#         self.dtype = dtype
#         # self.gtype = gdc.numpy_to_gdal_type(dtype)
#         # self.gtype = gtype

#         # Build the array as integers then cast to the output dtype
#         self.array = np.random.randint(v_min, v_max+1, size=grid.shape)\
#             .astype(self.dtype)
#         # self.array = np.random.uniform(v_min, v_max, size=grid.shape)\
#         #     .astype(self.dtype)

#         # self.nodata = gdc.numpy_type_nodata(dtype)
#         # self.nodata = nodata
#         if self.dtype in [np.float32, np.float64]:
#             self.array[self.array == v_nodata] = np.nan


class Raster(Grid):
    """RasterDS inherits all of the GeoArray properties"""
    def __init__(self, grid, filename, nodata, dtype, v_min, v_max,
                 v_nodata=None):
        # Copy properties from GeoArray
        # How do I get these values automatically (or inherit them)?
        self.extent = grid.extent
        self.geo = grid.geo
        self.shape = grid.shape
        self.cols = grid.cols
        self.rows = grid.rows
        self.wkt = grid.wkt

        # Set the nodata value using the default for the array type
        # If nodat is None, don't set the band nodata value
        if nodata == 'DEFAULT':
            self.nodata = gdc.numpy_type_nodata(dtype)
        elif nodata is None:
            self.nodata = None
        else:
            self.nodata = nodata

        # Array/Raster Type
        self.dtype = dtype
        # self.gtype = gdc.numpy_to_gdal_type(dtype)
        # self.gtype = gtype

        # Build the array as integers then cast to the output dtype
        self.array = np.random.randint(v_min, v_max + 1, size=self.shape)\
            .astype(self.dtype)
        # self.array = np.random.uniform(v_min, v_max, size=grid.shape)\
        #     .astype(self.dtype)

        # self.nodata = gdc.numpy_type_nodata(dtype)
        # self.nodata = nodata
        if self.dtype in [np.float32, np.float64] and v_nodata is not None:
            self.array[self.array == v_nodata] = np.nan

        self.path = filename
        if self.path:
            self.path = tmpdir.mkdir("rasters").join(self.path)
        driver = gdc.raster_driver(self.path)

        # Create the raster dataset
        self.gtype = gdc.numpy_to_gdal_type(dtype)
        self.ds = driver.Create(
            self.path, self.cols, self.rows, 1, self.gtype)
        self.ds.SetProjection(self.wkt)
        self.ds.SetGeoTransform(self.geo)

        # Write the array to the raster
        band = self.ds.GetRasterBand(1)
        if nodata is not None:
            band.SetNoDataValue(self.nodata)
        band.WriteArray(self.array, 0, 0)


@pytest.fixture(scope='module', params=grid_params)
def grid(request):
    return Grid(**request.param)


# @pytest.fixture(scope='module', params=array_params)
# def geoarray(request, grid):
#     return GeoArray(grid, **request.param)


@pytest.fixture(scope='module', params=raster_params)
def raster(request, grid):
    return Raster(grid, **request.param)


class TestNumpy:
    def test_numpy_to_gdal_type(self):
        """Return the GDAL raster data type based on the NumPy array data type"""
        assert gdc.numpy_to_gdal_type(np.bool) == gdal.GDT_Byte
        assert gdc.numpy_to_gdal_type(np.uint8) == gdal.GDT_Byte
        assert gdc.numpy_to_gdal_type(np.float32) == gdal.GDT_Float32
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.numpy_to_gdal_type(None)

    def test_numpy_type_nodata(self):
        """Return the default nodata value based on the NumPy array data type"""
        assert gdc.numpy_type_nodata(np.bool) == 0
        assert gdc.numpy_type_nodata(np.uint8) == 255
        assert gdc.numpy_type_nodata(np.int8) == -128
        assert gdc.numpy_type_nodata(np.float32) == float(
            np.finfo(np.float32).min)
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.numpy_type_nodata(None)


class TestGDAL:
    def test_raster_driver(self):
        """Return the GDAL driver from a raster path"""
        assert gdc.raster_driver('')
        assert gdc.raster_driver('test.img')
        assert gdc.raster_driver('d:\\test\\test.tif')
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.raster_driver('test.abc')

    def test_gdal_to_numpy_type(self):
        """Return the NumPy array data type based on a GDAL type"""
        assert gdc.gdal_to_numpy_type(gdal.GDT_Byte) == np.uint8
        assert gdc.gdal_to_numpy_type(gdal.GDT_Float32) == np.float32
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.gdal_to_numpy_type(None)


class TestOSR:
    """OSR specific tests"""

    def test_osr(self, epsg=32611):
        """Test building OSR from EPSG code to test GDAL_DATA environment variable

        This should fail if GDAL_DATA is not set or is invalid.
        """
        print('\nTesting GDAL_DATA environment variable')
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(epsg)
        wkt = sr.ExportToWkt()
        print('GDAL_DATA = {}'.format(os.environ['GDAL_DATA']))
        print('EPSG: {}'.format(epsg))
        print('WKT: {}'.format(wkt))
        assert wkt

    # Test the grid spatial reference parameters
    def test_osr_proj(self, grid):
        """Return the projection WKT of a spatial reference object"""
        assert gdc.osr_proj(grid.osr) == grid.wkt

    def test_epsg_osr(self, grid):
        """Return the spatial reference object of an EPSG code"""

        # Check that a bad EPSG code raises an exception
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.epsg_osr(-1)

        # Check that an OSR object is returned
        assert isinstance(
            gdc.epsg_osr(grid.epsg), type(osr.SpatialReference()))

    def test_proj4_osr(self, grid):
        """Return the spatial reference object of a PROj4 string"""
        # Check that a bad PROJ4 string raises an exception
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.proj4_osr('')

        # Check that an OSR object is returned
        assert isinstance(
            gdc.proj4_osr(grid.proj4), type(osr.SpatialReference()))

    # def test_matching_spatref(osr_a, osr_b):
    #     """Test if two spatial reference objects match"""
    #     assert False

    # # def test_matching_spatref(osr_a, osr_b):
    # #     """Test if two spatial reference objects match"""
    # #     assert False


class TestExtent:
    """GDAL Common Extent class specific tests"""

    def test_extent_properties(self, grid):
        extent = grid.extent
        extent_obj = gdc.Extent(extent)
        assert extent_obj.xmin == extent[0]
        assert extent_obj.ymin == extent[1]
        assert extent_obj.xmax == extent[2]
        assert extent_obj.ymax == extent[3]

    def test_extent_rounding(self,
                             extent=[0.111111, 0.111111, 0.888888, 0.888888],
                             ndigits=3, expected=[0.111, 0.111, 0.889, 0.889]):
        assert list(gdc.Extent(extent, ndigits=ndigits)) == expected

    def test_extent_str(self, grid):
        extent = grid.extent
        expected = ' '.join(['{:.1f}'.format(x) for x in extent])
        assert str(gdc.Extent(extent)) == expected

    def test_extent_list(self, grid):
        extent = grid.extent
        assert list(gdc.Extent(extent)) == extent

    def test_extent_adjust_to_snap_round(self, grid):
        extent_mod = grid.extent[:]
        # Adjust test extent out to the rounding limits
        extent_mod[0] = extent_mod[0] + 0.499 * grid.cellsize
        extent_mod[1] = extent_mod[1] - 0.5 * grid.cellsize
        extent_mod[2] = extent_mod[2] - 0.5 * grid.cellsize
        extent_mod[3] = extent_mod[3] + 0.499 * grid.cellsize
        extent_mod = list(gdc.Extent(extent_mod).adjust_to_snap(
            'ROUND', 0, 0, grid.cellsize))
        assert extent_mod == grid.extent

    def test_extent_adjust_to_snap_expand(self, grid):
        extent_mod = grid.extent[:]
        # Shrink the test extent in almost a full cellsize
        extent_mod[0] = extent_mod[0] + 0.99 * grid.cellsize
        extent_mod[1] = extent_mod[1] + 0.5 * grid.cellsize
        extent_mod[2] = extent_mod[2] - 0.5 * grid.cellsize
        extent_mod[3] = extent_mod[3] - 0.99 * grid.cellsize
        extent_mod = list(gdc.Extent(extent_mod).adjust_to_snap(
            'EXPAND', 0, 0, grid.cellsize))
        assert extent_mod == grid.extent

    def test_extent_adjust_to_snap_shrink(self, grid):
        extent_mod = grid.extent[:]
        # Expand the test extent out almost a full cellsize
        extent_mod[0] = extent_mod[0] - 0.99 * grid.cellsize
        extent_mod[1] = extent_mod[1] - 0.5 * grid.cellsize
        extent_mod[2] = extent_mod[2] + 0.5 * grid.cellsize
        extent_mod[3] = extent_mod[3] + 0.99 * grid.cellsize
        extent_mod = list(gdc.Extent(extent_mod).adjust_to_snap(
            'SHRINK', 0, 0, grid.cellsize))
        assert extent_mod == grid.extent

    @pytest.mark.parametrize("distance", [10, -10])
    def test_extent_buffer(self, distance, grid):
        expected = grid.extent[:]
        expected[0] = expected[0] - distance
        expected[1] = expected[1] - distance
        expected[2] = expected[2] + distance
        expected[3] = expected[3] + distance
        assert list(gdc.Extent(grid.extent).buffer(distance)) == expected

    # def test_extent_split(self, grid):
    #     """List of extent terms (xmin, ymin, xmax, ymax)"""
    #     assert gdc.Extent(grid.extent).split() == grid.extent

    def test_extent_copy(self, grid):
        """Return a copy of the extent"""
        orig_extent = gdc.Extent(grid.extent)
        copy_extent = orig_extent.copy()
        # Modify the original extent
        orig_extent = orig_extent.buffer(10)
        # Check that the copy hasn't changed
        assert list(copy_extent) == grid.extent

    def test_extent_corner_points(self, grid):
        """Corner points in clockwise order starting with upper-left point"""
        expected = [
            (grid.extent[0], grid.extent[3]),
            (grid.extent[2], grid.extent[3]),
            (grid.extent[2], grid.extent[1]),
            (grid.extent[0], grid.extent[1])]
        assert gdc.Extent(grid.extent).corner_points() == expected

    def test_extent_ul_lr_swap(self, grid):
        """Copy of extent object reordered as xmin, ymax, xmax, ymin

        Some gdal utilities want the extent described using upper-left and
        lower-right points.
            gdal_translate -projwin ulx uly lrx lry
            gdal_merge -ul_lr ulx uly lrx lry

        """
        expected = [
            grid.extent[0], grid.extent[3],
            grid.extent[2], grid.extent[1]]
        assert list(gdc.Extent(grid.extent).ul_lr_swap()) == expected

    def test_extent_ogrenv_swap(self, grid):
        """Copy of extent object reordered as xmin, xmax, ymin, ymax

        OGR feature (shapefile) extents are different than GDAL raster extents
        """
        expected = [
            grid.extent[0], grid.extent[2],
            grid.extent[1], grid.extent[3]]
        assert list(gdc.Extent(grid.extent).ogrenv_swap()) == expected

    def test_extent_origin(self, grid):
        """Origin (upper-left corner) of the extent"""
        assert gdc.Extent(grid.extent).origin() == grid.origin

    def test_extent_center(self, grid):
        """Centroid of the extent"""
        assert gdc.Extent(grid.extent).center() == grid.center

    def test_extent_shape(self, grid):
        """Return number of rows and columns of the extent

        Args:
            cs (int): cellsize
        Returns:
            tuple of raster rows and columns
        """
        extent = gdc.Extent(grid.extent)
        assert extent.shape(cs=grid.cellsize) == grid.shape

    def test_extent_geo(self, grid):
        """Geo-tranform of the extent"""
        extent = gdc.Extent(grid.extent)
        assert extent.geo(cs=grid.cellsize) == grid.geo

    def test_extent_geometry(self, grid):
        """Check GDAL geometry by checking if WKT matches"""
        extent_wkt = gdc.Extent(grid.extent).geometry().ExportToWkt()
        expected = [
            "{} {} 0".format(int(x), int(y))
            for x, y in gdc.Extent(grid.extent).corner_points()]
        # First point is repeated in geometry
        expected = "POLYGON (({}))".format(','.join(expected + [expected[0]]))
        assert extent_wkt == expected

    def test_extent_intersect_point(self, grid):
        """"Test if Point XY intersects the extent"""
        extent = gdc.Extent(grid.extent)
        origin = grid.origin
        cs = grid.cellsize
        assert not extent.intersect_point([origin[0] - cs, origin[1] + cs])
        assert extent.intersect_point([origin[0], origin[1]])
        assert extent.intersect_point([origin[0] + cs, origin[1] - cs])
        # assert extent.intersect_point(xy) == expected
        # assert extent.intersect_point(xy) == expected

    # Other extent related functions/tests
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            [[0, 0, 20, 20], [10, 10, 30, 30], True],
            [[0, 0, 20, 20], [30, 30, 50, 50], False]
        ]
    )
    def test_extents_overlap(self, a, b, expected):
        """Test if two extents overlap"""
        assert gdc.extents_overlap(gdc.Extent(a), gdc.Extent(b)) == expected

    @pytest.mark.parametrize(
        "extent_list,expected",
        [
            [[[0, 0, 20, 20], [10, 10, 30, 30]], [0, 0, 30, 30]],
            [[[0, 0, 20, 20], [10, 10, 30, 30], [20, 20, 40, 40]], [0, 0, 40, 40]]
        ]
    )
    def test_union_extents(self, extent_list, expected):
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
    def test_intersect_extents(self, extent_list, expected):
        """Return the intersection of all input extents"""
        extent_list = [gdc.Extent(extent) for extent in extent_list]
        assert list(gdc.intersect_extents(extent_list)) == expected

    # def test_project_extent(self, input_extent, input_osr, output_osr, cellsize):
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


class TestGeo:
    """GeoTransform specific tests"""
    def test_geo_cellsize(self, grid):
        assert gdc.geo_cellsize(
            grid.geo, x_only=False) == (grid.cellsize, -grid.cellsize)

    def test_geo_cellsize_x_only(self, grid):
        assert gdc.geo_cellsize(grid.geo, x_only=True) == grid.cellsize

    def test_geo_origin(self, grid):
        assert gdc.geo_origin(grid.geo) == grid.origin

    def test_geo_extent(self, grid):
        assert list(gdc.geo_extent(
            grid.geo, grid.rows, grid.cols)) == grid.extent

    # def test_round_geo(self, grid, n=10):
    #     """Round the values of a geotransform to n digits"""
    #     assert round_geo(grid.geo) == grid.geo

    def test_geo_2_ee_transform(self, grid):
        """ EE crs transforms are different than GDAL geo transforms

        EE: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]
        GDAL: [xTranslation, xScale, xShearing, yTranslation, yShearing, yScale]
        """
        assert gdc.geo_2_ee_transform(grid.geo) == grid.transform

    def test_ee_transform_2_geo(self, grid):
        """ EE crs transforms are different than GDAL geo transforms

        EE: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]
        GDAL: [xTranslation, xScale, xShearing, yTranslation, yShearing, yScale]
        """
        assert gdc.ee_transform_2_geo(grid.transform) == grid.geo

    # def test_array_offset_geo(self, test_geo, x_offset=test_cs, y_offset=test_cs,
    #                           expected):
    #     """Return sub_geo that is offset from full_geo"""
    #     assert gdc.array_offset_geo(test_geo, 15, )

    # def test_array_geo_offsets(self, full_geo, sub_geo, cs):
    #     """Return x/y offset of a gdal.geotransform based on another gdal.geotransform"""
    #     assert array_geo_offsets()


class TestFeature:
    """Feature specific tests"""
    pass
    # def test_feature_path_osr(self, feature_path):
    #     assert False

    # def test_feature_ds_osr(self, feature_ds):
    #     assert False

    # def test_feature_lyr_osr(self, feature_lyr):
    #     assert False

    # # def test_feature_ds_extent(self, feature_ds):
    # #     assert False

    # def test_feature_lyr_extent(self, feature_lyr):
    #     assert False

    # # def test_feature_path_fields(self, feature_path):
    # #     assert False

    # # def test_feature_ds_fields(self, feature_ds):
    # #     assert False

    # # def test_feature_lyr_fields(self, feature_lyr):
    # #     assert False


class TestArray:
    """Array specific tests"""
    pass

    # def test_project_array(self, input_array, resampling_type,
    #                        input_osr, input_cs, input_extent,
    #                        output_osr, output_cs, output_extent,
    #                        output_nodata=None):
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


class TestRaster:
    """Raster specific tests"""
    def test_raster_ds_geo(self, raster):
        assert gdc.raster_ds_geo(raster.ds) == raster.geo

    def test_raster_ds_extent(self, raster):
        assert list(gdc.raster_ds_extent(raster.ds)) == raster.extent

    # def test_raster_path_shape(self, raster_path):
    #     assert raster_path_shape(raster.path) == raster.shape

    def test_raster_ds_shape(self, raster):
        assert gdc.raster_ds_shape(raster.ds) == raster.shape

    # # def test_raster_path_set_nodata(self, raster, input_nodata):
    # #     """Set raster nodata value for all bands"""
    # #     assert gdc.raster_path_set_nodata(raster.path, input_nodata)

    # # def test_raster_ds_set_nodata(self, raster, input_nodata):
    # #     """Set raster dataset nodata value for all bands"""
    # #     assert gdc.raster_ds_set_nodata(raster.ds, input_nodata)

    # def test_raster_to_array(self, input_raster, band=1, mask_extent=None,
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

    def test_raster_ds_to_array(self, raster):
        """Test reading NumPy array from raster

        Output array size will match the mask_extent if mask_extent is set

        Args:
            input_raster_ds (): opened raster dataset as gdal raster
            mask_extent (): subset extent of the raster if desired
            fill_value (float): Value to Initialize empty array with
        """
        raster_array = gdc.raster_ds_to_array(
            raster.ds, return_nodata=False)
        assert np.array_equal(
            raster_array[np.isfinite(raster_array)],
            raster.array[np.isfinite(raster.array)])

    def test_raster_ds_to_array_1(self, raster):
        """Test reading raster array and nodata value"""
        raster_array, raster_nodata = gdc.raster_ds_to_array(
            raster.ds, return_nodata=True)
        assert np.array_equal(
            raster_array[np.isfinite(raster_array)],
            raster.array[np.isfinite(raster.array)])

        # Nodata value is always "nan" for float types
        if raster.dtype in [np.float32, np.float64]:
            assert np.isnan(raster_nodata)
        else:
            assert raster_nodata == raster.nodata

    # @pytest.mark.skipif(
    #     raster.nodata is not None,
    #     reason='Only test default_nodata_value if nodata value was not set')
    def test_raster_ds_to_array_2(self, raster, default_nodata_value=255):
        """Test reading raster with default_nodata_value set

        If the raster does not have a nodata value,
            use fill_value as the nodata value
        Only test if raster.nodata is
        """
        if raster.nodata is not None:
            pytest.skip('Only test if raster does not have a nodata value set')
        else:
            raster_array = gdc.raster_ds_to_array(
                raster.ds, default_nodata_value=default_nodata_value,
                return_nodata=False)
            expected = raster.array[:]
            expected[expected == default_nodata_value] = np.nan
            assert np.array_equal(
                raster_array[np.isfinite(raster_array)],
                expected[np.isfinite(expected)])

    def test_raster_ds_to_array_3(self, raster):
        """Test reading raster with mask_extent"""
        # raster_array, raster_nodata = gdc.raster_ds_to_array(
        #     raster.ds, mask_extent=[], return_nodata=False)
        pass


class TestGeoJson:
    @pytest.fixture(scope='class')
    def points(self, grid):
        """Convert grid corner points to GeoJson coordinates"""
        return map(list, gdc.Extent(grid.extent).corner_points())

    def test_json_reverse_polygon(self, points):
        """Reverse the point order from counter-clockwise to clockwise"""
        json_geom = {'type': 'Polygon', 'coordinates': [points]}
        expected = {'type': 'Polygon', 'coordinates': [points[::-1]]}
        assert gdc.json_reverse_func(json_geom) == expected

    def test_json_reverse_multipolygon(self, points):
        """Reverse the point order from counter-clockwise to clockwise"""
        json_geom = {'type': 'MultiPolygon', 'coordinates': [[points]]}
        expected = {'type': 'MultiPolygon', 'coordinates': [[points[::-1]]]}
        assert gdc.json_reverse_func(json_geom) == expected

    def test_json_strip_z_polygon(self, points):
        """Strip Z value from coordinates"""
        json_geom = {
            'type': 'Polygon', 'coordinates': [[p + [1.0] for p in points]]}
        expected = {'type': 'Polygon', 'coordinates': [points]}
        assert gdc.json_strip_z_func(json_geom) == expected

    def test_json_strip_z_multipolygon(self, points):
        """Strip Z value from coordinates"""
        json_geom = {
            'type': 'MultiPolygon',
            'coordinates': [[[p + [1.0] for p in points]]]}
        expected = {'type': 'MultiPolygon', 'coordinates': [[points]]}
        assert gdc.json_strip_z_func(json_geom) == expected


# def test_shapefile_2_geom_list_func(input_path, zone_field=None,
#                                     reverse_flag=False, simplify_flag=False):
#     """Return a list of feature geometries in the shapefile

#     Also return the FID and value in zone_field
#     FID value will be returned if zone_field is not set or does not exist
#     """
#     assert False
