# import datetime
# import json
import logging
# import math
# import sys

import ee
# from osgeo import ogr
import pytest

import ee_tools.ee_common as ee_common



# system_properties = ['system:index', 'system:time_start', 'system:time_end']

refl_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

# nldas_filter = ee.Filter.maxDifference(
#     1000 * 60 * 60 * 4,
#     "system:time_start", None, "system:time_start", None)
# nldas_prev_filter = nldas_filter.And(ee.Filter.greaterThan(
#     "system:time_start", None, "system:time_start", None))
# nldas_next_filter = nldas_filter.And(ee.Filter.lessThan(
#     "system:time_start", None, "system:time_start", None))


@pytest.fixture(scope="module")
def ee_init():
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    return ee.Initialize()


def image_value(image, band_name):
    return image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee.Geometry.Rectangle([0, 0, 10, 10], 'EPSG:32611', False),
        scale=1).getInfo()[band_name]


# def get_landsat_images(landsat4_flag=False, landsat5_flag=True,
#                        landsat7_flag=True, landsat8_flag=True,
#                        landsat_coll_args={}):
#     """Compute Landsat derived images

#     NLDAS must be merged with Landsat before calling images function
#     Since images functions are Landsat specific,
#         joining NLDAS images must be inside each Landsat conditional

#     Args:
#         landsat4_flag (bool): if True, include Landsat 4 images
#         landsat5_flag (bool): if True, include Landsat 5 images
#         landsat7_flag (bool): if True, include Landsat 7 images
#         landsat8_flag (bool): if True, include Landsat 8 images
#         landsat_coll_args (dict): keyword arguments for get_landst_collection

#     Returns:
#         ee.ImageCollection of Landsat TOA images
#     """
#     assert False


# def get_landsat_image(landsat, year, doy, landsat_coll_args={}):
#     """Return a single Landsat image

#     Mosaic images from different rows

#     Args:
#         image_id (str): Landsat scene ID
#         fmask_flag (bool): if True, apply FMASK cloud mask to image
#         acca_flag (bool): if True, apply ACCA cloud mask to image

#     Returns:
#         ee.Image
#     """
#     assert False


# def get_landsat_collection(landsat, landsat_type='TOA', fmask_type='none',
#                            fmask_flag=False, acca_flag=False,
#                            zone_geom=None, start_date=None, end_date=None,
#                            start_year=None, end_year=None,
#                            start_month=None, end_month=None,
#                            start_doy=None, end_doy=None,
#                            scene_id_keep_list=[], scene_id_skip_list=[],
#                            path_keep_list=[], row_keep_list=[],
#                            adjust_mode=''):
#     """Build and filter a Landsat collection

#     Args:
#         landsat ():
#         landsat_type (str): 'TOA' or 'SR' (not currently supported)
#             To support 'SR' would need to modify collection_name and
#             make WRS_PATH and WRS_ROW lower case.
#         fmask_type (str): 'none', 'fmask' or 'cfmask'
#         fmask_flag (bool): if True, mask Fmask cloud, shadow, and snow pixels
#         acca_flag (bool): if True, mask pixels with clouds scores > 50
#         zone_geom ():
#         start_date (str):
#         end_date (str):
#         start_year (int):
#         end_year (int):
#         start_month (int):
#         end_month (int):
#         start_doy (int):
#         end_doy (int):
#         scene_id_keep_list (list):
#         scene_id_skip_list (list):
#         path_keep_list (list):
#         row_keep_list (list):
#         adjust_mode (str): Adjust Landsat red and NIR bands
#             'ETM_2_OLI' or 'OLI_2_ETM'
#             This could probably be simplifed to a simple flag
#             This flag is not used directly in this function

#     Returns:
#         ee.ImageCollection
#     """

#     assert False


# def landsat_acca_band_func(refl_toa):
#     """Add ACCA like cloud score band to Landsat collection"""
#     assert False


# def landsat_fmask_band_func(refl_toa):
#     """Get Fmask band from the joined properties"""
#     assert False


# def landsat_empty_fmask_band_func(refl_toa):
#     """Add an empty fmask band"""
#     assert False


# def landsat45_images_func(refl_toa):
#     """EE mappable function for calling landsat_image_func for Landsat 4/5"""
#     assert False


# def landsat7_images_func(refl_toa):
#     """EE mappable function for calling landsat_image_func for Landsat 7"""
#     assert False


# def landsat8_images_func(refl_toa):
#     """EE mappable function for calling landsat_image_func for Landsat 8"""
#     assert False


# # DEADBEEF - This is an awful way of passing the adjust_mode to the function
# def landsat45_adjust_func(refl_toa):
#     """EE mappable function for calling landsat_image_func for Landsat 4/5"""
#     assert False

# def landsat7_adjust_func(refl_toa):
#     """EE mappable function for calling landsat_image_func for Landsat 7"""
#     assert False

# def landsat8_adjust_func(refl_toa):
#     """EE mappable function for calling landsat_image_func for Landsat 8"""
#     assert False


# def landsat_images_func(refl_toa_orig, landsat, adjust_mode=''):
#     """Calculate Landsat products

#     Args:
#         refl_toa_orig (ee.ImageCollection): Landsat TOA reflectance collection
#         landsat (str): Landsat type ('LT5', 'LE7', or 'LC8')
#         adjust_mode (str): Adjust Landsat red and NIR bands
#             'ETM_2_OLI' or 'OLI_2_ETM'
#             This could probably be simplifed to a simple flag

#     Returns:
#         ee.Image()
#     """
#     assert False


# def test_landsat45_toa_band_func(img):
#     """Rename Landsat 4 and 5 bands to common band names

#     Change band order to match Landsat 8
#     Set K1 and K2 coefficients used for computing land surface temperature
#     Set Tasseled cap coefficients
#     """
#     assert False


# def test_landsat7_toa_band_func(img):
#     """Rename Landsat 7 bands to common band names

#     For now, don't include pan-chromatic or high gain thermal band
#     Change band order to match Landsat 8
#     Set K1 and K2 coefficients used for computing land surface temperature
#     Set Tasseled cap coefficients
#     """
#     # ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2',
#     #  'B7', 'B8', 'cloud_score', 'fmask'],
#     # ['blue', 'green', 'red', 'nir', 'swir1', 'thermal1', 'thermal2',
#     #  'swir2', 'pan', 'cloud_score', 'fmask'])
#     assert False


# def test_landsat8_toa_band_func(img):
#     """Rename Landsat 8 bands to common band names

#     For now, don't include coastal, cirrus, pan-chromatic, or 2nd thermal band
#     Set K1 and K2 coefficients used for computing land surface temperature
#     Set Tasseled cap coefficients
#     """
#     # ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
#     #  'B9', 'B10', 'B11', 'cloud_score'],
#     # ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
#     #  'pan', 'cirrus', 'thermal1', 'thermal2', 'cloud_score'])
#     assert False


# def test_landsat45_sr_band_func(img):
#     """Rename Landsat 4 and 5 bands to common band names

#     Change band order to match Landsat 8
#     Scale values by 10000
#     """
#     assert False


# def test_landsat7_sr_band_func(img):
#     """Rename Landsat 7 bands to common band names

#     Change band order to match Landsat 8
#     For now, don't include pan-chromatic or high gain thermal band
#     Scale values by 10000
#     """
#     assert False


# def test_landsat8_sr_band_func(img):
#     """Rename Landsat 8 bands to common band names

#     For now, don't include coastal, cirrus, or pan-chromatic
#     Scale values by 10000
#     """
#     assert False


# def test_common_area_func(img):
#     """Only keep pixels that are common to all bands"""
#     assert False


# def test_erode_func(img):
#     """"""
#     assert False


# def landsat_acca_cloud_mask_func(img):
#     """Apply basic ACCA cloud mask to a daily Landsat TOA image

#     Only apply ACCA cloud mask to Landsat reflectance bands

#     For Landsat 8 images after Oct 31st, 2015, there is no LST data
#         so simpleCloudScore returns a fully masked image
#     This makes it appear as if there are no Landsat 8 TOA images/data
#     If simpleCloudScore doesn't work, this function should not mask any values
#         and instead return all pixels, even cloudy ones
#     Use "unmask(0)" to set all masked pixels as cloud free
#     This should have no impact on earlier Landsat TOA images and could be
#         removed once the LST issue is resolved
#     """
#     assert False


# def test_landsat_fmask_cloud_mask_func(img):
#     """Apply the Fmask band in the TOA FMASK reflectance collections

#     Only apply Fmask cloud mask to Landsat reflectance bands

#     0 - Clear land
#     1 - Clear water
#     2 - Cloud shadow
#     3 - Snow
#     4 - Cloud
#     """
#     assert False


# def test_landsat_cfmask_cloud_mask_func(img):
#     """Apply the CFmask band in the at-surface reflectance collections

#     Only apply Fmask cloud mask to reflectance bands

#     0 - Clear land
#     1 - Clear water
#     2 - Cloud shadow
#     3 - Snow
#     4 - Cloud
#     """
#     assert False


# # def test_acca_mask_func(refl_toa):
# #     """Apply ACCA cloud mask function"""
# #     assert False


# def test_cos_theta_flat_func(acq_doy, acq_time, lat=None, lon=None):
#     """Cos(theta) - Spatially varying flat Model

#     Args:
#         acq_doy: EarthEngine number of the image acquisition day of year
#             scene_date = ee.Algorithms.Date(ee_image.get("system:time_start"))
#             acq_doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
#         acq_time: EarthEngine number of the image acquisition UTC time in hours
#             i.e. 18:30 -> 18.5
#             Calcuatl
#             scene_date = ee.Algorithms.Date(ee_image.get("system:time_start"))
#             acq_time = ee.Number(scene_date.getFraction('day')).multiply(24)
#         lat: EarthEngine image of the latitude [radians]
#             lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi/180)
#         lon: EarthEngine image of the longitude [radians]
#             lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi/180)

#     Returns:
#         ee.Image()
#     """
#     assert False


# def test_cos_theta_mountain_func(acq_doy, acq_time, lat=None, lon=None,
#                             slope=None, aspect=None):
#     """Cos(theta) - Spatially varying moutain model

#     Args:
#         acq_doy: EarthEngine number of the image acquisition day of year
#             scene_date = ee.Algorithms.Date(ee_image.get("system:time_start"))
#             acq_doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
#         acq_time: EarthEngine number of the image acquisition UTC time in hours
#             i.e. 18:30 -> 18.5
#             Calcuatl
#             scene_date = ee.Algorithms.Date(ee_image.get("system:time_start"))
#             acq_time = ee.Number(scene_date.getFraction('day')).multiply(24)
#         lat: EarthEngine image of the latitude [radians]
#             lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi/180)
#         lon: EarthEngine image of the longitude [radians]
#             lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi/180)
#         slope: EarthEngine image of the slope [radians]
#             terrain = ee.call('Terrain', ee.Image("USGS/NED"))
#             slope = terrain.select(["slope"]).multiply(pi/180)
#         aspect: EarthEngine image of the aspect [radians]
#             0 is south, so subtract Pi from traditional aspect raster/calc
#             terrain = ee.call('Terrain', ee.Image("USGS/NED"))
#             aspect = terrain.select(["aspect"]).multiply(pi/180).subtract(math.pi)

#     Returns:
#         ee.Image()
#     """
#     assert False


# def test_refl_sur_tasumi_func(refl_toa, pair, ea, cos_theta, landsat,
#                          adjust_mode=''):
#     """Tasumi at-surface reflectance

#     Args:
#         refl_toa ():
#         pair ():
#         ea ():
#         cos_theta ():
#         landsat ():
#         adjust_mode (str): Adjust Landsat red and NIR bands
#             'ETM_2_OLI' or 'OLI_2_ETM'
#             This could probably be simplifed to a simple flag

#     Returns:
#         ee.Image of at-surface reflectance
#     """
#     assert False


@pytest.mark.parametrize(
    "refl_sur,landsat,expected",
    [
        [[1.0] * 6, 'LT5', sum([0.254, 0.149, 0.147, 0.311, 0.103, 0.036])],
        [[2.0] * 6, 'LE7', 2 * sum([0.254, 0.149, 0.147, 0.311, 0.103, 0.036])],
        [[1.0] * 6, 'LC8', sum([0.254, 0.149, 0.147, 0.311, 0.103, 0.036])]
    ]
)
def test_albedo_func(refl_sur, landsat, expected, tol=0.0001):
    """At-surface albedo"""
    refl_sur_image = ee.Image.constant(refl_sur).rename(refl_bands)
    albedo_image = ee_common.albedo_func(refl_sur_image, landsat).rename(['albedo'])
    output = image_value(albedo_image, 'albedo')
    logging.debug('  Target values: {}'.format(expected))
    logging.debug('  Output values: {}'.format(output))
    assert abs(output - expected) <= tol


# def test_landsat_ndvi_func(img):
#     """Calculate NDVI for a daily Landsat 4, 5, 7, or 8 image"""
#     assert False


# def test_landsat_savi_func(refl_image, L=0.1):
#     """Soil adjusted vegetation index (SAVI)"""
#     assert False

# # def test_savi_func(refl_image, L=0.1):
# #     """Soil adjusted vegetation index (SAVI)"""
# #     assert False


# def test_savi_lai_func(savi):
#     """Leaf area index (LAI) calculated from SAVI"""
#     assert False


# def test_ndvi_lai_func(ndvi):
#     """Leaf area index (LAI) calculated from NDVI"""
#     assert False


# def test_landsat_evi_func(img):
#     """Calculate EVI for a daily Landsat 4, 5, 7, or 8 image"""
#     assert False


# def test_etstar_func(evi, c0, c1, c2, expected):
#     """Beamer ET*"""
#     assert False


# def test_etg_func(etstar, eto, ppt):
#     assert False


# def test_et_func(etg, ppt):
#     assert False


# # def test_tasseled_cap_func(refl_toa):
# #     assert False


# def test_tc_bright_func(refl_toa, landsat='LE7'):
#     """Tasseled cap brightness

#     Top of atmosphere (at-satellite) reflectance

#     LT5 - http://www.gis.usu.edu/~doug/RS5750/assign/OLD/RSE(17)-301.pdf
#     LE7 - http://landcover.usgs.gov/pdf/tasseled.pdf
#     LC8 - http://www.tandfonline.com/doi/abs/10.1080/2150704X.2014.915434
#     https://www.researchgate.net/publication/262005316_Derivation_of_a_tasselled_cap_transformation_based_on_Landsat_8_at-_satellite_reflectance
#     """
#     assert False


# def test_tc_green_func(refl_toa, landsat='LE7'):
#     """Tasseled cap greeness"""
#     assert False


# def test_tc_wet_func(refl_toa, landsat='LE7'):
#     """Tasseled cap wetness"""
#     assert False


# def test_em_nb_func(ndvi, lai):
#     """Narrowband emissivity"""
#     assert False


# def test_em_wb_func(ndvi, lai):
#     """Broadband emissivity"""
#     assert False


# def test_ts_func(ts_brightness, em_nb, k1=607.76, k2=1260.56):
#     """Surface temperature"""
#     assert False


# def test_landsat_true_color_func(img):
#     """Calculate true color for a daily Landsat 4, 5, 7, or 8 image"""
#     assert False


# def test_landsat_false_color_func(img):
#     """Calculate false color for a daily Landsat 4, 5, 7, or 8 image"""
#     assert False


# def test_nldas_interp_func(ee_image):
#     """Interpolate NLDAS image at Landsat scene time

#     Args:
#         ee_image (ee.Image()):
#             NLDAS hourly image collection must have been joined to it
#             Previous NLDAS image must be selectable with "nldas_prev_match"
#             Next NLDAS image must be selectable with "nldas_next_match"
#     Returns
#         ee.Image() of NLDAS values interpolated at the image time
#     """
#     assert False


# def test_pair_func(elev_image):
#     """Elevation based air pressure"""
#     assert False


# def test_vapor_pressure_func(temperature_image):
#     """Vapor Pressure

#     in kPa with temperature in C
#     """
#     return temperature_image.expression(
#         '0.6108 * exp(17.27 * b() / (b() + 237.3))')


# def test_prism_ppt_func(prism_image):
#     """PRISM water year precipitation

#     Depends on maps engine assets
#     """
#     assert False


# def test_gridmet_ppt_func(gridmet_image):
#     """GRIDMET daily precipitation"""
#     assert False


# def test_gridmet_etr_func(gridmet_image):
#     """GRIDMET Daily ETr"""
#     assert False


# def test_gridmet_eto_func(gridmet_image):
#     """GRIDMET Daily ETo"""
#     assert False


# def test_daily_pet_func(doy, tmin, tmax, ea, rs, uz, zw, cn=900, cd=0.34):
#     """Daily ASCE Penman Monteith Standardized Reference ET

#     Daily ETo cn=900, cd=0.34
#     Daily ETr cn=1600, cd=0.38

#     doy -- day of year
#     tmin -- minimum daily temperature [C]
#     tmax -- maximum daily temperature [C]
#     ea -- vapor pressure [?]
#     rs -- incoming solar radiation [MJ m-2 day]
#     uz -- wind speed [m s-1]
#     zw -- wind speed height [m]
#     cn -- coefficient
#     cd -- coefficient

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


# def test_shapefile_2_geom_list_func(input_path, zone_field=None,
#                                reverse_flag=False, simplify_flag=False):
#     """Return a list of feature geometries in the shapefile

#     Also return the FID and value in zone_field
#     FID value will be returned if zone_field is not set or does not exist
#     """
#     assert False


# def test_json_reverse_func(json_obj):
#     """Reverse the point order from counter-clockwise to clockwise"""
#     assert False


# def test_geo_2_ee_transform(gdal_geo):
#     """ EE crs transforms are different than GDAL geo transforms
#     EE: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]
#     GDAL: [xTranslation, xScale, xShearing, yTranslation, yShearing, yScale]
#     """
#     assert False


# def test_ee_transform_2_geo(gdal_geo):
#     """ EE crs transforms are different than GDAL geo transforms
#     EE: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]
#     GDAL: [xTranslation, xScale, xShearing, yTranslation, yShearing, yScale]
#     """
#     assert False