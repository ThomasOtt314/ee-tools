#--------------------------------
# Name:         ee_common.py
# Purpose:      Common EarthEngine support functions
# Author:       Charles Morton
# Created       2017-01-25
# Python:       2.7
#--------------------------------

import datetime
import logging
import math
import sys

import ee


ee.Initialize()

system_properties = ['system:index', 'system:time_start', 'system:time_end']

refl_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

nldas_filter = ee.Filter.maxDifference(
    difference=1000 * 60 * 60 * 4,
    leftField='system:time_start', rightField='system:time_start')
nldas_prev_filter = ee.Filter.And(
    nldas_filter,
    ee.Filter.greaterThan(
        leftField='system:time_start', rightField='system:time_start'))
nldas_next_filter = ee.Filter.And(
    nldas_filter,
    ee.Filter.lessThan(
        leftField='system:time_start', rightField='system:time_start'))


def show_thumbnail(ee_image):
    """Show the EarthEngine image thumbnail in a window"""
    output_url = ee_image.getThumbUrl({'format': 'jpg', 'size': '600'})
    logging.debug(output_url)

    # import webbrowser
    # webbrowser.open(output_url)

    import io
    # import Image, ImageTk
    from PIL import Image, ImageTk
    import Tkinter as tk
    import urllib2
    window = tk.Tk()
    output_file = Image.open(io.BytesIO(urllib2.urlopen(output_url).read()))
    output_photo = ImageTk.PhotoImage(output_file)
    label = tk.Label(window, image=output_photo)
    label.pack()
    window.mainloop()


# Earth Engine calculation functions

def get_landsat_images(landsat4_flag=False, landsat5_flag=True,
                       landsat7_flag=True, landsat8_flag=True,
                       mosaic_method=None, landsat_coll_args={}):
    """Compute Landsat derived images

    NLDAS must be merged with Landsat before calling images function
    Since images functions are Landsat specific,
        joining NLDAS images must be inside each Landsat conditional

    Args:
        landsat4_flag (bool): if True, include Landsat 4 images.
        landsat5_flag (bool): if True, include Landsat 5 images.
        landsat7_flag (bool): if True, include Landsat 7 images.
        landsat8_flag (bool): if True, include Landsat 8 images.
        mosaic_method (str):
        landsat_coll_args (dict): keyword arguments for get_landsat_collection.

    Returns:
        ee.ImageCollection
    """
    # Assign nearest 4 NLDAS images to each Landsat image
    nldas_coll = ee.ImageCollection('NASA/NLDAS/FORA0125_H002')
    try:
        nldas_coll = nldas_coll.filterDate(
            landsat_coll_args['start_date'], landsat_coll_args['end_date'])
    except KeyError:
        pass
    nldas_prev_join = ee.Join.saveBest('nldas_prev_match', 'nldas_prev_metric')
    nldas_next_join = ee.Join.saveBest('nldas_next_match', 'nldas_next_metric')

    if landsat4_flag:
        l4_coll = get_landsat_collection('LT4', **landsat_coll_args)
        l4_coll = ee.ImageCollection(nldas_prev_join.apply(
            l4_coll, nldas_coll, nldas_prev_filter))
        l4_coll = ee.ImageCollection(nldas_next_join.apply(
            l4_coll, nldas_coll, nldas_next_filter))
        # DEADBEEF - This is an awful way to set which adjustment to use
        if ('adjust_method' in landsat_coll_args.keys() and
                landsat_coll_args['adjust_method'] and
                landsat_coll_args['adjust_method'].upper()):
            l4_coll = ee.ImageCollection(l4_coll.map(landsat45_adjust_func))
        else:
            l4_coll = ee.ImageCollection(l4_coll.map(landsat45_images_func))
    if landsat5_flag:
        l5_coll = get_landsat_collection('LT5', **landsat_coll_args)
        # Exclude 2012 Landsat 5 images
        l5_coll = l5_coll.filter(ee.Filter.calendarRange(1984, 2011, 'year'))
        l5_coll = ee.ImageCollection(nldas_prev_join.apply(
            l5_coll, nldas_coll, nldas_prev_filter))
        l5_coll = ee.ImageCollection(nldas_next_join.apply(
            l5_coll, nldas_coll, nldas_next_filter))
        # DEADBEEF - This is an awful way to set which adjustment to use
        if ('adjust_method' in landsat_coll_args.keys() and
                landsat_coll_args['adjust_method'] and
                landsat_coll_args['adjust_method'].lower() == 'etm_2_oli'):
            l5_coll = ee.ImageCollection(l5_coll.map(landsat45_adjust_func))
        else:
            l5_coll = ee.ImageCollection(l5_coll.map(landsat45_images_func))
    if landsat7_flag:
        l7_coll = get_landsat_collection('LE7', **landsat_coll_args)
        l7_coll = ee.ImageCollection(nldas_prev_join.apply(
            l7_coll, nldas_coll, nldas_prev_filter))
        l7_coll = ee.ImageCollection(nldas_next_join.apply(
            l7_coll, nldas_coll, nldas_next_filter))
        # DEADBEEF - This is an awful way to set which adjustment to use
        if ('adjust_method' in landsat_coll_args.keys() and
                landsat_coll_args['adjust_method'] and
                landsat_coll_args['adjust_method'].lower() == 'etm_2_oli'):
            l7_coll = ee.ImageCollection(l7_coll.map(landsat7_adjust_func))
        else:
            l7_coll = ee.ImageCollection(l7_coll.map(landsat7_images_func))
    if landsat8_flag:
        l8_coll = get_landsat_collection('LC8', **landsat_coll_args)
        l8_coll = ee.ImageCollection(nldas_prev_join.apply(
            l8_coll, nldas_coll, nldas_prev_filter))
        l8_coll = ee.ImageCollection(nldas_next_join.apply(
            l8_coll, nldas_coll, nldas_next_filter))
        # DEADBEEF - This is an awful way to set which adjustment to use
        if ('adjust_method' in landsat_coll_args.keys() and
                landsat_coll_args['adjust_method'] and
                landsat_coll_args['adjust_method'].lower() == 'oli_2_etm'):
            l8_coll = ee.ImageCollection(l8_coll.map(landsat8_adjust_func))
        else:
            l8_coll = ee.ImageCollection(l8_coll.map(landsat8_images_func))

    # Merging the image collections changes the SCENE_ID
    # The SCENE_ID is extracted below using a regular expression
    # LT504303222011158EDC00 -> 1_1_LT504303222011158EDC00
    landsat_coll = ee.ImageCollection([])
    if landsat5_flag:
        landsat_coll = landsat_coll.merge(l5_coll)
    if landsat4_flag:
        landsat_coll = landsat_coll.merge(l4_coll)
    if landsat7_flag:
        landsat_coll = landsat_coll.merge(l7_coll)
    if landsat8_flag:
        landsat_coll = landsat_coll.merge(l8_coll)

    return landsat_coll


def get_landsat_image(landsat, year, doy, path=None, row=None,
                      mosaic_method=None, landsat_coll_args={}):
    """Return a single mosaiced Landsat image

    Mosaic images from different rows from the same date (same path)

    Args:
        landsat (str):
        year (int):
        doy (int): day of year
        path (int): Landsat path number
        row (int): Landsat row number
        mosaic_method (str):
        landsat_coll_args (dict): keyword arguments for get_landst_collection

    Returns:
        ee.Image
    """
    image_start_dt = datetime.datetime.strptime(
        '{:04d}_{:03d}'.format(int(year), int(doy)), '%Y_%j')
    image_end_dt = image_start_dt + datetime.timedelta(days=1)

    # Adjust the default keyword arguments for a single image date
    image_args = landsat_coll_args.copy()
    image_args['start_date'] = image_start_dt.date().isoformat()
    image_args['end_date'] = image_end_dt.date().isoformat()
    # image_args['start_year'] = year
    # image_args['end_dear'] = year
    # image_args['start_doy'] = doy
    # image_args['end_doy'] = doy
    if path:
        image_args['path_keep_list'] = [int(path)]
    if row:
        image_args['row_keep_list'] = [int(row)]

    landsat_coll = get_landsat_collection(landsat, **image_args)

    # Filter the NLDAS collection
    nldas_coll = ee.ImageCollection('NASA/NLDAS/FORA0125_H002') \
        .filterDate(image_start_dt, image_end_dt)

    # Assign nearest 4 NLDAS images to each Landsat image
    # landsat_coll = ee.ImageCollection(
    #     ee.Join.saveBest('nldas_match', 'nldas_metric').apply(
    #         landsat_coll, nldas_coll, nldas_filter));
    landsat_coll = ee.ImageCollection(ee.Join.saveBest(
        'nldas_prev_match', 'nldas_prev_metric').apply(
            landsat_coll, nldas_coll, nldas_prev_filter))
    landsat_coll = ee.ImageCollection(ee.Join.saveBest(
        'nldas_next_match', 'nldas_next_metric').apply(
            landsat_coll, nldas_coll, nldas_next_filter))

    # Compute derived images
    if landsat.upper() in ['LT4', 'LT5']:
        # DEADBEEF - This is an awful way to set which adjustment to use
        if ('adjust_method' in landsat_coll_args.keys() and
                landsat_coll_args['adjust_method'] and
                landsat_coll_args['adjust_method'].lower() == 'etm_2_oli'):
            landsat_coll = landsat_coll.map(landsat45_adjust_func)
        else:
            landsat_coll = landsat_coll.map(landsat45_images_func)
    elif landsat.upper() == 'LE7':
        # DEADBEEF - This is an awful way to set which adjustment to use
        if ('adjust_method' in landsat_coll_args.keys() and
                landsat_coll_args['adjust_method'] and
                landsat_coll_args['adjust_method'].lower() == 'etm_2_oli'):
            landsat_coll = landsat_coll.map(landsat7_adjust_func)
        else:
            landsat_coll = landsat_coll.map(landsat7_images_func)
    elif landsat.upper() == 'LC8':
        # DEADBEEF - This is an awful way to set which adjustment to use
        if ('adjust_method' in landsat_coll_args.keys() and
                landsat_coll_args['adjust_method'] and
                landsat_coll_args['adjust_method'].lower() == 'oli_2_etm'):
            landsat_coll = landsat_coll.map(landsat8_adjust_func)
        else:
            landsat_coll = landsat_coll.map(landsat8_images_func)

    # Return first image from collection (don't mosaic)
    if not mosaic_method:
        landsat_image = ee.Image(landsat_coll.first())
    # Mosaic overlapping images
    elif mosaic_method.lower() == 'mean':
        landsat_image = ee.Image(landsat_coll.mean())
    elif mosaic_method.lower() == 'mosaic':
        landsat_image = ee.Image(landsat_coll.mosaic())
    elif mosaic_method.lower() == 'min':
        landsat_image = ee.Image(landsat_coll.min())
    elif mosaic_method.lower() == 'max':
        landsat_image = ee.Image(landsat_coll.max())
    elif mosaic_method.lower() == 'median':
        landsat_image = ee.Image(landsat_coll.median())
    else:
        logging.error('\nERROR: Unsupported mosaic method: {}'.format(
            mosaic_method))
        sys.exit()

    return landsat_image


def get_landsat_collection(landsat, landsat_type='toa', fmask_type=None,
                           fmask_flag=False, acca_flag=False,
                           zone_geom=None, start_date=None, end_date=None,
                           start_year=None, end_year=None,
                           start_month=None, end_month=None,
                           start_doy=None, end_doy=None,
                           scene_id_keep_list=[], scene_id_skip_list=[],
                           path_keep_list=[], row_keep_list=[],
                           adjust_method=None):
    """Build and filter a Landsat collection

    If fmask_type is 'fmask', an fmask collection is built but not used.
    This was done to avoid including lots of conditionals and to
        make the collection filtering logic easier to read/follow.

    Args:
        landsat ():
        landsat_type (str): 'toa'
            'sr' not currently supported.
            To support 'sr' would need to modify collection_name and
                make WRS_PATH and WRS_ROW lower case.
        fmask_type (str): 'none', 'fmask' or 'cfmask'
        fmask_flag (bool): if True, mask Fmask cloud, shadow, and snow pixels
        acca_flag (bool): if True, mask pixels with clouds scores > 50
        zone_geom ():
        start_date (str):
        end_date (str):
        start_year (int):
        end_year (int):
        start_month (int):
        end_month (int):
        start_doy (int):
        end_doy (int):
        scene_id_keep_list (list):
        scene_id_skip_list (list):
        path_keep_list (list): Landsat path numbers (as int)
        row_keep_list (list): Landsat row numbers (as int)
        adjust_method (str): Adjust Landsat red and NIR bands.
            Choices: 'etm_2_oli' or 'oli_2_etm'.
            This could probably be simplifed to a flag.
            This flag is passed through and not used directly in this function

    Returns:
        ee.ImageCollection
    """

    def scene_id_func(input_image):
        scene_id = ee.String(
            input_image.get('system:index')).slice(0, 16)
        #     input_image.get('LANDSAT_SCENE_ID')).slice(0, 16)
        return input_image.setMulti({'SCENE_ID': scene_id})
        # mosaic_id = scene_id.slice(0, 6).cat('XXX').cat(scene_id.slice(9, 16))
        # return input_image.setMulti({
        #     'SCENE_ID': scene_id, 'MOSAIC_ID': mosaic_id})

    landsat_sr_name = 'LANDSAT/{}_SR'.format(landsat.upper())
    landsat_toa_name = 'LANDSAT/{}_L1T_TOA'.format(landsat.upper())
    landsat_fmask_name = 'LANDSAT/{}_L1T_TOA_FMASK'.format(landsat.upper())

    if (landsat_type.lower() == 'toa' and
            (not fmask_type or fmask_type.lower() == 'none')):
        landsat_coll = ee.ImageCollection(landsat_toa_name)
        # Add empty fmask band
        landsat_coll = landsat_coll.map(landsat_empty_fmask_band_func)
        # Build fmask_coll so filtering is cleaner, but don't use it
        fmask_coll = ee.ImageCollection(landsat_sr_name) \
            .select(['cfmask'], ['fmask'])
    elif landsat_type.lower() == 'toa' and fmask_type.lower() == 'cfmask':
        # Join Fmask band from SR collection to TOA collection
        landsat_coll = ee.ImageCollection(landsat_toa_name)
        fmask_coll = ee.ImageCollection(landsat_sr_name) \
            .select(['cfmask'], ['fmask'])
    elif landsat_type.lower() == 'toa' and fmask_type.lower() == 'fmask':
        landsat_coll = ee.ImageCollection(landsat_fmask_name)
        # This fmask collection will not be used
        fmask_coll = ee.ImageCollection(landsat_sr_name) \
            .select(['cfmask'], ['fmask'])
    else:
        logging.error(
            '\nERROR: Unknown Landsat/Fmask type combination, exiting\n'
            '  Landsat: {}  Fmask: {}'.format(landsat, landsat_type))
        sys.exit()

    if path_keep_list:
        landsat_coll = landsat_coll.filter(
            ee.Filter.inList('WRS_PATH', path_keep_list))
        fmask_coll = fmask_coll.filter(
            ee.Filter.inList('wrs_path', path_keep_list))
    if row_keep_list:
        landsat_coll = landsat_coll.filter(
            ee.Filter.inList('WRS_ROW', row_keep_list))
        fmask_coll = fmask_coll.filter(
            ee.Filter.inList('wrs_row', row_keep_list))

    # Filter by date
    if start_date and end_date:
        landsat_coll = landsat_coll.filterDate(start_date, end_date)
        fmask_coll = fmask_coll.filterDate(start_date, end_date)
    # Filter by year
    if start_year and end_year:
        year_filter = ee.Filter.calendarRange(
            start_year, end_year, 'year')
        landsat_coll = landsat_coll.filter(year_filter)
        fmask_coll = fmask_coll.filter(year_filter)
    # Filter by month
    if ((start_month and start_month != 1) and
            (end_month and end_month != 12)):
        month_filter = ee.Filter.calendarRange(
            start_month, end_month, 'month')
        landsat_coll = landsat_coll.filter(month_filter)
        fmask_coll = fmask_coll.filter(month_filter)
    # Filter by day of year
    if ((start_doy and start_doy != 1) and
            (end_doy and end_doy != 365)):
        doy_filter = ee.Filter.calendarRange(
            start_doy, end_doy, 'day_of_year')
        landsat_coll = landsat_coll.filter(doy_filter)
        fmask_coll = fmask_coll.filter(doy_filter)

    # Filter by geometry
    if zone_geom:
        landsat_coll = landsat_coll.filterBounds(zone_geom)
        fmask_coll = fmask_coll.filterBounds(zone_geom)

    # Set SCENE_ID property for joining and filtering
    landsat_coll = landsat_coll.map(scene_id_func)
    fmask_coll = fmask_coll.map(scene_id_func)

    # Filter by SCENE_ID
    if scene_id_keep_list:
        scene_id_keep_filter = ee.Filter.inList(
            'SCENE_ID', scene_id_keep_list)
        landsat_coll = landsat_coll.filter(scene_id_keep_filter)
        fmask_coll = fmask_coll.filter(scene_id_keep_filter)
    if scene_id_skip_list:
        scene_id_skip_filter = ee.Filter.inList(
            'SCENE_ID', scene_id_skip_list).Not()
        landsat_coll = landsat_coll.filter(scene_id_skip_filter)
        fmask_coll = fmask_coll.filter(scene_id_skip_filter)

    # Join the at-surface reflectance CFmask collection if necessary
    if fmask_type and fmask_type.lower() == 'cfmask':
        scene_id_filter = ee.Filter.equals(
            leftField='SCENE_ID', rightField='SCENE_ID')
        landsat_coll = ee.ImageCollection(ee.Join.saveFirst('fmask').apply(
            landsat_coll, fmask_coll, scene_id_filter))

        # Add Fmask band from joined property
        landsat_coll = landsat_coll.map(landsat_fmask_band_func)

    # Add ACCA band (must be done before bands are renamed)
    landsat_coll = landsat_coll.map(landsat_acca_band_func)

    # Modify landsat collections to have same band names
    if landsat.upper() in ['LT4', 'LT5']:
        landsat_coll = landsat_coll.map(landsat45_toa_band_func)
    elif landsat.upper() in ['LE7']:
        landsat_coll = landsat_coll.map(landsat7_toa_band_func)
    elif landsat.upper() in ['LC8']:
        landsat_coll = landsat_coll.map(landsat8_toa_band_func)

    # Apply cloud masks
    if fmask_flag:
        landsat_coll = landsat_coll.map(landsat_fmask_cloud_mask_func)
    if acca_flag:
        landsat_coll = landsat_coll.map(landsat_acca_cloud_mask_func)

    return ee.ImageCollection(landsat_coll)


def landsat_acca_band_func(refl_toa):
    """Add ACCA like cloud score band to Landsat collection"""
    cloud_score = ee.Algorithms.Landsat.simpleCloudScore(refl_toa) \
        .select(['cloud'], ['cloud_score'])
    return refl_toa.addBands(cloud_score)


def landsat_fmask_band_func(refl_toa):
    """Get Fmask band from the joined properties"""
    return refl_toa.addBands(
        ee.Image(refl_toa.get('fmask')).select([0], ['fmask']))


def landsat_empty_fmask_band_func(refl_toa):
    """Add an empty fmask band"""
    return refl_toa.addBands(
        refl_toa.select([0]).multiply(0).select([0], ['fmask']))


def landsat45_images_func(refl_toa):
    """EE mappable function for calling landsat_image_func for Landsat 4/5"""
    return landsat_images_func(refl_toa, landsat='LT5', adjust_method='')


def landsat7_images_func(refl_toa):
    """EE mappable function for calling landsat_image_func for Landsat 7"""
    return landsat_images_func(refl_toa, landsat='LE7', adjust_method='')


def landsat8_images_func(refl_toa):
    """EE mappable function for calling landsat_image_func for Landsat 8"""
    return landsat_images_func(refl_toa, landsat='LC8', adjust_method='')


# DEADBEEF - This seems like an awful way of passing the adjust_method
#   to the function
def landsat45_adjust_func(refl_toa):
    """EE mappable function for calling landsat_image_func for Landsat 4/5"""
    return landsat_images_func(
        refl_toa, landsat='LT5', adjust_method='etm_2_oli')

def landsat7_adjust_func(refl_toa):
    """EE mappable function for calling landsat_image_func for Landsat 7"""
    return landsat_images_func(
        refl_toa, landsat='LE7', adjust_method='etm_2_oli')

def landsat8_adjust_func(refl_toa):
    """EE mappable function for calling landsat_image_func for Landsat 8"""
    return landsat_images_func(
        refl_toa, landsat='LC8', adjust_method='oli_2_etm')


def landsat_images_func(refl_toa_orig, landsat, adjust_method=''):
    """Calculate Landsat products

    Args:
        refl_toa_orig (ee.ImageCollection): Landsat TOA reflectance collection
        landsat (str): Landsat type ('LT4', 'LT5', 'LE7', or 'LC8')
        adjust_method (str): Adjust Landsat red and NIR bands.
            Choices are 'etm_2_oli' or 'oli_2_etm'.
            This could probably be simplifed to a flag

    Returns:
        ee.Image()
    """
    scene_date = ee.Date(refl_toa_orig.get('system:time_start'))
    doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
    hour = ee.Number(scene_date.getFraction('day')).multiply(24)

    # Clip to common area of all bands
    # Make a copy so original Fmask and cloud score can be passed through
    refl_toa = common_area_func(refl_toa_orig)

    # Brightness temperature must be > 250 K
    # refl_toa = refl_toa.updateMask(refl_toa.select(['thermal']).gt(250))

    # At-surface reflectance
    pair = pair_func(ee.Image('USGS/NED'))
    # Interpolate NLDAS data to scene time
    nldas_image = ee.Image(nldas_interp_func(refl_toa))
    # Specific humidity (kg/kg)
    q = nldas_image.select(['specific_humidity'])
    # q = ee.Image(refl_toa.get('match')).select(['specific_humidity'])
    # q = nldas_interp_func(refl_toa).select(['specific_humidity'])
    ea = pair.expression(
        'q * pair / (0.622 + 0.378 * q)', {'q': q, 'pair': pair})
    refl_sur = ee.Image(refl_sur_tasumi_func(
        refl_toa, pair, ea, cos_theta_flat_func(doy, hour),
        landsat, adjust_method))

    # At-surface albedo
    albedo_sur = albedo_func(refl_sur, landsat)

    # NDVI
    ndvi_toa = refl_toa.normalizedDifference(['nir', 'red'])
    ndvi_sur = refl_sur.normalizedDifference(['nir', 'red'])

    # NDWI - McFeeters 1996
    ndwi_green_nir_toa = refl_toa.normalizedDifference(['green', 'nir'])
    ndwi_green_nir_sur = refl_sur.normalizedDifference(['green', 'nir'])

    # NDWI - Xu 2006 (MNDWI) doi: 10.1080/01431160600589179
    # Equivalent to NDSI Hall et al 1995 and 1998
    # http://modis-snow-ice.gsfc.nasa.gov/uploads/pap_dev95.pdf
    # http://modis-snow-ice.gsfc.nasa.gov/uploads/pap_assmnt98.pdf
    ndwi_green_swir1_toa = refl_toa.normalizedDifference(['green', 'swir1'])
    ndwi_green_swir1_sur = refl_sur.normalizedDifference(['green', 'swir1'])

    # NDWI Gao doi: 10.1016/S0034-4257(96)00067-3
    # Inverse of NDSI (Soil) in Rogers & Keraney 2004
    ndwi_nir_swir1_toa = refl_toa.normalizedDifference(['nir', 'swir1'])
    ndwi_nir_swir1_sur = refl_sur.normalizedDifference(['nir', 'swir1'])

    # NDWI - Allen 2007
    # Return this NDWI as the default ndwi_sur and ndwi_toa
    ndwi_swir1_green_toa = refl_toa.normalizedDifference(['swir1', 'green'])
    ndwi_swir1_green_sur = refl_sur.normalizedDifference(['swir1', 'green'])

    # LAI (for computing Ts) (Empirical function from Allen et al 2007)
    lai_toa = ndvi_lai_func(ndvi_toa)

    # EVI
    evi_sur = landsat_evi_func(refl_sur)

    # Surface temperature
    ts = ts_func(
        refl_toa.select('thermal'),
        em_nb_func(ndvi_toa, lai_toa),
        k1=ee.Number(refl_toa.get('k1_constant')),
        k2=ee.Number(refl_toa.get('k2_constant')))

    # Tasseled cap
    tc_bright = tc_bright_func(refl_toa, landsat)
    tc_green = tc_green_func(refl_toa, landsat)
    tc_wet = tc_wet_func(refl_toa, landsat)

    # For backwards compatability, output NDWI SWIR1-Green (Allen) as NDWI?
    return ee.Image(
        [
            refl_toa.select('blue'), refl_toa.select('green'),
            refl_toa.select('red'), refl_toa.select('nir'),
            refl_toa.select('swir1'), refl_toa.select('swir2'),
            refl_sur.select('blue'), refl_sur.select('green'),
            refl_sur.select('red'), refl_sur.select('nir'),
            refl_sur.select('swir1'), refl_sur.select('swir2'),
            ndvi_toa, ndvi_sur, evi_sur, albedo_sur, ts,
            ndwi_green_nir_sur, ndwi_green_swir1_sur, ndwi_nir_swir1_sur,
            # ndwi_green_nir_toa, ndwi_green_swir1_toa, ndwi_nir_swir1_toa,
            ndwi_swir1_green_toa, ndwi_swir1_green_sur,
            tc_bright, tc_green, tc_wet,
            refl_toa_orig.select('cloud_score'), refl_toa_orig.select('fmask')
        ]).select(
            range(27),
            [
                'toa_blue', 'toa_green', 'toa_red',
                'toa_nir', 'toa_swir1', 'toa_swir2',
                'sur_blue', 'sur_green', 'sur_red',
                'sur_nir', 'sur_swir1', 'sur_swir2',
                'ndvi_toa', 'ndvi_sur', 'evi_sur', 'albedo_sur', 'ts',
                'ndwi_green_nir_sur', 'ndwi_green_swir1_sur', 'ndwi_nir_swir1_sur',
                # 'ndwi_green_nir_toa', 'ndwi_green_swir1_toa', 'ndwi_nir_swir1_toa',
                'ndwi_swir1_green_toa', 'ndwi_swir1_green_sur',
                # # 'ndwi_toa', 'ndwi_sur',
                'tc_bright', 'tc_green', 'tc_wet',
                'cloud_score', 'fmask'
            ]) \
        .copyProperties(refl_toa, system_properties + ['SCENE_ID'])


def landsat45_toa_band_func(img):
    """Rename Landsat 4 and 5 bands to common band names

    Change band order to match Landsat 8
    Set K1 and K2 coefficients used for computing land surface temperature
    Set Tasseled cap coefficients
    """
    return ee.Image(img.select(
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B7',
         'B6', 'cloud_score', 'fmask'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
         'thermal', 'cloud_score', 'fmask']))\
        .setMulti({'k1_constant': 607.76, 'k2_constant': 1260.56})\
        .copyProperties(img, system_properties)


def landsat7_toa_band_func(img):
    """Rename Landsat 7 bands to common band names

    For now, don't include pan-chromatic or high gain thermal band
    Change band order to match Landsat 8
    Set K1 and K2 coefficients used for computing land surface temperature
    Set Tasseled cap coefficients
    """
    # ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2',
    #  'B7', 'B8', 'cloud_score', 'fmask'],
    # ['blue', 'green', 'red', 'nir', 'swir1', 'thermal1', 'thermal2',
    #  'swir2', 'pan', 'cloud_score', 'fmask'])
    return ee.Image(img.select(
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B7',
         'B6_VCID_1', 'cloud_score', 'fmask'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
         'thermal', 'cloud_score', 'fmask']))\
        .setMulti({'k1_constant': 666.09, 'k2_constant': 1282.71}) \
        .copyProperties(img, system_properties)


def landsat8_toa_band_func(img):
    """Rename Landsat 8 bands to common band names

    For now, don't include coastal, cirrus, pan-chromatic, or 2nd thermal band
    Set K1 and K2 coefficients used for computing land surface temperature
    Set Tasseled cap coefficients
    """
    # ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
    #  'B9', 'B10', 'B11', 'cloud_score'],
    # ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
    #  'pan', 'cirrus', 'thermal1', 'thermal2', 'cloud_score'])
    return ee.Image(img.select(
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
         'B10', 'cloud_score', 'fmask'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
         'thermal', 'cloud_score', 'fmask']))\
        .setMulti({
            'k1_constant': img.get('K1_CONSTANT_BAND_10'),
            'k2_constant': img.get('K2_CONSTANT_BAND_10')}) \
        .copyProperties(img, system_properties)


def landsat45_sr_band_func(img):
    """Rename Landsat 4 and 5 bands to common band names

    Change band order to match Landsat 8
    Scale values by 10000
    """
    sr_image = img.select(
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']) \
        .divide(10000.0)
    # Cloud mask bands must be set after scaling
    return ee.Image(sr_image) \
        .addBands(
            img.select(['cloud_score', 'cfmask'], ['cloud_score', 'fmask'])) \
        .copyProperties(img, system_properties)


def landsat7_sr_band_func(img):
    """Rename Landsat 7 bands to common band names

    Change band order to match Landsat 8
    For now, don't include pan-chromatic or high gain thermal band
    Scale values by 10000
    """
    # ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B8'],
    # ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pan'])
    sr_image = img.select(
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])\
        .divide(10000.0)
    # Cloud mask bands must be set after scaling
    return ee.Image(sr_image) \
        .addBands(
            img.select(['cloud_score', 'cfmask'], ['cloud_score', 'fmask'])) \
        .copyProperties(img, system_properties)


def landsat8_sr_band_func(img):
    """Rename Landsat 8 bands to common band names

    For now, don't include coastal, cirrus, or pan-chromatic
    Scale values by 10000
    """
    # ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9'],
    # ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
    #  'pan', 'cirrus'])
    sr_image = img \
        .select(
            ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
            ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])\
        .divide(10000.0)
    # Cloud mask bands must be set after scaling
    return ee.Image(sr_image) \
        .addBands(
            img.select(['cloud_score', 'cfmask'], ['cloud_score', 'fmask'])) \
        .copyProperties(img, system_properties)


def common_area_func(img):
    """Only keep pixels that are common to all bands"""
    common_mask = img.mask().reduce(ee.Reducer.And())
    return img.updateMask(common_mask)
    # common_mask = img \
    #     .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']) \
    #     .mask().reduce(ee.Reducer.And())
    # # common_mask = img.select(['fmask']).mask()
    # return img.updateMask(common_mask)
    # return img \
    #     .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']) \
    #     .updateMask(common_mask) \
    #     .addBands(img.select(['cloud_score', 'cfmask'])) \
    #     .copyProperties(img, system_properties)


def erode_func(img):
    """"""
    input_mask = img.mask().reduceNeighborhood(
        ee.Reducer.min(), ee.call("Kernel.circle", 120, 'meters'))
    return img.updateMask(input_mask)


def landsat_acca_cloud_mask_func(img):
    """Apply basic ACCA cloud mask to a daily Landsat TOA image

    Only apply ACCA cloud mask to Landsat reflectance bands

    For Landsat 8 images after Oct 31st, 2015, there is no LST data
        so simpleCloudScore returns a fully masked image
    This makes it appear as if there are no Landsat 8 TOA images/data
    If simpleCloudScore doesn't work, this function should not mask any values
        and instead return all pixels, even cloudy ones
    Use "unmask(0)" to set all masked pixels as cloud free
    This should have no impact on earlier Landsat TOA images and could be
        removed once the LST issue is resolved
    """
    cloud_mask = img.select(['cloud_score']).unmask(0).lt(50)
    refl_img = img \
        .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']) \
        .updateMask(cloud_mask)
    return refl_img.addBands(img.select(['cloud_score', 'fmask'])) \
        .copyProperties(img, system_properties)


def landsat_fmask_cloud_mask_func(img):
    """Apply the Fmask band in the TOA FMASK reflectance collections

    Only apply Fmask cloud mask to Landsat reflectance bands

    0 - Clear land
    1 - Clear water
    2 - Cloud shadow
    3 - Snow
    4 - Cloud
    """
    fmask = ee.Image(img.select(['fmask']))
    cloud_mask = fmask.lt(2)
    # cloud_mask = fmask.eq(2).Or(fmask.eq(3)).Or(fmask.eq(4)).Not()
    refl_img = img \
        .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']) \
        .updateMask(cloud_mask)
    return refl_img.addBands(img.select(['cloud_score', 'fmask'])) \
        .copyProperties(img, system_properties)


def landsat_cfmask_cloud_mask_func(img):
    """Apply the CFmask band in the at-surface reflectance collections

    Only apply Fmask cloud mask to reflectance bands

    0 - Clear land
    1 - Clear water
    2 - Cloud shadow
    3 - Snow
    4 - Cloud
    """
    fmask = ee.Image(img.select(['cfmask']))
    cloud_mask = fmask.lt(2)
    # cloud_mask = fmask.eq(2).Or(fmask.eq(3)).Or(fmask.eq(4)).Not()
    refl_img = img \
        .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']) \
        .updateMask(cloud_mask)
    return refl_img \
        .addBands(
            img.select(['cloud_score', 'cfmask'], ['cloud_score', 'fmask'])) \
        .copyProperties(img, system_properties)


# def acca_mask_func(refl_toa):
#     """Apply ACCA cloud mask function"""
#     cloud_mask = ee.Algorithms.Landsat.simpleCloudScore(refl_toa) \
#         .select(['cloud']).lt(ee.Image.constant(50))
#     cloud_mask = cloud_mask.updateMask(cloud_mask)
#     return refl_toa.updateMask(cloud_mask)


def cos_theta_flat_func(acq_doy, acq_time, lat=None, lon=None):
    """Cos(theta) - Spatially varying flat Model

    Args:
        acq_doy (ee.Number): Image acquisition day of year.
            scene_date = ee.Date(ee_image.get('system:time_start'))
            acq_doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
        acq_time (ee.Number): Image acquisition UTC time in hours.
            i.e. 18:30 -> 18.5
            scene_date = ee.Date(ee_image.get('system:time_start'))
            acq_time = ee.Number(scene_date.getFraction('day')).multiply(24)
        lat (ee.Image): Latitude [radians].
            lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi/180)
        lon (ee.Image): Longitude [radians].
            lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi/180)

    Returns:
        ee.Image()
    """
    pi = math.pi
    if lat is None:
        lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi / 180)
    if lon is None:
        lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi / 180)
    delta = acq_doy.multiply(2 * pi / 365).subtract(1.39435).sin().multiply(0.40928)
    sc_b = acq_doy.subtract(81).multiply(2 * pi / 364)
    sc = sc_b.multiply(2).sin().multiply(0.1645) \
        .subtract(sc_b.cos().multiply(0.1255)) \
        .subtract(sc_b.sin().multiply(0.025))
    solar_time = lon.multiply(12 / pi).add(acq_time).add(sc)
    # solar_time = lon.expression(
    #     't + (lon * 12 / pi) + sc',
    #     {'pi':pi, 't':ee.Image.constant(acq_time),
    #      'lon':lon, 'sc':ee.Image.constant(sc)})
    omega = solar_time.subtract(12).multiply(pi / 12)
    cos_theta = lat.expression(
        'sin(delta) * sin(lat) + cos(delta) * cos(lat) * cos(omega)',
        {'delta': ee.Image.constant(delta), 'lat': lat, 'omega': omega})
    return cos_theta.select([0], ['cos_theta'])


def cos_theta_mountain_func(acq_doy, acq_time, lat=None, lon=None,
                            slope=None, aspect=None):
    """Cos(theta) - Spatially varying moutain model

    Args:
        acq_doy: EarthEngine number of the image acquisition day of year
            scene_date = ee.Algorithms.Date(ee_image.get('system:time_start'))
            acq_doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
        acq_time: EarthEngine number of the image acquisition UTC time in hours
            i.e. 18:30 -> 18.5
            Calcuatl
            scene_date = ee.Algorithms.Date(ee_image.get('system:time_start'))
            acq_time = ee.Number(scene_date.getFraction('day')).multiply(24)
        lat: EarthEngine image of the latitude [radians]
            lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi/180)
        lon: EarthEngine image of the longitude [radians]
            lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi/180)
        slope: EarthEngine image of the slope [radians]
            terrain = ee.call('Terrain', ee.Image("USGS/NED"))
            slope = terrain.select(["slope"]).multiply(pi/180)
        aspect: EarthEngine image of the aspect [radians]
            0 is south, so subtract Pi from traditional aspect raster/calc
            terrain = ee.call('Terrain', ee.Image("USGS/NED"))
            aspect = terrain.select(["aspect"]).multiply(pi/180).subtract(math.pi)

    Returns:
        ee.Image()
    """
    pi = math.pi
    if lat is None:
        lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi / 180)
    if lon is None:
        lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi / 180)
    if slope is None or aspect is None:
        terrain = ee.call('Terrain', ee.Image('USGS/NED'))
    if slope is None:
        slope = terrain.select(['slope']).multiply(pi / 180)
    if aspect is None:
        aspect = terrain.select(['aspect']).multiply(pi / 180).subtract(pi)
    delta = acq_doy.multiply(2 * math.pi / 365).subtract(1.39435).sin().multiply(0.40928)
    b = acq_doy.subtract(81).multiply(2 * pi / 364)
    sc = b.multiply(2).sin().multiply(0.1645)\
        .subtract(b.cos().multiply(0.1255))\
        .subtract(b.sin().multiply(0.025))
    solar_time = lon.multiply(12 / pi).add(acq_time).add(sc)
    # solar_time = lon.expression(
    #   't + (lon * 12 / pi) + sc',
    #   {'pi':pi, 't':ee.Image.constant(acq_time),
    #    'lon':lon, 'sc':ee.Image.constant(sc)})
    omega = solar_time.subtract(12).multiply(pi / 12)
    slope_c = slope.cos()
    slope_s = slope.sin()
    cos_theta = lat.expression(
        '(sin(lat) * slope_c * delta_s) - '
        '(cos(lat) * slope_s * cos(aspect) * delta_s) + '
        '(cos(lat) * slope_c * cos(omega) * delta_c) + '
        '(sin(lat) * slope_s * cos(aspect) * cos(omega) * delta_c) + '
        '(sin(aspect) * slope_s * sin(omega) * delta_c)',
        {'lat': lat, 'aspect': aspect,
         'slope_c': slope_c, 'slope_s': slope_s, 'omega': omega,
         'delta_c': ee.Image.constant(delta.cos()),
         'delta_s': ee.Image.constant(delta.sin())})
    cos_theta = cos_theta.divide(slope_c).max(ee.Image.constant(0.1))
    return cos_theta.select([0], ['cos_theta'])


def refl_sur_tasumi_func(refl_toa, pair, ea, cos_theta, landsat,
                         adjust_method=''):
    """Tasumi at-surface reflectance

    Args:
        refl_toa (ee.Image):
        pair (ee.Image):
        ea (ee.Image):
        cos_theta (ee.Image):
        landsat (str): Landsat type
        adjust_method (str): Adjust Landsat red and NIR bands.
            Choices are 'etm_2_oli' or 'oli_2_etm'.
            This could probably be simplifed to a flag

    Returns:
        ee.Image: at-surface reflectance
    """
    w = pair.multiply(0.14).multiply(ea).add(2.1)
    if landsat.upper() in ['LT5', 'LT4', 'LE7']:
        c1 = [0.987, 2.319, 0.951, 0.375, 0.234, 0.365]
        c2 = [-0.00071, -0.000164, -0.000329, -0.000479, -0.001012, -0.000966]
        c3 = [0.000036, 0.000105, 0.00028, 0.005018, 0.004336, 0.004296]
        c4 = [0.088, 0.0437, 0.0875, 0.1355, 0.056, 0.0155]
        c5 = [0.0789, -1.2697, 0.1014, 0.6621, 0.7757, 0.639]
        cb = [0.640, 0.31, 0.286, 0.189, 0.274, -0.186]
        # c1 = ee.Image([0.987, 2.319, 0.951, 0.375, 0.234, 0.365])
        # c2 = ee.Image([-0.00071, -0.000164, -0.000329, -0.000479, -0.001012, -0.000966])
        # c3 = ee.Image([0.000036, 0.000105, 0.00028, 0.005018, 0.004336, 0.004296])
        # c4 = ee.Image([0.088, 0.0437, 0.0875, 0.1355, 0.056, 0.0155])
        # c5 = ee.Image([0.0789, -1.2697, 0.1014, 0.6621, 0.7757, 0.639])
        # cb = ee.Image([0.640, 0.31, 0.286, 0.189, 0.274, -0.186])
    elif landsat.upper() == 'LC8':
        c1 = [0.987, 2.148, 0.942, 0.248, 0.260, 0.315]
        c2 = [-0.000727, -0.000199, -0.000261, -0.000410, -0.001084, -0.000975]
        c3 = [0.000037, 0.000058, 0.000406, 0.000563, 0.000675, 0.004012]
        c4 = [0.0869, 0.0464, 0.0928, 0.2256, 0.0632, 0.0116]
        c5 = [0.0788, -1.0962, 0.1125, 0.7991, 0.7549, 0.6906]
        cb = [0.640, 0.310, 0.286, 0.189, 0.274, -0.186]
        # c1 = ee.Image([0.987, 2.148, 0.942, 0.248, 0.260, 0.315])
        # c2 = ee.Image([-0.000727, -0.000199, -0.000261, -0.000410, -0.001084, -0.000975])
        # c3 = ee.Image([0.000037, 0.000058, 0.000406, 0.000563, 0.000675, 0.004012])
        # c4 = ee.Image([0.0869, 0.0464, 0.0928, 0.2256, 0.0632, 0.0116])
        # c5 = ee.Image([0.0788, -1.0962, 0.1125, 0.7991, 0.7549, 0.6906])
        # cb = ee.Image([0.640, 0.310, 0.286, 0.189, 0.274, -0.186])

    # Incoming/outgoing narrowband transmittance
    # IN  (C1*exp(((C2*pair)/(Kt*cos_theta))-((C3*W+C4)/cos_theta))+C5)
    # OUT (C1*exp(((C2*pair)/(Kt*1.0))-((C3*W+C4)/1.0))+C5)
    # These broke when I made them expressions, need to try again
    tau_in = pair.multiply(c2).subtract(w.multiply(c3)).subtract(c4) \
        .divide(cos_theta).exp().multiply(c1).add(c5)
    tau_out = pair.multiply(c2).subtract(w.multiply(c3)).subtract(c4) \
        .exp().multiply(c1).add(c5)
    refl_sur = ee.Image(refl_toa).select(refl_bands) \
        .expression(
            '(b() + cb * (tau_in - 1.0)) / (tau_in * tau_out)',
            {'cb': cb, 'tau_in': tau_in, 'tau_out': tau_out})

    if (adjust_method.upper() == 'etm_2_oli' and
            landsat.upper() in ['LT5', 'LT4', 'LE7']):
        # http://www.sciencedirect.com/science/article/pii/S0034425716302619
        # Coefficients for scaling ETM+ to OLI
        refl_sur = ee.Image(refl_sur) \
            .subtract([0, 0, 0.0024, -0.0003, 0, 0]) \
            .divide([1, 1, 1.0047, 1.0036, 1, 1])
    elif (adjust_method.upper() == 'oli_2_etm' and
            landsat.upper() in ['LC8']):
        # http://www.sciencedirect.com/science/article/pii/S0034425716302619
        # Coefficients for scaling OLI to ETM+
        refl_sur = ee.Image(refl_sur) \
            .multiply([1, 1, 1.0047, 1.0036, 1, 1]) \
            .add([0, 0, 0.0024, -0.0003, 0, 0])

    return refl_sur.clamp(0.0001, 1).copyProperties(
        refl_toa, system_properties)


def albedo_func(refl_sur, landsat):
    """At-surface albedo"""
    if landsat.upper() in ['LT5', 'LT4']:
        wb_coef = [0.254, 0.149, 0.147, 0.311, 0.103, 0.036]
    elif landsat.upper() == 'LE7':
        wb_coef = [0.254, 0.149, 0.147, 0.311, 0.103, 0.036]
    elif landsat.upper() == 'LC8':
        wb_coef = [0.254, 0.149, 0.147, 0.311, 0.103, 0.036]
    return ee.Image(refl_sur).select(refl_bands).multiply(wb_coef) \
        .reduce(ee.Reducer.sum())


def landsat_ndvi_func(img):
    """Calculate NDVI for a daily Landsat 4, 5, 7, or 8 image"""
    # Removed .clamp(-0.1, 1)
    return ee.Image(img)\
        .normalizedDifference(['nir', 'red']).select([0], ['NDVI'])\
        .copyProperties(img, system_properties)


def landsat_savi_func(refl_image, L=0.1):
    """Soil adjusted vegetation index (SAVI)"""
    return refl_image.expression(
        '(1.0 + L) * (NIR - RED) / (L + NIR + RED)',
        {'RED': refl_image.select('red'),
         'NIR': refl_image.select('nir'), 'L': L})

# def savi_func(refl_image, L=0.1):
#     """Soil adjusted vegetation index (SAVI)"""
#     return refl_image.expression(
#         '(1.0 + L) * (NIR - RED) / (L + NIR + RED)',
#         {'RED': refl_image.select('red'),
#          'NIR': refl_image.select('nir'), 'L': L})


def savi_lai_func(savi):
    """Leaf area index (LAI) calculated from SAVI"""
    return savi.pow(3).multiply(11.0).clamp(0, 6)


def ndvi_lai_func(ndvi):
    """Leaf area index (LAI) calculated from NDVI"""
    return ndvi.pow(3).multiply(7.0).clamp(0, 6)


def landsat_evi_func(img):
    """Calculate EVI for a daily Landsat 4, 5, 7, or 8 image"""
    evi = ee.Image(img).expression(
        '(2.5 * (b("nir") - b("red"))) / '
        '(b("nir") + 6 * b("red") - 7.5 * b("blue") + 1)')
    return evi.select([0], ['EVI']).copyProperties(
        img, system_properties)


def etstar_func(evi, c0, c1, c2):
    """Beamer ET*"""
    etstar = ee.Image(evi).expression(
        'c0 + c1 * evi + c2 * (evi ** 2)',
        {'evi': evi, 'c0': c0, 'c1': c1, 'c2': c2})
    return etstar.max(0)


def etg_func(etstar, eto, ppt):
    return etstar.multiply(eto.subtract(ppt))


def et_func(etg, ppt):
    return etg.add(ppt)


# def tasseled_cap_func(refl_toa):
#     refl_toa_sub = refl_toa.select(refl_bands)
#     tc_bright_coef = ee.List(refl_toa.get('tc_bright'))
#     tc_green_coef = ee.List(refl_toa.get('tc_green'))
#     tc_wet_coef = ee.List(refl_toa.get('tc_wet'))
#     return ee.Image([
#         refl_toa_sub.multiply(tc_bright_coef).reduce(ee.Reducer.sum()),
#         refl_toa_sub.multiply(tc_green_coef).reduce(ee.Reducer.sum()),
#         refl_toa_sub.multiply(tc_wet_coef).reduce(ee.Reducer.sum())])\
#         .select([0, 1, 2], ['tc_bright', 'tc_green', 'tc_wet'])


def tc_bright_func(refl_toa, landsat='LE7'):
    """Tasseled cap brightness

    Top of atmosphere (at-satellite) reflectance

    LT5 - http://www.gis.usu.edu/~doug/RS5750/assign/OLD/RSE(17)-301.pdf
    LE7 - http://landcover.usgs.gov/pdf/tasseled.pdf
    LC8 - http://www.tandfonline.com/doi/abs/10.1080/2150704X.2014.915434
    https://www.researchgate.net/publication/262005316_Derivation_of_a_tasselled_cap_transformation_based_on_Landsat_8_at-_satellite_reflectance
    """
    if landsat in ['LT4', 'LT5']:
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_bright_coef = [0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303]
    elif landsat == 'LE7':
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_bright_coef = [0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0.1596]
    elif landsat == 'LC8':
        refl_toa_sub = refl_toa.select(refl_bands)
        # refl_toa_sub = refl_toa_sub.multiply(0.0001)
        tc_bright_coef = [0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872]
    return refl_toa_sub.multiply(tc_bright_coef).reduce(ee.Reducer.sum())


def tc_green_func(refl_toa, landsat='LE7'):
    """Tasseled cap greeness"""
    if landsat in ['LT4', 'LT5']:
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_green_coef = [-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446]
    elif landsat == 'LE7':
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_green_coef = [-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, -0.2630]
    elif landsat == 'LC8':
        refl_toa_sub = refl_toa.select(refl_bands)
        # refl_toa_sub = refl_toa_sub.multiply(0.0001)
        tc_green_coef = [-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608]
    return refl_toa_sub.multiply(tc_green_coef).reduce(ee.Reducer.sum())


def tc_wet_func(refl_toa, landsat='LE7'):
    """Tasseled cap wetness"""
    if landsat in ['LT4', 'LT5']:
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_wet_coef = [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]
    elif landsat == 'LE7':
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_wet_coef = [0.2626, 0.2141, 0.0926, 0.0656, -0.7629, -0.5388]
    elif landsat == 'LC8':
        refl_toa_sub = refl_toa.select(refl_bands)
        # refl_toa_sub = refl_toa_sub.multiply(0.0001)
        tc_wet_coef = [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559]
    return refl_toa_sub.multiply(tc_wet_coef).reduce(ee.Reducer.sum())


def em_nb_func(ndvi, lai):
    """Narrowband emissivity"""
    # Initial values are for NDVI > 0 and LAI <= 3
    return lai.divide(300).add(0.97) \
        .where(ndvi.lte(0), 0.99) \
        .where(ndvi.gt(0).And(lai.gt(3)), 0.98)


def em_wb_func(ndvi, lai):
    """Broadband emissivity"""
    # Initial values are for NDVI > 0 and LAI <= 3
    return lai.divide(100).add(0.95) \
        .where(ndvi.lte(0), 0.985) \
        .where(ndvi.gt(0).And(lai.gt(3)), 0.98)


def ts_func(ts_brightness, em_nb, k1=607.76, k2=1260.56):
    """Surface temperature"""
    # First back out radiance from brightness temperature
    # Then recalculate emissivity corrected Ts
    thermal_rad_toa = ts_brightness.expression(
        'k1 / (exp(k2 / ts_brightness) - 1.0)',
        {'ts_brightness': ts_brightness, 'k1': k1, 'k2': k2})
    rc = thermal_rad_toa.expression(
        '((thermal_rad_toa - rp) / tnb) - ((1.0 - em_nb) * rsky)',
        {"thermal_rad_toa": thermal_rad_toa, "em_nb": em_nb,
         "rp": 0.91, "tnb": 0.866, 'rsky': 1.32})
    ts = rc.expression(
        'k2 / log(em_nb * k1 / rc + 1.0)',
        {'em_nb': em_nb, 'rc': rc, 'k1': k1, "k2": k2})
    return ts


def landsat_true_color_func(img):
    """Calculate true color for a daily Landsat 4, 5, 7, or 8 image"""
    return ee.Image(img.select(['blue', 'green', 'red']))\
        .copyProperties(img, system_properties)


def landsat_false_color_func(img):
    """Calculate false color for a daily Landsat 4, 5, 7, or 8 image"""
    return ee.Image(img.select(['green', 'red', 'nir']))\
        .copyProperties(img, system_properties)


def nldas_interp_func(img):
    """Interpolate NLDAS image at Landsat scene time

    Args:
        img (ee.Image()):
            NLDAS hourly image collection must have been joined to it
            Previous NLDAS image must be selectable with 'nldas_prev_match'
            Next NLDAS image must be selectable with 'nldas_next_match'

    Returns
        ee.Image(): NLDAS values interpolated at the image time
    """
    scene_time = ee.Number(img.get('system:time_start'))
    nldas_prev_image = ee.Image(img.get('nldas_prev_match'))
    nldas_next_image = ee.Image(img.get('nldas_next_match'))
    nldas_prev_time = ee.Number(nldas_prev_image.get('system:time_start'))
    nldas_next_time = ee.Number(nldas_next_image.get('system:time_start'))

    # Calculate time ratio of Landsat image between NLDAS images
    time_ratio = scene_time.subtract(nldas_prev_time).divide(
        nldas_next_time.subtract(nldas_prev_time))
    # time_ratio_image = ee.Image.constant(scene_time.subtract(nldas_prev_time) \
    #     .divide(nldas_next_time.subtract(nldas_prev_time)))

    # Interpolate NLDAS values at Landsat image time
    return nldas_next_image.subtract(nldas_prev_image) \
        .multiply(time_ratio).add(nldas_prev_image) \
        .setMulti({
            'system:time_start': scene_time,
            'system:time_end': scene_time})


def pair_func(elev_image):
    """Elevation based air pressure"""
    return elev_image.expression(
        '101.3 * pow((293 - 0.0065 * b()) / 293, 5.26)')


def vapor_pressure_func(temperature_image):
    """Vapor Pressure

    in kPa with temperature in C
    """
    return temperature_image.expression(
        '0.6108 * exp(17.27 * b() / (b() + 237.3))')


def prism_ppt_func(prism_image):
    """PRISM water year precipitation

    Depends on maps engine assets
    """
    return prism_image.select([0], ['PPT']).copyProperties(
        prism_image, system_properties)


def gridmet_ppt_func(gridmet_image):
    """GRIDMET daily precipitation"""
    return gridmet_image.select(["pr"], ['PPT']).copyProperties(
        gridmet_image, system_properties)


def gridmet_etr_func(gridmet_image):
    """GRIDMET Daily ETr"""
    scene_date = ee.Algorithms.Date(gridmet_image.get('system:time_start'))
    doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()

    # Read in GRIDMET layers
    tmin = gridmet_image.select(['tmmn']).subtract(273.15)  # K to C
    tmax = gridmet_image.select(['tmmx']).subtract(273.15)  # K to C
    # rhmin = gridmet_image.select(['rmin']).multiply(0.01)  # % to decimal
    # rhmax = gridmet_image.select(['rmax']).multiply(0.01)  # % to decimal
    q = gridmet_image.select(['sph'])                      # kg kg-1
    rs = gridmet_image.select(['srad']).multiply(0.0864)   # W m-2 to MJ m-2 day-1
    uz = gridmet_image.select(['vs'])                      # m/s?
    zw = 10.0    # Windspeed measurement/estimated height (GRIDMET=10m)

    # Vapor pressure from RHmax and RHmin (Eqn 11)
    # ea = es_tmin.multiply(rhmax).add(es_tmax.multiply(rhmin)).multiply(0.5)
    # Vapor pressure from specific humidity (Eqn )
    # To match standardized form, ea is calculated from elevation based pair
    pair = pair_func(ee.Image('USGS/NED'))
    ea = pair.expression(
        'q * pair / (0.622 + 0.378 * q)', {'pair': pair, 'q': q})

    return daily_pet_func(
        doy, tmin, tmax, ea, rs, uz, zw, 1600, 0.38).copyProperties(
            gridmet_image, system_properties)


def gridmet_eto_func(gridmet_image):
    """GRIDMET Daily ETo"""
    scene_date = ee.Algorithms.Date(gridmet_image.get('system:time_start'))
    doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()

    # Read in GRIDMET layers
    tmin = gridmet_image.select(['tmmn']).subtract(273.15)  # K to C
    tmax = gridmet_image.select(['tmmx']).subtract(273.15)  # K to C
    # rhmin = gridmet_image.select(['rmin']).multiply(0.01)  # % to decimal
    # rhmax = gridmet_image.select(['rmax']).multiply(0.01)  # % to decimal
    q = gridmet_image.select(['sph'])                      # kg kg-1
    rs = gridmet_image.select(['srad']).multiply(0.0864)   # W m-2 to MJ m-2 day-1
    uz = gridmet_image.select(['vs'])                      # m/s?
    zw = 10.0  # Windspeed measurement/estimated height (GRIDMET=10m)

    # Vapor pressure from RHmax and RHmin (Eqn 11)
    # ea = es_tmin.multiply(rhmax).add(es_tmax.multiply(rhmin)).multiply(0.5)
    # Vapor pressure from specific humidity (Eqn )
    # To match standardized form, ea is calculated from elevation based pair
    pair = pair_func(ee.Image('USGS/NED'))
    ea = pair.expression(
        'q * pair / (0.622 + 0.378 * q)', {'pair': pair, 'q': q})

    return daily_pet_func(
        doy, tmin, tmax, ea, rs, uz, zw, 900, 0.34).copyProperties(
            gridmet_image, system_properties)


def daily_pet_func(doy, tmin, tmax, ea, rs, uz, zw, cn=900, cd=0.34):
    """Daily ASCE Penman Monteith Standardized Reference ET

    Daily ETo cn=900, cd=0.34
    Daily ETr cn=1600, cd=0.38

    doy -- day of year
    tmin -- minimum daily temperature [C]
    tmax -- maximum daily temperature [C]
    ea -- vapor pressure [?]
    rs -- incoming solar radiation [MJ m-2 day]
    uz -- wind speed [m s-1]
    zw -- wind speed height [m]
    cn -- coefficient
    cd -- coefficient

    """
    # Globals in playground/javascript
    pi = math.pi
    lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi / 180)
    lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi / 180)
    elev = ee.Image('USGS/NED')
    pair = pair_func(elev)

    # Calculations
    tmean = tmin.add(tmax).multiply(0.5)  # C
    psy = pair.multiply(0.000665)
    es_tmax = vapor_pressure_func(tmax)  # C
    es_tmin = vapor_pressure_func(tmin)  # C
    es_tmean = vapor_pressure_func(tmean)
    es_slope = es_tmean.expression(
        '4098 * es / (pow((t + 237.3), 2))', {'es': es_tmean, 't': tmean})
    es = es_tmin.add(es_tmax).multiply(0.5)

    # Extraterrestrial radiation (Eqn 24, 27, 23, 21)
    delta = ee.Image.constant(
        doy.multiply(2 * pi / 365).subtract(1.39435).sin().multiply(0.40928))
    omegas = lat.expression(
        'acos(-tan(lat) * tan(delta))', {'lat': lat, 'delta': delta})
    theta = omegas.expression(
        'omegas * sin(lat) * sin(delta) + cos(lat) * cos(delta) * sin(b())',
        {'omegas': omegas, 'lat': lat, 'delta': delta})
    dr = ee.Image.constant(
        doy.multiply(2 * pi / 365).cos().multiply(0.033).add(1))
    ra = theta.expression(
        '(24 / pi) * gsc * dr * theta',
        {'pi': pi, 'gsc': 4.92, 'dr': dr, 'theta': theta})

    # Simplified clear sky solar formulation (Eqn 19)
    # var rso = elev.expression(
    #     '(0.75 + 2E-5 * elev) * ra', {'elev':elev, 'ra':ra})

    # This is the full clear sky solar formulation
    # sin of the angle of the sun above the horizon (D.5 and Eqn 62)
    sin_beta_24 = lat.expression(
        'sin(0.85 + 0.3 * lat * delta / 0.40928 - 0.42 * lat ** 2)',
        {'lat': lat, 'delta': delta})

    # Precipitable water (Eqn D.3)
    w = pair.expression(
        '0.14 * ea * pair + 2.1', {'pair': pair, 'ea': ea})

    # Clearness index for direct beam radiation (Eqn D.2)
    # Limit sin_beta >= 0.01 so that KB does not go undefined
    kb = pair.expression(
        '0.98 * exp((-0.00146 * pair) / (kt * sin_beta) - '
        '0.075 * pow((w / sin_beta), 0.4))',
        {'pair': pair, 'kt': 1.0, 'sin_beta': sin_beta_24.max(0.01), 'w': w})

    # Transmissivity index for diffuse radiation (Eqn D.4)
    kd = kb.multiply(-0.36).add(0.35).min(kb.multiply(0.82).add(0.18))
    # var kd = kb.multiply(-0.36).add(0.35)
    #     .where(kb.lt(0.15), kb.multiply(0.82).add(0.18))

    # (Eqn D.1)
    rso = ra.multiply(kb.add(kd))
    # Cloudiness fraction (Eqn 18)
    fcd = rs.divide(rso).clamp(0.3, 1).multiply(1.35).subtract(0.35)

    # Net long-wave radiation (Eqn 17)
    rnl = ea.expression(
        ('4.901E-9 * fcd * (0.34 - 0.14 * sqrt(ea)) * '
         '(pow(tmax_k, 4) + pow(tmin_k, 4)) / 2'),
        {'ea': ea, 'fcd': fcd,
         'tmax_k': tmax.add(273.15), 'tmin_k': tmin.add(273.15)})

    # Net radiation (Eqns 15 and 16)
    rn = rs.multiply(0.77).subtract(rnl)

    # Wind speed (Eqn 33)
    u2 = uz.expression('b() * 4.87 / log(67.8 * zw - 5.42)', {'zw': zw})

    # Daily ETo (Eqn 1)
    return tmin.expression(
        '(0.408 * slope * (rn - g) + (psy * cn * u2 * (es - ea) / (t + 273))) / '
        '(slope + psy * (cd * u2 + 1))',
        {'slope': es_slope, 'rn': rn, 'g': 0, 'psy': psy, 'cn': cn,
         't': tmean, 'u2': u2, 'es': es, 'ea': ea, 'cd': cd})
