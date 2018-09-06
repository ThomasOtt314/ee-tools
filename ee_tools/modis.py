#--------------------------------
# Name:         modis.py
# Purpose:      Common EarthEngine MODIS functions
# Python:       3.6
#--------------------------------

import datetime
import logging
import pprint
import sys

import ee

system_properties = ['system:index', 'system:time_start']

# refl_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
# refl_bands = ['blue', 'green', 'red', 'nir']
# mod09ga_bands = ['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b01', 'sur_refl_b02']
# myd09ga_bands = ['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b01', 'sur_refl_b02']

daily_collections = [
    'MOD09GA', 'MYD09GA', 'MOD09GQ', 'MYD09GQ', 'MCD43A4',
    'MOD11A1', 'MYD11A1']


class MODIS():
    """"""

    def __init__(self, args):

        """Initialize the class with the user specified arguments

        All argument strings should be lower case

        Args: dictionary with the following key/values
            cloud_flag (bool): if True, mask Fmask cloud, shadow, and snow pixels
            start_date (str): ISO format start date (YYYY-MM-DD)
            end_date (str): ISO format end date (YYYY-MM-DD) (inclusive)
            start_year (int): start year
            end_year (int): end year
            start_month (int): start month
            end_month (int): end month
            start_doy (int): start day of year
            end_doy (int): end day of year
            zone_geom (ee.Geometry): apply filterBounds using this geometry
            products (list): MODIS bands to compute/return

        """
        arg_list = [
            'zone_geom', 'products', 'cloud_flag',
            'start_date', 'end_date', 'start_year', 'end_year',
            'start_month', 'end_month', 'start_doy', 'end_doy',

        ]
        int_args = [
            'start_year', 'end_year', 'start_month', 'end_month',
            'start_doy', 'end_doy'
        ]
        # list_args = ['products']

        # Set default products list if it was not set
        if 'products' not in args:
            args['products'] = []

        # # Set start and end date if they are not set
        # # This is needed for selecting MODIS collections below
        # if not args['start_date'] and args['start_year']:
        #     args['start_date'] = '{}-01-01'.format(args['start_year'])
        # elif not args['start_date'] and args['start_year']:
        #     args['start_date'] = '1982-01-01'
        # if not args['end_date'] and args['end_year']:
        #     args['end_date'] = '{}-12-31'.format(args['end_year'])
        # elif not args['end_date'] and args['end_date']:
        #     args['end_date'] = datetime.datetime.now().strftime('%Y-%m-%d')

        # logging.debug('  Init Args')
        for key in arg_list:
            try:
                if str(key) in int_args:
                    value = int(args[key])
                else:
                    value = args[key]
            except KeyError:
                # Argument was not passed in or set
                value = None
            except TypeError:
                # Argument is not integer type
                value = None
            setattr(self, str(key), value)
            # if key not in ['zone_geom']:
            #     logging.debug('  {}: {}'.format(key, value))

        # # MODIS list should not be directly set by the user
        # # It will be computed from the flags
        # self.set_modis_from_flags()

        # today = datetime.date.today().isoformat()
        # self.dates = {
        #     'MOD09GA': {'start': '2000-02-24', 'end': today},
        #     'MYD09GQ': {'start': '2002-07-04', 'end': today},
        #     'MOD09GA': {'start': '2000-02-24', 'end': today},
        #     'MYD09GQ': {'start': '2002-07-04', 'end': today},
        #     'MCD43A1': {'start': '2000-02-24', 'end': today},
        #     'MOD11A1': {'start': '2000-02-24', 'end': today},
        #     'MYD11A1': {'start': '2002-07-04', 'end': today},
        # }

    # def set_modis_from_products(self):
    #     """Set MODIS type list based on INI flags"""
    #     modis_list = []
    #     if self.mod09ga_flag:
    #         modis_list.append('MOD09GA')
    #     if self.myd09ga_flag:
    #         modis_list.append('MYD09GA')
    #     self._modis_list = sorted(modis_list)

    # def get_image(self, modis, year, doy):
    #     """Return a single MODIS image
    #
    #     Parameters
    #     ----------
    #     modis : str
    #     year : int
    #     doy : int
    #         Day of year.
    #
    #     Returns
    #     -------
    #     ee.Image
    #
    #     """
    #     image_start_dt = datetime.datetime.strptime(
    #         '{:04d}_{:03d}'.format(int(year), int(doy)), '%Y_%j')
    #     image_end_dt = image_start_dt + datetime.timedelta(days=1)
    #
    #     # Adjust the default keyword arguments for a single image date
    #     self.start_date = image_start_dt.date().isoformat()
    #     self.end_date = image_end_dt.date().isoformat()
    #     # self.start_year = year
    #     # self.end_dear = year
    #     # self.start_doy = doy
    #     # self.end_doy = doy
    #
    #     # Image collection for a single date
    #     output_coll = self.get_daily_collection()
    #     return ee.Image(output_coll.first())

    def get_daily_collection(self, product):
        """Build and filter a full MODIS daily collection

        Parameters
        ----------
        args : dict
            Keyword arguments for get_modis_collection.

        Returns
        -------
        ee.ImageCollection

        """
        # logging.debug('  Daily Product: {}'.format(product))

        # Load collection
        # I could build the collection name from the modis value directly
        if 'MOD09GA' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MOD09GA')
        elif 'MYD09GA' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MYD09GA')
        elif 'MOD09GQ' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MOD09GQ')
        elif 'MYD09GQ' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MYD09GQ')
        elif 'MOD11A1' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MOD11A1')
        elif 'MYD11A1' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MYD11A1')
        elif 'MCD43A4' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MCD43A4')
        else:
            raise ValueError('\nUnsupported MODIS collection: {}'.format(product))
        # Limit the collections to a later starting date
        # if 'mod' in modis.lower():
        #     modis_coll = modis_coll.filter(ee.Filter.gt(
        #         'system:time_start', ee.Date('2000-02-24').millis()))
        # elif 'myd' in modis.lower():
        #     modis_coll = modis_coll.filter(ee.Filter.gt(
        #         'system:time_start', ee.Date('2002-07-04').millis()))
        # elif 'mcd' in modis.lower():
        #     modis_coll = modis_coll.filter(ee.Filter.gt(
        #         'system:time_start', ee.Date('2000-02-24').millis()))

        # Assume iteration will generally be controlled by changing
        #   start_date and end_date
        if self.start_date and self.end_date:
            # End date is inclusive but filterDate is exclusive
            end_date = (
                    datetime.datetime.strptime(self.end_date, '%Y-%m-%d') +
                    datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            modis_coll = modis_coll.filterDate(self.start_date, end_date)
        # Filter by year
        if self.start_year and self.end_year:
            year_filter = ee.Filter.calendarRange(
                self.start_year, self.end_year, 'year')
            modis_coll = modis_coll.filter(year_filter)
        # Filter by month
        if ((self.start_month and self.start_month != 1) or
                (self.end_month and self.end_month != 12)):
            month_filter = ee.Filter.calendarRange(
                self.start_month, self.end_month, 'month')
            modis_coll = modis_coll.filter(month_filter)
        # Filter by day of year
        if ((self.start_doy and self.start_doy != 1) or
                (self.end_doy and self.end_doy != 365)):
            doy_filter = ee.Filter.calendarRange(
                self.start_doy, self.end_doy, 'day_of_year')
            modis_coll = modis_coll.filter(doy_filter)

        # Compute derived images
        if product.upper() in ['NDVI_MOD09GA', 'NDVI_MYD09GA',
                               'NDVI_MOD09GQ', 'NDVI_MYD09GQ']:
            output_coll = ee.ImageCollection(modis_coll.map(self.mod09_ndvi_func))\
                .select(['NDVI'], [product.upper()])
        elif product.upper() in ['NDVI_MCD43A4']:
            output_coll = ee.ImageCollection(modis_coll.map(self.mcd43_ndvi_func))\
                .select(['NDVI'], [product.upper()])
        elif product.upper() in ['LST_MOD11A1', 'LST_MYD11A1']:
            output_coll = ee.ImageCollection(modis_coll.map(self.mod11_lst_func))\
                .select(['NDVI'], [product.upper()])
        # elif variable == 'STATE_1KM':
        #     product_coll = ee.ImageCollection(
        #         modis_coll.map(self.mcd43_ndvi_func)) \
        #         .select(['ndvi'], [product])
        # elif variable == 'SensorZenith':
        #     product_coll = ee.ImageCollection(
        #         modis_coll.map(self.mcd43_ndvi_func)) \
        #         .select(['ndvi'], [product])
        else:
            raise ValueError('\nUnsupported MODIS product: {}'.format(product))
            sys.exit()

        # # Apply cloud masks
        # if self.cloud_flag:
        #     output_coll = output_coll.map(modis_state_qa_mask_func)
        # pprint.pprint(ee.Image(output_coll.first()).getInfo())

        return output_coll

    # def get_8day_collection(self):
    #     pass

    # def get_16day_collection(self):
    #     pass

    def mod09_ndvi_func(self, input_image):
        date = ee.Date(input_image.get('system:time_start')).format('yyyy-MM-dd')
        return input_image\
            .normalizedDifference(['sur_refl_b02', 'sur_refl_b01']) \
            .rename(['NDVI']) \
            .copyProperties(input_image, system_properties) \
            .set('DATE', date)

    def mcd43_ndvi_func(self, input_image):
        date = ee.Date(input_image.get('system:time_start')).format('yyyy-MM-dd')
        return input_image\
            .normalizedDifference(['Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band1']) \
            .rename(['NDVI'])\
            .copyProperties(input_image, system_properties) \
            .set('DATE', date)

    def mod11_lst_func(self, input_image):
        date = ee.Date(input_image.get('system:time_start')).format('yyyy-MM-dd')
        return input_image\
            .select(['LST_Day_1km'], ['LST'])\
            .copyProperties(input_image, system_properties) \
            .set('DATE', date)

    # def modis_images_func(self, input_image):
    #     """Calculate MODIS products
    #
    #     Parameters
    #     ----------
    #     input_image : ee.Image
    #         MODIS image.
    #
    #     Returns
    #     -------
    #     ee.Image
    #
    #     """
    #     output_images = []
    #
    #     # NDVI
    #     if 'ndvi' in self.products:
    #         ndvi = input_image.normalizedDifference(['nir', 'red']) \
    #             .rename(['ndvi'])
    #         output_images.append(ndvi)
    #
    #     # # EVI (used to compute ET*)
    #     # if ('evi_sur' in self.products or
    #     #         any([True for p in self.products if 'etstar_' in p]) or
    #     #         any([True for p in self.products if 'etg_' in p])):
    #     #     evi_sur = ee.Image(modis_evi_func(refl_sur)) \
    #     #         .rename(['evi_sur'])
    #     #     output_images.append(evi_sur)
    #     #
    #     # # NDWI - McFeeters 1996
    #     # if 'ndwi_green_nir_toa' in self.products:
    #     #     ndwi_green_nir_toa = refl_toa \
    #     #         .normalizedDifference(['green', 'nir']) \
    #     #         .rename(['ndwi_green_nir_toa'])
    #     #     output_images.append(ndwi_green_nir_toa)
    #     # if 'ndwi_green_nir_sur' in self.products:
    #     #     ndwi_green_nir_sur = refl_sur \
    #     #         .normalizedDifference(['green', 'nir']) \
    #     #         .rename(['ndwi_green_nir_sur'])
    #     #     output_images.append(ndwi_green_nir_sur)
    #     #
    #     # # NDWI - Xu 2006 (MNDWI) doi: 10.1080/01431160600589179
    #     # # Equivalent to NDSI Hall et al 1995 and 1998
    #     # # http://modis-snow-ice.gsfc.nasa.gov/uploads/pap_dev95.pdf
    #     # # http://modis-snow-ice.gsfc.nasa.gov/uploads/pap_assmnt98.pdf
    #     # if 'ndwi_green_swir1_toa' in self.products:
    #     #     ndwi_green_swir1_toa = refl_toa \
    #     #         .normalizedDifference(['green', 'swir1']) \
    #     #         .rename(['ndwi_green_swir1_toa'])
    #     #     output_images.append(ndwi_green_swir1_toa)
    #     # if 'ndwi_green_swir1_sur' in self.products:
    #     #     ndwi_green_swir1_sur = refl_sur \
    #     #         .normalizedDifference(['green', 'swir1']) \
    #     #         .rename(['ndwi_green_swir1_sur'])
    #     #     output_images.append(ndwi_green_swir1_sur)
    #     #
    #     # # NDWI - Gao 1996 doi: 10.1016/S0034-4257(96)00067-3
    #     # # Inverse of NDSI (Soil) in Rogers & Keraney 2004
    #     # if 'ndwi_nir_swir1_toa' in self.products:
    #     #     ndwi_nir_swir1_toa = refl_toa \
    #     #         .normalizedDifference(['nir', 'swir1']) \
    #     #         .rename(['ndwi_nir_swir1_toa'])
    #     #     output_images.append(ndwi_nir_swir1_toa)
    #     # if 'ndwi_nir_swir1_sur' in self.products:
    #     #     ndwi_nir_swir1_sur = refl_sur \
    #     #         .normalizedDifference(['nir', 'swir1']) \
    #     #         .rename(['ndwi_nir_swir1_sur'])
    #     #     output_images.append(ndwi_nir_swir1_sur)
    #     #
    #     # # NDWI - Allen 2007
    #     # # Return this NDWI as the default ndwi_sur and ndwi_toa below
    #     # if 'ndwi_swir1_green_toa' in self.products:
    #     #     ndwi_swir1_green_toa = refl_toa \
    #     #         .normalizedDifference(['swir1', 'green']) \
    #     #         .rename(['ndwi_swir1_green_toa'])
    #     #     output_images.append(ndwi_swir1_green_toa)
    #     # if 'ndwi_swir1_green_sur' in self.products:
    #     #     ndwi_swir1_green_sur = refl_sur \
    #     #         .normalizedDifference(['swir1', 'green']) \
    #     #         .rename(['ndwi_swir1_green_sur'])
    #     #     output_images.append(ndwi_swir1_green_sur)
    #     #
    #     # if 'ndwi_toa' in self.products:
    #     #     ndwi_toa = refl_toa \
    #     #         .normalizedDifference(['swir1', 'green']) \
    #     #         .rename(['ndwi_toa'])
    #     #     output_images.append(ndwi_toa)
    #     # if 'ndwi_sur' in self.products:
    #     #     ndwi_sur = refl_sur \
    #     #         .normalizedDifference(['swir1', 'green']) \
    #     #         .rename(['ndwi_sur'])
    #     #     output_images.append(ndwi_sur)
    #
    #     # # Surface temperature
    #     # if 'ts' in self.products:
    #     #     ts = ee.Image(ts_func(
    #     #         ts_brightness=input_image.select('lst'),
    #     #         em_nb=em_nb_func(ndvi_toa, lai_toa),
    #     #         k1=ee.Number(refl_toa.get('k1_constant')),
    #     #         k2=ee.Number(refl_toa.get('k2_constant'))))
    #     #     output_images.append(ts)
    #
    #     # # Add additional bands
    #     # output_images.extend([
    #     #     input_image.select('fmask'),
    #     # ])
    #
    #     return ee.Image(output_images) \
    #         .copyProperties(input_image, system_properties) \
    #         .set('SCENE_ID', input_image.get('id'))


# def modis_lst_func(img):
#     """Converted from unsigned 16-bit integer"""
#     variable = 'LST_Day_1km'
#     return img.select(variable).multiply(0.02) \
#         .select([0], [variable]) \
#         .copyProperties(img, property_list)


def modis_state_qa_mask_func(img):
    """Parse the MODIS State QA band to build a cloud mask"""

    def getQABits(image, start, end, newName):
        """
        Tyler's function from https://ee-api.appspot.com/#97ab9a8f694b28128a5a5ca2e2df7841
        """
        pattern = 0
        for i in range(start, end + 1):
            pattern += int(2 ** i)
        return image.select([0], [newName]) \
            .bitwise_and(pattern).right_shift(start)

    # Extract the various masks from the QA band
    qa_image = img.select('state_1km')
    mod35_cloud_mask = getQABits(qa_image, 0, 1, 'cloud_state') \
        .expression('b(0) == 1 || b(0) == 2')
    # shadow_mask = getQABits(qa_image, 2, 2, 'shadow_flag')
    ocean_mask = getQABits(qa_image, 3, 5, 'ocean_flag') \
        .expression('b(0) == 0 || b(0) >= 6')
    # 2 == average cirrus, 3 == high cirrus
    # cirrus_mask = getQABits(qa_image, 8, 9, 'cirrus_flag').gte(2)
    # internal_cloud_mask = getQABits(qa_image, 10, 10, 'internal_cloud_flag')
    # mod35_snow_mask = getQABits(qa_image, 12, 12, 'MOD35_snow_flag')
    # adjacent_mask = getQABits(qa_image, 13, 13, 'adjacent_flag')
    # internal_snow_mask = getQABits(qa_image, 15, 15, 'internal_snow_flag')

    # Masks are positive for clouds, so compute logical not for setting mask
    # cloud_mask = mod35_cloud_mask.Or(cirrus_mask).Not()
    cloud_mask = mod35_cloud_mask.Or(ocean_mask).Not()

    return img.updateMask(cloud_mask)
    #     .copyProperties(img, property_list)
