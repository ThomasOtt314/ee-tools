#--------------------------------
# Name:         modis.py
# Purpose:      Common EarthEngine MODIS functions
# Python:       3.6
#--------------------------------

import datetime
import pprint
import sys

import ee

system_properties = ['system:index', 'system:time_start']

collections_daily = ['MOD09GA', 'MYD09GA', 'MOD09GQ', 'MYD09GQ',
                     'MOD11A1', 'MYD11A1']
collections_8day = ['MOD11A2', 'MYD11A2']
collections_16day = ['MOD13Q1', 'MYD13Q1', 'MOD13A1', 'MYD13A1',
                     'MCD43A4']


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
            end_year (int): endyear
            start_month (int): start month
            end_month (int): end month
            start_doy (int): start day of year
            end_doy (int): end day of year
            zone_geom (ee.Geometry): apply filterBounds using this geometry
            products (list): MODIS bands to compute/return

        """
        arg_list = [
            'cloud_flag',
            'start_date', 'end_date', 'start_year', 'end_year',
            'start_month', 'end_month', 'start_doy', 'end_doy',
            'date_keep_list',

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
        product : str

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

        # Filter by date keep list
        if self.date_keep_list:
            date_keep_filter = ee.Filter.inList(
                'system:index', self.date_keep_list)
            modis_coll = modis_coll.filter(date_keep_filter)

        # Compute product images
        if product.upper() in ['NDVI_MOD09GA', 'NDVI_MYD09GA']:
            if self.cloud_flag:
                modis_coll = modis_coll.map(state_1km_mask_func)
            output_coll = ee.ImageCollection(modis_coll.map(mod09_ndvi_func))\
                .select(['NDVI'], [product.upper()])
        elif product.upper() in ['NDVI_MOD09GQ', 'NDVI_MYD09GQ']:
            output_coll = ee.ImageCollection(modis_coll.map(mod09_ndvi_func)) \
                .select(['NDVI'], [product.upper()])
        elif product.upper() in ['LST_MOD11A1', 'LST_MYD11A1']:
            output_coll = ee.ImageCollection(modis_coll.map(mod11_lst_func))\
                .select(['LST'], [product.upper()])
        elif product.upper() in ['CLOUD_MOD09GA', 'CLOUD_MYD09GA',
                                 'CLOUD_MOD09GQ', 'CLOUD_MYD09GQ']:
            output_coll = ee.ImageCollection(modis_coll.map(mod09_cloud_func)) \
                .select(['CLOUD'], [product])
        elif product.upper() in ['ZENITH_MOD09GA', 'ZENITH_MYD09GA',
                                 'ZENITH_MOD09GQ', 'ZENITH_MYD09GQ']:
            output_coll = ee.ImageCollection(modis_coll.map(mod09_zenith_func)) \
                .select(['ZENITH'], [product])
        else:
            raise ValueError('\nUnsupported MODIS product: {}'.format(product))
            sys.exit()

        return output_coll

    def get_8day_collection(self, product):
        """Build and filter a full MODIS daily collection

        Parameters
        ----------
        product : str

        Returns
        -------
        ee.ImageCollection

        """
        # logging.debug('  Daily Product: {}'.format(product))

        # Load collection
        # I could build the collection name from the modis value directly
        if 'MOD11A2' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MOD11A2')
        elif 'MYD11A2' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MYD11A2')
        else:
            raise ValueError('\nUnsupported MODIS collection: {}'.format(product))

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

        # Filter by date keep list
        if self.date_keep_list:
            date_keep_filter = ee.Filter.inList(
                'system:index', self.date_keep_list)
            modis_coll = modis_coll.filter(date_keep_filter)

        # # Apply cloud mask to input bands before computing product
        # if self.cloud_flag:
        #     if 'MOD11' in product.upper() or 'MYD11' in product.upper():
        #         modis_coll = modis_coll.map(state_1km_mask_func)

        # Compute product images
        if product.upper() in ['LST_MOD11A2', 'LST_MYD11A2']:
            output_coll = ee.ImageCollection(modis_coll.map(self.mod11_lst_func))\
                .select(['LST'], [product.upper()])
        else:
            raise ValueError('\nUnsupported MODIS product: {}'.format(product))
            sys.exit()

        return output_coll

    def get_16day_collection(self, product):
        """Build and filter a full MODIS daily collection

        Parameters
        ----------
        product : str

        Returns
        -------
        ee.ImageCollection

        """
        # logging.debug('  Daily Product: {}'.format(product))

        # Load collection
        # I could build the collection name from the modis value directly
        if 'MOD13A1' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MOD13A1')
        elif 'MYD13A1' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MYD13A1')
        elif 'MOD13Q1' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MOD13Q1')
        elif 'MYD13Q1' in product.upper():
            modis_coll = ee.ImageCollection('MODIS/006/MYD13Q1')
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

        # Filter by date keep list
        if self.date_keep_list:
            date_keep_filter = ee.Filter.inList(
                'system:index', self.date_keep_list)
            modis_coll = modis_coll.filter(date_keep_filter)

        # # Apply cloud mask to input bands before computing product
        # if self.cloud_flag:
        #     if 'MOD13' in product.upper() or 'MYD13' in product.upper():
        #         modis_coll = modis_coll.map(state_1km_mask_func)

        # Compute product images
        if product.upper() in ['NDVI_MOD13A1', 'NDVI_MYD13A1',
                               'NDVI_MOD13Q1', 'NDVI_MYD13Q1']:
            output_coll = ee.ImageCollection(modis_coll.map(mod13_ndvi_func))\
                .select(['NDVI'], [product.upper()])
        elif product.upper() in ['EVI_MOD13A1', 'EVI_MYD13A1',
                                 'EVI_MOD13Q1', 'EVI_MYD13Q1']:
            output_coll = ee.ImageCollection(modis_coll.map(mod13_evi_func)) \
                .select(['EVI'], [product.upper()])
        elif product.upper() in ['NDVI_MCD43A4']:
            output_coll = ee.ImageCollection(modis_coll.map(mcd43_ndvi_func))\
                .select(['NDVI'], [product.upper()])
        else:
            raise ValueError('\nUnsupported MODIS product: {}'.format(product))
            sys.exit()

        # # Apply cloud masks
        # if self.cloud_flag:
        #     output_coll = output_coll.map(modis_state_qa_mask_func)

        return output_coll

def mod09_ndvi_func(input_img):
    date = ee.Date(input_img.get('system:time_start')).format('yyyy-MM-dd')
    return input_img\
        .normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])\
        .rename(['NDVI'])\
        .copyProperties(input_img, system_properties)\
        .set('DATE', date)


def mod09_cloud_func(input_img):
    date = ee.Date(input_img.get('system:time_start')).format('yyyy-MM-dd')
    return state_1km_func(input_img).rename(['CLOUD'])\
        .copyProperties(input_img, system_properties)\
        .set('DATE', date)


def mod09_zenith_func(input_img):
    date = ee.Date(input_img.get('system:time_start')).format('yyyy-MM-dd')
    return input_img.select(['SensorZenith'], ['ZENITH']) \
        .multiply(0.01) \
        .copyProperties(input_img, system_properties)\
        .set('DATE', date)


def mod11_lst_func(input_img):
    date = ee.Date(input_img.get('system:time_start')).format('yyyy-MM-dd')
    return input_img.select(['LST_Day_1km'], ['LST'])\
        .multiply(0.02)\
        .copyProperties(input_img, system_properties)\
        .set('DATE', date)


def mod13_ndvi_func(input_img):
    date = ee.Date(input_img.get('system:time_start')).format('yyyy-MM-dd')
    return input_img.select(['NDVI'])\
        .copyProperties(input_img, system_properties)\
        .set('DATE', date)


def mod13_evi_func(input_img):
    date = ee.Date(input_img.get('system:time_start')).format('yyyy-MM-dd')
    return input_img.select(['EVI'])\
        .copyProperties(input_img, system_properties)\
        .set('DATE', date)


def mcd43_ndvi_func(input_img):
    date = ee.Date(input_img.get('system:time_start')).format('yyyy-MM-dd')
    return input_img\
        .normalizedDifference(['Nadir_Reflectance_Band2',
                               'Nadir_Reflectance_Band1'])\
        .rename(['NDVI'])\
        .copyProperties(input_img, system_properties)\
        .set('DATE', date)


def state_1km_mask_func(input_img):
    """Apply the MODIS State QA band cloud mask to the image"""
    # Masks are positive for clouds, so compute logical not for setting mask
    cloud_mask = state_1km_func(input_img).Not()
    return input_img.updateMask(cloud_mask)


def state_1km_func(input_img):
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
    qa_image = input_img.select(['state_1km'])
    mod35_cloud_mask = getQABits(qa_image, 0, 1, 'cloud_state') \
        .expression('b(0) == 1 || b(0) == 2')
    shadow_mask = getQABits(qa_image, 2, 2, 'shadow_flag')
    ocean_mask = getQABits(qa_image, 3, 5, 'ocean_flag') \
        .expression('b(0) == 0 || b(0) >= 6')
    # 2 == average cirrus, 3 == high cirrus
    cirrus_mask = getQABits(qa_image, 8, 9, 'cirrus_flag').gte(3)
    # internal_cloud_mask = getQABits(qa_image, 10, 10, 'internal_cloud_flag')
    # internal_fire_mask = getQABits(qa_image, 11, 11, 'internal_fire_flag')
    mod35_snow_mask = getQABits(qa_image, 12, 12, 'MOD35_snow_flag')
    adjacent_mask = getQABits(qa_image, 13, 13, 'adjacent_flag')
    # brdf_adjust_mask = getQABits(qa_image, 14, 14, 'brdf_adjust_mask')
    # internal_snow_mask = getQABits(qa_image, 15, 15, 'internal_snow_flag')

    return mod35_cloud_mask.Or(shadow_mask).Or(cirrus_mask).Or(ocean_mask)
    # return mod35_cloud_mask.Or(shadow_mask).Or(ocean_mask).Or(cirrus_mask)\
    #     .Or(adjacent_mask)
