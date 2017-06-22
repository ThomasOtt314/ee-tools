#--------------------------------
# Name:         ee_common.py
# Purpose:      Common EarthEngine support functions
# Author:       Charles Morton
# Created       2017-06-21
# Python:       3.6
#--------------------------------

import datetime
import logging
import math
import sys

import ee


ee.Initialize()

system_properties = ['system:index', 'system:time_start']

refl_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']


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


class Landsat():
    mosaic_options = ['min', 'max', 'median', 'mean', 'mosaic']
    cellsize = 30

    def __init__(self, args):
        """Initialize the class with the user specified arguments

        All argument strings should be lower case

        Args: dictionary with the following key/values
            refl_source (str):
                Choices: 'tasumi'
                Eventually support collection 1 at-surface reflectance
            fmask_source (str): FMask data source
                Choices: 'none', 'cfmask', 'fmask'
                if 'none',
                if 'fmask', get from EE dyanamic TOA Fmask collection
                if'cfmask', get from EE at-surface reflectance collection
            fmask_flag (bool): if True, mask Fmask cloud, shadow, and snow pixels
            acca_flag (bool): if True, mask pixels with clouds scores > 50
            start_date (str): ISO format start date (YYYY-MM-DD)
            end_date (str): ISO format end date (YYYY-MM-DD) (inclusive)
            start_year (int): start year
            end_year (int): end year
            start_month (int): start month
            end_month (int): end month
            start_doy (int): start day of year
            end_doy (int): end day of year
            zone_geom (ee.Geometry): apply filterBounds using this geometry
            scene_id_keep_list (list): SCENE_IDs to explicitly include
                SCENE_IDs do not include version or downlink station
                Example: "LT05_041032_1984214"
            scene_id_skip_list (list): SCENE_IDs to explicitly skip/exclude
                SCENE_IDs do not include version or downlink station
                Example: "LT05_041032_1984214"
            path_keep_list (list): Landsat path numbers (as int)
            row_keep_list (list): Landsat row numbers (as int)
            path_row_geom (ee.Geometry):
            adjust_method (str): Adjust Landsat red and NIR bands.
                Choices: 'etm_2_oli' or 'oli_2_etm'.
                This could probably be simplifed to a flag.
                This flag is passed through and not used directly in this function
            mosaic_method (str):
            products (list): Landsat bands to compute/return
            landsat4_flag (bool): if True, include Landsat 4 images
            landsat5_flag (bool): if True, include Landsat 5 images
            landsat7_flag (bool): if True, include Landsat 7 images
            landsat8_flag (bool): if True, include Landsat 8 images

        """
        arg_list = [
            'refl_source', 'fmask_source', 'fmask_flag', 'acca_flag',
            'start_date', 'end_date', 'start_year', 'end_year',
            'start_month', 'end_month', 'start_doy', 'end_doy',
            'zone_geom', 'scene_id_keep_list', 'scene_id_skip_list',
            'path_keep_list', 'row_keep_list', 'path_row_geom',
            'adjust_method', 'mosaic_method', 'products',
            'landsat4_flag', 'landsat5_flag',
            'landsat7_flag', 'landsat8_flag'
        ]
        int_args = [
            'start_year', 'end_year', 'start_month', 'end_month',
            'start_doy', 'end_doy'
        ]
        # list_args = ['products']

        if 'products' not in args:
            args['products'] = []

        # Currently only using TOA collections and comput Tasumi at-surface
        #   reflectance is supported
        if 'refl_type' not in args:
            args['refl_type'] = 'toa'

        # # Set start and end date if they are not set
        # # This is needed for selecting Landsat collections below
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

        # Is there a cleaner way of building a list of Landsat types
        #   from the flags?
        landsat_list = []
        if self.landsat4_flag:
            landsat_list.append('LT04')
        if self.landsat5_flag:
            landsat_list.append('LT05')
        if self.landsat7_flag:
            landsat_list.append('LE07')
        if self.landsat8_flag:
            landsat_list.append('LC08')
            landsat_list.append('LC08_PRE')
        self.landsat_list = landsat_list

        today = datetime.date.today().isoformat()
        self.dates = {
            'LT04': {'start': '1982-01-01', 'end': '1993-12-31'},
            'LT05': {'start': '1984-01-01', 'end': '2011-12-31'},
            'LE07': {'start': '1999-01-01', 'end': today},
            'LC08_PRE': {'start': '2013-01-01', 'end': '2015-01-01'},
            'LC08': {'start': '2015-01-01', 'end': today}
        }

    def get_image(self, landsat, year, doy, path=None, row=None):
        """Return a single Landsat image

        Mosaic images from different rows from the same date (same path)

        Args:
            landsat (str):
            year (int): year
            doy (int): day of year
            path (int): Landsat path number
            row (int): Landsat row number

        Returns:
            ee.Image
        """
        image_start_dt = datetime.datetime.strptime(
            '{:04d}_{:03d}'.format(int(year), int(doy)), '%Y_%j')
        image_end_dt = image_start_dt + datetime.timedelta(days=1)

        # Adjust the default keyword arguments for a single image date
        self.start_date = image_start_dt.date().isoformat()
        self.end_date = image_end_dt.date().isoformat()
        # self.start_year = year
        # self.end_dear = year
        # self.start_doy = doy
        # self.end_doy = doy
        if path:
            self.path_keep_list = [int(path)]
        if row:
            self.row_keep_list = [int(row)]
        # if path and row:
        #     self.pathrow_keep_list = [
        #         'p{:03d}r{:03d}'.format(int(path), int(row))]

        # Landsat collection for a single date
        landsat_coll = self.get_collection()
        return ee.Image(landsat_coll.first())

    def get_collection(self):
        """Build and filter a full Landsat collection

        Args:
            args (dict): keyword arguments for get_landst_collection

        Returns:
            ee.ImageCollection
        """

        # Process each Landsat type and append to the output collection
        output_coll = ee.ImageCollection([])
        for landsat in self.landsat_list:
            # Assume ieration will be controlled by changing start_date and end_date
            # Skip Landsat collections that are outside these date ranges
            if self.end_date and self.end_date < self.dates[landsat]['start']:
                continue
            elif self.start_date and self.start_date > self.dates[landsat]['end']:
                continue
            # Is it necessary or helpful to check year also?
            elif (self.end_year and
                    self.end_year < int(self.dates[landsat]['start'][:4])):
                continue
            elif (self.start_year and
                    self.start_year > int(self.dates[landsat]['end'][:4])):
                continue
            # logging.debug('  Landsat: {}'.format(landsat))

            if landsat in ['LE07', 'LC08']:
                # Collection 1
                # landsat_sr_name = 'LANDSAT/{}/C01/T1_SR'.format(landsat)
                landsat_toa_name = 'LANDSAT/{}/C01/T1_RT_TOA'.format(landsat)

                # Currently only using TOA collection with Tasumi at-surface
                #   reflectance is supported
                self.refl_source = 'tasumi'
                if self.refl_source == 'tasumi':
                    landsat_coll = ee.ImageCollection(landsat_toa_name)
                    # Keep fmask_coll for filtering below
                    # Once Collection 1 is fully supported, this could be removed
                    # fmask_coll = ee.ImageCollection(landsat_toa_name)
                    # fmask_coll = ee.ImageCollection([])
                else:
                    logging.error(
                        '\nERROR: Unknown Landsat/Fmask type combination, exiting\n'
                        '  Landsat: {}  Reclectance: {}  Fmask: {}'.format(
                            landsat, self.refl_source, self.fmask_source))
                    sys.exit()

            elif landsat in ['LT04', 'LT05', 'LC08_PRE']:
                # Pre-collection
                landsat_pre = landsat.replace('0', '').replace('_PRE', '')
                landsat_sr_name = 'LANDSAT/{}_SR'.format(landsat_pre)
                landsat_toa_name = 'LANDSAT/{}_L1T_TOA'.format(landsat_pre)
                landsat_fmask_name = 'LANDSAT/{}_L1T_TOA_FMASK'.format(
                    landsat_pre)

                # Currently only using TOA collection with Tasumi at-surface
                #   reflectance is supported
                self.refl_source = 'tasumi'
                if (self.refl_source == 'tasumi' and
                        (not self.fmask_source or self.fmask_source == 'none')):
                    landsat_coll = ee.ImageCollection(landsat_toa_name)
                    # Add empty fmask band
                    landsat_coll = landsat_coll.map(
                        landsat_empty_fmask_band_func)
                    # Build fmask_coll so filtering is cleaner, but don't use it
                    fmask_coll = ee.ImageCollection(landsat_sr_name) \
                        .select(['cfmask'], ['fmask'])
                elif (self.refl_source == 'tasumi' and
                        self.fmask_source == 'cfmask'):
                    # Join Fmask band from SR collection to TOA collection
                    landsat_coll = ee.ImageCollection(landsat_toa_name)
                    fmask_coll = ee.ImageCollection(landsat_sr_name) \
                        .select(['cfmask'], ['fmask'])
                elif (self.refl_source == 'tasumi' and
                        self.fmask_source == 'fmask'):
                    landsat_coll = ee.ImageCollection(landsat_fmask_name)
                    # This fmask collection will not be used
                    fmask_coll = ee.ImageCollection(landsat_sr_name) \
                        .select(['cfmask'], ['fmask'])
                else:
                    logging.error(
                        '\nERROR: Unknown Landsat/Fmask type combination, exiting\n'
                        '  Landsat: {}  Reclectance: {}  Fmask: {}'.format(
                            landsat, self.refl_source, self.fmask_source))
                    sys.exit()

            # Filter non-L1T/L1TP images
            # There are a couple of non-L1TP images in LE07 collection 1
            if landsat in ['LE07']:
                landsat_coll = landsat_coll.filterMetadata(
                    'DATA_TYPE', 'equals', 'L1TP')
            # if landsat in ['LE07', 'LC08']:
            #     landsat_coll = landsat_coll.filterMetadata(
            #         'DATA_TYPE', 'equals', 'L1TP')
            # if landsat in ['LT04', 'LT05', 'LC08_PRE']:
            #     landsat_coll = landsat_coll.filterMetadata(
            #         'DATA_TYPE', 'equals', 'L1T')

            # Exclude 2012 Landsat 5 images
            if landsat in ['LT05']:
                landsat_coll = landsat_coll.filter(
                    ee.Filter.calendarRange(1984, 2011, 'year'))
                fmask_coll = fmask_coll.filter(
                    ee.Filter.calendarRange(1984, 2011, 'year'))

            # DEADBEEF - Landsat 8 collection 1 is not fully ingested
            if landsat in ['LC08_PRE']:
                landsat_coll = landsat_coll.filter(
                    ee.Filter.calendarRange(2013, 2015, 'year'))
                fmask_coll = fmask_coll.filter(
                    ee.Filter.calendarRange(2013, 2015, 'year'))
            if landsat in ['LC08']:
                landsat_coll = landsat_coll.filter(
                    ee.Filter.calendarRange(2015, 2020, 'year'))

            # Filter by date
            if self.start_date and self.end_date:
                landsat_coll = landsat_coll.filterDate(
                    self.start_date, self.end_date)
                if landsat in ['LT04', 'LT05', 'LC08_PRE']:
                    fmask_coll = fmask_coll.filterDate(
                        self.start_date, self.end_date)
            # Filter by year
            if self.start_year and self.end_year:
                year_filter = ee.Filter.calendarRange(
                    self.start_year, self.end_year, 'year')
                landsat_coll = landsat_coll.filter(year_filter)
                if landsat in ['LT04', 'LT05', 'LC08_PRE']:
                    fmask_coll = fmask_coll.filter(year_filter)
            # Filter by month
            if ((self.start_month and self.start_month != 1) and
                    (self.end_month and self.end_month != 12)):
                month_filter = ee.Filter.calendarRange(
                    self.start_month, self.end_month, 'month')
                landsat_coll = landsat_coll.filter(month_filter)
                if landsat in ['LT04', 'LT05', 'LC08_PRE']:
                    fmask_coll = fmask_coll.filter(month_filter)
            # Filter by day of year
            if ((self.start_doy and self.start_doy != 1) and
                    (self.end_doy and self.end_doy != 365)):
                doy_filter = ee.Filter.calendarRange(
                    self.start_doy, self.end_doy, 'day_of_year')
                landsat_coll = landsat_coll.filter(doy_filter)
                if landsat in ['LT04', 'LT05', 'LC08_PRE']:
                    fmask_coll = fmask_coll.filter(doy_filter)

            if self.path_keep_list:
                # path_keep_list = sorted(path_keep_list)
                # landsat_coll = landsat_coll.filter(ee.Filter.rangeContains(
                #     'WRS_ROW', path_keep_list[0], path_keep_list[-1]))
                # fmask_coll = fmask_coll.filter(ee.Filter.rangeContains(
                #     'wrs_row', path_keep_list[0], path_keep_list[-1]))
                landsat_coll = landsat_coll.filter(
                    ee.Filter.inList('WRS_PATH', self.path_keep_list))
                if landsat in ['LT04', 'LT05', 'LC08_PRE']:
                    fmask_coll = fmask_coll.filter(
                        ee.Filter.inList('wrs_path', self.path_keep_list))
            if self.row_keep_list:
                # row_keep_list = sorted(row_keep_list)
                # landsat_coll = landsat_coll.filter(ee.Filter.rangeContains(
                #     'WRS_ROW', row_keep_list[0], row_keep_list[-1]))
                # fmask_coll = fmask_coll.filter(ee.Filter.rangeContains(
                #     'wrs_row', row_keep_list[0], row_keep_list[-1]))
                landsat_coll = landsat_coll.filter(
                    ee.Filter.inList('WRS_ROW', self.row_keep_list))
                if landsat in ['LT04', 'LT05', 'LC08_PRE']:
                    fmask_coll = fmask_coll.filter(
                        ee.Filter.inList('wrs_row', self.row_keep_list))
            if self.path_row_geom:
                landsat_coll = landsat_coll.filterBounds(self.path_row_geom)
                if landsat in ['LT04', 'LT05', 'LC08_PRE']:
                    fmask_coll = fmask_coll.filterBounds(self.path_row_geom)

            # Filter by geometry
            if self.zone_geom:
                landsat_coll = landsat_coll.filterBounds(self.zone_geom)
                if landsat in ['LT04', 'LT05', 'LC08_PRE']:
                    fmask_coll = fmask_coll.filterBounds(self.zone_geom)

            # Set SCENE_ID property for joining and filtering
            if landsat in ['LE07', 'LC08']:
                landsat_coll = landsat_coll.map(c1_scene_id_func)
            elif landsat in ['LT04', 'LT05', 'LC08_PRE']:
                landsat_coll = landsat_coll.map(pre_scene_id_func)
                fmask_coll = fmask_coll.map(pre_scene_id_func)

            # Filter by SCENE_ID
            if self.scene_id_keep_list:
                scene_id_keep_filter = ee.Filter.inList(
                    'SCENE_ID', self.scene_id_keep_list)
                landsat_coll = landsat_coll.filter(scene_id_keep_filter)
                if landsat in ['LT04', 'LT05', 'LC08_PRE']:
                    fmask_coll = fmask_coll.filter(scene_id_keep_filter)
            if self.scene_id_skip_list:
                scene_id_skip_filter = ee.Filter.inList(
                    'SCENE_ID', self.scene_id_skip_list).Not()
                landsat_coll = landsat_coll.filter(scene_id_skip_filter)
                if landsat in ['LT04', 'LT05', 'LC08_PRE']:
                    fmask_coll = fmask_coll.filter(scene_id_skip_filter)

            # Join the at-surface reflectance CFmask collection if necessary
            if (self.fmask_source and self.fmask_source == 'cfmask' and
                    landsat in ['LT04', 'LT05', 'LC08_PRE']):
                scene_id_filter = ee.Filter.equals(
                    leftField='SCENE_ID', rightField='SCENE_ID')
                landsat_coll = ee.ImageCollection(
                    ee.Join.saveFirst('fmask').apply(
                        landsat_coll, fmask_coll, scene_id_filter))
                # Add Fmask band from joined property
                landsat_coll = landsat_coll.map(landsat_fmask_band_func)
            elif landsat in ['LE07']:
                # Extract Fmask band from QA band
                landsat_coll = landsat_coll.map(landsat_qa_band_func)
            elif landsat in ['LC08']:
                # Extract Fmask band from QA band
                landsat_coll = landsat_coll.map(landsat_qa_band_func)

            # Add ACCA band (must be done before bands are renamed)
            landsat_coll = landsat_coll.map(landsat_acca_band_func)

            # Modify landsat collections to have same band names
            if landsat in ['LT04', 'LT05']:
                landsat_coll = landsat_coll.map(landsat5_toa_band_func)
            elif landsat in ['LE07']:
                landsat_coll = landsat_coll.map(landsat7_toa_band_func)
            elif landsat in ['LC08', 'LC08_PRE']:
                landsat_coll = landsat_coll.map(landsat8_toa_band_func)

            # Apply cloud masks
            if self.fmask_flag:
                landsat_coll = landsat_coll.map(landsat_fmask_cloud_mask_func)
            if self.acca_flag:
                landsat_coll = landsat_coll.map(landsat_acca_cloud_mask_func)

            # # Get the output image URL
            # output_url = ee.Image(landsat_coll.first()) \
            #     .select(['red', 'green', 'blue']) \
            #     .visualize(min=[0, 0, 0], max=[0.4, 0.4, 0.4]) \
            #     .getThumbUrl({'format': 'png', 'size': '600'})
            # # This would load the image in your browser
            # import webbrowser
            # webbrowser.open(output_url)
            # # webbrowser.read(output_url)

            # # Set properties
            # # These could be combined with SCENE_ID function above?
            # def landsat_properties(input_image):
            #     return input_image.setMulti({
            #         'type': landsat
            #     })
            # landsat_coll = landsat_coll.map(scene_id_func)

            # Compute derived images
            if landsat in ['LT04', 'LT05']:
                landsat_coll = landsat_coll.map(self.landsat5_images_func)
            elif landsat in ['LE07']:
                landsat_coll = landsat_coll.map(self.landsat7_images_func)
            elif landsat in ['LC08', 'LC08_PRE']:
                landsat_coll = landsat_coll.map(self.landsat8_images_func)

            # Mosaic overlapping images
            if (self.mosaic_method and
                    self.mosaic_method in self.mosaic_options):
                landsat_coll = mosaic_landsat_images(
                    landsat_coll, self.mosaic_method)

            # Merge Landsat specific collection into output collection
            output_coll = ee.ImageCollection(
                output_coll.merge(landsat_coll))
            # logging.info('{}'.format([
            #     f['properties']['MOSAIC_ID']
            #     for f in output_coll.getInfo()['features']]))
            # raw_input('ENTER')

        return output_coll

    def landsat5_images_func(self, refl_toa):
        """EE mappable function for calling landsat_image_func for Landsat 4/5"""
        return self.landsat_images_func(refl_toa, landsat='LT05')

    def landsat7_images_func(self, refl_toa):
        """EE mappable function for calling landsat_image_func for Landsat 7"""
        return self.landsat_images_func(refl_toa, landsat='LE07')

    def landsat8_images_func(self, refl_toa):
        """EE mappable function for calling landsat_image_func for Landsat 8"""
        return self.landsat_images_func(refl_toa, landsat='LC08')

    def landsat_images_func(self, refl_toa_orig, landsat):
        """Calculate Landsat products

        Send Landsat ROW number back as an image for determining "dominant"
            row in zones that overlap multiple images.

        Args:
            refl_toa_orig (ee.ImageCollection): Landsat TOA reflectance collection
            landsat (str): Landsat type ('LT04', 'LT05', 'LE07', or 'LC08')
                This is not an input argument

        Self Properties
            adjust_method (str): Adjust Landsat red and NIR bands.
                Choices are 'etm_2_oli' or 'oli_2_etm'.
                This could probably be simplifed to a flag

        Returns:
            ee.Image()
        """
        output_images = []

        # Clip to common area of all bands
        # Make a copy so original Fmask and cloud score can be passed through
        refl_toa = common_area_func(refl_toa_orig)

        # Eventually use Fmask band to set common area instead
        # refl_toa = refl_toa_orig.updateMask(
        #     refl_toa_orig.select(['fmask']).gte(0))

        # Brightness temperature must be > 250 K
        # refl_toa = refl_toa.updateMask(refl_toa.select(['thermal']).gt(250))

        # Output individual TOA reflectance bands
        for band in refl_bands:
            if '{}_toa'.format(band) in self.products:
                output_images.append(refl_toa.select([band], [band + '_toa']))

        # At-surface reflectance
        if any([p for p in self.products if '_sur' in p]):
            refl_sur = ee.Image(refl_sur_tasumi_func(
                refl_toa, landsat, self.adjust_method))
        # Output individual at-surfrace reflectance bands
        for band in refl_bands:
            if band + '_sur' in self.products:
                output_images.append(refl_sur.select([band], [band + '_sur']))

        # At-surface albedo
        if 'albedo_sur' in self.products:
            albedo_sur = ee.Image(albedo_func(refl_sur, landsat)) \
                .rename(['albedo_sur'])
            output_images.append(albedo_sur)

        # NDVI
        if ('ndvi_toa' in self.products or 'lai_toa' in self.products or
                'ts' in self.products):
            ndvi_toa = refl_toa.normalizedDifference(['nir', 'red']) \
                .rename(['ndvi_toa'])
            output_images.append(ndvi_toa)
        if 'ndvi_sur' in self.products or 'lai_sur' in self.products:
            ndvi_sur = refl_sur.normalizedDifference(['nir', 'red']) \
                .rename(['ndvi_sur'])
            output_images.append(ndvi_sur)

        # NDWI - McFeeters 1996
        if 'ndwi_green_nir_toa' in self.products:
            ndwi_green_nir_toa = refl_toa \
                .normalizedDifference(['green', 'nir']) \
                .rename(['ndwi_green_nir_toa'])
            output_images.append(ndwi_green_nir_toa)
        if 'ndwi_green_nir_sur' in self.products:
            ndwi_green_nir_sur = refl_sur \
                .normalizedDifference(['green', 'nir']) \
                .rename(['ndwi_green_nir_sur'])
            output_images.append(ndwi_green_nir_sur)

        # NDWI - Xu 2006 (MNDWI) doi: 10.1080/01431160600589179
        # Equivalent to NDSI Hall et al 1995 and 1998
        # http://modis-snow-ice.gsfc.nasa.gov/uploads/pap_dev95.pdf
        # http://modis-snow-ice.gsfc.nasa.gov/uploads/pap_assmnt98.pdf
        if 'ndwi_green_swir1_toa' in self.products:
            ndwi_green_swir1_toa = refl_toa \
                .normalizedDifference(['green', 'swir1']) \
                .rename(['ndwi_green_swir1_toa'])
            output_images.append(ndwi_green_swir1_toa)
        if 'ndwi_green_swir1_sur' in self.products:
            ndwi_green_swir1_sur = refl_sur \
                .normalizedDifference(['green', 'swir1']) \
                .rename(['ndwi_green_swir1_sur'])
            output_images.append(ndwi_green_swir1_sur)

        # NDWI - Gao 1996 doi: 10.1016/S0034-4257(96)00067-3
        # Inverse of NDSI (Soil) in Rogers & Keraney 2004
        if 'ndwi_nir_swir1_toa' in self.products:
            ndwi_nir_swir1_toa = refl_toa \
                .normalizedDifference(['nir', 'swir1']) \
                .rename(['ndwi_nir_swir1_toa'])
            output_images.append(ndwi_nir_swir1_toa)
        if 'ndwi_nir_swir1_sur' in self.products:
            ndwi_nir_swir1_sur = refl_sur \
                .normalizedDifference(['nir', 'swir1']) \
                .rename(['ndwi_nir_swir1_sur'])
            output_images.append(ndwi_nir_swir1_sur)

        # NDWI - Allen 2007
        # Return this NDWI as the default ndwi_sur and ndwi_toa below
        if 'ndwi_swir1_green_toa' in self.products:
            ndwi_swir1_green_toa = refl_toa \
                .normalizedDifference(['swir1', 'green']) \
                .rename(['ndwi_swir1_green_toa'])
            output_images.append(ndwi_swir1_green_toa)
        if 'ndwi_swir1_green_sur' in self.products:
            ndwi_swir1_green_sur = refl_sur \
                .normalizedDifference(['swir1', 'green']) \
                .rename(['ndwi_swir1_green_sur'])
            output_images.append(ndwi_swir1_green_sur)

        if 'ndwi_toa' in self.products:
            ndwi_toa = refl_toa \
                .normalizedDifference(['swir1', 'green']) \
                .rename(['ndwi_toa'])
            output_images.append(ndwi_toa)
        if 'ndwi_sur' in self.products:
            ndwi_sur = refl_sur \
                .normalizedDifference(['swir1', 'green']) \
                .rename(['ndwi_sur'])
            output_images.append(ndwi_sur)

        # LAI (for computing Ts) (Empirical function from Allen et al 2007)
        if 'lai_toa' in self.products or 'ts' in self.products:
            lai_toa = ee.Image(ndvi_lai_func(ndvi_toa)) \
                .rename(['lai_toa'])
            output_images.append(lai_toa)
        if 'lai_sur' in self.products:
            lai_sur = ee.Image(ndvi_lai_func(ndvi_sur)) \
                .rename(['lai_sur'])
            output_images.append(lai_sur)

        # EVI
        if ('evi_sur' in self.products or
                any([True for p in self.products if 'etstar_' in p]) or
                any([True for p in self.products if 'etg_' in p])):
            evi_sur = ee.Image(landsat_evi_func(refl_sur)) \
                .rename(['evi_sur'])
            output_images.append(evi_sur)

        # Surface temperature
        if 'ts' in self.products:
            ts = ee.Image(ts_func(
                ts_brightness=refl_toa.select('thermal'),
                em_nb=em_nb_func(ndvi_toa, lai_toa),
                k1=ee.Number(refl_toa.get('k1_constant')),
                k2=ee.Number(refl_toa.get('k2_constant'))))
            output_images.append(ts)

        # Tasseled cap
        if 'tc_bright' in self.products:
            tc_bright = ee.Image(tc_bright_func(refl_toa, landsat))
            output_images.append(tc_bright)
        if 'tc_green' in self.products:
            tc_green = ee.Image(tc_green_func(refl_toa, landsat))
            output_images.append(tc_green)
        if 'tc_wet' in self.products:
            tc_wet = ee.Image(tc_wet_func(refl_toa, landsat))
            output_images.append(tc_wet)

        # Beamer ET* and ETg
        if 'etstar_mean' in self.products or 'etg_mean' in self.products:
            etstar_mean = ee.Image(etstar_func(evi_sur, etstar_type='mean')) \
                .rename(['etstar_mean'])
        if 'etstar_lpi' in self.products or 'etg_lpi' in self.products:
            etstar_lpi = ee.Image(etstar_func(evi_sur, etstar_type='lpi')) \
                .rename(['etstar_lpi'])
        if 'etstar_upi' in self.products or 'etg_upi' in self.products:
            etstar_upi = ee.Image(etstar_func(evi_sur, etstar_type='upi')) \
                .rename(['etstar_upi'])
        if 'etstar_lci' in self.products or 'etg_lci' in self.products:
            etstar_lci = ee.Image(etstar_func(evi_sur, etstar_type='lci')) \
                .rename(['etstar_lci'])
        if 'etstar_uci' in self.products or 'etg_uci' in self.products:
            etstar_uci = ee.Image(etstar_func(evi_sur, etstar_type='uci')) \
                .rename(['etstar_uci'])
        if 'etstar_mean' in self.products:
            output_images.append(etstar_mean)
        if 'etstar_lpi' in self.products:
            output_images.append(etstar_lpi)
        if 'etstar_upi' in self.products:
            output_images.append(etstar_upi)
        if 'etstar_lci' in self.products:
            output_images.append(etstar_lci)
        if 'etstar_uci' in self.products:
            output_images.append(etstar_uci)

        # # For each Landsat scene, I need to calculate water year PPT and ETo sums
        # # ppt = ee.Image.constant(100)
        # # eto = ee.Image.constant(1000)
        # if any([p for p in self.products if 'etg_' in p]):
        #     ppt = ee.Image.constant(refl_toa_orig.get('wy_ppt'))
        #     eto = ee.Image.constant(refl_toa_orig.get('wy_eto'))

        # # ETg
        # if 'etg_mean' in self.products:
        #     etg_mean = ee.Image(etg_func(etstar_mean, eto, ppt)) \
        #         .rename(['etg_mean'])
        #     output_images.append(etg_mean)
        # if 'etg_lpi' in self.products:
        #     etg_lpi = ee.Image(etg_func(etstar_lpi, eto, ppt)) \
        #         .rename(['etg_lpi'])
        #     output_images.append(etg_lpi)
        # if 'etg_upi' in self.products:
        #     etg_upi = ee.Image(etg_func(etstar_upi, eto, ppt)) \
        #         .rename(['etg_upi'])
        #     output_images.append(etg_upi)
        # if 'etg_lci' in self.products:
        #     etg_lci = ee.Image(etg_func(etstar_lci, eto, ppt)) \
        #         .rename(['etg_lci'])
        #     output_images.append(etg_lci)
        # if 'etg_uci' in self.products:
        #     etg_uci = ee.Image(etg_func(etstar_uci, eto, ppt)) \
        #         .rename(['etg_uci'])
        #     output_images.append(etg_uci)

        # Add additional bands
        output_images.extend([
            refl_toa_orig.select('cloud_score'),
            refl_toa_orig.select('fmask'),
            refl_toa_orig.metadata('WRS_ROW', 'row')
        ])

        return ee.Image(output_images) \
            .copyProperties(refl_toa, system_properties + ['SCENE_ID'])


def c1_scene_id_func(img):
    """Construct Collecton 1 short SCENE_ID for collection 1 images

    LT05_PPPRRR_YYYYMMDD
    Format matches EE collection 1 system:index
    Split on '_' in case the collection was merged first
    """
    scene_id = ee.List(ee.String(
        img.get('system:index')).split('_')).slice(-3)
    scene_id = ee.String(scene_id.get(0)).cat('_') \
        .cat(ee.String(scene_id.get(1))).cat('_') \
        .cat(ee.String(scene_id.get(2)))
    return img.setMulti({'SCENE_ID': scene_id})


def pre_scene_id_func(img):
    """Construct Collecton 1 short SCENE_ID for pre-collection images

    LT05_PPPRRR_YYYYMMDD
    Format matches EE collection 1 system:index
    Split on '_' in case the collection was merged first"""
    scene_id = ee.String(ee.List(ee.String(
        img.get('system:index')).split('_')).slice(-1).get(0))
    scene_id = scene_id.slice(0, 2).cat('0') \
        .cat(scene_id.slice(2, 3)).cat('_') \
        .cat(scene_id.slice(3, 9)).cat('_') \
        .cat(ee.Date(img.get('system:time_start')).format('yyyyMMdd'))
    return img.setMulti({'SCENE_ID': scene_id})


def refl_sur_tasumi_func(refl_toa, landsat, adjust_method=None):
    """Tasumi at-surface reflectance

    Args:
        refl_toa (ee.Image):
        landsat (str): Landsat type
        adjust_method (str): Adjust Landsat red and NIR bands.
            Choices are 'etm_2_oli', 'oli_2_etm', 'none'.
            Default is 'none'.

    Returns:
        ee.Image: at-surface reflectance
    """
    scene_date = ee.Date(refl_toa.get('system:time_start'))
    doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
    hour = ee.Number(scene_date.getFraction('day')).multiply(24)
    cos_theta = ee.Image(cos_theta_flat_func(doy, hour))

    pair = ee.Image(pair_func(ee.Image('USGS/SRTMGL1_003')))
    # pair = ee.Image(pair_func(ee.Image('USGS/NED')))

    # Interpolate NLDAS data to scene time
    nldas_image = ee.Image(nldas_interp_func(refl_toa))

    # Specific humidity (kg/kg)
    q = nldas_image.select(['specific_humidity'])
    # q = ee.Image(refl_toa.get('match')).select(['specific_humidity'])
    # q = nldas_interp_func(refl_toa).select(['specific_humidity'])
    ea = pair.expression(
        'q * pair / (0.622 + 0.378 * q)', {'q': q, 'pair': pair})

    # Precipitable water?
    w = pair.multiply(0.14).multiply(ea).add(2.1)

    if landsat in ['LT05', 'LT04', 'LE07']:
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
    elif landsat in ['LC08', 'LC08_PRE']:
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

    if (adjust_method and adjust_method.lower() == 'etm_2_oli' and
            landsat in ['LT05', 'LT04', 'LE07']):
        # http://www.sciencedirect.com/science/article/pii/S0034425716302619
        # Coefficients for scaling ETM+ to OLI
        refl_sur = ee.Image(refl_sur) \
            .subtract([0, 0, 0.0024, -0.0003, 0, 0]) \
            .divide([1, 1, 1.0047, 1.0036, 1, 1])
    elif (adjust_method and adjust_method.lower() == 'oli_2_etm' and
            landsat in ['LC08', 'LC08_PRE']):
        # http://www.sciencedirect.com/science/article/pii/S0034425716302619
        # Coefficients for scaling OLI to ETM+
        refl_sur = ee.Image(refl_sur) \
            .multiply([1, 1, 1.0047, 1.0036, 1, 1]) \
            .add([0, 0, 0.0024, -0.0003, 0, 0])

    return refl_sur \
        .clamp(0.0001, 1) \
        .copyProperties(refl_toa, system_properties)


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
        lat = ee.Image.pixelLonLat() \
            .select(['latitude']).multiply(pi / 180)
    if lon is None:
        lon = ee.Image.pixelLonLat() \
            .select(['longitude']).multiply(pi / 180)
    delta = acq_doy.multiply(2 * pi / 365).subtract(1.39435) \
        .sin().multiply(0.40928)
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
        lat = ee.Image.pixelLonLat() \
            .select(['latitude']).multiply(pi / 180)
    if lon is None:
        lon = ee.Image.pixelLonLat() \
            .select(['longitude']).multiply(pi / 180)
    if slope is None or aspect is None:
        terrain = ee.call('Terrain', ee.Image('USGS/NED'))
    if slope is None:
        slope = terrain.select(['slope']).multiply(pi / 180)
    if aspect is None:
        aspect = terrain.select(['aspect']).multiply(pi / 180).subtract(pi)
    delta = acq_doy.multiply(2 * math.pi / 365).subtract(1.39435) \
        .sin().multiply(0.40928)
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


def albedo_func(refl_sur, landsat):
    """At-surface albedo"""
    if landsat in ['LT05', 'LT04']:
        wb_coef = [0.254, 0.149, 0.147, 0.311, 0.103, 0.036]
    elif landsat in ['LE07']:
        wb_coef = [0.254, 0.149, 0.147, 0.311, 0.103, 0.036]
    elif landsat in ['LC08', 'LC08_PRE']:
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


def etstar_func(evi, etstar_type='mean'):
    """Compute Beamer ET* from EVI (assuming at-surface reflectance)"""
    def etstar(evi, c0, c1, c2):
        """Beamer ET*"""
        return ee.Image(evi) \
            .expression(
                'c0 + c1 * evi + c2 * (evi ** 2)',
                {'evi': evi, 'c0': c0, 'c1': c1, 'c2': c2}) \
            .max(0)
    if etstar_type == 'mean':
        return etstar(evi, -0.1955, 2.9042, -1.5916)
    elif etstar_type == 'lpi':
        return etstar(evi, -0.2871, 2.9192, -1.6263)
    elif etstar_type == 'upi':
        return etstar(evi, -0.1039, 2.8893, -1.5569)
    elif etstar_type == 'lci':
        return etstar(evi, -0.2142, 2.9175, -1.6554)
    elif etstar_type == 'uci':
        return etstar(evi, -0.1768, 2.8910, -1.5278)


def etg_func(etstar, eto, ppt):
    """Compute groundwater ET (ETg) (ET* x (ETo - PPT))"""
    return etstar.multiply(eto.subtract(ppt))


def et_func(etg, ppt):
    """Compute net ET (ETg + PPT)"""
    return etg.add(ppt)


# def tasseled_cap_func(self, refl_toa):
#     refl_toa_sub = refl_toa.select(refl_bands)
#     tc_bright_coef = ee.List(refl_toa.get('tc_bright'))
#     tc_green_coef = ee.List(refl_toa.get('tc_green'))
#     tc_wet_coef = ee.List(refl_toa.get('tc_wet'))
#     return ee.Image([
#         refl_toa_sub.multiply(tc_bright_coef).reduce(ee.Reducer.sum()),
#         refl_toa_sub.multiply(tc_green_coef).reduce(ee.Reducer.sum()),
#         refl_toa_sub.multiply(tc_wet_coef).reduce(ee.Reducer.sum())])\
#         .select([0, 1, 2], ['tc_bright', 'tc_green', 'tc_wet'])


def tc_bright_func(refl_toa, landsat):
    """Tasseled cap brightness

    Top of atmosphere (at-satellite) reflectance

    LT04/LT05 - http://www.gis.usu.edu/~doug/RS5750/assign/OLD/RSE(17)-301.pdf
    LE07 - http://landcover.usgs.gov/pdf/tasseled.pdf
    LC08 - http://www.tandfonline.com/doi/abs/10.1080/2150704X.2014.915434
    https://www.researchgate.net/publication/262005316_Derivation_of_a_tasselled_cap_transformation_based_on_Landsat_8_at-_satellite_reflectance
    """
    if landsat in ['LT04', 'LT05']:
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_bright_coef = [0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303]
    elif landsat in ['LE07']:
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_bright_coef = [0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0.1596]
    elif landsat in ['LC08', 'LC08_PRE']:
        refl_toa_sub = refl_toa.select(refl_bands)
        # refl_toa_sub = refl_toa_sub.multiply(0.0001)
        tc_bright_coef = [0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872]
    return refl_toa_sub.multiply(tc_bright_coef).reduce(ee.Reducer.sum()) \
        .rename(['tc_bright'])


def tc_green_func(refl_toa, landsat):
    """Tasseled cap greeness"""
    if landsat in ['LT04', 'LT05']:
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_green_coef = [-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446]
    elif landsat in ['LE07']:
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_green_coef = [-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, -0.2630]
    elif landsat in ['LC08', 'LC08_PRE']:
        refl_toa_sub = refl_toa.select(refl_bands)
        # refl_toa_sub = refl_toa_sub.multiply(0.0001)
        tc_green_coef = [-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608]
    return refl_toa_sub.multiply(tc_green_coef).reduce(ee.Reducer.sum()) \
        .rename(['tc_green'])


def tc_wet_func(refl_toa, landsat):
    """Tasseled cap wetness"""
    if landsat in ['LT04', 'LT05']:
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_wet_coef = [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]
    elif landsat in ['LE07']:
        refl_toa_sub = refl_toa.select(refl_bands)
        tc_wet_coef = [0.2626, 0.2141, 0.0926, 0.0656, -0.7629, -0.5388]
    elif landsat in ['LC08', 'LC08_PRE']:
        refl_toa_sub = refl_toa.select(refl_bands)
        # refl_toa_sub = refl_toa_sub.multiply(0.0001)
        tc_wet_coef = [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559]
    return refl_toa_sub.multiply(tc_wet_coef).reduce(ee.Reducer.sum()) \
        .rename(['tc_wet'])


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
    return ts.rename(['ts'])


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
        img (ee.Image):

    Returns
        ee.Image(): NLDAS values interpolated at the image time
    """
    scene_time = ee.Number(img.get('system:time_start'))
    scene_datetime = ee.Date(scene_time)
    nldas_coll = ee.ImageCollection('NASA/NLDAS/FORA0125_H002')
    nldas_prev_image = ee.Image(nldas_coll.filterDate(
        scene_datetime.advance(-1, 'hour'), scene_datetime).first())
    nldas_next_image = ee.Image(nldas_coll.filterDate(
        scene_datetime, scene_datetime.advance(1, 'hour')).first())
    # print(nldas_prev_image.getInfo()['features'][0]['properties']['system:index'])
    # print(nldas_prev_image.getInfo()['features'][0]['properties']['system:index'])
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
        .setMulti({'system:time_start': scene_time})


def landsat_acca_band_func(refl_toa_img):
    """Add ACCA like cloud score band to Landsat collection"""
    cloud_score = ee.Algorithms.Landsat.simpleCloudScore(refl_toa_img) \
        .select(['cloud'], ['cloud_score'])
    return refl_toa_img.addBands(cloud_score)


def landsat_fmask_band_func(refl_toa_img):
    """Get Fmask band from the joined properties"""
    return refl_toa_img.addBands(
        ee.Image(refl_toa_img.get('fmask')).rename(['fmask']))


def landsat_empty_fmask_band_func(refl_toa_img):
    """Add an empty fmask band"""
    return refl_toa_img.addBands(
        refl_toa_img.select([0]).multiply(0).rename(['fmask']))


def landsat_qa_band_func(refl_toa_img):
    """Get Fmask band from the joined properties

    https://landsat.usgs.gov/collectionqualityband

    Confidence values
    00 = "Not Determined" = Algorithm did not determine the status of this condition
    01 = "No" = Algorithm has low to no confidence that this condition exists (0-33 percent confidence)
    10 = "Maybe" = Algorithm has medium confidence that this condition exists (34-66 percent confidence)
    11 = "Yes" = Algorithm has high confidence that this condition exists (67-100 percent confidence
    """
    qa_img = ee.Image(refl_toa_img.select(['BQA']))

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
    fill_mask = getQABits(qa_img, 0, 0, 'designated_fill')
    # drop_mask = getQABits(qa_img, 1, 1, 'dropped_pixel')
    # Landsat 8 only
    # terrain_mask = getQABits(qa_img, 1, 1, 'terrain_occlusion')
    # saturation_mask = getQABits(qa_img, 2, 3, 'saturation_confidence').gte(2)
    # cloud_mask = getQABits(qa_img, 4, 4, 'cloud')
    cloud_mask = getQABits(qa_img, 7, 8, 'cloud_confidence').gte(2)
    shadow_mask = getQABits(qa_img, 7, 8, 'shadow_confidence').gte(3)
    snow_mask = getQABits(qa_img, 9, 10, 'snow_confidence').gte(3)
    # Landsat 8 only
    # cirrus_mask = getQABits(qa_img, 11, 12, 'cirrus_confidence').gte(3)

    # Convert masks to old style Fmask values
    # 0 - Clear land
    # 1 - Clear water
    # 2 - Cloud shadow
    # 3 - Snow
    # 4 - Cloud
    fmask_img = fill_mask \
        .add(shadow_mask.multiply(2)) \
        .add(snow_mask.multiply(3)) \
        .add(cloud_mask.multiply(4))

    return refl_toa_img.addBands(fmask_img.rename(['fmask']))


def landsat5_toa_band_func(img):
    """Rename Landsat 4 and 5 bands to common band names

    Change band order to match Landsat 8
    Set K1 and K2 coefficients used for computing land surface temperature
    Set Tasseled cap coefficients
    """
    return img \
        .select(
            ['B1', 'B2', 'B3', 'B4', 'B5', 'B7',
             'B6', 'cloud_score', 'fmask'],
            ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
             'thermal', 'cloud_score', 'fmask'])\
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
    return img \
        .select(
            ['B1', 'B2', 'B3', 'B4', 'B5', 'B7',
             'B6_VCID_1', 'cloud_score', 'fmask'],
            ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
             'thermal', 'cloud_score', 'fmask']) \
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
    return img \
        .select(
            ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
             'B10', 'cloud_score', 'fmask'],
            ['blue', 'green', 'red', 'nir', 'swir1', 'swir2',
             'thermal', 'cloud_score', 'fmask']) \
        .setMulti({
            'k1_constant': img.get('K1_CONSTANT_BAND_10'),
            'k2_constant': img.get('K2_CONSTANT_BAND_10')}) \
        .copyProperties(img, system_properties)


# def landsat5_sr_band_func(img):
#     """Rename Landsat 4 and 5 bands to common band names

#     Change band order to match Landsat 8
#     Scale values by 10000
#     """
#     sr_image = img \
#         .select(
#             ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
#             ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']) \
#         .divide(10000.0)
#     # Cloud mask bands must be set after scaling
#     return ee.Image(sr_image) \
#         .addBands(img.select(
#             ['cloud_score', 'cfmask'], ['cloud_score', 'fmask'])) \
#         .copyProperties(img, system_properties)


# def landsat7_sr_band_func(img):
#     """Rename Landsat 7 bands to common band names

#     Change band order to match Landsat 8
#     For now, don't include pan-chromatic or high gain thermal band
#     Scale values by 10000
#     """
#     # ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B8'],
#     # ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pan'])
#     sr_image = img \
#         .select(
#             ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
#             ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])\
#         .divide(10000.0)
#     # Cloud mask bands must be set after scaling
#     return ee.Image(sr_image) \
#         .addBands(img.select(
#             ['cloud_score', 'cfmask'], ['cloud_score', 'fmask'])) \
#         .copyProperties(img, system_properties)


# def landsat8_sr_band_func(img):
#     """Rename Landsat 8 bands to common band names

#     For now, don't include coastal, cirrus, or pan-chromatic
#     Scale values by 10000
#     """
#     # ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9'],
#     # ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
#     #  'pan', 'cirrus'])
#     sr_image = img \
#         .select(
#             ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
#             ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])\
#         .divide(10000.0)
#     # Cloud mask bands must be set after scaling
#     return ee.Image(sr_image) \
#         .addBands(img.select(
#             ['cloud_score', 'cfmask'], ['cloud_score', 'fmask'])) \
#         .copyProperties(img, system_properties)


def common_area_func(img):
    """Only keep pixels that are common to all bands"""
    common_mask = ee.Image(img).mask().reduce(ee.Reducer.And())
    return img.updateMask(common_mask)
    # common_mask = img \
    #     .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']) \
    #     .mask().reduce(ee.Reducer.And())
    # # common_mask = img.select(['fmask']).mask()
    # return img \
    #     .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']) \
    #     .updateMask(common_mask) \
    #     .addBands(img.select(['cloud_score', 'cfmask'])) \
    #     .copyProperties(img, system_properties)


def erode_func(img):
    """"""
    input_mask = ee.Image(img).mask().reduceNeighborhood(
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
    return img \
        .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']) \
        .updateMask(cloud_mask) \
        .addBands(img.select(['cloud_score', 'fmask'])) \
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
    return img \
        .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']) \
        .updateMask(cloud_mask) \
        .addBands(img.select(['cloud_score', 'fmask'])) \
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
        .addBands(img.select(
            ['cloud_score', 'cfmask'], ['cloud_score', 'fmask'])) \
        .copyProperties(img, system_properties)


# def acca_mask_func(refl_toa):
#     """Apply ACCA cloud mask function"""
#     cloud_mask = ee.Algorithms.Landsat.simpleCloudScore(refl_toa) \
#         .select(['cloud']).lt(ee.Image.constant(50))
#     cloud_mask = cloud_mask.updateMask(cloud_mask)
#     return refl_toa.updateMask(cloud_mask)


def prism_ppt_func(prism_image):
    """PRISM water year precipitation

    Depends on maps engine assets
    """
    return prism_image.select([0], ['PPT']) \
        .copyProperties(prism_image, system_properties)


def gridmet_ppt_func(gridmet_image):
    """GRIDMET daily precipitation"""
    return gridmet_image.select(['pr'], ['PPT']) \
        .copyProperties(gridmet_image, system_properties)


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


def mosaic_landsat_images(landsat_coll, mosaic_method):
    """"""
    def mosaic_id_func(image):
        """Set MOSAIC_ID with row set to XXX

        Using GEE Collection 1 system:index naming convention
        LT05_PPPRRR_YYYYMMDD
        """
        scene_id = ee.String(image.get('SCENE_ID'))
        mosaic_id = scene_id.slice(0, 8).cat('XXX').cat(scene_id.slice(11, 20))
        # If mosaicing after merging, SCENE_ID is at end
        # scene_id = ee.String(ee.List(ee.String(
        #     image.get('system:index')).split('_')).get(-1))
        # Build product ID from old style scene ID
        # scene_id = ee.String(img.get('system:index'))
        # scene_id = scene_id.slice(0, 2).cat('0') \
        #     .cat(scene_id.slice(2, 3)).cat('_') \
        #     .cat(scene_id.slice(3, 9)).cat('_') \
        #     .cat(ee.Date(img.get('system:time_start')).format('yyyyMMdd'))
        return image.setMulti({'MOSAIC_ID': mosaic_id})
    landsat_coll = landsat_coll.map(mosaic_id_func)

    mosaic_id_list = ee.List(ee.Dictionary(ee.FeatureCollection(
        landsat_coll.aggregate_histogram('MOSAIC_ID'))).keys())

    def set_mosaic_id(mosaic_id):
        return ee.Feature(None, {'MOSAIC_ID': ee.String(mosaic_id)})
    mosaic_id_coll = ee.FeatureCollection(mosaic_id_list.map(set_mosaic_id))

    join_coll = ee.Join.saveAll('join').apply(
        mosaic_id_coll, landsat_coll,
        ee.Filter.equals(leftField='MOSAIC_ID', rightField='MOSAIC_ID'))

    def aggregate_func(ftr):
        # The composite image time will be 0 UTC (not Landsat time)
        coll = ee.ImageCollection.fromImages(ftr.get('join'))
        time = ee.Image(ee.List(ftr.get('join')).get(0)).get('system:time_start')
        if mosaic_method == 'mean':
            image = coll.mean()
        elif mosaic_method == 'median':
            image = coll.median()
        elif mosaic_method == 'mosaic':
            image = coll.mosaic()
        elif mosaic_method == 'min':
            image = coll.min()
        elif mosaic_method == 'max':
            image = coll.max()
        else:
            image = coll.first()
        return ee.Image(image).setMulti({
            'SCENE_ID': ee.String(ftr.get('MOSAIC_ID')),
            'system:time_start': time})
    mosaic_coll = ee.ImageCollection(join_coll.map(aggregate_func))

    return mosaic_coll
