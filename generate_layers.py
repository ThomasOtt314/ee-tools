import argparse
import datetime
import logging
import os
import shutil
import sys
import tempfile

import arcpy


def main():
    """Generate Layer Files for each Landsat image from templates"""

    # Search for Landsat images to build templates for in the input workspace
    # Code currently is assuming images are in separate folders for each year
    input_ws = r'Y:\justinh\Projects\SNWA\ee\images\spring_valley\landsat'

    # Set a different raster workspace in the layers
    layer_ws = r'Y:\justinh\Projects\SNWA\ee\images\spring_valley\landsat'
    # layer_ws = r'R:\Image_Regional\SpringValley\Hearings_2017_GDE\ee\images\spring_valley\landsat'

    # Save layer files to the output workspace
    #   (separate folders for each year)
    output_ws = r'Y:\justinh\Projects\SNWA\ee\images\spring_valley\layers'

    # Folder where the template layers are stored
    template_ws = r'Y:\justinh\Projects\SNWA\ee\layers'


    # This would need to be appended to process other products
    template_dict = {
        'refl_sur': 'template.refl_sur.tif.lyr'
        # 'ndvi_sur': 'template.ndvi_sur.tif.lyr'
    }

    # Need a temporary folder because of a bug in replaceDataSource
    temp_ws = tempfile.mkdtemp()
    logging.debug('\nTemp folder: {}'.format(temp_ws))
    if not os.path.isdir(temp_ws):
        os.makedirs(temp_ws)

    # Check that templates exists
    for product, template in template_dict.items():
        template_path = os.path.join(template_ws, template)
        if not os.path.isfile(template_path):
            logging.error(
                '\nERROR: The {} template layer does not exist\m    {}'.format(
                    product, template_path))

    # Process each year separately
    for year in os.listdir(input_ws):
        logging.info('\n{}'.format(year))
        input_year_ws = os.path.join(input_ws, str(year))
        layer_year_ws = os.path.join(layer_ws, str(year))
        output_year_ws = os.path.join(output_ws, str(year))
        if not os.path.isdir(input_year_ws):
            continue
        if not os.path.isdir(output_year_ws):
            os.makedirs(output_year_ws)

        for item in os.listdir(input_year_ws):
            if not item.endswith('.tif'):
                continue

            for product, template in template_dict.items():
                # logging.debug('{}'.format(product))
                if not item.endswith(product + '.tif'):
                    continue
                logging.info('{}'.format(item))

                template_path = os.path.join(template_ws, template)
                layer_path = os.path.join(
                    output_year_ws, item.replace('.tif', '.lyr'))
                logging.debug('  Template: {}'.format(template_path))
                logging.debug('  Layer:    {}'.format(layer_path))

                # There is a bug in replaceDataSource (ArcGIS 10.3)
                # There is a problem with file names that have extra dots (".")
                #   that causes replaceDataSource to defaults to the 1st raster
                #   in the workspace.
                # To get around this, I am creating one temp raster with the
                #   same name as the target raster
                temp_path = os.path.join(temp_ws, item)
                if arcpy.Exists(temp_path):
                    arcpy.Delete_management(temp_path)
                arcpy.CreateRasterDataset_management(
                    temp_ws, item, "1", "8_BIT_UNSIGNED", "", "1")

                # Open the template layer
                lyr = arcpy.mapping.Layer(template_path)

                # First set the DataSource to the temp folder raster
                lyr.replaceDataSource(temp_ws, 'RASTER_WORKSPACE', item, False)

                # Then change the workspace to the correct workspace
                lyr.findAndReplaceWorkspacePath(
                    os.path.dirname(lyr.datasetName), layer_year_ws, False)
                lyr.name = item
                lyr.saveACopy(layer_path)

                # Delete the temp raster
                arcpy.Delete_management(temp_path)

                del lyr

    # Try to remove the temp folder
    try:
        shutil.rmtree(temp_ws)
    except:
        pass


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate Layer Files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main()
