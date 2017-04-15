import argparse
import logging
import math
import os

from osgeo import gdal, ogr, osr


def main(input_path, output_path, output_epsg=None,
         cellsize=30, snap=(15, 15), overwrite_flag=False):
    """Adjust shapefile polygon borders to follow raster cell outlines
      in an arbitrary projection and snap.

    Args:
        input_path (str):
        output_path (str):
        epsg (int): EPSG code
        cellsize (float): cellsize
        snap (list/tuple):
        overwrite_flag (bool): if True, overwrite existing files
    """
    logging.info('Rasterizing Polygon Geometry')

    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    gdal_mem_driver = gdal.GetDriverByName('MEM')
    ogr_mem_driver = ogr.GetDriverByName('MEMORY')

    # Copy the input shapefile
    input_ds = ogr.Open(input_path)
    input_lyr = input_ds.GetLayer()
    input_osr = input_lyr.GetSpatialRef()
    output_ds = shp_driver.CopyDataSource(input_ds, output_path)
    output_ds = None
    input_lyr = None

    # Open the output shapefiles
    output_ds = ogr.Open(output_path, 1)
    output_lyr = output_ds.GetLayer()

    # Build the output spatial reference object
    if output_epsg is None:
        logging.debug('  Using input spatial reference: {}'.format(
            input_osr.ExportToProj4()))
        output_osr = input_osr.Clone()
    else:
        output_osr = osr.SpatialReference()
        output_osr.ImportFromEPSG(int(output_epsg))

    # Projection coordinate transformation
    output_tx = osr.CoordinateTransformation(input_osr, output_osr)

    # Read the input layer into memory and project
    project_ds = ogr_mem_driver.CreateDataSource('project')
    input_lyr = project_ds.CopyLayer(
        input_ds.GetLayer(), 'project', ['OVERWRITE=YES'])
    for input_ftr in input_lyr:
        input_geom = input_ftr.GetGeometryRef()
        input_geom.Transform(output_tx)
        input_ftr.SetGeometry(input_geom)
        input_lyr.SetFeature(input_ftr)
    input_lyr.ResetReading()
    input_ds = None

    # Rasterize each feature separately
    for output_ftr in output_lyr:
        output_fid = output_ftr.GetFID()
        logging.debug('  FID: {}'.format(output_fid))

        # Selec the current feature from the input layer
        # input_lyr = input_ds.GetLayer()
        # input_lyr.SetAttributeFilter("{0} = {1}".format('FID', output_fid))
        input_lyr.SetAttributeFilter("{0} = {1}".format('FID', output_fid))
        input_ftr = input_lyr.GetNextFeature()
        input_geom = input_ftr.GetGeometryRef()
        # logging.debug('  Geom: {}'.format(input_geom.ExportToWkt()))

        # Compute snapped extent (in the projected coordinate system)
        # input_geom = output_ftr.GetGeometryRef()

        output_extent = ogrenv_swap(input_geom.GetEnvelope())
        # logging.debug('  Extent: {}'.format(output_extent))
        output_extent = adjust_to_snap(
            output_extent, 'EXPAND', snap[0], snap[1], cellsize)
        output_geo = extent_geo(output_extent, cellsize)
        # logging.debug('  Extent: {}'.format(output_extent))
        # logging.debug('  Geo: {}'.format(output_geo))

        # Create the in-memory raster to rasterize into
        raster_rows, raster_cols = extent_shape(output_extent, cellsize)
        # logging.debug('  Shape: {}x{}'.format(raster_rows, raster_cols))
        raster_ds = gdal_mem_driver.Create(
            '', raster_cols, raster_rows, 1, gdal.GDT_Byte)
        raster_ds.SetProjection(output_osr.ExportToWkt())
        raster_ds.SetGeoTransform(output_geo)
        raster_band = raster_ds.GetRasterBand(1)
        raster_band.Fill(0)
        raster_band.SetNoDataValue(0)

        # Rasterize the current feature
        gdal.RasterizeLayer(
            raster_ds, [1], input_lyr, burn_values=[1])
        raster_band = raster_ds.GetRasterBand(1)

        # Polygonize the raster
        polygon_ds = ogr_mem_driver.CreateDataSource('memData')
        # polygon_ds = ogr_mem_driver.Open('memData', 1)
        polygon_lyr = polygon_ds.CreateLayer('memLayer', srs=None)
        gdal.Polygonize(
            raster_band, raster_band, polygon_lyr, -1, [], callback=None)
        raster_ds, raster_band = None, None

        # Get the new geometry from the in memory polygon
        output_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        # if polygon_lyr.GetFeatureCount() > 1:
        #     output_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        # else:
        #     output_geom = ogr.Geometry(ogr.wkbPolygon)
        for polygon_ftr in polygon_lyr:
            output_geom.AddGeometry(polygon_ftr.GetGeometryRef())

        # Replace the original geometry with the new geometry
        output_ftr.SetGeometry(output_geom)
        output_lyr.SetFeature(output_ftr)

        polygon_lyr, polygon_ds = None, None

    output_ds, output_lyr = None, None
    project_ds = None

    # Set the output spatial reference
    output_osr.MorphToESRI()
    with open(output_path.replace('.shp', '.prj'), 'w') as output_f:
        output_f.write(output_osr.ExportToWkt())


def adjust_to_snap(extent, method, snap_x, snap_y, cellsize):
    extent[0] = math.floor((extent[0] - snap_x) / cellsize) * cellsize + snap_x
    extent[1] = math.floor((extent[1] - snap_y) / cellsize) * cellsize + snap_y
    extent[2] = math.ceil((extent[2] - snap_x) / cellsize) * cellsize + snap_x
    extent[3] = math.ceil((extent[3] - snap_y) / cellsize) * cellsize + snap_y
    return extent


def ogrenv_swap(extent):
    return [extent[0], extent[2], extent[1], extent[3]]


def extent_shape(extent, cellsize):
    """Return number of rows and columns of the extent"""
    cols = int(round(abs((extent[0] - extent[2]) / cellsize), 0))
    rows = int(round(abs((extent[3] - extent[1]) / -cellsize), 0))
    return rows, cols


def extent_geo(extent, cellsize):
    return (extent[0], cellsize, 0., extent[3], 0., -cellsize)


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Rasterize Polygon Geometry',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'src', help='Input file path')
    parser.add_argument(
        'dst', help='Output file path')
    parser.add_argument(
        '-a_srs',
        help='Output spatial reference')
    parser.add_argument(
        '-e', '--epsg', default=None,
        help='Output spatial reference EPSG code')
    parser.add_argument(
        '-cs', '--cellsize', default=30, type=float,
        help='Output spatial reference')
    parser.add_argument(
        '-s', '--snap', default=[15, 15], type=float, nargs=2,
        help='Snap point (x, y)', metavar=('X', 'Y'))
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if args.src and os.path.isfile(os.path.abspath(args.src)):
        args.src = os.path.abspath(args.src)
    if args.dst and os.path.isfile(os.path.abspath(args.dst)):
        args.dst = os.path.abspath(args.dst)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    main(
        input_path=args.src, output_path=args.dst,
        output_epsg=args.epsg, cellsize=args.cellsize,
        snap=args.snap, overwrite_flag=args.overwrite)
