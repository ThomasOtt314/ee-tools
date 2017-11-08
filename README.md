# ee-tools
Earth Engine Zonal Stats and Image Download Tools

The ee-tools can be separated into three main components:
+ Download time series of zonal statistics
+ Download daily Landsat and GRIDMET PPT/ETo imagery

## Requirements

#### Python Dependencies
See the [Dependencies](## dependencies) section below for additional details on the Python specific requirements needed to run the ee-tools.

For information on installing Python and Pandas or details on how to run the Python scripts, please see the [Python README](PYTHON.md).

#### Earth Engine
To run the ee-tools you must have an Earth Engine account.

#### Google Drive
To run the zonal stats and image download scripts, you must have Google Drive installed on your computer.  The scripts first initiate Earth Engine export tasks that will write to your Google Drive, and then copy the files from Google Drive to the output workspace set in the INI file.

## INI Files
All of the scripts are controlled using INI files.  The INI file is structured into sections (defined by square brackets, i.e. [INPUTS]) and key/value pairs separated by an equals sign (i.e. "start_year = 1985").  Additional details on the INI structure can be found in the [Python configparser module documentation](https://docs.python.org/3/library/configparser.html#supported-ini-file-structure).  Example INI files are provided in the example folder.

#### Sections
Each of the scripts reads a different combination of INI sections.  There are seven sections currently used in the scripts:
+ INPUTS - Used by all of the ee-tools
+ EXPORT - Export specific parameters and is read by the zonal statistics and image download scripts.
+ ZONAL_STATS - Zonal stats specific parameters.
+ SPATIAL - Spatial reference (projection, snap point, cellsize) parameters.
+ IMAGES - Image download specific parameters.
+ SUMMARY - Summary specific parameters and is read by the summary figures and summary tables scripts.
+ FIGURES - Summary figure specific parameters.
+ TABLES - Summary table specific parameters.

## Study Area Zones
The user must provide a shapefile of the zones they wish to analyze.  Zonal statistics for each feature in the shapefile will be computed.  Images can be downloaded separately for each zone separately (set IMAGES parameter merge_geometries=False) or as a single image that includes all zones (merge_geometries=True).

The user is strongly encouraged to "rasterize" the study area zones to the UTM Zones of the interecting Landsat path/rows using the provided tool (miscellaneous/polygon_rasterize.py).  This script will adjust the zone geometries to follow the Landsat pixels.  For example, a field in Mason Valley, NV is in the overlap of two Landsat path/rows with different UTM Zones (42/33 - WGS84 Zone 11N / EPSG:32611 and 43/44 - Zone 10N / EPSG:32610), so separate rasterized shapefiles should be generated for each UTM zone.

To rasterize the example shapefile to EPSG 32610 and EPSG 32611, execute the following:
```
> python ..\miscellaneous\polygon_rasterize.py example.shp example_wgs84z10.shp --epsg 32610 -o
> python ..\miscellaneous\polygon_rasterize.py example.shp example_wgs84z11.shp --epsg 32611 -o
```

#### Zone Field
The user must indicate which field in the shapefile to use for setting the "Zone ID".  The field must be an integer or string type and the values must be unique for each feature/zone.  A good default is use the "FID" field since this is guaranteed to be unique and makes it easy to join the output tables to the shapefile.

#### Spatial Reference / Projection
Currently, the output spatial reference set in the INI file (EXPORT parameter "output_proj") must match exactly with the spatial reference of the zones shapefile.  The code should prompt you if they do not match, in which case you should reproject the zones shapefile to the output spatial reference (see [Study Area Zones](#study-area-zones)).  Eventually the code will be able to project the zones geometries to the output projection automatically.

## Zonal Stats
To initiate Earth Engine zonal statistics export tasks, execute the following:
```
> python ee_shapefile_zonal_stats_export.py -i example\example_ee_zs.ini
```

As the export tasks finish, the zonal stats CSV files will be written to your Google drive.  Once all of the exports have finished, rerun the script, and the CSV files will be copied to the output workspace set in the INI file.

#### Output

EE Output field desciptions:
PIXEL_TOTAL - Number of pixels that could nominally be in the zone.
PIXEL_COUNT - Number of pixels with data used in the computation of mean NDVI, Ts, etc.  PIXEL_COUNT should always be <= PIXEL_TOTAL.  PIXEL_COUNT will be lower than PIXEL_TOTAL for zones that are near the edge of the image or cross the scan-line corrector gaps in Landsat 7 images.  Zones that are fully contained within cloud free Landsat 5 and 8 images can have PIXEL_COUNTS equal to PIXEL_TOTAL.
FMASK_TOTAL - Number of pixels with an FMASK value.  FMASK_TOTAL should be equal to PIXEL_COUNT, but may be slightly different for LE7 SCL-off images.
FMASK_COUNT - Number of pixels with FMASK values of 2, 3, or 4 (shadow, snow, and cloud).  FMASK_COUNT should always be <= FMASK_TOTAL.  Cloudy scenes will have high FMASK_COUNTs relative to FMASK_TOTAL.

## QA/QC
The QA/QC script will add the following fields to the daily Landsat CSV file:
FMASK_PCT - Percentage of available pixels that are cloudy (FMASK_COUNT / FMASK_TOTAL)
QA - QA/QC value (higher values are more likely to be cloudy or bad data)
OUTLIER_SCORE - Experimental - Values will be relative to distribution of data

To compute QA/QC values, execute the following:
```
> python summary_qaqc.py -i example\example_summary.ini
```

## Image Download
To download Landsat images, execute the following:
```
> python ee_landsat_image_download.py -i example\example_images.ini
```

To download GRIDMET ETo/PPT images, execute the following:
```
> python ee_gridmet_image_download.py -i example\example_images.ini
```

The download scripts must be run twice (like the zonal stats script) in order to first export the TIF files to your Google drive and then copy them to the output workspace.

## Landsat Thumbnail Download
To download Landsat thumbnail images for each zone, execute the following:
```
> python ee_landsat_thumbnail_download.py -i example\example_summary.ini
```

Currently you must use the "summary" or "zonal stats" INI file to set the output workspace.
The Landsat thumbnail script must also be run after zonal statistics have been computed, since it reads the landsat_daily.csv to determine which images to download.

## Dependencies
The EE-Tools have been tested using both Python 3.6 and Python 2.7 (using the "configparser" backport and "future" module, see below).

The following modules must be present to run all of the EE-Tools:
* [numpy](http://www.numpy.org)
* [pandas](http://pandas.pydata.org)
* [gdal](http://gdal.org/)
* [dateutil](http://dateutil.readthedocs.io/en/stable/relativedelta.html)
* [earthengine-api](https://github.com/google/earthengine-api)
* [requests](http://docs.python-requests.org/en/master/)

The following module is used to run the test suite
* [pytest](http://doc.pytest.org/en/latest/)

The following modules must be present if using Python 2.7
* [configparser](https://docs.python.org/3/library/configparser.html) (backport of Python 3.X configparser module)
* [future](http://python-future.org/)

#### EarthEngine-API / PIP
The EarthEngine API must be installed through pip:
```
> pip install earthengine-api
```

After installing the EarthEngine API module, you will need to authenticate the Earth Engine API (see [setting-up-authentication-credentials](https://developers.google.com/earth-engine/python_install#setting-up-authentication-credentials)):
```
> python -c "import ee; ee.Initialize()"
```

#### GDAL
After installing GDAL, you may need to manually set the GDAL_DATA user environmental variable.

###### Windows
You can check the current value of the variable at the command prompt:
```
echo %GDAL_DATA%
```

If GDAL_DATA is set, this will return a folder path (something similar to C:\Anaconda2\Library\share\gdal)

If GDAL_DATA is not set, it can be set from the command prompt (note, your path may vary):
```
> setx GDAL_DATA "C:\Anaconda2\Library\share\gdal"
```

The GDAL_DATA environment variable can also be set through the Windows Control Panel (System -> Advanced system settings -> Environment Variables).

#### ArcPy
Currently the ArcGIS ArcPy module is used for computing raster statistics in some of the modules.  This dependency will eventually be removed.

## Code

#### Style Guide
All Python code should follow the [PEP8 Style Guide](https://www.python.org/dev/peps/pep-0008/).

#### Tests
The full test suite can be run using [Pytest](http://doc.pytest.org/en/latest/):
```
> python -m pytest
```
