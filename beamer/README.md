# Beamer ET*/ETg

Note, the only difference between the image download INI files is the output workspace.

## ETg zonal stats

#### ee_beamer_zonal_stats.py

This script will compute Beamer ET*/ETg specific zonal statistics.

## ETg Images

There are three scripts for downloading ETg images.  The scripts will attempt to download separate images for each feature in the zones shapefile, so if you want a single image for all zones, the features must be dissolved/merged before running the script.

#### ee_beamer_image_download.py

This script will download each ETg image (based on the start/end date in the INI file) from Earth Engine.  The script will then compute the mean annual ETg for each year (water year) and the median ETg for all years using the local images.

#### ee_beamer_annual_mean_download.py

This script will download the mean annual ETg for each year (water year) from Earth Engine.  The script will then compute the median ETg for all years from the annuals using the local annual images.

#### ee_beamer_median_download.py

This script will download the median ETg for all years (water years) directly from Earth Engine.

