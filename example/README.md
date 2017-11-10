# Example
To run EE-Tools example, execute the following scripts in order.

Compute zonal statistics and export CSV files to Google Drive
```
> python ee_zonal_stats_by_zone.py -i example\example_ee_zs.ini
```

Rerun the zonal stats script to copy the CSV files from Google Drive to the local folder
```
> python ee_zonal_stats_by_zone.py -i example\example_ee_zs.ini
```

Compute QA/QC flags
```
> python summary_qaqc.py -i example\example_summary.ini
```

Generate summary tables
```
> python summary_tables.py -i example\example_summary.ini
```

Generate summary figures
```
> python summary_figures.py -i example\example_summary.ini
```

Generate summary timeseries Bokeh plots
```
> python summary_timeseries.py -i example\example_summary.ini
```

Download Landsat thumbnails
```
> python ee_landsat_thumbnail_download.py -i example\example_summary.ini
```

Download full Landsat images (one image for all zones)
```
> python ee_landsat_image_download.py -i example\example_ee_images.ini
```

Download full GRIDMET images (one image for all zones)
```
> python ee_gridmet_image_download.py -i example\example_ee_images.ini
```
