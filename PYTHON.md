# Python

The cloud free scene count Python scripts have been tested using both Python 3.6 and Python 2.7.

## Anaconda

The easiest way of obtaining Python and all of the necessary external modules, is to install [Anaconda](https://www.continuum.io/downloads). Another (more prefereable) option is to download miniconda.

It is important to double check that you are calling the Anaconda version, especially if you have two or more version of Python installed (e.g. Anaconda and ArcGIS).

+ Windows: "where python"
+ Linux/Mac: "which python"

 After installing Anaconda or miniconda, add the conda-forge channel by entering the following in the command prompt or terminal:
```
> conda config --add channels conda-forge
```

#### Installing/Updating Python Modules

Most of the modules needed for these scripts are installed by default with Anaconda but additional modules will need to be installed (and/or updated) using "conda".  For example to install the pandas module, enter the following in a command prompt or terminal window:

```
conda install pandas
```

To update the pandas module to the latest version, enter the following in a command prompt or terminal window:

```
conda update pandas
```

The external modules can also be installed all together with the following command:
```
> conda install configparser gdal numpy pandas
```

## Command Prompt / Terminal

The python scripts can be run from the terminal (mac/linux) or command prompt (windows).

In some cases the scripts can also be run by double clicking directly on the script.  The script will open a GUI asking you select an INI file.  Be advised, if you have multiple versions of Python installed (for example if you have ArcGIS and you install Anaconda), this may try to use a different different version of Python.

#### Help
To see what arguments are available for a script, and their default values, pass the "-h" argument to the script.
```
> python ee_shapefile_zonal_stats_export.py -h
usage: ee_shapefile_zonal_stats_export.py [-h] [-i PATH] [-d] [-o]

Earth Engine Zonal Statistics

optional arguments:
  -h, --help            show this help message and exit
  -i FILE, --ini FILE   Input file (default: None)
  -d, --debug           Debug level logging (default: 20)
  -o, --overwrite       Force overwrite of existing files (default: False)
```

#### Input file
To set the input file, use the "-i" or "--ini" argument.  The INI file path can be absolute or relative to the current working directory.
```
> python ee_shapefile_zonal_stats_export.py -i example\example_ee_zs.ini
```

#### Overwrite

#### Debug
