# ee-tools
Earth Engine Zonal Stats and Image Download Tools

## Zonal Stats



## Image Download



## Using the Tools



## Dependencies

The EE-Tools have only been tested using Python 2.7 but they may work with Python 3.x.

The following modules must be present to run all of the EE-Tools:
* [numpy](http://www.numpy.org)
* [pandas](http://pandas.pydata.org)
* [configparser](https://docs.python.org/3/library/configparser.html) (backport from Python 3.X)
* [gdal](http://gdal.org/)
* [relativedelta](http://dateutil.readthedocs.io/en/stable/relativedelta.html)
* [earthengine-api](https://github.com/google/earthengine-api)

#### Anaconda

The easiest way of obtaining Python and all of the necessary external modules, is to install [Anaconda](https://www.continuum.io/downloads).

It is important to double check that you are calling the Anaconda version, especially if you have two or more version of Python installed (e.g. Anaconda and ArcGIS).

+ Windows: "where python"
+ Linux/Mac: "which python"

After installing Anaconda, add the conda-forge channel by entering the following in the command prompt or terminal:

```
> conda config --add channels conda-forge
```

The external modules can then be installed one by one:
```
> conda install numpy
> conda install gdal
...
```

or all together:
```
> conda install gdal numpy pandas configparser
```

#### EarthEngine-API / PIP

The EarthEngine API must be installed through pip:
```
> pip install earthengine-api
```

#### ArcPy

The ArcGIS ArcPy module may be needed for some operations.
