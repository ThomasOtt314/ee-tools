#--------------------------------
# Name:         ee_common.py
# Purpose:      Common EarthEngine support functions
# Python:       3.6
#--------------------------------

import logging
import math

import ee


system_properties = ['system:index', 'system:time_start']

refl_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
refl_toa_bands = [b + '_toa' for b in refl_bands]
refl_sur_bands = [b + '_sur' for b in refl_bands]


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


# def scene_id_func(img):
#     """Construct Collecton 1 short SCENE_ID for collection 1 images
#
#     LT05_PPPRRR_YYYYMMDD
#     Format matches EE collection 1 system:index
#     Split on '_' in case the collection was merged first
#     """
#     scene_id = ee.List(ee.String(
#         img.get('system:index')).split('_')).slice(-3)
#     scene_id = ee.String(scene_id.get(0)).cat('_') \
#         .cat(ee.String(scene_id.get(1))).cat('_') \
#         .cat(ee.String(scene_id.get(2)))
#     return img.setMulti({'SCENE_ID': scene_id})



def pair_func(elev_image):
    """Elevation based air pressure"""
    return elev_image.expression(
        '101.3 * pow((293 - 0.0065 * b()) / 293, 5.26)')


# def prism_ppt_func(prism_image):
#     """PRISM water year precipitation
#
#     Depends on maps engine assets
#     """
#     return prism_image.select([0], ['ppt']) \
#         .copyProperties(prism_image, system_properties)


# DEADBEEF - Using GRIDMET precipitation band directly
# def gridmet_ppt_func(gridmet_image):
#     """GRIDMET daily precipitation"""
#     return gridmet_image.select(['pr'], ['ppt']).max(0) \
#         .copyProperties(gridmet_image, system_properties)


# DEADBEEF - Using GRIDMET ETo/ETr bands directly
# DEADBEEF - Not using geerefet for ETo/ETr calculation
# def gridmet_eto_func(gridmet_image):
#     """GRIDMET Daily ETo"""
#     return ee.Image(geerefet.Daily.gridmet(gridmet_image).eto()).max(0)\
#         .copyProperties(gridmet_image, system_properties)
#
# def gridmet_etr_func(gridmet_image):
#     """GRIDMET Daily ETr"""
#     return ee.Image(geerefet.Daily.gridmet(gridmet_image).etr()).max(0)\
#         .copyProperties(gridmet_image, system_properties)


# DEADBEEF - Using GRIDMET ETo/ETr bands directly
# DEADBEEF - Not computing ETo/ETr from component bands
# def gridmet_etr_func(gridmet_image):
#     """GRIDMET Daily ETr"""
#     scene_date = ee.Algorithms.Date(gridmet_image.get('system:time_start'))
#     doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
#
#     # Read in GRIDMET layers
#     tmin = gridmet_image.select(['tmmn']).subtract(273.15)  # K to C
#     tmax = gridmet_image.select(['tmmx']).subtract(273.15)  # K to C
#     # rhmin = gridmet_image.select(['rmin']).multiply(0.01)  # % to decimal
#     # rhmax = gridmet_image.select(['rmax']).multiply(0.01)  # % to decimal
#     q = gridmet_image.select(['sph'])                      # kg kg-1
#     rs = gridmet_image.select(['srad']).multiply(0.0864)   # W m-2 to MJ m-2 day-1
#     uz = gridmet_image.select(['vs'])                      # m/s?
#     zw = 10.0    # Windspeed measurement/estimated height (GRIDMET=10m)
#
#     # Vapor pressure from RHmax and RHmin (Eqn 11)
#     # ea = es_tmin.multiply(rhmax).add(es_tmax.multiply(rhmin)).multiply(0.5)
#     # Vapor pressure from specific humidity (Eqn )
#     # To match standardized form, ea is calculated from elevation based pair
#     pair = pair_func(ee.Image('USGS/NED'))
#     ea = pair.expression(
#         'q * pair / (0.622 + 0.378 * q)', {'pair': pair, 'q': q})
#
#     return daily_pet_func(
#         doy, tmin, tmax, ea, rs, uz, zw, 1600, 0.38).copyProperties(
#             gridmet_image, system_properties)
#
# def gridmet_eto_func(gridmet_image):
#     """GRIDMET Daily ETo"""
#     scene_date = ee.Algorithms.Date(gridmet_image.get('system:time_start'))
#     doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
#
#     # Read in GRIDMET layers
#     tmin = gridmet_image.select(['tmmn']).subtract(273.15)  # K to C
#     tmax = gridmet_image.select(['tmmx']).subtract(273.15)  # K to C
#     # rhmin = gridmet_image.select(['rmin']).multiply(0.01)  # % to decimal
#     # rhmax = gridmet_image.select(['rmax']).multiply(0.01)  # % to decimal
#     q = gridmet_image.select(['sph'])                      # kg kg-1
#     rs = gridmet_image.select(['srad']).multiply(0.0864)   # W m-2 to MJ m-2 day-1
#     uz = gridmet_image.select(['vs'])                      # m/s?
#     zw = 10.0  # Windspeed measurement/estimated height (GRIDMET=10m)
#
#     # Vapor pressure from RHmax and RHmin (Eqn 11)
#     # ea = es_tmin.multiply(rhmax).add(es_tmax.multiply(rhmin)).multiply(0.5)
#     # Vapor pressure from specific humidity (Eqn )
#     # To match standardized form, ea is calculated from elevation based pair
#     pair = pair_func(ee.Image('USGS/NED'))
#     ea = pair.expression(
#         'q * pair / (0.622 + 0.378 * q)', {'pair': pair, 'q': q})
#
#     return daily_pet_func(
#         doy, tmin, tmax, ea, rs, uz, zw, 900, 0.34).copyProperties(
#             gridmet_image, system_properties)
#
#
# def daily_pet_func(doy, tmin, tmax, ea, rs, uz, zw, cn=900, cd=0.34):
#     """Daily ASCE Penman Monteith Standardized Reference ET
#
#     Daily ETo cn=900, cd=0.34
#     Daily ETr cn=1600, cd=0.38
#
#     doy -- day of year
#     tmin -- minimum daily temperature [C]
#     tmax -- maximum daily temperature [C]
#     ea -- vapor pressure [?]
#     rs -- incoming solar radiation [MJ m-2 day]
#     uz -- wind speed [m s-1]
#     zw -- wind speed height [m]
#     cn -- coefficient
#     cd -- coefficient
#
#     """
#     # Globals in playground/javascript
#     pi = math.pi
#     lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi / 180)
#     lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi / 180)
#     elev = ee.Image('USGS/NED')
#     pair = pair_func(elev)
#
#     # Calculations
#     tmean = tmin.add(tmax).multiply(0.5)  # C
#     psy = pair.multiply(0.000665)
#     es_tmax = vapor_pressure_func(tmax)  # C
#     es_tmin = vapor_pressure_func(tmin)  # C
#     es_tmean = vapor_pressure_func(tmean)
#     es_slope = es_tmean.expression(
#         '4098 * es / (pow((t + 237.3), 2))', {'es': es_tmean, 't': tmean})
#     es = es_tmin.add(es_tmax).multiply(0.5)
#
#     # Extraterrestrial radiation (Eqn 24, 27, 23, 21)
#     delta = ee.Image.constant(
#         doy.multiply(2 * pi / 365).subtract(1.39435).sin().multiply(0.40928))
#     omegas = lat.expression(
#         'acos(-tan(lat) * tan(delta))', {'lat': lat, 'delta': delta})
#     theta = omegas.expression(
#         'omegas * sin(lat) * sin(delta) + cos(lat) * cos(delta) * sin(b())',
#         {'omegas': omegas, 'lat': lat, 'delta': delta})
#     dr = ee.Image.constant(
#         doy.multiply(2 * pi / 365).cos().multiply(0.033).add(1))
#     ra = theta.expression(
#         '(24 / pi) * gsc * dr * theta',
#         {'pi': pi, 'gsc': 4.92, 'dr': dr, 'theta': theta})
#
#     # Simplified clear sky solar formulation (Eqn 19)
#     # var rso = elev.expression(
#     #     '(0.75 + 2E-5 * elev) * ra', {'elev':elev, 'ra':ra})
#
#     # This is the full clear sky solar formulation
#     # sin of the angle of the sun above the horizon (D.5 and Eqn 62)
#     sin_beta_24 = lat.expression(
#         'sin(0.85 + 0.3 * lat * delta / 0.40928 - 0.42 * lat ** 2)',
#         {'lat': lat, 'delta': delta})
#
#     # Precipitable water (Eqn D.3)
#     w = pair.expression(
#         '0.14 * ea * pair + 2.1', {'pair': pair, 'ea': ea})
#
#     # Clearness index for direct beam radiation (Eqn D.2)
#     # Limit sin_beta >= 0.01 so that KB does not go undefined
#     kb = pair.expression(
#         '0.98 * exp((-0.00146 * pair) / (kt * sin_beta) - '
#         '0.075 * pow((w / sin_beta), 0.4))',
#         {'pair': pair, 'kt': 1.0, 'sin_beta': sin_beta_24.max(0.01), 'w': w})
#
#     # Transmissivity index for diffuse radiation (Eqn D.4)
#     kd = kb.multiply(-0.36).add(0.35).min(kb.multiply(0.82).add(0.18))
#     # var kd = kb.multiply(-0.36).add(0.35)
#     #     .where(kb.lt(0.15), kb.multiply(0.82).add(0.18))
#
#     # (Eqn D.1)
#     rso = ra.multiply(kb.add(kd))
#     # Cloudiness fraction (Eqn 18)
#     fcd = rs.divide(rso).clamp(0.3, 1).multiply(1.35).subtract(0.35)
#
#     # Net long-wave radiation (Eqn 17)
#     rnl = ea.expression(
#         ('4.901E-9 * fcd * (0.34 - 0.14 * sqrt(ea)) * '
#          '(pow(tmax_k, 4) + pow(tmin_k, 4)) / 2'),
#         {'ea': ea, 'fcd': fcd,
#          'tmax_k': tmax.add(273.15), 'tmin_k': tmin.add(273.15)})
#
#     # Net radiation (Eqns 15 and 16)
#     rn = rs.multiply(0.77).subtract(rnl)
#
#     # Wind speed (Eqn 33)
#     u2 = uz.expression('b() * 4.87 / log(67.8 * zw - 5.42)', {'zw': zw})
#
#     # Daily ETo (Eqn 1)
#     return tmin.expression(
#         '(0.408 * slope * (rn - g) + (psy * cn * u2 * (es - ea) / (t + 273))) / '
#         '(slope + psy * (cd * u2 + 1))',
#         {'slope': es_slope, 'rn': rn, 'g': 0, 'psy': psy, 'cn': cn,
#          't': tmean, 'u2': u2, 'es': es, 'ea': ea, 'cd': cd})


# def vapor_pressure_func(temperature_image):
#     """Vapor Pressure
#
#     in kPa with temperature in C
#     """
#     return temperature_image.expression(
#         '0.6108 * exp(17.27 * b() / (b() + 237.3))')
