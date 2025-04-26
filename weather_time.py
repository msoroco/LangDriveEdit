#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import glob
import os
import sys

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

import carla

import argparse
import math


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)



class time_and_weather():
    def __init__(self, world, speed_factor=0.005,  **kwargs):
        """
        if `random` is True, the weather is set to random values based on values in `kwargs`. 
        
        TODO: For realistic weather the only parameter that can be set is the time...
        
        ### Realistic:
        Change Sun position smoothly with time and generate storms occasionally.
        There are two types of storms: light rain and heavy rain.
        By default the rate at which the weather changes (`speed_factor`) is slowed down to be almost constant weather during shorter videos.
        ### Unrealistic:
        individual parameters are modified without considering the others.
        The parameters will be constant.        
        """
        self.world = world
        self.kwargs = kwargs
        self.speed_factor = speed_factor

        update_freq = 0.1 / speed_factor

        # world.set_weather(carla.WeatherParameters.WetCloudySunset)
        # https://carla.readthedocs.io/en/0.9.8/python_api/#carla.WeatherParameters:~:text=Note%3A-,ClearNoon,-%2C%20CloudyNoon%2C%20WetNoon%2C%20WetCloudyNoon

        # if self.realistic_weather:
        #     if 'profile' in kwargs:
        #         world.set_weather(getattr(carla.WeatherParameters, kwargs['profile']))
        #         self.weather = Weather(world.get_weather())
        #     else:
        #         weather = world.get_weather()
        #         for key, value in kwargs.items():
        #             setattr(weather, key, value)
        #         world.set_weather(weather)
        #         self.weather = Weather(world.get_weather())
        # else:
        #     # set the weather via settings using kwards specified:
        #     weather = world.get_weather()
        #     for key, value in kwargs.items():
        #         setattr(weather, key, value)
        #     world.set_weather(weather)

    def set_weather(self, **kwargs):
        if 'profile' in kwargs:
            if kwargs['profile'] != 'MidRainyNoon':
                if 'Rainy' in kwargs['profile']:
                    # replace Rainy with Rain
                    kwargs['profile'] = kwargs['profile'].replace('Rainy', 'Rain')
            self.profile = kwargs['profile']
            self.world.set_weather(getattr(carla.WeatherParameters, kwargs['profile']))
            self.weather = Weather(self.world.get_weather())
            self.initial_settings = self.world.get_weather()
            self.constant = True
        else:
            weather = self.world.get_weather()
            for key, value in kwargs.items():
                setattr(weather, key, value)
            self.world.set_weather(weather)
            self.weather = Weather(self.world.get_weather())
            self.initial_settings = self.world.get_weather()
            self.constant = False
        self.tick(0.5)

    def tick(self, elapsed_time):
        if not self.constant:
            self.weather.tick(self.speed_factor * elapsed_time)
            self.world.set_weather(self.weather.weather)

    def __str__(self):
        return str(self.weather) + 12 * ' '