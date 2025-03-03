


import random

import carla


class Editor():
    def __init__(self, world, args, client):
        """
        'time_of_day',
        'weather',
        'weather_and_time_of_day',
        'lane_marking',
        'building_texture',
        'vehicle_lights',
        'vehicle_color',
        'vehicle_replacement',
        'vehicle_deletion',
        'building_deletion',
        'pedestrian_deletion',
        """
        self.world = world
        self.args = args
        self.client = client
        self.recording_mode = True
        self.records = {}

    def set_recording_mode(self):
        """Turns on recording mode. Records all objects that can be edited when apply is called."""
        if not self.recording_mode:
            self.records = {}
        self.recording_mode = True
        

    def record(self, key, editable_object):
        """Records the object to be edited if recording mode is on."""
        if self.recording_mode:
            self.records[key] = editable_object


    def apply(self, key, editable_object):
        """Applies the edits to the object if recording mode is off, else returns the object."""
        if not self.recording_mode:
            if getattr(self.args, 'time_of_day', False) and key == 'weather':
                editable_object.set_weather(**self.edits_to_insert['time_of_day'])
            if getattr(self.args, 'weather', False) and key == 'weather':
                editable_object.set_weather(**self.edits_to_insert['weather'])
            if getattr(self.args, 'weather_and_time_of_day', False) and key == 'weather':
                editable_object.set_weather(**self.edits_to_insert['weather_and_time_of_day'])
            
            if getattr(self.args, 'vehicle_replacement', False) and key == 'npc_vehicles':
                vehicle_types = editable_object            
                new_vehicle_bp = self.edits_to_insert['vehicle_replacement']['new_vehicle']
                i = self.edits_to_insert['vehicle_replacement']['index']
                old = vehicle_types[i]
                vehicle_types[i] = new_vehicle_bp
                return vehicle_types
        else:       
            return editable_object
            



    def edit(self):
        """Turns off recording mode. Samples an edit to apply when apply is called."""
        self.recording_mode = False
        edits_to_insert = {}
        if getattr(self.args, 'time_of_day', False):
            k, v = self.edit_time_of_day()
            edits_to_insert[k] = v
        if getattr(self.args, 'weather', False):
            k, v = self.edit_weather()
            edits_to_insert[k] = v
        if getattr(self.args, 'weather_and_time_of_day', False):
            k, v = self.edit_weather_and_time_of_day()
            edits_to_insert[k] = v
        if getattr(self.args, 'lane_marking', False):
            k, v = self.edit_lane_marking()
            edits_to_insert[k] = v
        if getattr(self.args, 'building_texture', False):
            k, v = self.edit_building_texture()
            edits_to_insert[k] = v
        if getattr(self.args, 'vehicle_lights', False):
            k, v = self.edit_vehicle_lights()
            edits_to_insert[k] = v
        if getattr(self.args, 'vehicle_color', False):
            k, v = self.edit_vehicle_color()
            edits_to_insert[k] = v
        if getattr(self.args, 'vehicle_replacement', False):
            k, v = self.edit_vehicle_replacement()
            edits_to_insert[k] = v
        if getattr(self.args, 'vehicle_deletion', False):
            k, v = self.edit_vehicle_deletion()
            edits_to_insert[k] = v
        if getattr(self.args, 'building_deletion', False):
            k, v = self.edit_building_deletion()
            edits_to_insert[k] = v
        if getattr(self.args, 'pedestrian_deletion', False):
            k, v = self.edit_pedestrian_deletion()
            edits_to_insert[k] = v

        self.edits_to_insert = edits_to_insert

    def edit_time_of_day(self):
        """sample a time of day and add it to a weather object"""
        time_and_weather_instance = self.records['weather']
        if 'profile' in time_and_weather_instance.kwargs:
            old_profile = time_and_weather_instance.kwargs['profile']
            if old_profile.endswith('Sunset'):
                new_profile = old_profile.replace('Sunset', 'Noon')
            else:
                new_profile = old_profile.replace('Noon', 'Sunset')
            return 'time_of_day', {'profile': new_profile}
        else:
            old_azimuth = time_and_weather_instance.initial_settings.sun_azimuth_angle
            old_altitude = time_and_weather_instance.initial_settings.sun_altitude_angle
            if old_altitude < 0:
                new_altitude = random.uniform(10, 70)
            if old_altitude > 20:
                if random.choice([0, 1]) == 0:
                    new_altitude = random.uniform(-20, 10)
                else:
                    new_altitude = random.uniform(40, 70)
            if old_altitude > 40:
                new_altitude = random.uniform(-20, 10)
            new_azimuth = random.uniform(0, 360)
            return 'time_of_day', {'sun_azimuth_angle': new_azimuth, 'sun_altitude_angle': new_altitude}


    def edit_weather(self):
        time_and_weather_instance = self.records['weather']
        if 'profile' in time_and_weather_instance.kwargs:
            old_profile = time_and_weather_instance.kwargs['profile']
            if old_profile.endswith('Sunset'):
                options = ['ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'SoftRainSunset', 'MidRainSunset', 'HardRainSunset'].remove(old_profile)
            else:
                options = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'SoftRainNoon', 'MidRainyNoon', 'HardRainNoon'].remove(old_profile)
            new_profile = random.choice(options)
            return 'weather_and_time_of_day', {'profile': new_profile}
        else:
            old_precipitation = time_and_weather_instance.initial_settings.precipitation
            if old_precipitation > 0:
                new_precipitation = random.uniform(-100, -50)
            else:
                new_precipitation = random.uniform(10, 99)
            return 'weather', {'precipitation': new_precipitation}

    def edit_weather_and_time_of_day(self):
        time_and_weather_instance = self.records['weather']
        if 'profile' in time_and_weather_instance.kwargs:
            old_profile = time_and_weather_instance.kwargs['profile']
            options = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'SoftRainNoon', 'MidRainyNoon', 'HardRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'SoftRainSunset', 'MidRainSunset', 'HardRainSunset'].remove(old_profile)
            new_profile = random.choice(options)
            return 'weather_and_time_of_day', {'profile': new_profile}
        else:
            # TODO: don't force weather and time to both change always
            old_azimuth = time_and_weather_instance.initial_settings.sun_azimuth_angle
            old_altitude = time_and_weather_instance.initial_settings.sun_altitude_angle
            if old_altitude < 0:
                new_altitude = random.uniform(10, 70)
            if old_altitude > 20:
                if random.choice([0, 1]) == 0:
                    new_altitude = random.uniform(-20, 10)
                else:
                    new_altitude = random.uniform(40, 70)
            if old_altitude > 40:
                new_altitude = random.uniform(-20, 10)
            new_azimuth = random.uniform(0, 360)
            old_precipitation = time_and_weather_instance.initial_settings.precipitation
            if old_precipitation > 0:
                new_precipitation = random.uniform(-100, -50)
            else:
                new_precipitation = random.uniform(10, 99)

            new_weather = {'sun_azimuth_angle': new_azimuth, 'sun_altitude_angle': new_altitude, 'precipitation': new_precipitation}
            return 'weather_and_time_of_day', new_weather

    def edit_lane_marking(self):
        pass

    def edit_building_texture(self):
        pass

    def edit_vehicle_lights(self):
        pass

    def edit_vehicle_color(self):
        pass

    def edit_vehicle_replacement(self):
        """Returns the index of the vehicle to edit and the vehicle blueprint for the replacement vehicle."""
        dictionary = self.records['npc_vehicles']
        list_vehicles = dictionary['vehicles']
        list_indices = dictionary['indices']
        old_vehicles = [(i, v.attributes) for i, v in zip(list_indices, list_vehicles)]
        vehicle_to_edit_tuple = random.choice(old_vehicles)
        vehicle_bp = self.world.get_blueprint_library().filter('*vehicle*')
        new_vehicle = random.choice(vehicle_bp)
        while new_vehicle.id == vehicle_to_edit_tuple[1]['ros_name']:
            new_vehicle = random.choice(vehicle_bp)

        return 'vehicle_replacement', {'index': vehicle_to_edit_tuple[0], 'new_vehicle': new_vehicle}

    def edit_vehicle_deletion(self):
        pass

    def edit_building_deletion(self):
        pass

    def edit_pedestrian_deletion(self):
        pass
