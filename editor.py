


import os
import random
from PIL import Image
import carla
from utils import synthesize_texture


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
            if key == 'weather':
                if getattr(self.args, 'time_of_day', False):
                    # editable_object.set_weather(**self.edits_to_insert['time_of_day'])
                    return self.edits_to_insert['time_of_day']
                if getattr(self.args, 'weather', False):
                    # editable_object.set_weather(**self.edits_to_insert['weather'])
                    return self.edits_to_insert['weather']
                if getattr(self.args, 'weather_and_time_of_day', False):
                    # editable_object.set_weather(**self.edits_to_insert['weather_and_time_of_day'])
                    return self.edits_to_insert['weather_and_time_of_day']
                return editable_object
            elif key == 'npc_vehicles':            
                if getattr(self.args, 'vehicle_replacement', False):
                    vehicle_types = editable_object['vehicle_names']
                    edits = self.edits_to_insert['vehicle_replacement']
                    for edit in edits:
                        i = edit['index']
                        new_vehicle_bp = edit['new_vehicle']
                        old = vehicle_types[i]
                        vehicle_types[i] = new_vehicle_bp       
                    return {'vehicle_names': vehicle_types, 'spawn_points': editable_object['spawn_points'], 'vehicle_colors': editable_object['vehicle_colors']}
                if getattr(self.args, 'vehicle_color', False):
                    vehicle_colors = editable_object['vehicle_colors']
                    edits = self.edits_to_insert['vehicle_color']
                    for edit in edits:
                        i = edit['index']
                        new_vehicle_color = edit['new_vehicle_color']
                        old = vehicle_colors[i]
                        vehicle_colors[i] = new_vehicle_color
                    return {'vehicle_names': editable_object['vehicle_names'], 'spawn_points': editable_object['spawn_points'], 'vehicle_colors': vehicle_colors}
                if getattr(self.args, 'vehicle_deletion', False):
                    vehicle_types = editable_object['vehicle_names']
                    vehicle_spawn_points = editable_object['spawn_points']
                    edits = self.edits_to_insert['vehicle_deletion']
                    edits = sorted(edits, key=lambda x: x['index'], reverse=True)
                    for edit in edits:
                        # delete the indices in reverse order
                        i = edit['index']
                        vehicle_types.pop(i)
                        vehicle_spawn_points.pop(i)
                    return {'vehicle_names': vehicle_types, 'spawn_points': vehicle_spawn_points, 'vehicle_colors': editable_object['vehicle_colors']}
                else:
                    return editable_object
            elif key == 'npc_walkers':
                if getattr(self.args, 'walker_replacement', False):
                    walker_types = editable_object['walker_names']
                    edits = self.edits_to_insert['walker_replacement']
                    for edit in edits:
                        i = edit['index']
                        new_walker_bp = edit['new_walker']
                        old = walker_types[i]
                        walker_types[i] = new_walker_bp
                    return {'walker_names': walker_types, 'spawn_points': editable_object['spawn_points']}
                if getattr(self.args, 'walker_color', False):
                    walker_types = editable_object['walker_names']
                    edits = self.edits_to_insert['walker_color']
                    for edit in edits:
                        i = edit['index']
                        new_walker_bp = edit['new_walker_id']
                        old = walker_types[i]
                        walker_types[i] = new_walker_bp
                    return {'walker_names': walker_types, 'spawn_points': editable_object['spawn_points']}
                if getattr(self.args, 'walker_deletion', False):
                    walker_types = editable_object['walker_names']
                    walker_spawn_points = editable_object['spawn_points']
                    edits = self.edits_to_insert['walker_deletion']
                    edits = sorted(edits, key=lambda x: x['index'], reverse=True)
                    for edit in edits:
                        i = edit['index']
                        walker_types.pop(i)
                        walker_spawn_points.pop(i)
                    return {'walker_names': walker_types, 'spawn_points': walker_spawn_points}
                else:
                    return editable_object
            elif key == 'road_texture':    
                if getattr(self.args, 'road_texture', False):
                    edits = self.edits_to_insert['road_texture']
                    for edit in edits:
                        road_object = edit['road']
                        road_texture = edit['texture']
                        rotation = edit['rotation']
                        # TODO: height and width
                        self.apply_texture(road_object, road_texture, rotate=rotation)
            elif key == 'sidewalk_texture':    
                if getattr(self.args, 'sidewalk_texture', False):
                    edits = self.edits_to_insert['sidewalk_texture']
                    for edit in edits:
                        road_object = edit['sidewalk']
                        road_texture = edit['texture']
                        # TODO: height and width
                        self.apply_texture(road_object, road_texture)
            
            elif key == 'building_texture':
                if getattr(self.args, 'building_texture', False):
                    building_object = editable_object['building']
                    building_texture = editable_object['texture']
                    integer = editable_object['integer']
                    rotation = 90 * (integer % 4)
                    # TODO: height and width
                    self.apply_texture(building_object, building_texture, rotate=rotation)
        else:       
            return editable_object
            



    def edit(self):
        """Turns off recording mode. Samples an edit to apply which will be applied when apply() is called."""
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
        # TODO: implement this to eid the building texture more realistically?
        # if getattr(self.args, 'building_texture', False):
        #     k, v = self.edit_building_texture()
        #     edits_to_insert[k] = v
        if getattr(self.args, 'road_texture', False):
            k, v = self.edit_road_texture()
            edits_to_insert[k] = v
        if getattr(self.args, 'sidewalk_texture', False):
            k, v = self.edit_sidewalk_texture()
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
        if getattr(self.args, 'walker_color', False):
            k, v = self.edit_walker_color()
            edits_to_insert[k] = v
        if getattr(self.args, 'walker_replacement', False):
            k, v = self.edit_walker_replacement()
            edits_to_insert[k] = v
        if getattr(self.args, 'walker_deletion', False):
            k, v = self.edit_walker_deletion()
            edits_to_insert[k] = v
        if getattr(self.args, 'building_deletion', False):
            k, v = self.edit_building_deletion()
            edits_to_insert[k] = v

        self.edits_to_insert = edits_to_insert
        # print(self.records)

    def edit_time_of_day(self):
        """sample a time of day and add it to a weather object"""
        weather_kwargs = self.records['weather']
        if 'profile' in weather_kwargs:
            old_profile = weather_kwargs['profile']
            if old_profile.endswith('Sunset'):
                new_profile = old_profile.replace('Sunset', 'Noon')
            else:
                new_profile = old_profile.replace('Noon', 'Sunset')
            return 'time_of_day', {'profile': new_profile}
        else:
            old_azimuth = weather_kwargs.sun_azimuth_angle
            old_altitude = weather_kwargs.sun_altitude_angle
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
        weather_kwargs = self.records['weather']
        if 'profile' in weather_kwargs:
            old_profile = weather_kwargs['profile']
            # print(old_profile)
            if old_profile.endswith('Sunset'):
                options = ['ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'SoftRainSunset', 'MidRainSunset', 'HardRainSunset']
            else:
                options = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'SoftRainNoon', 'MidRainyNoon', 'HardRainNoon']
            options.remove(old_profile)
            # print(options)
            new_profile = random.choice(options)
            return 'weather', {'profile': new_profile}
        else:
            old_precipitation = weather_kwargs.precipitation
            if old_precipitation > 0:
                new_precipitation = random.uniform(-100, -50)
            else:
                new_precipitation = random.uniform(10, 99)
            return 'weather', {'precipitation': new_precipitation}

    def edit_weather_and_time_of_day(self):
        weather_kwargs = self.records['weather']
        if 'profile' in weather_kwargs:
            old_profile = weather_kwargs['profile']
            options = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'SoftRainNoon', 'MidRainyNoon', 'HardRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'SoftRainSunset', 'MidRainSunset', 'HardRainSunset']
        
            options.remove(old_profile)
            new_profile = random.choice(options)
            return 'weather_and_time_of_day', {'profile': new_profile}
        else:
            # TODO: don't force weather and time to both change always
            old_azimuth = weather_kwargs.sun_azimuth_angle
            old_altitude = weather_kwargs.sun_altitude_angle
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
            old_precipitation = weather_kwargs.precipitation
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

    def edit_building_deletion(self):
        pass

    def edit_vehicle_color(self):
        dictionary = self.records['npc_vehicles']
        list_indices = dictionary['indices']
        tuples = random.sample(list_indices, len(list_indices))
        edits = []
        for index in tuples:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color = f"{color[0]},{color[1]},{color[2]}"

            edits.append({'index': index, 'new_vehicle_color': color})
        return 'vehicle_color', edits

    def edit_vehicle_replacement(self):
        """Returns the index of the vehicle to edit and the vehicle blueprint for the replacement vehicle."""
        dictionary = self.records['npc_vehicles']
        list_vehicles = dictionary['vehicle_names']
        list_indices = dictionary['indices']
        old_vehicles = [(i, v) for i, v in zip(list_indices, list_vehicles)]

        vehicle_bp = self.world.get_blueprint_library().filter('*vehicle*')
        tuples = random.sample(old_vehicles, len(old_vehicles))
        edits = []
        for index, old_name in tuples:
            new_vehicle = random.choice(vehicle_bp)
            while new_vehicle.id == old_name:
                new_vehicle = random.choice(vehicle_bp)

            edits.append({'index': index, 'new_vehicle': new_vehicle.id})
        return 'vehicle_replacement', edits

    def edit_vehicle_deletion(self):
        dictionary = self.records['npc_vehicles']
        list_vehicles = dictionary['vehicle_names']
        list_indices = dictionary['indices']
        indices_to_delete = list_indices #random.sample(list_indices, random.randint(1, len(list_indices)))
        edits = []
        indices_to_delete = sorted(list_indices, reverse=True)  # sort indices in reverse order
        for i in indices_to_delete:
            edits.append({'index': i})
        return 'vehicle_deletion', edits
    

    def edit_walker_color(self):
        dictionary = self.records['npc_walkers']
        list_walkers = dictionary['walker_names']
        list_indices = dictionary['indices']
        old_walkers = [(i, w) for i, w in zip(list_indices, list_walkers)]
        tuples = random.sample(old_walkers, len(old_walkers))
        edits = []
        for index, old_type in tuples:
            # print("old_type", old_type)
            # old_walker_id = int(old_type.split('.')[-1])
            parts = old_type.split('.')
            if len(parts) == 3 and parts[0] == "walker" and parts[1] == "pedestrian":
                try:
                    old_walker_id = int(parts[2])
                except ValueError:
                    # print(f"Invalid walker ID format: {old_type}")
                    continue
            else:
                # print(f"Invalid walker type format: {old_type}")
                continue
            # print("old_id", old_walker_id)
            new_walker_id = self.get_walker_variant(old_walker_id)
            # print("new_id", new_walker_id)
            if new_walker_id is None:
                walker_id = old_walker_id
                # walker_bp = self.world.get_blueprint_library().find("walker.pedestrian." + f"{old_walker_id:04d}")
            else:
                walker_id = new_walker_id
                # walker_bp = self.world.get_blueprint_library().find("walker.pedestrian." + new_walker_id)

            edits.append({'index': index, 'new_walker_id': "walker.pedestrian." + f"{int(walker_id):04d}"})
        return 'walker_color', edits


    def edit_walker_replacement(self):
        dictionary = self.records['npc_walkers']
        list_walkers = dictionary['walker_names']
        list_indices = dictionary['indices']
        old_walkers = [(i, w) for i, w in zip(list_indices, list_walkers)]

        walker_bp = self.world.get_blueprint_library().filter('*walker*')
        tuples = random.sample(old_walkers, len(old_walkers))
        edits = []
        for index, old_type in tuples:
            new_walker = random.choice(walker_bp)
            while new_walker.id == old_type:
                new_walker = random.choice(walker_bp)
            edits.append({'index': index, 'new_walker': new_walker.id})
        return 'walker_replacement', edits
    
    def edit_walker_deletion(self):
        dictionary = self.records['npc_walkers']
        list_walkers = dictionary['walker_names']
        list_indices = dictionary['indices']
        indices_to_delete = list_indices # random.sample(list_indices, random.randint(1, len(list_indices)))
        edits = []
        indices_to_delete = sorted(list_indices, reverse=True)
        for i in indices_to_delete:
            edits.append({'index': i})
        return 'walker_deletion', edits


    def edit_road_texture(self):
        dictionary = self.records['road_texture']
        roads = dictionary['roads']
        road_texture_dir = os.path.join(os.path.dirname(__file__), 'textures', 'road')
        texture_files = os.listdir(road_texture_dir)
        edits = []
        texture_file = random.choice(texture_files)
        for road in roads:
            # texture_file = random.choice(texture_files)
            rotation = random.choice([0, 90, 180, 270])
            texture_path = os.path.join(road_texture_dir, texture_file)
            edits.append({'road': road, 'texture': texture_path, 'rotation': rotation})
        return 'road_texture', edits

    def edit_sidewalk_texture(self):
        dictionary = self.records['sidewalk_texture']
        roads = dictionary['sidewalks']
        road_texture_dir = os.path.join(os.path.dirname(__file__), 'textures', 'road')
        texture_files = os.listdir(road_texture_dir)
        edits = []
        texture_file = random.choice(texture_files)
        for road in roads:
            # texture_file = random.choice(texture_files)
            texture_path = os.path.join(road_texture_dir, texture_file)
            edits.append({'sidewalk': road, 'texture': texture_path})
        return 'sidewalk_texture', edits


    def apply_texture(self, object_to_edit, texture_path, height=256, width=256, rotate=0):
        image = Image.open(texture_path)
        image = image.convert("RGB")
        # rotate the image by rotate degrees:
        image = image.rotate(rotate)
        # image = Image.open('/home/mai/msoroco/carla/grass_texture.jpeg')
        image = image.resize((width, height))
        height = image.size[1]
        width = image.size[0]

        texture = carla.TextureColor(width, height)
        for x in range(0, width):
            for y in range(0, height):
                color = image.getpixel((x, y))
                r = int(color[0])
                g = int(color[1])
                b = int(color[2])
                a = 255
                texture.set(x, y, carla.Color(r, g, b, a))

        print(f"Applying texture to actor: {object_to_edit}")
        self.world.apply_color_texture_to_object(object_to_edit, carla.MaterialParameter.Diffuse, texture)


    def apply_texture_efros_leung(self, object_to_edit, texture_path, height=256, width=256, window_size=64):
        image = Image.open(texture_path)
        image = image.resize((width, height))
        height = image.size[1]
        width = image.size[0]

        # window_size = min(height / window_size, width / window_size)
        synthesized_texture = synthesize_texture(image, (width, height), window_size)

        texture = carla.TextureColor(width, height)
        for x in range(0, width):
            for y in range(0, height):
                color = synthesized_texture[y, x]
                r = int(color[0])
                g = int(color[1])
                b = int(color[2])
                a = 255
                texture.set(x, y, carla.Color(r, g, b, a))

        # print(f"Applying texture to actor: {object_to_edit}")
        self.world.apply_color_texture_to_object(object_to_edit, carla.MaterialParameter.Diffuse, texture)




    def get_walker_variant(self, variant_number):
        # ue4 older carla.
        variants = [
            ('Adult pedestrian - 1', [1, 5, 6, 7, 8]),
            ('Adult pedestrian - 2', [4,3,2]),
            ('Adult pedestrian - 3', [15,19,]),
            ('Adult pedestrian - 4', [16, 17]),
            ('Adult pedestrian - 5', [26, 18]),
            ('Adult pedestrian - 6', [21, 20]),
            ('Adult pedestrian - 7', [23, 22]),
            ('Adult pedestrian - 8', [24, 25]),
            ('Adult pedestrian - 9', [27, 29, 28]),
            ('Adult pedestrian - 10', [41, 40, 33, 31]),
            ('Adult pedestrian - 11', [34, 38]),
            ('Adult pedestrian - 12', [35,36,37]),
            ('Adult pedestrian - 13', [39]),
            ('Adult pedestrian - 14', [42,43, 44]),
            ('Adult pedestrian - 15', [47, 46, 45]),
            ('Child pedestrian - 1', [11, 10, 9]),
            ('Child pedestrian - 2', [14, 13, 12]),
            ('Child pedestrian - 3', [48]),
            ('Child pedestrian - 4', [49]),
            ('Child pedestrian - 1', [50]), #ue5
            ('Child pedestrian - 2', [51]), #ue5
            ('Police - 1', [30]),
            ('Police - 2', [32])
        ]
        # ## ue5 carla
        # variants = [
        #     ('Adult pedestrian - 1', [15, 19]),
        #     ('Adult pedestrian - 2', [16, 17]),
        #     ('Adult pedestrian - 3', [26, 18]),
        #     ('Adult pedestrian - 4', [21, 20]),
        #     ('Adult pedestrian - 5', [23, 22]),
        #     ('Adult pedestrian - 6', [24, 25]),
        #     ('Adult pedestrian - 7', [27, 29, 28]),
        #     ('Adult pedestrian - 8', [41, 40, 33, 31]),
        #     ('Adult pedestrian - 9', [34, 38]),
        #     ('Adult pedestrian - 10', [35, 36, 37]),
        #     ('Adult pedestrian - 11', [39]),
        #     ('Adult pedestrian - 12', [42, 43, 44]),
        #     ('Adult pedestrian - 13', [47, 46, 45]),
        #     # ('Adult pedestrian - 14', [45]), # looks the same as 13
        #     ('Child pedestrian - 1', [50]),
        #     ('Child pedestrian - 2', [51]),
        #     ('Child pedestrian - 3', [48]),
        #     ('Child pedestrian - 4', [49]),
        #     ('Police pedestrian - 1', [30]),
        #     ('Police pedestrian - 2', [32])
        # ]
        all_indices = [index for _, indices in variants for index in indices]
        duplicates = [index for index in set(all_indices) if all_indices.count(index) > 1]
        assert not duplicates, f"Duplicate indices found: {duplicates}"

        # Convert the 4-digit number to an index
        index = int(variant_number) - 1
        # Find the type and variant number
        current_type = None
        current_variant = None
        for variant_type, indices in variants:
            if index + 1 in indices:
                current_type = variant_type
                current_variant = index + 1
                break

        # Get a different variant of the same type
        if current_type is not None and current_variant is not None:
            available_variants = [v for v in indices if v != current_variant]
            if not available_variants:
                # no other clothing types available
                return None
            new_variant = random.choice(available_variants)
            return f"{new_variant:04d}"
        else:
            return None