import gc
import logging
import queue
import re
import carla
import math
import random
import time
import imageio
import os
from tqdm import tqdm
# debugging imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import display
import argparse

from weather_time import time_and_weather
from utils import create_video_from_images, overlay_instances, get_instance_bounding_boxes
from track_metadata import MetadataTracker_simple, PositionTracker, LightTracker #MetadataTracker
from editor import Editor

from pdb import set_trace as st

world = None
client = None
editor = None
EDITS = [
    'time_of_day',
    'weather',
    'weather_and_time_of_day',
    # 'lane_marking',
    'building_texture',
    # 'vehicle_lights',
    'vehicle_color',
    'vehicle_replacement',
    'vehicle_deletion',
    'walker_color',
    'walker_replacement',
    'walker_deletion',
    # 'building_deletion',
    'road_texture',
    # 'sidewalk_texture',
    'traffic_light_state',
]

VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")

def verbose_debug(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)

logging.Logger.verbose_debug = verbose_debug
logger = logging.getLogger(__name__)

def delete_empty_dirs(path):
    """Recursively delete empty directories."""
    if not os.path.isdir(path):
        return
    # check if the directory is empty
    if not os.listdir(path):
        os.rmdir(path)
        logger.info('Deleted empty directory: %s', path)
        return
    # recursively check subdirectories
    for subdir in os.listdir(path):
        full_path = os.path.join(path, subdir)
        delete_empty_dirs(full_path)
    # check again if the directory is empty after deleting subdirectories
    if not os.listdir(path):
        os.rmdir(path)
        logger.info('Deleted empty directory: %s', path)
        

def prepare_output_dir(global_output_dir):
    """Creates a directory in `global_output_dir` to store the output images and metadata.
        Returns the path to the `global_output_dir`/out_dir/subdirectory if `subdirectory` not None, otherwise returns a path to the directory `global_output_dir`/out_dir.
    """
    os.makedirs(global_output_dir , exist_ok=True)

    run_dirs = [name for name in os.listdir(global_output_dir) if os.path.isdir(os.path.join(global_output_dir, name))]
    run_numbers = [int(re.search(r'run_(\d+)', name).group(1)) for name in run_dirs if re.search(r'run_(\d+)', name)]
    base_count = max(run_numbers, default=0)
    logger.debug('Next run number: %s', base_count)

    # create a new directory for the current run
    out_dir = os.path.join(global_output_dir, 'run_' + str(base_count+1))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        logger.info('All simulations for this run will be saved to: %s', out_dir)
    return out_dir

def sample_editing_operation(args):
    """ Samples an editing operation to apply to the simulation.
        The edit will set a flag in the args object to indicate the type of edit to apply.

        Note that multiple edits can be applied at once if these flags are activated elsewhere in the code:
    """

    if args.edit == 'random':
        logger.info('Sampling an editing operation.')
        edit = random.choice(EDITS)
        logger.info('Sampled edit: %s', edit)
    else:
        logger.info(f'No edit sampled. User specifed the edit: {args.edit}.')
        edit = args.edit
    setattr(args, edit, True)
    return




def attach_sensor(vehicle, sensor_type, height=1.75, fps=1.0, attachement_type=carla.AttachmentType.Rigid):
    """Attaches a sensor to a vehicle at a certain height above the vehicle.
        The height is specified by `height`.

        To avoid having the vehicle in the frame, the sensor (x,y) position is calibrated for z=1.75.
        The sensor is attached to the ego vehicle with different attachment types:
         * A spring arm so the movement is smooth and "hops" are avoided. Only recommended to record videos from the simulation
         * Rigid so movement is strict regarding its parent location. This is the proper attachment to retrieve data from the simulation.

        Returns:
            1. a dictionary of sensors where each key is a sensor type and the value is a a sensors representing that part of a 360 view.
            2. a dictionary of queues where each key is a sensor type and the value is a queue to hold the images from the sensor.
    """
    sensors = {}
    queues = {}
    for i, role in enumerate(['front', 'front_right', 'rear_right', 'rear', 'rear_left', 'front_left']):
        # create a transform to place the camera on top of the vehicle.
        x_value = 1.4 * math.cos(math.radians(60*i)) - 0.28
        y_value = 0.28 * math.sin(math.radians(60*i))
        camera_init_trans = carla.Transform(carla.Location(z=height, x=x_value, y=y_value), carla.Rotation(yaw=60*i))
        # create the camera through a blueprint that defines its properties
        camera_bp = world.get_blueprint_library().find(sensor_type)
        # for semantic and instance sensors, grab raw 32-bit data
        
        # if 'segmentation' in sensor_type:
        #     camera_bp.set_attribute('format', 'Raw')
        
        # set the time in seconds between sensor captures
        camera_bp.set_attribute('sensor_tick', str(round(1/fps, 5)))
        camera_bp.set_attribute('role_name', role)
        # spawn the camera and attach it to our ego vehicle
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle, attachment_type=attachement_type)
        sensors[role] = camera
        queues[role] = queue.Queue()
        logger.debug(f'Created {role} camera: {camera}')
    return sensors, queues


def activate_sensors(sensors, sensor_queues, output_dir=None, tracker=None):
    """Activates the sensors to start listening for data.
        `sensors` is a dictionary of sensors where each key is a sensor type and the value is a dictionary of sensors representing a 360 view.
        `sensor_queues` is a dictionary of queues where each key is a sensor type and the value is a dictionary of queues to hold the images from each view of the sensor type.
    """
    def save_to_disk(image, sensor_type, role, directory, tracker=None):
        """Saves the image to disk."""
        name = f'{sensor_type}_{role}_%06d.png' % image.frame
        path = os.path.join(directory, name)
        if tracker is not None:
            tracker.track_metadata(image.frame)
        image.save_to_disk(path)
        logger.verbose_debug(f'Saved image to disk: {name}')

    def add_to_queue(image, sensor_queue, tracker=None):
        if tracker is not None:
            tracker.track_metadata(image.frame)
        sensor_queue.put(image)


    logger.debug('Activating sensors and saving images to queues.')
    for sensor_type, sensor_dict in sensors.items():
        for role, sensor_view in sensor_dict.items():
            sensor_queue = sensor_queues[sensor_type][role]
            # if sensor_type == 'sensor.camera.rgb' and role == 'front':
            #     sensor_view.listen(lambda image, sq=sensor_queue: add_to_queue(image, sq, tracker))
            # else:
            #     sensor_view.listen(sensor_queue.put)

            sensor_view.listen(sensor_queue.put)
            # sensor_view.listen(lambda image, st=sensor_type, r=role, o=output_dir: save_to_disk(image, st, r, o))
            logger.debug('Activated %s sensor: %s', sensor_type, sensor_view)


def build_instance2semantic_majority(semantic_map: np.ndarray,
                                     instance_id_map: np.ndarray):
    """
    Map each non‑zero instance ID → the semantic ID that occurs most
    frequently within its mask.
    """
    mapping = {}
    # 1. gather all instance IDs (skip background 0)
    instance_ids = np.unique(instance_id_map)
    instance_ids = instance_ids[instance_ids != 0]

    for inst_id in instance_ids:
        # 2. mask out pixels belonging to this instance
        mask = (instance_id_map == inst_id)
        sem_vals = semantic_map[mask]  # flat array of semantic IDs under that instance

        # 3. count occurrences of each semantic ID
        labels, counts = np.unique(sem_vals, return_counts=True)
        logger.verbose_debug(f'Instance ID: {inst_id}, Labels: {labels}, Counts: {counts}')
        # 4. pick the label with the maximum count
        majority_sem = labels[np.argmax(counts)]
        # check if 6 is the majority (poles) if so check if its attached to a light
        if majority_sem == 6:
            logger.debug(f'Instance ID: {inst_id} is a pole.')
            # check if the instance is a traffic light
            actor = world.get_actor(int(inst_id))
            logger.debug(f'Actor ID: {inst_id}, Actor: {actor}')
            if actor is not None and 'traffic_light' in actor.type_id:
                # check if the instance is a traffic light
                logger.verbose_debug(f'Instance ID: {inst_id} is a traffic light.')
                mapping[int(inst_id)] = 7
            else:
                # if not, set it to 0
                logger.verbose_debug(f'Instance ID: {inst_id} is not a traffic light.')

        else:
            mapping[int(inst_id)] = int(majority_sem)

    return mapping


def save_images_from_queues(sensor_queues, output_dir, tracker, frame_id):
    """Reads images from the sensor queues and saves them to disk.
        `sensor_queues` is a dictionary of queues where each key is a sensor type and the value is a dictionary of queues to hold the images from the sensor.
        `output_dir` is the directory where the images will be saved.
    """
    def save_to_disk(image, sensor_type, role, directory, cc=None):
        """Saves the image to disk."""
        name = f'{sensor_type}_{role}_%06d.png' % image.frame
        path = os.path.join(directory, name)
        if cc is None:
            image.save_to_disk(path)
        else:
            image.save_to_disk(path, cc)
        logger.verbose_debug(f'Saved image to disk: {name}')
        
        return path

    frames_saved = []
    # for sensor_type, queue_dict in sensor_queues.items():
    #     sensor_directory = os.path.join(output_dir, sensor_type)
    #     os.makedirs(sensor_directory, exist_ok=True)
    #     for role, sensor_queue in queue_dict.items():
    #         sensor_view_subdirectory = os.path.join(sensor_directory, role)
    #         os.makedirs(sensor_view_subdirectory, exist_ok=True)
    #         while not sensor_queue.empty():
    #             image = sensor_queue.get()
    #             # if sensor_type == 'sensor.camera.depth' : ## visualize the output
    #             #     cc = carla.ColorConverter.Depth
    #             #     save_to_disk(image, sensor_type, role, sensor_view_subdirectory, cc)
    #             #     continue
    #             save_to_disk(image, sensor_type, role, sensor_view_subdirectory)  
    #             if image.frame not in frames_saved:
    #                 frames_saved.append(image.frame)
    
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points() # spawn points may be occupied already
    vehicle_blueprints = bp_lib.filter('*vehicle*')
    walker_blueprints = bp_lib.filter('*walker*')
    
    ordered_sensor_types = ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'sensor.camera.instance_segmentation']
    remaining_sensor_types = [sensor_type for sensor_type in sensor_queues if sensor_type not in ordered_sensor_types]
    sorted_sensor_types = ordered_sensor_types + remaining_sensor_types

    for sensor_type in sorted_sensor_types:
        if sensor_type in sensor_queues:
            queue_dict = sensor_queues[sensor_type]
    # for sensor_type, queue_dict in sensor_queues.items():
            sensor_directory = os.path.join(output_dir, sensor_type)
            os.makedirs(sensor_directory, exist_ok=True)

            for role, sensor_queue in queue_dict.items():
                role_dir = os.path.join(sensor_directory, role)
                os.makedirs(role_dir, exist_ok=True)

                while not sensor_queue.empty():
                    image = sensor_queue.get()
                    rgb_path = save_to_disk(image, sensor_type, role, role_dir)
                    if image.frame not in frames_saved:
                        frames_saved.append(image.frame)
                        
                    if sensor_type == 'sensor.camera.instance_segmentation':
                        # 1. Interpret raw_data as a flat uint8 array
                        array = np.frombuffer(image.raw_data, dtype=np.uint8)
                        # 2. Reshape to H×W×4 (BGRA)
                        array = array.reshape((image.height, image.width, 4))
                        # 3. Drop alpha and convert BGR → RGB
                        rgb = array[:, :, :3][:, :, ::-1]
                        # Now rgb[y,x] is [R, G, B]:
                        #   R = semantic class, G = low‐order byte of instance ID, B = high‐order byte

                        # (Optional) extract semantic tag map
                        semantic_map = rgb[:, :, 0]
                        # semantic_map_path = os.path.join(output_dir, "sensor.camera.semantic_segmentation", role, f'sensor.camera.semantic_segmentation_{role}_{image.frame:06d}.png')
                        # semantic_map = Image.open(rgb_path)
                        # # get the R channel:
                        # semantic_map = np.array(semantic_map)[:, :, 0]

                        # (Optional) reconstruct 16‐bit instance ID
                        instance_low  = rgb[:, :, 1].astype(np.uint16)
                        instance_high = rgb[:, :, 2].astype(np.uint16)
                        instance_id_map = instance_low + (instance_high << 8)

                        # 4) save the full maps if you like
                        np.save(os.path.join(role_dir, f'{image.frame:06d}_inst_map.npy'), instance_id_map)
                        np.save(os.path.join(role_dir, f'{image.frame:06d}_sem_map.npy'), semantic_map)

                        # # 5) visualize the instance mask 
                        # # rgb_path = os.path.join(output_dir, "sensor.camera.rgb", role, f'sensor.camera.rgb_{role}_{image.frame:06d}.png')
                        # # rgb_path = os.path.join(role_dir, 'sensor.camera.rgb', role, f'rgb_{role}_{image.frame:06d}.png')
                        rgb = Image.open(rgb_path)
                        
                        # arr = rgb
                        for instance_id in np.unique(instance_id_map):
                            # semantic_id = instance2semantic[instance_id]
                            
                            actor = world.get_actor(int(instance_id))
                            if actor is not None:
                                actor_type = actor.type_id
                                logger.debug(f'Actor ID: {instance_id}, Actor Type: {actor_type}')
                            else:
                                logger.debug(f'Actor ID: {instance_id} not found in the world.')
                        
                        
                        instance2semantic = {}
                        instance2semantic = build_instance2semantic_majority(semantic_map, instance_id_map)
                        
                        # 12: pedestrians, 13: Drivers, 14: vehicles, 15: trucks, 16: bus, 7: traffic lights, 18: motorcycles
                        interesting_semantic_classes = [12, 13, 14, 7, 15, 16, 18, 19, 17]
                        interesting_instances = {inst_id: sem_id for inst_id, sem_id in instance2semantic.items() if sem_id in interesting_semantic_classes}
                        # st()
                        actor_names = {}
                        for inst_id in interesting_instances:
                            actor = world.get_actor(int(inst_id))
                            if actor is not None:
                                actor_type = actor.type_id
                                actor_names[inst_id] = actor_type
                            else:
                                actor_names[inst_id] = inst_id
                        
                        overlayed = overlay_instances(rgb, instance_id_map, 
                                                    interesting_instances=interesting_instances,
                                                    actor_names=actor_names,
                                                    semantic_map=semantic_map,
                                                    alpha=0.0)
                        # save:
                        overlay_dir = os.path.join(role_dir, 'overlays')
                        os.makedirs(overlay_dir, exist_ok=True)
                        overlayed.save(os.path.join(overlay_dir, f'{image.frame:06d}_overlay.png'))
                        # st()
                        
                        # inst_dir = os.path.join(role_dir, 'instance_masks')
                        # os.makedirs(inst_dir, exist_ok=True)
                        # for inst_id, mask in masks.items():
                        #     mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                        #     mask_img.save(os.path.join(inst_dir, f'{image.frame:06d}_actor_{inst_id:04d}.png'))

                        # 6) record in your tracker
                        if tracker is not None:
                            # Each key is the instance_id, and the value is a tuple: (x0, y0, x1, y1)
                            bboxes_dict = get_instance_bounding_boxes(instance_id_map, interesting_instances, semantic_map)
                            tracker.track_metadata(image.frame, view=role, intersting_actors=interesting_instances, bboxes=bboxes_dict)

                        continue

                

    
    return frames_saved          
                

def stop_sensors(sensors):
    """Destroys all the sensors."""
    for sensor_dict in sensors.values():
        for sensor in sensor_dict.values():
            sensor.stop()
            logger.debug(f'Stopped sensor: {sensor}')
    logger.info('stopped all sensors.')

def weather_parameters_to_dict(weather):
    return {
        'cloudiness': weather.cloudiness,
        'precipitation': weather.precipitation,
        'precipitation_deposits': weather.precipitation_deposits,
        'wind_intensity': weather.wind_intensity,
        'sun_azimuth_angle': weather.sun_azimuth_angle,
        'sun_altitude_angle': weather.sun_altitude_angle,
        'fog_density': weather.fog_density,
        'fog_distance': weather.fog_distance,
        'fog_falloff': weather.fog_falloff,
        'wetness': weather.wetness,
        'scattering_intensity': weather.scattering_intensity,
        'mie_scattering_scale': weather.mie_scattering_scale,
        'rayleigh_scattering_scale': weather.rayleigh_scattering_scale,
        'dust_storm': weather.dust_storm
    }

def set_weather_and_time_of_day(**weather_kwargs):
    """Sets the weather for the simulation.
        Any values not specified in the `weather_kwargs` will be set to random values.
        If `realistic` is True, the weather will change according to present profiles.

        Note: for realistic weather, the profiles are automatically determined by only:
        `sun_azimuth_angle`, `sun_altitude_angle`, and `precipitation` or a `profile` in 
            `[ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset, CloudySunset, WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, HardRainSunset]`

        Will tick once to update the weather.
    """        

    weather_kwargs = editor.apply('weather', weather_kwargs)
    time_and_weather_instance = time_and_weather(world)
    time_and_weather_instance.set_weather(**weather_kwargs)
    world.tick()

    weather = world.get_weather()
    weather_dict = weather_parameters_to_dict(weather)
    weather_kwargs.update(weather_dict)
    editor.record('weather', weather_kwargs)    
    logger.debug(f'Weather: {weather}')
    logger.debug('Weather basics: %s', str(time_and_weather_instance))
    return time_and_weather_instance

def spawn_vehicle_actors(spawn_points, vehicle_blueprints, num_actors=None, positionTracker=None):
    """Spawns vehicle actors in the simulation. Note that the spawn points may be occupied already so the number of actors spawned may be less than `num_actors`.
        `spawn_points` is a list of spawn points.
        `vehicle_blueprints` is a list of vehicle blueprints.
        `num_actors` is the number of actors to spawn. If None, a random number will be chosen.
    """
    if num_actors is None:
        num_actors = random.randint(10, 30)
        if 'vehicle' in args.edit:
            num_actors = random.randint(25, 40)
    logger.debug(f'Spawning {num_actors} vehicle actors.')

    spawn_points = random.sample(spawn_points, num_actors)
    vehicle_types = [random.choice(vehicle_blueprints) for _ in range(num_actors)]
    indices = []
    vehicles = []
    spawned_locations = []

    vehicle_names = []
    vehicle_colors = []
    for vehicle_type in vehicle_types:
        vehicle_names.append(vehicle_type.id)
        if vehicle_type.has_attribute('color'):
            color_value = vehicle_type.get_attribute('color').as_color()
            r, g, b = color_value.r, color_value.g, color_value.b
            color_string = f"{r},{g},{b}"
            vehicle_colors.append(color_string)
        else:  
            vehicle_colors.append(None)


    dict = editor.apply('npc_vehicles', {'vehicle_names': vehicle_names, 'spawn_points': spawn_points, 'vehicle_colors': vehicle_colors})
    vehicle_names = dict['vehicle_names']
    spawn_points = dict['spawn_points']
    removed_indices = dict.get('removed_indices', [])
    colors = dict['vehicle_colors']

    if len(removed_indices) > 0 and positionTracker is not None:
        if positionTracker.is_active():
            positionTracker.ignore_actors(removed_indices)

    spawned_vehicle_names = []
    spawned_vehicle_colors = []
    for i, (spawn_point, vehicle_name, color) in enumerate(zip(spawn_points, vehicle_names, colors)):
        vehicle_type = vehicle_blueprints.find(vehicle_name)
        if vehicle_type.has_attribute('color'):
            vehicle_type.set_attribute('color', color)
        vehicle = world.try_spawn_actor(vehicle_type, spawn_point)
        if vehicle is None:
            logger.debug(f'Failed to spawn vehicle actor at spawn point: {spawn_point}.')
        else:
            indices.append(i)
            spawned_vehicle_names.append(vehicle.attributes['ros_name'])
            if 'color' in vehicle.attributes:
                spawned_vehicle_colors.append(vehicle.attributes['color'])
            else:
                spawned_vehicle_colors.append(None)
            vehicles.append(vehicle)
            spawned_locations.append(spawn_point)
            logger.verbose_debug(f'Spawned vehicle actor number {i}: {vehicle}')

    editor.record('npc_vehicles', {'vehicle_names': spawned_vehicle_names, 'indices': indices, 'colors': spawned_vehicle_colors})
    world.tick() # tick once to make sure the vehicles are spawned

    for vehicle in vehicles:
        logger.verbose_debug("Vehicle location: %s, ROS name: %s", vehicle.get_location(), vehicle.attributes['ros_name']) 
    logger.info(f'Spawned {len(vehicles)} vehicle actors.')
    return vehicles

def spawn_walker_actors(spawn_points, walker_blueprints, num_actors=None, ego_vehicle=None):
    """Spawns walker actors in the simulation.
        `num_actors` is the number of actors to spawn.
        `spawn_points` is a list of spawn points.
        `walker_blueprints` is a list of walker blueprints.
    """
    if num_actors is None:
        num_actors = random.randint(10, 40)
        if 'walker' in args.edit:
            num_actors = random.randint(55, 85)
    
    if ego_vehicle is None:
        spawn_points = []
        for _ in range(num_actors):
            for attempt in range(10):
                spawn_location = world.get_random_location_from_navigation()
                if spawn_location:
                    random_yaw = random.uniform(0, 360)
                    spawn_transform = carla.Transform(spawn_location, carla.Rotation(yaw=random_yaw))
                    spawn_points.append(spawn_transform)
                    break
    else:
        ego_location = ego_vehicle.get_location()
        spawn_points = []
        for _ in range(num_actors):
            for attempt in range(20):
                spawn_location = world.get_random_location_from_navigation()
                if spawn_location:
                    distance = ego_location.distance(spawn_location)
                    if distance <= 55:  # Check if within 100 meters
                        random_yaw = random.uniform(0, 360)
                        spawn_transform = carla.Transform(spawn_location, carla.Rotation(yaw=random_yaw))
                        spawn_points.append(spawn_transform)
                        break

    walker_types = [random.choice(walker_blueprints) for _ in range(num_actors)]    
    walker_names = [walker.id for walker in walker_types]
    indices = []
    walkers = []
    spawned_locations = []
    dict = editor.apply('npc_walkers', {'walker_names': walker_names, 'spawn_points': spawn_points})
    walker_names = dict['walker_names']
    spawn_points = dict['spawn_points']
    if len(walker_names) > len(spawn_points):
        walker_names = walker_names[:len(spawn_points)]
    logger.debug(f'Spawning {num_actors} walker actors.')

    spawned_walker_names = []
    for i, (spawn_point, walker_name) in enumerate(zip(spawn_points, walker_names)):
        walker_type = walker_blueprints.find(walker_name)
        walker = world.try_spawn_actor(walker_type, spawn_point)
        if walker is None:
            logger.debug(f'Failed to spawn walker actor at spawn point: {spawn_point}.')
        else:
            indices.append(i)
            walkers.append(walker)
            spawned_walker_names.append(walker.attributes['ros_name'])
            spawned_locations.append(spawn_point)
            logger.verbose_debug(f'Spawned walker actor number {i}: {walker}')
    editor.record('npc_walkers', {'walker_names': spawned_walker_names, 'indices': indices})
    world.tick() # tick once to make sure the walkers are spawned

    for walker in walkers:
        logger.verbose_debug("Walker location: %s, ROS name: %s", walker.get_location(), walker.attributes['ros_name'])
    logger.info(f'Spawned {num_actors} walker actors.')
    return walkers




def animate_vehicle_actors(vehicles):
    """Animates the vehicle actors in the simulation."""
    logger.debug('Animating NPC vehicle actors.')
    for vehicle in vehicles:
        vehicle.set_autopilot(True, args.tm_port)
        logger.verbose_debug(f'Animating vehicle actor: {vehicle}. {vehicle.attributes}')
    world.tick() # tick once to make sure the vehicles are animating
    logger.info('Animated all NPC vehicle actors.')

def animate_ego_vehicle(ego_vehicle):
    """Animates the ego vehicle actor in the simulation."""
    logger.debug('Animating ego vehicle actor.')
    ego_vehicle.set_autopilot(True, args.tm_port)
    world.tick() # tick once to make sure the ego vehicle is animating
    logger.info('Animated ego vehicle actor.')
    

def add_texture_to_buildings():
    """Adds textures to the buildings in the simulation.
        * samples a building
        * samples a texture from a list of textures
        Note that only the editor (not this method) applies the texture to the building.
    """
    names = world.get_names_of_all_objects()
    include_substrings = ['Apartment', 'Building', 'House', 'Hotel', 'concrete', 'Office', 'Shop', 'Skyscraper', 'building', 'Block', 'Garage', 'GuardShelter', 'Hall', 'WallBridge', 'WallTunnel']
    exclude_substrings = ['Lights', 'Win', 'Door']
    # Barel Bench, Container, Fountain, Kiosk, Parking barrier Perkola, Chair, Awning, Secwater, StreetBarrier, TrafficPole, BillBoard, StreetLight, BusStop, Lamppost, StreetLight, HighwayLightm Stop, SpeedLimit,
    # exclude_substrings = []
    
    # Filter names to include only those with the specified substrings and exclude others
    filtered_names = list(filter(lambda k: any(sub in k for sub in include_substrings) and not any(sub in k for sub in exclude_substrings), names))
    
    texture_files = [
        'black_brick.jpg',
        'blue_glass.jpg',
        'bluish_brick.jpg',
        'brown_rock.jpeg',
        'grey_cement.jpg',
        'light_brick.jpg',
        'metal_panels2.jpeg',
        'metal_panels.jpeg',
        'red-brick2.jpg',
        'red_brick.jpeg',
    ]
    
    for name in filtered_names:
        texture = random.choice(texture_files)
        integer = random.randint(0, 100)
        # texture_path = os.path.join('textures', 'building', texture)
        texture_path = os.path.join(os.path.dirname(__file__), 'textures', 'building', texture)
        # editor.record('building_texture', {'building': name, 'texture': texture_path})
        editor.apply('building_texture', {'building': name, 'texture': texture_path, 'integer': integer})

        
def add_texture_to_roads():
    """Adds textures to the roads in the simulation."""
    names = world.get_names_of_all_objects()
    include_substrings = ['Road_Road'] # 'Lane', 'Sidewalk' ]
    exclude_substrings = ['Lights', 'Win', 'Door', 'Crosswalk', 'Grass', 'Gutter', 'Curb']
    filtered_names = list(filter(lambda k: any(sub in k for sub in include_substrings), names))
    ## for now, change all roads to random textures
    # road_texture_dir = os.path.join('textures', 'road')
    # texture_files = os.listdir(road_texture_dir)
    editor.record('road_texture', {'roads': filtered_names})
    editor.apply('road_texture', {'roads': filtered_names})

    # for name in filtered_names:
    #     texture = random.choice(texture_files)
    #     texture_path = os.path.join('textures', 'road', texture)
    #     editor.record('road_texture', {'road': name})
    #     editor.apply('road_texture', {'road': name})


def add_texture_to_sidewalks():
    """Adds textures to the sidewalks in the simulation."""
    names = world.get_names_of_all_objects()
    include_substrings = ['Lane', 'Sidewalk' ]
    exclude_substrings = ['Lights', 'Win', 'Door', 'Crosswalk', 'Grass', 'Gutter', 'Curb']
    filtered_names = list(filter(lambda k: any(sub in k for sub in include_substrings), names))
    editor.record('sidewalk_texture', {'sidewalks': filtered_names})
    editor.apply('sidewalk_texture', {'sidewalks': filtered_names})


def get_trafficlight_actors():
    # get all traffic lights:
    list_tl = world.get_actors().filter('traffic.traffic_light')
    return list_tl





def run(args, **kwargs):
    global world, client
    world = client.get_world()

    vehicle_positions_tracker = kwargs.get('vehicle_positions_tracker', None)
    traffic_light_tracker = kwargs.get('traffic_light_tracker', None)
    
    # set synchronous mode
    old_settings = world.get_settings()
    settings = world.get_settings()

    # Apply synchronous mode
    settings.synchronous_mode = True


    settings.fixed_delta_seconds = 0.05  # Set a fixed time step (e.g., 0.05 seconds)
    logger.debug("Fixed delta seconds: %s", settings.fixed_delta_seconds) # each simulation tick will advance the simulation time by this amount
    logger.debug("substepping: %s", settings.substepping) # substepping improves the accuracy of the physics simulation by dividing each simulation step into smaller substeps
    logger.debug("Max substep delta time: %s", settings.max_substep_delta_time) # maximum duration (coarseness) of each substep when substepping is enabled.
    logger.debug("Max substeps: %s", settings.max_substeps) # maximum number of substeps (granularity) to perform within each simulation tick.

    assert settings.fixed_delta_seconds <= settings.max_substep_delta_time * settings.max_substeps, f"Fixed delta seconds of {settings.fixed_delta_seconds} should be less than {settings.max_substep_delta_time * settings.max_substeps}."


    # apply the settings
    world.apply_settings(settings)
    logger.debug("Synchronous mode: %s", settings.synchronous_mode)
    assert world.get_settings().synchronous_mode, "Synchronous mode not set."
    # set the Traffic Manager to sync mode
    traffic_manager = client.get_trafficmanager(int(args.tm_port))
    traffic_manager.set_synchronous_mode(True)
    # Set the seeds for determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    traffic_manager.set_random_device_seed(args.seed)
    # Set the seed value for pedestrian ** positions **
    world.set_pedestrians_seed(args.seed)

    traffic_manager.set_hybrid_physics_mode(True) # to speed up simulation
    world.tick()


    ##### -------------------------------- Perform edits -------------------------------- #####
    



    ### --------------------- Simulation setup --------------------- ###



    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points() # spawn points may be occupied already
    vehicle_blueprints = bp_lib.filter('*vehicle*')
    walker_blueprints = bp_lib.filter('*walker*')

    # place the ego car
    ego_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    ego_bp.set_attribute('role_name', 'hero')
    ego_spawn_point = random.choice(spawn_points)
    ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
    world.tick()

    # place the ego cameras
    # All sensors use queues to avoid dropping frames.
    ego_sensors = {} # Each key is a sensor type, and the value is a dictionary of sensors representing a 360 view. Use the `role_name` attribute to distinguish between them.
    ego_sensor_queues = {} # Each key is a sensor type, and the value is a dictionary of queues to hold the images from the sensor.
    # TODO: add the other sensors
    # for sensor in ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'sensor.camera.depth']:
    for sensor in [
        'sensor.camera.rgb',
        'sensor.camera.semantic_segmentation',
        'sensor.camera.depth',
        'sensor.camera.instance_segmentation' 
    ]:
        sensors_360, queues_360 = attach_sensor(ego_vehicle, sensor, height=1.75, fps=args.fps)
        ego_sensors[sensor] = sensors_360
        ego_sensor_queues[sensor] = queues_360

    

    # set the weather and time of day
    weather_profile = random.choice(['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'SoftRainNoon', 'MidRainyNoon', 'HardRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'SoftRainSunset', 'MidRainSunset', 'HardRainSunset'])
    time_and_weather_instance = set_weather_and_time_of_day(profile=weather_profile)

    args.weather_profile = time_and_weather_instance.profile
    # add weather details to the args object
    weather_dict = weather_parameters_to_dict(world.get_weather())
    for key, value in weather_dict.items():
        setattr(args, f"weather_{key}", value)


    # set the NPC actors (random if no edits are specified).
    vehicles = spawn_vehicle_actors(spawn_points, vehicle_blueprints, getattr(args, 'num_vehicle_actors', None))
    walkers = spawn_walker_actors(spawn_points, walker_blueprints, getattr(args, 'num_walker_actors', None), ego_vehicle)

    tracker = MetadataTracker_simple(world, args.run_output_dir)

    ### note that pedestrians are controlled by an AI, not a traffic manager. This also needs to be set.
    ### pedestrians use different spawn points than vehicles. see docs

    ## animations
    if vehicle_positions_tracker is not None and vehicle_positions_tracker.is_active():
        logger.info("Disabling autopilot for NPCs and Ego since setting positions using tracker is active.")
    else:
        animate_vehicle_actors(vehicles)
        animate_ego_vehicle(ego_vehicle)


    add_texture_to_roads()
    add_texture_to_buildings()
    add_texture_to_sidewalks()
    try:

        def tick_world(tick=None):
            """Wrapper function for world.tick() to advance the simulation. To use in the for-loop to run simulation.
            
            Returns the frame ID of the new frame computed by the server."""
            if tick is not None and vehicle_positions_tracker is not None and vehicle_positions_tracker.is_active():
                # vehicle_positions_tracker.set_location([ego_vehicle] + vehicles, tick)
                vehicle_positions_tracker.set_location_batch([ego_vehicle] + vehicles, tick, client)
            if tick is not None and traffic_light_tracker is not None and traffic_light_tracker.is_active():
                traffic_light_tracker.set_light_states(get_trafficlight_actors(), tick)

            frame_id = world.tick()
            logger.verbose_debug(f"World ticked. Frame ID: {frame_id}")
            # time_and_weather_instance.tick(world.get_settings().fixed_delta_seconds)
            logger.verbose_debug(str(time_and_weather_instance))

            if tick is not None and vehicle_positions_tracker is not None and not vehicle_positions_tracker.is_active():
                vehicle_positions_tracker.track_positions([ego_vehicle] + vehicles, tick)
            if tick is not None and traffic_light_tracker is not None and not traffic_light_tracker.is_active():
                traffic_light_tracker.track_light_states(get_trafficlight_actors(), tick)
            # print("Frame ID: ", frame_id)
            return frame_id

        ### --------------------- Run the simulation --------------------- ###

        # Do a few ticks to get the simulation started
        for _ in range(5):
            tick_world()

        activate_sensors(ego_sensors, ego_sensor_queues, args.run_output_dir, tracker=tracker)
        world.tick() # tick once to activate the sensors
        logger.debug('Activated sensors.')



        total_seconds = args.length
        num_ticks = math.ceil(total_seconds / world.get_settings().fixed_delta_seconds)
        logger.info(f"Total seconds: {total_seconds}, Num ticks: {num_ticks}")
        frames_saved = []
        # for _ in tqdm(range(num_ticks), desc="Simulating frames"):
        #     frame_id = tick_world(_)
        #     if _ % 20 == 0 or _ == num_ticks - 1:
        #         frames_saved_subsample = save_images_from_queues(ego_sensor_queues, args.run_output_dir, tracker, frame_id)
        #         frames_saved.extend(frames_saved_subsample)
        #     if tracker is not None:
        #         tracker.track_metadata(frame_id)
        #     print(frame_id)
        with tqdm(range(num_ticks), desc="Simulating frames") as pbar:
            for i in pbar:
                frame_id = tick_world(i)
                pbar.set_description(f"Simulating frames {frame_id}")
                # if i % 20 == 0 or i == num_ticks - 1:
                #     # the only camera that must be saved every tick is the instance segmentation camera
                #     frames_saved_subsample = set()
                #     for sensor_type, queue_dict in ego_sensor_queues.items():
                #         if sensor_type != 'sensor.camera.instance_segmentation':
                #             frames_saved_subsample.update(save_images_from_queues({sensor_type: queue_dict}, args.run_output_dir, tracker, frame_id))
                #     frames_saved.extend(frames_saved_subsample)
                if tracker is not None:
                    # # the instance segmentation camera is saved every tick
                    # # tracker.track_metadata(frame_id) will be called in the save_images_from_queues function
                    # instance_segmentation_queue = ego_sensor_queues.get('sensor.camera.instance_segmentation', None)
                    # save_images_from_queues({'sensor.camera.instance_segmentation': instance_segmentation_queue}, args.run_output_dir, tracker, frame_id)

                    frames_saved_subsample = save_images_from_queues(ego_sensor_queues, args.run_output_dir, tracker, frame_id)
                    frames_saved.extend(frames_saved_subsample)
                    
        frames_saved = list(set(frames_saved)) # unique
        tracker.remove_frame_not_listed(frames_saved)

    finally:
        logger.info('stopping sensors')
        stop_sensors(ego_sensors)
    
        tracker.save_args(args)
        tracker.save_metadata()

        # logger.info('destroying actors')
        # # TODO: destroy the actors (walkers, vehicles, etc.)
        # actor_list = list(world.get_actors()) # causes simulation to shut down
        # # actor_list = [ego_vehicle] + vehicles + walkers
        # batch = [carla.command.DestroyActor(x) for x in actor_list]
        # results = client.apply_batch_sync(batch, True)
        # for i, result in enumerate(results):
        #     if result.error:
        #         logger.error(f"Failed to destroy actor {actor_list[i].id}: {result.error}")

        # Always disable sync mode before the script ends to prevent the server blocking whilst waiting for a tick
        logger.info('Disabling synchronous mode')
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(settings)
        logger.debug('Synchronous mode: %s', settings.synchronous_mode)
        assert not world.get_settings().synchronous_mode, "Synchronous mode not disabled."



        traffic_manager.shut_down()

        client.reload_world()
        # workaround: give time to UE4 to clean memory after loading (old assets)
        time.sleep(5)
        
        
    world.apply_settings(old_settings)
    logger.info('done.')
    # time.sleep(5) # give time for the simulation to shut down




def main(args):
    """Main method"""
    if args.seed_edit is None:
        args.seed_edit = random.randint(0, 2**32 - 1)
    logging.info(f"Using global edit seed {args.seed_edit}")

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    logging.info(f"Using simulation seed {args.seed}")

    random.seed(args.seed_edit)
    np.random.seed(args.seed_edit)
    # TODO: etc...
    

    trial_output_dir = prepare_output_dir(args.output)
    args.output = trial_output_dir

    if args.port is not None:
        if args.tm_port is None:
            args.tm_port = int(int(args.port) + 3000)
        logger.info('Using traffic manager port: %s', args.tm_port)
    global world, client, editor
    client = carla.Client(args.host, int(args.port))
    
    # print("Client (Python API) version:", carla.__version__)
    print("Server version:           ", client.get_server_version())
    
    client.set_timeout(60.0) 
    world = client.get_world()
    editor = Editor(world, args, client)

    maps = client.get_available_maps()
    logger.debug('Available maps: %s', maps)
    if args.map:
        map_name = args.map
    else:
        if args.edit == 'buidling_texture':
            maps = [map_item for map_item in maps if '4' not in map_item]
        map_name = random.choice(maps)

    if args.edit == 'road_texture' or args.edit == 'sidewalk_texture':
        map_name = '/Game/Carla/Maps/Town10HD_Opt'

    args.map = map_name
    world = client.load_world(map_name)
    logger.info('Loaded map: %s', map_name)
    # workaround: give time to UE4 to clean memory after loading (old assets)
    time.sleep(8)


    # save random state:
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    # to guarantee determinism in actor positions:
    vehicle_pos_tracker = PositionTracker("vehicles") ## very slow
    traffic_light_tracker = LightTracker("traffic_lights")
    args.traffic_light_mapping = traffic_light_tracker.get_permutation()
    print(args.traffic_light_mapping)
    # vehicle_pos_tracker = None

    ### ---------------------- main edit loop ---------------------- ###
    for r in range(2):
        logger.info(f'{"="*14} Starting simulation {r+1}. {"="*14}')

        subdirectory = os.path.join(args.output, str(r))
        os.makedirs(subdirectory, exist_ok=False)
        logger.info('Simulation subdirectory created: %s', subdirectory)
        args.run_output_dir = subdirectory

        # Each simulation will also set the seeds for determinism
        if r == 0:
            pass
            editor.set_recording_mode()
            if vehicle_pos_tracker is not None: vehicle_pos_tracker.deactivate()
            if traffic_light_tracker is not None: traffic_light_tracker.deactivate()
            # simulate
            # TODO: any return values?
            run(args, vehicle_positions_tracker=vehicle_pos_tracker, traffic_light_tracker=traffic_light_tracker)
        else:
            pass
            # restore the random state
            random.setstate(random_state)
            np.random.set_state(np_random_state)
            # sample an editing operation
            logger.info('Sampling an editing operation.')
            sample_editing_operation(args)
            editor.edit()
            if vehicle_pos_tracker is not None: vehicle_pos_tracker.activate()

            if args.edit == 'traffic_light_state':
                if traffic_light_tracker is not None:
                    traffic_light_tracker.activate()
                else:
                    logger.error("Traffic light tracker is not active.")
            # TODO
            # save the random state
            random_state = random.getstate()
            np_random_state = np.random.get_state()

            # simulate
            # TODO: any return values?
            # print("Reloading world")
            # client.reload_world()
            # workaround: give time to UE4 to clean memory after loading (old assets)
            # time.sleep(5)
            # print("World reloaded")
            run(args, vehicle_positions_tracker=vehicle_pos_tracker, traffic_light_tracker=traffic_light_tracker)
        if vehicle_pos_tracker is not None: vehicle_pos_tracker.reset_but_keep_record()
        if traffic_light_tracker is not None: traffic_light_tracker.reset_but_keep_record()

        create_video_from_images(f"{args.run_output_dir}/sensor.camera.rgb/front", args.output, fps=args.fps)
        logger.info('Created video from images.')



    editor = None
    del editor
    client = None
    del client
    world = None
    del world
    gc.collect()
    return args, trial_output_dir

    

if __name__ == '__main__':


    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        nargs='?',
        const=0,
        type=int,
        default=None,
        help='Set verbosity level for debug information (0 for verbose, 1 for super verbose, default is None)')
    argparser.add_argument(
        '-o', '--output',
        metavar='DIR',
        default='out',
        help='Set the global output directory')
    argparser.add_argument(
        '-f', '--fps',
        metavar='float',
        default='4.0',
        type=float,
        help='Set the frames per second for the sensors')
    argparser.add_argument(
        '-l', '--length',
        metavar='secs',
        default='20.0',
        type=float,
        help='The length of the simulation in seconds')
    argparser.add_argument(
        '-s', '--seed',
        metavar='int',
        default=None,
        type=int,
        help='The seed value for the simulation')
    argparser.add_argument(
        '--seed_edit',
        metavar='int',
        default=None,
        type=int,
        help='The seed value for editing and setting up the world before simulations')
    argparser.add_argument(
        '-m', '--map',
        metavar='MAP',
        default=None,
        help='The map to load.')
    argparser.add_argument(
        '--host',
        metavar='HOST',
        default='localhost',
        help='The IP of the host server')
    argparser.add_argument(
        '--port',
        metavar='PORT',
        default='2000',
        help='The port of the host server')
    argparser.add_argument(
        '-e', '--edit',
        metavar='EDIT',
        default='random',
        choices=['random'] + EDITS,
        help='The edit to apply to the simulation')
    argparser.add_argument(
        '--tm_port',
        metavar='PORT',
        default=None,
        type=int,
        help='The port of the traffic manager server. If None, and the host server port is set, then host server port + 1 is used.')
    
    args = argparser.parse_args()


    if args.verbose is None:
        log_level = logging.INFO
    elif args.verbose == 0:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = VERBOSE
    else:
        log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logger.info('listening to server %s:%s', args.host, args.port)


    try:
        output = args.output
        args, trial_output_dir = main(args)

        # open a txt file to write the output directory and sampled edit
        with open(os.path.join(output, 'description.txt'), 'a') as f:
            # output basename
            basename = os.path.basename(trial_output_dir)
            f.write(f"Directory: {basename}, Edit: {args.edit}\n")

    finally:
        # delete the output directory if it is empty
        delete_empty_dirs(args.output)