import logging
import queue
import re
import carla
import math
import random
import time
import imageio
import os
# debugging imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import display
import argparse

from weather_time import time_and_weather

world = None
client = None


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

def attach_sensor(vehicle, sensor_type, height=0.5, fps=1.0):
    """Attaches a sensor to a vehicle at a certain height above the vehicle.
        The height is specified by `height`.

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
        # We create the camera through a blueprint that defines its properties
        camera_bp = world.get_blueprint_library().find(sensor_type)
        # Set the time in seconds between sensor captures
        camera_bp.set_attribute('sensor_tick', str(round(1/fps, 5)))
        camera_bp.set_attribute('role_name', role)
        # We spawn the camera and attach it to our ego vehicle
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        sensors[role] = camera
        queues[role] = queue.Queue()
        logger.debug(f'Created {role} camera: {camera}')
    return sensors, queues

# def activate_sensors(sensors, output_dir):
#     """Activates the sensors to start listening for data.
#         `sensors` is a dictionary of sensors where each key is a sensor type and the value is a dictionary of sensors representing a 360 view.
#     """
#     # TODO: add support for other sensor types (non-image types).

#     logger.debug('Saving images to: %s', output_dir)

#     def save_to_disk(image, sensor_type, sensor_view, directory):
#         """Saves the image to disk."""
#         name = f'{sensor_type}_{sensor_view.attributes["role_name"]}_%06d.png' % image.frame
#         path = os.path.join(directory, name)
#         image.save_to_disk(path)
#         logger.debug(f'Saved image to disk: {name}')

       

#     for sensor_type, sensor_dict in sensors.items():
#         sensor_directory = os.path.join(output_dir, sensor_type)
#         os.makedirs(sensor_directory, exist_ok=True)
#         for sensor_view in sensor_dict.values():
#             sensor_view_subdirectory = os.path.join(sensor_directory, sensor_view.attributes['role_name'])
#             os.makedirs(sensor_view_subdirectory, exist_ok=True)
#             sensor_view.listen(lambda image, st=sensor_type, sv=sensor_view, sd=sensor_view_subdirectory: save_to_disk(image, st, sv, sd))
#             logger.debug('Activated %s sensor: %s', sensor_type, sensor_view)


def activate_sensors(sensors, sensor_queues):
    """Activates the sensors to start listening for data.
        `sensors` is a dictionary of sensors where each key is a sensor type and the value is a dictionary of sensors representing a 360 view.
        `sensor_queues` is a dictionary of queues where each key is a sensor type and the value is a dictionary of queues to hold the images from each view of the sensor type.
    """
    logger.debug('Activating sensors and saving images to queues.')
    for sensor_type, sensor_dict in sensors.items():
        for role, sensor_view in sensor_dict.items():
            sensor_queue = sensor_queues[sensor_type][role]
            sensor_view.listen(sensor_queue.put)
            logger.debug('Activated %s sensor: %s', sensor_type, sensor_view)

def save_images_from_queues(sensor_queues, output_dir):
    """Reads images from the sensor queues and saves them to disk.
        `sensor_queues` is a dictionary of queues where each key is a sensor type and the value is a dictionary of queues to hold the images from the sensor.
        `output_dir` is the directory where the images will be saved.
    """
    def save_to_disk(image, sensor_type, role, directory):
        """Saves the image to disk."""
        name = f'{sensor_type}_{role}_%06d.png' % image.frame
        path = os.path.join(directory, name)
        image.save_to_disk(path)
        logger.verbose_debug(f'Saved image to disk: {name}')

    for sensor_type, queue_dict in sensor_queues.items():
        sensor_directory = os.path.join(output_dir, sensor_type)
        os.makedirs(sensor_directory, exist_ok=True)
        for role, sensor_queue in queue_dict.items():
            sensor_view_subdirectory = os.path.join(sensor_directory, role)
            os.makedirs(sensor_view_subdirectory, exist_ok=True)
            while not sensor_queue.empty():
                image = sensor_queue.get()
                save_to_disk(image, sensor_type, role, sensor_view_subdirectory)





def stop_and_destroy_sensors(sensors):
    """Destroys all the sensors."""
    for sensor_dict in sensors.values():
        for sensor in sensor_dict.values():
            sensor.stop()
            sensor.destroy()
            logger.debug(f'Destroyed sensor: {sensor}')
    logger.info('Destroyed all sensors.')


def set_weather_and_time_of_day(realistic=True, **weather_kwargs):
    """Sets the weather for the simulation.
        Any values not specified in the `weather_kwargs` will be set to random values.
        If `realistic` is True, the weather will change according to present profiles.

        Note: for realistic weather, the profiles are automatically determined by only:
        `sun_azimuth_angle`, `sun_altitude_angle`, and `precipitation`.

        Will tick once to update the weather.
    """
    if not realistic:
        logger.info('Weather may be unrealistic if individual parameters are modified without considering the others.')
        
    time_and_weather_instance = time_and_weather(world, realistic=realistic, **weather_kwargs)
    world.tick()
    weather = world.get_weather()
    logger.debug(f'Weather: {weather}')
    logger.debug('Weather basics: %s', str(time_and_weather_instance))
    return time_and_weather_instance






def run(args):
    global world, client
    # set synchronous mode
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
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    world.tick()


    # Set the seeds for determinism
    random.seed(args.seed)
    traffic_manager.set_random_device_seed(args.seed)
    # Set the seed value for pedestrian ** positions **
    # TODO
    # world.set_pedestrian_seed(args.seed)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_blueprints = bp_lib.filter('*vehicle*')
    walker_blueprints = bp_lib.filter('*walker*')

    # place the ego car
    ego_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    ego_bp.set_attribute('role_name', 'hero')
    ego_spawn_point = random.choice(spawn_points)
    ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
    world.tick()


    # place the ego cameras
    ego_sensors = {} # Each key is a sensor type, and the value is a dictionary of sensors representing a 360 view. Use the `role_name` attribute to distinguish between them.
    ego_sensor_queues = {} # Each key is a sensor type, and the value is a dictionary of queues to hold the images from the sensor.
    # TODO: add the other sensors
    for sensor in ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'sensor.camera.depth']:
        sensors_360, queues_360 = attach_sensor(ego_vehicle, sensor, height=1.75, fps=args.fps)
        ego_sensors[sensor] = sensors_360
        ego_sensor_queues[sensor] = queues_360

    # set the weather and time of day
    time_and_weather_instance = set_weather_and_time_of_day()

    # set the buidlings (do nothing if no edits are specified)
    # TODO
    # set the NPC actors (random if no edits are specified).
    # valid spawn points (translating vehicles??)
    # NPC vehicles, pedestrians, etc. Apply traffic manager.
    # TODO


    def tick_world():
        """Wrapper function for world.tick() to detect when the simulation advances. To use in the for-loop to run simulation"""
        frame_id = world.tick()
        logger.verbose_debug(f"World ticked. Frame ID: {frame_id}")
        time_and_weather_instance.tick(world.get_settings().fixed_delta_seconds)
        logger.verbose_debug(str(time_and_weather_instance))
        return frame_id

    ### --------------------- Run the simulation --------------------- ###
    # call camera.listen on all sensors
    activate_sensors(ego_sensors, ego_sensor_queues)
    world.tick() # tick once to activate the sensors
    logger.debug('Activated sensors.')

    total_seconds = args.length
    num_ticks = math.ceil(total_seconds / world.get_settings().fixed_delta_seconds)
    logger.debug(f"Total seconds: {total_seconds}, Num ticks: {num_ticks}")
    for _ in range(num_ticks):
        frame_id = tick_world() # returns the id of the new frame computed by the server
        save_images_from_queues(ego_sensor_queues, args.run_output_dir)
        # Track the metadata of the simulation
        # TODO


    # stop the simulation
    logger.info('stopping and destroying sensors')
    stop_and_destroy_sensors(ego_sensors)
    # destroy agents gracefully
    logger.info('destroying actors')

    # TODO: destroy the actors
    actor_list = [ego_vehicle]
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

    # Always disable sync mode before the script ends to prevent the server blocking whilst waiting for a tick
    logger.info('Disabling synchronous mode')
    settings = world.get_settings()
    settings.synchronous_mode = False
    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(settings)
    logger.debug('Synchronous mode: %s', settings.synchronous_mode)
    assert not world.get_settings().synchronous_mode, "Synchronous mode not disabled."
    logger.info('done.')




def main(args):
    """Main method"""

    # set the seeds for determinism
    seed_value_1 = 1234
    random.seed(seed_value_1)
    np.random.seed(seed_value_1)
    # TODO: etc...

    # prepare the output directory
    run_output_dir = prepare_output_dir(args.output)
    args.output = run_output_dir

    global world, client
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # load world by sampling a random map unless specified
    maps = client.get_available_maps()
    logger.debug('Available maps: %s', maps)
    if args.map:
        map_name = args.map
    else:
        map_name = random.choice(maps)
    world = client.load_world(map_name)
    logger.info('Loaded map: %s', map_name)


    # save random state:
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    for r in range(2):
        logger.info(f'{"="*14} Starting simulation {r+1}. {"="*14}')
        # prepare the output directory
        subdirectory = os.path.join(args.output, str(r))
        os.makedirs(subdirectory, exist_ok=False)
        logger.info('Simulation subdirectory created: %s', subdirectory)
        args.run_output_dir = subdirectory

        # Each simulation will also set the seeds for determinism
        if r == 0:
            pass
            # simulate
            # TODO: any return values?
            run(args)
        else:
            pass
            # restore the random state
            random.setstate(random_state)
            np.random.set_state(np_random_state)
            # sample an editing operation
            # TODO
            # save the random state
            random_state = random.getstate()
            np_random_state = np.random.get_state()

            # simulate
            # TODO: any return values?
            run(args)



    

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
        default='1234',
        type=int,
        help='The seed value for the simulation')
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
        main(args)
    finally:
        # delete the output directory if it is empty
        delete_empty_dirs(args.output)