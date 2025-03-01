import logging
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


def prepare_output_dir(global_output_dir):
    """Creates a subdirectory to store the output images and metadata.
        Returns the path to the directory.
    """
    os.makedirs(global_output_dir , exist_ok=True)

    run_dirs = [name for name in os.listdir(global_output_dir) if os.path.isdir(os.path.join(global_output_dir, name))]
    run_numbers = [int(re.search(r'run_(\d+)', name).group(1)) for name in run_dirs if re.search(r'run_(\d+)', name)]
    base_count = max(run_numbers, default=0)
    logging.debug('Next run number: %s', base_count)

    # create a new directory for the current run
    out_dir = os.path.join(global_output_dir, 'run_' + str(base_count+1))
    os.makedirs(out_dir, exist_ok=True)

    logging.info('Saving images to %s', out_dir)
    return out_dir

def attach_sensor(vehicle, sensor_type, height=0.5, fps=1.0):
    """Attaches a sensor to a vehicle at a certain height above the vehicle.
        The height is specified by `height`.
    """
    sensors = {}
    for i, role in enumerate(['front', 'front_right', 'rear_right', 'rear', 'rear_left', 'front_left']):
        # create a transform to place the camera on top of the vehicle.
        x_value = 1.4 * math.cos(math.radians(60*i)) - 0.28
        y_value = 0.28 * math.sin(math.radians(60*i))
        camera_init_trans = carla.Transform(carla.Location(z=height, x=x_value, y=y_value), carla.Rotation(yaw=60*i))
        # We create the camera through a blueprint that defines its properties
        camera_bp = world.get_blueprint_library().find(sensor_type)
        # Set the time in seconds between sensor captures
        camera_bp.set_attribute('sensor_tick', str(fps))
        camera_bp.set_attribute('role_name', role)
        # We spawn the camera and attach it to our ego vehicle
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        sensors[role] = camera
        logging.debug(f'Created {role} camera: {camera}')
    return sensors

def activate_sensors(sensors, output_dir):
    """Activates the sensors to start listening for data.
        `sensors` is a dictionary of sensors where each key is a sensor type and the value is a dictionary of sensors representing a 360 view.
    """
    # TODO: add support for other sensor types (non-image types).

    logging.debug('Saving images to: %s', output_dir)

    def save_to_disk(image, sensor_type, sensor_view, directory):
        """Saves the image to disk."""
        name = f'{sensor_type}_{sensor_view.attributes["role_name"]}_%06d.png' % image.frame
        path = os.path.join(directory, name)
        image.save_to_disk(path)
        logging.debug(f'Saved image to disk: {name}')

       

    for sensor_type, sensor_dict in sensors.items():
        sensor_directory = os.path.join(output_dir, sensor_type)
        os.makedirs(sensor_directory, exist_ok=True)
        for sensor_view in sensor_dict.values():
            sensor_view_subdirectory = os.path.join(sensor_directory, sensor_view.attributes['role_name'])
            os.makedirs(sensor_view_subdirectory, exist_ok=True)
            sensor_view.listen(lambda image, st=sensor_type, sv=sensor_view, sd=sensor_view_subdirectory: save_to_disk(image, st, sv, sd))
            logging.debug('Activated %s sensor: %s', sensor_type, sensor_view)

def stop_and_destroy_sensors(sensors):
    """Destroys all the sensors."""
    for sensor_dict in sensors.values():
        for sensor in sensor_dict.values():
            sensor.stop()
            sensor.destroy()
            logging.debug(f'Destroyed sensor: {sensor}')
    logging.debug('Destroyed all sensors.')


def set_weather_and_time_of_day(realistic=True, **weather_kwargs):
    """Sets the weather for the simulation.
        Any values not specified in the `weather_kwargs` will be set to random values.
        If `realistic` is True, the weather will change according to present profiles.

        Note: for realistic weather, the profiles are automatically determined by only:
        `sun_azimuth_angle`, `sun_altitude_angle`, and `precipitation`.
    """
    if not realistic:
        logging.info('Weather may be unrealistic if individual parameters are modified without considering the others.')
        
    time_and_weather_instance = time_and_weather(world, realistic=realistic, **weather_kwargs)
    weather = world.get_weather()
    logging.debug(f'Weather: {weather}')
    logging.debug('Weather basics: %s', str(time_and_weather_instance))
    return time_and_weather_instance






def run(args):
    global world, client
    # set synchronous mode
    settings = world.get_settings()

    # Apply synchronous mode
    settings.synchronous_mode = True


    settings.fixed_delta_seconds = 0.05  # Set a fixed time step (e.g., 0.05 seconds)
    logging.debug("Fixed delta seconds: %s", settings.fixed_delta_seconds) # each simulation tick will advance the simulation time by this amount
    logging.debug("substepping: %s", settings.substepping) # substepping improves the accuracy of the physics simulation by dividing each simulation step into smaller substeps
    logging.debug("Max substep delta time: %s", settings.max_substep_delta_time) # maximum duration (coarseness) of each substep when substepping is enabled.
    logging.debug("Max substeps: %s", settings.max_substeps) # maximum number of substeps (granularity) to perform within each simulation tick.

    assert settings.fixed_delta_seconds <= settings.max_substep_delta_time * settings.max_substeps, f"Fixed delta seconds of {settings.fixed_delta_seconds} should be less than {settings.max_substep_delta_time * settings.max_substeps}."


    # apply the settings
    world.apply_settings(settings)
    logging.debug("Synchronous mode: %s", settings.synchronous_mode)
    assert world.get_settings().synchronous_mode, "Synchronous mode not set."
    # set the Traffic Manager to sync mode
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)




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
    # TODO: add the other sensors
    for sensor in ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'sensor.camera.depth']:
        sensors_360 = attach_sensor(ego_vehicle, sensor, height=1.75, fps=args.fps)
        ego_sensors[sensor] = sensors_360


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
        logging.debug(f"World ticked: {frame_id}")
        time_and_weather_instance.tick(world.get_settings().fixed_delta_seconds)
        logging.debug(str(time_and_weather_instance))
        return frame_id

    ### --------------------- Run the simulation --------------------- ###
    # call camera.listen on all sensors
    activate_sensors(ego_sensors, args.output)
    world.tick() # tick once to activate the sensors
    logging.debug('Activated sensors.')

    total_seconds = args.length
    num_ticks = math.ceil(total_seconds / world.get_settings().fixed_delta_seconds)
    for _ in range(num_ticks):
        frame_id = tick_world() # returns the id of the new frame computed by the server
        logging.debug(f"Frame id: {frame_id}")
        # Track the metadata of the simulation
        # TODO


    # stop the simulation
    logging.info('stopping and destroying sensors')
    stop_and_destroy_sensors(ego_sensors)
    # destroy agents gracefully
    logging.info('destroying actors')

    # TODO: destroy the actors
    actor_list = [ego_vehicle]
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

    # Always disable sync mode before the script ends to prevent the server blocking whilst waiting for a tick
    logging.info('Disabling synchronous mode')
    settings = world.get_settings()
    settings.synchronous_mode = False
    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(settings)
    logging.debug('Synchronous mode: %s', settings.synchronous_mode)
    assert not world.get_settings().synchronous_mode, "Synchronous mode not disabled."
    logging.info('done.')




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
    logging.debug('Available maps: %s', maps)
    if args.map:
        map_name = args.map
    else:
        map_name = random.choice(maps)
    world = client.load_world(map_name)
    logging.info('Loaded map: %s', map_name)


    # save random state:
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    for r in range(2):
        logging.info(f'{"="*14} Starting simulation {r+1}. {"="*14}')
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
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '-o', '--output',
        metavar='DIR',
        default='out',
        help='Set the global output directory')
    argparser.add_argument(
        '-f', '--fps',
        metavar='float',
        default='10.0',
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

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)


    try:
        main(args)
    finally:
        # delete the output directory if it is empty
        if not os.listdir(args.output):
            os.rmdir(args.output)
            logging.info('Deleted empty output run directory: %s', args.output)