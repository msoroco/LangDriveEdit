import json
import os
import random
from re import match
from collections import Counter
import carla
from carla import command
import numpy as np
import math
from utils import get_visible_actors



class MetadataTracker_simple():
    def __init__(self, world, run_output_dir):
        """
        Class tracks metadata in global files by using the instances of actors visible in the frame of an instance segmentation sensor.

        The metadata is tracked in two files that describe the whole simulation.
        The first file is a jsonl file that contains the summary metadata of all actors that appear in some frame of the simulation. It also stores the list of the frames where the actor appears.

        The second file is a json file that contains, per-frame, the detailed metadata of all actors including the ego vehicle.
        """
        self.world = world
        self.output_dir = run_output_dir
        # self.apperances = {}
        self.metadata = []
        # self.apperances_file = os.path.join(self.output_dir, 'apperances.jsonl')
        self.metadata_file = os.path.join(self.output_dir, 'frame_metadata.json')

    def get_frame_number(self, frame):
        offset = getattr(self, 'frame_offset', None)
        if offset is None:
            self.frame_offset = int(frame)
            return 0
        else:
            return frame - offset

    def save_args(self, args):
        """save the argparse arguments at the top of the metadata."""
        args_dict = vars(args)
        args_json = json.dumps(args_dict)
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                existing_content = f.read()
        else:
            existing_content = ""
        with open(self.metadata_file, 'w') as f:
            f.write(args_json + '\n')
            f.write(existing_content)

    def get_walker_description(self, variant_number):
        # ue4 older carla.
        variant_number = str(variant_number).zfill(4)
        pedestrian_dict = {
            '0001': {'catalogue': 'Adult pedestrian - 1', 'age': 'adult', 'sex': 'female', 'hair': 'blond', 'top': 'beige tank top', 'pants': 'dark grey jeans', 'footwear': 'brown shoes'},
            '0005': {'catalogue': 'Adult pedestrian - 1', 'age': 'adult', 'sex': 'female', 'hair': 'blond', 'top': 'turquoise tank top', 'pants': 'red jeans', 'footwear': 'light blue shoes'},
            '0006': {'catalogue': 'Adult pedestrian - 1', 'age': 'adult', 'sex': 'female', 'hair': 'blond', 'top': 'green tank top', 'pants': 'black jeans', 'footwear': 'purple shoes'},
            '0007': {'catalogue': 'Adult pedestrian - 1', 'age': 'adult', 'sex': 'female', 'hair': 'blond', 'top': 'burgandy tank top', 'pants': 'grey jeans', 'footwear': 'orange shoes'},
            '0008': {'catalogue': 'Adult pedestrian - 1', 'age': 'adult', 'sex': 'female', 'hair': 'blond', 'top': 'red tank top', 'pants': 'orange jeans', 'footwear': 'yellow shoes'},
            '0004': {'catalogue': 'Adult pedestrian - 2', 'age': 'adult', 'sex': 'male', 'hair': 'blond', 'top': 'yellow shirt', 'pants': 'light blue jeans', 'footwear': 'brown shoes'},
            '0003': {'catalogue': 'Adult pedestrian - 2', 'age': 'adult', 'sex': 'male', 'hair': 'blond', 'top': 'grey shirt', 'pants': 'light blue jeans', 'footwear': 'brown shoes'},
            '0002': {'catalogue': 'Adult pedestrian - 2', 'age': 'adult', 'sex': 'male', 'hair': 'blond', 'top': 'white blue', 'pants': 'light blue jeans', 'footwear': 'brown shoes'},
            '0015': {'catalogue': 'Adult pedestrian - 3', 'age': 'adult', 'sex': 'female', 'hair': 'brown', 'top': 'brown leather jacket', 'pants': 'black jeans', 'footwear': 'purple shoes'},
            '0019': {'catalogue': 'Adult pedestrian - 3', 'age': 'adult', 'sex': 'female', 'hair': 'brown', 'top': 'light brown jacket', 'pants': 'navy jeans', 'footwear': 'yellow shoes'},
            '0016': {'catalogue': 'Adult pedestrian - 4', 'age': 'adult', 'sex': 'male', 'hair': 'brown', 'top': 'brown jacket', 'pants': 'grey pants', 'footwear': 'brown shoes'},
            '0017': {'catalogue': 'Adult pedestrian - 4', 'age': 'adult', 'sex': 'male', 'hair': 'brown', 'top': 'blue jacket', 'pants': 'green pants', 'footwear': 'brown shoes'},
            '0026': {'catalogue': 'Adult pedestrian - 5', 'age': 'adult', 'sex': 'male', 'hair': 'bald', 'top': 'yellow jacket', 'pants': 'grey pants', 'footwear': 'brown shoes'},
            '0018': {'catalogue': 'Adult pedestrian - 5', 'age': 'adult', 'sex': 'male', 'hair': 'bald', 'top': 'blue jacket', 'pants': 'green pants', 'footwear': 'brown shoes'},
            '0021': {'catalogue': 'Adult pedestrian - 6', 'age': 'adult', 'sex': 'female', 'hair': 'blond', 'top': 'red jacket', 'pants': 'navy jeans', 'footwear': 'brown shoes'},
            '0020': {'catalogue': 'Adult pedestrian - 6', 'age': 'adult', 'sex': 'female', 'hair': 'blond', 'top': 'beige jacket', 'pants': 'black jeans', 'footwear': 'light shoes'},
            '0023': {'catalogue': 'Adult pedestrian - 7', 'age': 'adult', 'sex': 'female', 'hair': 'brown', 'top': 'green jacket', 'pants': 'grey jeans', 'footwear': 'blue shoes'},
            '0022': {'catalogue': 'Adult pedestrian - 7', 'age': 'adult', 'sex': 'female', 'hair': 'brown', 'top': 'grey jacket', 'pants': 'red jeans', 'footwear': 'brown shoes'},
            '0024': {'catalogue': 'Adult pedestrian - 8', 'age': 'adult', 'sex': 'male', 'hair': 'brown', 'top': 'yellow jacket', 'pants': 'grey pants', 'footwear': 'brown shoes'},
            '0025': {'catalogue': 'Adult pedestrian - 8', 'age': 'adult', 'sex': 'male', 'hair': 'brown', 'top': 'brown leather jacket', 'pants': 'camo pants', 'footwear': 'beige shoes'},
            '0027': {'catalogue': 'Adult pedestrian - 9', 'age': 'adult', 'sex': 'male', 'hair': 'black', 'top': 'green shirt', 'pants': 'black pants', 'footwear': 'brown shoes'},
            '0029': {'catalogue': 'Adult pedestrian - 9', 'age': 'adult', 'sex': 'male', 'hair': 'black', 'top': 'blue shirt', 'pants': 'beige pants', 'footwear': 'orange shoes'},
            '0028': {'catalogue': 'Adult pedestrian - 9', 'age': 'adult', 'sex': 'male', 'hair': 'black', 'top': 'grey shirt', 'pants': 'navy pants', 'footwear': 'brown sneakers'},
            '0041': {'catalogue': 'Adult pedestrian - 10', 'age': 'adult', 'sex': 'female', 'hair': 'black', 'top': 'turquoise jacket', 'pants': 'brown pants', 'footwear': 'grey sneakers'},
            '0040': {'catalogue': 'Adult pedestrian - 10', 'age': 'adult', 'sex': 'female', 'hair': 'black', 'top': 'beige jacket', 'pants': 'beige pants', 'footwear': 'black sneakers'},
            '0033': {'catalogue': 'Adult pedestrian - 10', 'age': 'adult', 'sex': 'female', 'hair': 'black', 'top': 'green jacket', 'pants': 'brown pants', 'footwear': 'grey sneakers'},
            '0031': {'catalogue': 'Adult pedestrian - 10', 'age': 'adult', 'sex': 'female', 'hair': 'black', 'top': 'brown jacket', 'pants': 'grey pants', 'footwear': 'grey sneakers'},
            '0034': {'catalogue': 'Adult pedestrian - 11', 'age': 'adult', 'sex': 'male', 'hair': 'black', 'top': 'brown sweater', 'pants': 'grey pants', 'footwear': 'grey sneakers'},
            '0038': {'catalogue': 'Adult pedestrian - 11', 'age': 'adult', 'sex': 'male', 'hair': 'black', 'top': 'red sweater', 'pants': 'black pants', 'footwear': 'red sneakers'},
            '0035': {'catalogue': 'Adult pedestrian - 12', 'age': 'adult', 'sex': 'female', 'hair': 'red', 'top': 'long white sweater', 'pants': 'pink pants', 'footwear': 'yellow boots'},
            '0036': {'catalogue': 'Adult pedestrian - 12', 'age': 'adult', 'sex': 'female', 'hair': 'red', 'top': 'long rainbow sweater', 'pants': 'grey pants', 'footwear': 'green boots'},
            '0037': {'catalogue': 'Adult pedestrian - 12', 'age': 'adult', 'sex': 'female', 'hair': 'red', 'top': 'long grey sweater', 'pants': 'grey pants', 'footwear': 'grey boots'},
            '0039': {'catalogue': 'Adult pedestrian - 13', 'age': 'adult', 'sex': 'male', 'hair': 'grey', 'top': 'brown shirt', 'pants': 'grey pants', 'footwear': 'brown shoes'},
            '0042': {'catalogue': 'Adult pedestrian - 14', 'age': 'adult', 'sex': 'female', 'hair': 'grey', 'top': 'yellow shirt', 'pants': 'grey pants', 'footwear': 'brown shoes'},
            '0043': {'catalogue': 'Adult pedestrian - 14', 'age': 'adult', 'sex': 'female', 'hair': 'grey', 'top': 'white shirt', 'pants': 'white pants', 'footwear': 'grey shoes'},
            '0044': {'catalogue': 'Adult pedestrian - 14', 'age': 'adult', 'sex': 'female', 'hair': 'grey', 'top': 'grey shirt', 'pants': 'blue pants', 'footwear': 'brown shoes'},
            '0047': {'catalogue': 'Adult pedestrian - 15', 'age': 'adult', 'sex': 'female', 'hair': 'black', 'top': 'grey shirt', 'pants': 'beige pants', 'footwear': 'brown shoes'},
            '0046': {'catalogue': 'Adult pedestrian - 15', 'age': 'adult', 'sex': 'female', 'hair': 'grey', 'top': 'light grey shirt', 'pants': 'black pants', 'footwear': 'brown shoes'},
            '0045': {'catalogue': 'Adult pedestrian - 15', 'age': 'adult', 'sex': 'female', 'hair': 'grey', 'top': 'blue shirt', 'pants': 'white pants', 'footwear': 'brown shoes'},
            '0011': {'catalogue': 'Child pedestrian - 1', 'age': 'child', 'sex': 'male', 'hair': 'brown', 'top': 'yellow shirt', 'pants': 'black shorts', 'footwear': 'grey sneakers'},
            '0010': {'catalogue': 'Child pedestrian - 1', 'age': 'child', 'sex': 'male', 'hair': 'brown', 'top': 'light blue shirt', 'pants': 'black shorts', 'footwear': 'grey sneakers'},
            '0009': {'catalogue': 'Child pedestrian - 1', 'age': 'child', 'sex': 'male', 'hair': 'brown', 'top': 'white shirt', 'pants': 'black shorts', 'footwear': 'white sneakers'},
            '0014': {'catalogue': 'Child pedestrian - 2', 'age': 'child', 'sex': 'male', 'hair': 'brown', 'top': 'white shirt', 'pants': 'yellow overalls', 'footwear': 'white shoes'},
            '0013': {'catalogue': 'Child pedestrian - 2', 'age': 'child', 'sex': 'male', 'hair': 'brown', 'top': 'purple shirt', 'pants': 'grey overalls', 'footwear': 'grey shoes'},
            '0012': {'catalogue': 'Child pedestrian - 2', 'age': 'child', 'sex': 'male', 'hair': 'brown', 'top': 'grey shirt', 'pants': 'blue overalls', 'footwear': 'grey shoes'},
            '0048': {'catalogue': 'Child pedestrian - 3', 'age': 'child', 'sex': 'female', 'hair': 'brown', 'top': 'yellow green jacket', 'pants': 'pink shorts', 'footwear': 'red sneakers'},
            '0049': {'catalogue': 'Child pedestrian - 4', 'age':'child', 'sex': 'male', 'hair': 'black', 'top': 'blue jacket', 'pants': 'green shorts', 'footwear': 'red sneakers'},
            '0050': {'catalogue': 'Child pedestrian - 1', 'age': 'child', 'sex': 'female', 'hair': 'black', 'top': 'pink shirt', 'pants': 'blue shorts', 'footwear': 'white shoes'},
            '0051': {'catalogue': 'Child pedestrian - 2', 'age': 'child', 'sex': 'female', 'hair': 'black', 'top': 'grey shirt', 'pants': 'light blue pants', 'footwear': 'white shoes'},
            '0030': {'catalogue': 'Police - 1', 'age': 'adult', 'sex': 'male', 'hair': 'police hat', 'top': 'police uniform', 'pants': 'police uniform', 'footwear': 'police uniform'},
            '0032': {'catalogue': 'Police - 2', 'age': 'adult', 'sex': 'female', 'hair': 'police hat', 'top': 'police uniform', 'pants': 'police uniform', 'footwear': 'police uniform'}
        }

        clothing = pedestrian_dict.get(variant_number)
        if clothing is None:
            print(f"Unknown walker variant: {variant_number}")
        return clothing

    def track_metadata(self, frame, view, intersting_actors=None, bboxes=None):
        """
        Track metadata in global files by using the instances of actors visible in the frame.
        :param frame: the framenumber which is output of a camera.instance segmentation sensor.
        :param view: the view of the camera.
        :param intersting_actors: the list of actors to track. If None, all actors are tracked.
        :param bboxes: the bounding boxes of intersting_actors to track.  Each key is the instance_id, and the value is a tuple: (x0, y0, x1, y1)
        """
        entry = {}
        entry['world_frame'] = frame # to map to sensors
        entry['frame_number'] = self.get_frame_number(frame)
        entry['view'] = view
        world_snapshot = self.world.get_snapshot()
        # assert world_snapshot.frame == frame
        # for actor_snapshot in world_snapshot: ### for all actors
        vehicle_actors = self.world.get_actors().filter('vehicle.*')
        walker_actors = self.world.get_actors().filter('walker.*')
        traffic_actors = self.world.get_actors().filter('traffic.traffic_light')
        sensor_actors = self.world.get_actors().filter('sensor.*')

        all_actors = list(vehicle_actors) + list(walker_actors) + list(sensor_actors) + list(traffic_actors)
        if intersting_actors is not None:
            all_actors = [actor for actor in all_actors if actor.id in intersting_actors]
            all_actors += list(sensor_actors)

        for actual_actor in all_actors:
            actual_actor_id = actual_actor.id
            actor_snapshot = world_snapshot.find(actual_actor_id)

            # actual_actor = self.world.get_actor(actor_snapshot.id)

            actual_actor_type = actual_actor.type_id
            actor_class = self.get_actor_type(actual_actor_type)
            bounding_box = actual_actor.bounding_box
            transform = actor_snapshot.get_transform()
            velocity = actor_snapshot.get_velocity()
            angular_velocity = actor_snapshot.get_angular_velocity()
            acceleration = actor_snapshot.get_acceleration()
            
            actor_metadata = {
                'id': actual_actor_id,
                'type': actual_actor.type_id,
                'semantic_class': actual_actor.semantic_tags,
                'position': {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z
                },
                'rotation': {
                    'pitch': transform.rotation.pitch,
                    'yaw': transform.rotation.yaw,
                    'roll': transform.rotation.roll
                }
            }
            if bboxes and actual_actor_id in bboxes:
                bbox = bboxes[actual_actor_id]
                actor_metadata['image_bbox_2d'] = {
                    'x0': bbox[0],
                    'y0': bbox[1],
                    'x1': bbox[2],
                    'y1': bbox[3]
                }
            if actor_class == 'vehicle' or actor_class == 'walker' or actor_class == 'traffic':
                actor_metadata.update({
                    'bounding_box': {
                        'extent': {
                            'x': bounding_box.extent.x,
                            'y': bounding_box.extent.y,
                            'z': bounding_box.extent.z
                        },
                        'location': {
                            'x': bounding_box.location.x,
                            'y': bounding_box.location.y,
                            'z': bounding_box.location.z
                        },
                        'rotation': {
                            'pitch': bounding_box.rotation.pitch,
                            'yaw': bounding_box.rotation.yaw,
                            'roll': bounding_box.rotation.roll
                        }
                    }
                })
            if actor_class == 'vehicle' or actor_class == 'walker':
                actor_metadata.update({
                        'velocity': {
                            'x': velocity.x,
                            'y': velocity.y,
                            'z': velocity.z
                        },
                        'angular_velocity': {
                            'x': angular_velocity.x,
                            'y': angular_velocity.y,
                            'z': angular_velocity.z
                        },
                        'acceleration': {
                            'x': acceleration.x,
                            'y': acceleration.y,
                            'z': acceleration.z
                        }
                        # 'control': actor_snapshot.get_control()
                    })
            if actor_class == 'walker':
                # extract clothing colours which must be mapped manually:
                walker_type = actual_actor.type_id
                # print("walker_type: ", walker_type)
                parts = walker_type.split('.')
                if len(parts) == 3 and parts[0] == "walker" and parts[1] == "pedestrian":
                    walker_variant = int(parts[2])
                    clothing = self.get_walker_description(walker_variant)


                # add clothing dict to actor_metadata
                if clothing is not None:
                    actor_metadata.update({
                        'clothing': {
                            'catalogue': clothing['catalogue'],
                            # 'age': clothing['age'], # already included by carla
                            # 'sex': clothing['sex'], # already included by carla
                            'hair': clothing['hair'],
                            'top': clothing['top'],
                            'pants': clothing['pants'],
                            'footwear': clothing['footwear']
                        }
                    })
            if actor_class == 'traffic':
                # extract traffic light state
                state = actual_actor.get_state()
                if state == carla.TrafficLightState.Red:
                    actor_metadata['light_colour'] = 'red'
                elif state == carla.TrafficLightState.Yellow:
                    actor_metadata['light_colour'] = 'yellow'
                elif state == carla.TrafficLightState.Green:
                    actor_metadata['light_colour'] = 'green'
                else:
                    actor_metadata['light_colour'] = 'off'





            # for attribute in actual_actor.attributes:
            #     actor_metadata[attribute] = actual_actor.attributes[attribute]
            actor_metadata.update(actual_actor.attributes)
            if hasattr(actual_actor, 'get_state'):
                actor_metadata['actor_state'] = actual_actor.get_state()
            if f"{actor_class}_metadata" not in  entry:
                entry[f"{actor_class}_metadata"] = []

            entry[f"{actor_class}_metadata"].append(actor_metadata)
        self.metadata.append(entry)


    def get_actor_type(self, actor_type):
        if 'vehicle' in actor_type:
            return 'vehicle'
        if 'walker' in actor_type:
            return 'walker'
        if "sensor" in actor_type:
            return 'sensor'
        else:
            return 'traffic'
        
    def remove_frame_not_listed(self, frames_to_keep):
        """
        Remove all frames that are not in the list of *world* frame ids  to keep.
        """
        self.metadata = [entry for entry in self.metadata if entry['world_frame'] in frames_to_keep]

    def save_metadata(self):
        """
        Save the (outstanding) metadata to a json file.
        """
        try:
            with open(self.metadata_file, 'a') as f:
                for entry in self.metadata:
                    json.dump(entry, f)
                    f.write('\n')
            # Clear the metadata list after saving
            self.metadata.clear()
        except Exception as e:
            print(f"Failed to save metadata: {e}")


class ActorRecord():
    def __init__(self, location, rotation):
        self.location = location
        self.rotation = rotation


class PositionTracker():
    def __init__(self, label):
        """
        Class tracks the position of the actors in the world.
        """
        # self.world = world
        # self.output_dir = run_output_dir
        # self.positions = {}
        # self.positions_file = os.path.join(self.output_dir, 'positions.jsonl')
        self.label = label
        self.active = False
        self.simulation_record = {}
        self.frame_offset = None

    def track_positions(self, actor_list, frame_id):
        """
        If inactive: track the positions of the actors in the world.
        """
        if not self.active:
            frame = self.get_frame_number(frame_id)
            # print("setting frame: ", frame)
            for i, actor in enumerate(actor_list):
                transform = actor.get_transform()
                location = {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    # 'z': transform.location.z
                }
                rotation = {
                    'pitch': transform.rotation.pitch,
                    'yaw': transform.rotation.yaw,
                    'roll': transform.rotation.roll
                }
                if i not in self.simulation_record:
                    self.simulation_record[i] = {}
                self.simulation_record[i].update({frame: ActorRecord(location, rotation)})

    def set_location(self, actor_list, frame_id):
        """
        If active: Set the location of the actors in the world.
        """
        if self.active:
            frame = self.get_frame_number(frame_id)
            # print("setting frame: ", frame)
            records = self.simulation_record_copy
            for i, actor in enumerate(actor_list):
                actor_location = records[i][frame].location
                actor_rotation = records[i][frame].rotation
                existing_location = actor.get_transform().location
                # print("actor_location: ", existing_location)
                # print("new actor_location: ", actor_location)
                actor.set_transform(carla.Transform(carla.Location(z=existing_location.z, **actor_location), carla.Rotation(**actor_rotation)))

    def set_location_batch(self, actor_list, frame_id, client):
        """
        If active: Set the location of the actors in the world.
        """
        if self.active:
            frame = self.get_frame_number(frame_id)
            # print("setting frame: ", frame)
            records = self.simulation_record_copy
            batch = []
            for i, actor in enumerate(actor_list):
                actor_location = records[i][frame].location
                actor_rotation = records[i][frame].rotation
                existing_location = actor.get_transform().location
                # print("actor_location: ", existing_location)
                # print("new actor_location: ", actor_location)
                new_transform = carla.Transform(
                    carla.Location(x=actor_location['x'], y=actor_location['y'], z=existing_location.z),
                    carla.Rotation(pitch=actor_rotation['pitch'], yaw=actor_rotation['yaw'], roll=actor_rotation['roll'])
                )
                batch.append(command.ApplyTransform(actor.id, new_transform))

            # Apply the batch of commands synchronously
            client.apply_batch_sync(batch, False)


    def ignore_actors(self, actor_indices):
        """
        Ignores these actors when reading the position logs to update actors.
        """
        if not self.active:
            raise ValueError("Position tracker is not active")
    
        actor_indices_copy = actor_indices.copy()
        print("Ignoring actors: ", actor_indices_copy)
        # remove the actors from the simulation record copy
        actor_indices_copy.sort(reverse=True)
        for actor_index in actor_indices_copy:
            self.simulation_record_copy.pop(actor_index, None)

    def activate(self):
        """Stop tracking the positions of the actors and prepare to use the records of actors tracked so far to update positions."""
        self.active = True
        self.simulation_record_copy = self.simulation_record.copy()

    def reset_but_keep_record(self):
        """Reset the state of the tracker for using the records of the actors to update a new starting simulation."""
        self.frame_offset = None
        self.simulation_record_copy = {}

    def deactivate(self):
        """Delete all records of the actors. Resets the tracker to start tracking the positions of the actors again."""
        self.active = False
        self.frame_offset = None
        self.simulation_record = {}

    def is_active(self):
        """Returns wherther the tracker has a complete log (active) or is in the progress of tracking the positions of the actors (inactive)."""
        return self.active
    
    def get_frame_number(self, frame):
        return frame
        # offset = getattr(self, 'frame_offset', None)
        # if offset is None:
        #     self.frame_offset = int(frame)
        #     return 0
        # else:
        #     return frame - offset



class LightTracker(PositionTracker):
    def __init__(self, label):
        super().__init__(label)
        self.permutation = self.generate_random_mapping()

    def get_permutation(self):
        return self.permutation

    def generate_random_mapping(self):
        """ Mapping of [red, yellow, green] to new traffic light colours"""
        bernoulli = random.random()
        if bernoulli > 0.5:
            return ['yellow', 'green', 'red']
        else:
            return ['green', 'red', 'yellow']

    def track_light_states(self, actor_list, frame_id):
        """
        If inactive: track the light colour of the traffic lights in the world.
        """
        if not self.active:
            frame = self.get_frame_number(frame_id)
            for i, actor in enumerate(actor_list):
                state = actor.get_state()
                if state == carla.TrafficLightState.Red:
                    state = 'red'
                elif state == carla.TrafficLightState.Yellow:
                    state = 'yellow'
                elif state == carla.TrafficLightState.Green:
                    state = 'green'
                else:
                    state = 'off' # or unknown

                if i not in self.simulation_record:
                    self.simulation_record[i] = {}
                self.simulation_record[i].update({frame: state})

    def set_light_states(self, actor_list, frame_id):
        """
        If active: Set the light colour of the traffic lights in the world.
        """
        if self.active:
            frame = self.get_frame_number(frame_id)
            records = self.simulation_record_copy
            for i, actor in enumerate(actor_list):
                state = records[i][frame]
                if state == 'red':
                    new_state = self.permutation[0]
                elif state == 'yellow':
                    new_state = self.permutation[1]
                elif state == 'green':
                    new_state = self.permutation[2]
                else:
                    new_state = 'yellow' # least likely colour

                if new_state == 'red':
                    actor.set_state(carla.TrafficLightState.Red)
                elif new_state == 'yellow':
                    actor.set_state(carla.TrafficLightState.Yellow)
                elif new_state == 'green':
                    actor.set_state(carla.TrafficLightState.Green)

    # def set_light_states_batch(self, actor_list, frame_id, client):
    #     """
    #     If active: Set the light colour of the traffic lights in the world.
    #     """
    #     if self.active:
    #         frame = self.get_frame_number(frame_id)
    #         records = self.simulation_record_copy
    #         batch = []
    #         for i, actor in enumerate(actor_list):
    #             state = records[i][frame]
    #             if state == 'red':
    #                 new_state = carla.TrafficLightState.Yellow
    #             elif state == 'yellow':
    #                 new_state = carla.TrafficLightState.Green
    #             elif state == 'green':
    #                 new_state = carla.TrafficLightState.Red
    #             else:
    #                 new_state = carla.TrafficLightState.Yellow
    #             batch.append(command.SetTrafficLightState(actor.id, new_state))

    #         # Apply the batch of commands synchronously
    #         client.apply_batch_sync(batch, False)



# class MetadataTracker():
#     def __init__(self, world, run_output_dir):
#         """
#         Class tracks metadata in global files by using the instances of actors visible in the frame of an instance segmentation sensor.

#         The metadata is tracked in two files that describe the whole simulation.
#         The first file is a jsonl file that contains the summary metadata of all actors that appear in some frame of the simulation. It also stores the list of the frames where the actor appears.

#         The second file is a json file that contains, per-frame, the detailed metadata of all actors including the ego vehicle.
#         """
#         self.world = world
#         self.output_dir = run_output_dir
#         self.apperances = {}
#         self.frame_metadata = {}
#         self.apperances_file = os.path.join(self.output_dir, 'apperances.jsonl')
#         self.frame_metadata_file = os.path.join(self.output_dir, 'frame_metadata.json')

#         # self.instance_to_actor = self.create_mapping_from_instance_id_to_actor_id()

#     def save_metadata(self):
#         pass
#         print("Not implemented yet")
        
#     def track_metadata(self, frame, view, output_dir):
#         return self.track_metadata_TODO(frame, view, output_dir)

#     def track_metadata_TODO_BB(self, frame, view, output_dir, camera):
#         """
#         Track metadata in global files by using the instances of actors visible in the frame.
#         :param frame: the output of a camera.instance segmentation sensor.
#         :param view: the view of the camera.
#         :param output_dir: the directory where the metadata files are stored.
#         """
#         frame_number = frame.frame
#         key = f'{frame_number}.{view}'
#         # for each pixel in the frame, the R value is a semantic tag, and the G & B values are the instance ID.
#         frame_data = frame.raw_data # Flattened array of pixel data, use reshape to create an image array.
#         frame_data = np.reshape(np.copy(frame.raw_data), (frame.height, frame.width, 4))
#         print(view)
#         print("frame_data: ", frame_data.shape)
#         actors = self.world.get_actors()
#         for actor in actors:
#             print("actor: ", actor.id)

#         visible_actors = get_visible_actors(camera, self.world, actors)

#         for actor in visible_actors:
#             print(f"Visible actor ID: {actor.id}")
#             print(f"Visible actor type: {actor.type_id}")
        
#         raise ValueError("Stop")

#     def create_mapping_from_instance_id_to_actor_id(self):
#         """
#         Create a mapping from the instance ID of an actor to the actor ID.
#         Spawns five segmentation cameeras around the actor: front, back, left, right, and top.
#         Get the instance segmentation ID from each camera at the centre pixel.
#         The corresponding instance segmentation ID is the ID appearing in the majority of the cameras.

#         An instance segmentation id is composed of a semantic class and an instance id which is only unique within the semantic class.

#         NOTE: This function needs to be called before the simulation recordings start because it will tick the world.

#         Returns:
#             dict: A mapping from the instance ID to the actor ID.        
#         """
#         def process_image(image, role, votes):
#             # Process the instance segmentation image to extract the instance ID
#             image_data = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) # BGRA
#             # GB -> instance ID
#             # R -> semantic tag
#             centre_pixel = image_data[image.height//2, image.width//2]
#             instance_id = int(str(centre_pixel[1]) + str(centre_pixel[0]))
#             semantic_tag = centre_pixel[2]
#             votes[role] = (semantic_tag, instance_id)

#             image.save_to_disk(f"debugging/{semantic_tag}_{instance_id}_{role}.png")



#         actors = self.world.get_actors()
#         instance_id_to_actor_id = {}
#         for actor in actors:
#             votes = {}
#             camera_bp = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
#             for i, role in enumerate(['front', 'back', 'left', 'right', 'top']):
#                 if role == 'top':
#                     x = 0
#                     y = 0
#                     z = 1.5
#                     camera_transform = carla.Transform(
#                         actor.get_transform().location + carla.Location(x=x, y=y, z=z),
#                         carla.Rotation(
#                             pitch=actor.get_transform().rotation.pitch + -90,
#                             yaw=actor.get_transform().rotation.yaw,
#                             roll=actor.get_transform().rotation.roll
#                         ))
#                 else:
#                     x = math.cos(math.radians(90 *i))
#                     y = math.sin(math.radians(90 *i))
#                     camera_transform = carla.Transform(
#                         actor.get_transform().location + carla.Location(x=x, y=y),
#                         carla.Rotation(
#                             pitch=actor.get_transform().rotation.pitch,
#                             yaw=actor.get_transform().rotation.yaw + 90*i,
#                             roll=actor.get_transform().rotation.roll
#                         ))
                    
#                 camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=actor)
#                 camera.listen(lambda image, role=role, votes=votes: process_image(image, role, votes))
#                 c = 0
#                 while len(votes) < i+1:
#                     self.world.tick()
#                     c = c+1
#                     if c > 12: # patience
#                         break
#                 camera.destroy()
#             if len(votes) != 5: raise Warning(f"Actor {actor.id} did not have 5 votes")
#             # once all votes are in, get the tuple that appears the most.
#             winner = Counter(votes.values()).most_common(1)[0]
#             instance_id = winner[0][1]
#             semantic_tag = winner[0][0]
#             print(f"Actor {actor.id} has instance ID {(semantic_tag, instance_id)}")
#             print(actor.attributes)
#             if (semantic_tag, instance_id) in instance_id_to_actor_id:
#                 if instance_id_to_actor_id[(semantic_tag, instance_id)] != actor.id:
#                     print(actor.attributes)
#                     a = self.world.get_actor(instance_id_to_actor_id[(semantic_tag, instance_id)])
#                     print(a.attributes)
#                     raise ValueError(f"Instance ID {(semantic_tag, instance_id)} already associated with actor {instance_id_to_actor_id[(semantic_tag, instance_id)]}")
#             instance_id_to_actor_id[(semantic_tag, instance_id)] = actor.id
#             self.world.tick() # destroy camera
#         return instance_id_to_actor_id



#     def track_metadata_TODO(self, frame, view, output_dir):
#         """
#         Track metadata in global files by using the instances of actors visible in the frame.
#         :param frame: the output of a camera.instance segmentation sensor.
#         :param view: the view of the camera.
#         :param output_dir: the directory where the metadata files are stored.
#         """
#         frame_number = frame.frame
#         key = f'{frame_number}.{view}'
#         # for each pixel in the frame, the R value is a semantic tag, and the G & B values are the instance ID.
#         frame_data = frame.raw_data # Flattened array of pixel data, use reshape to create an image array.
#         frame_data = np.reshape(np.copy(frame.raw_data), (frame.height, frame.width, 4))
#         print(view)
#         print("frame_data: ", frame_data.shape)
#         actors = self.world.get_actors()
#         vehicle_actors = actors.filter('vehicle.*')
#         for actor in vehicle_actors:
#             print(actor.id)
#         env_objs = self.world.get_environment_objects(carla.CityObjectLabel.Buildings)
#         for actor in env_objs:
#             actor_id = actor.id
#             boundary_value = 65536
#             if actor_id > boundary_value:
#                 actor_id -= math.floor(actor_id / boundary_value) * boundary_value
#             print(actor_id, actor.id)
#         actors_seen = []
#         for i in range(frame_data.shape[0]):
#             for j in range(frame_data.shape[1]):
#                 actor_class = frame_data[i, j, 2]
#                 G = int(frame_data[i, j, 1])
#                 B = int(frame_data[i, j, 0])
#                 # actor_id = int(str(G) + str(B))
#                 actor_id = ((G << 0) & 0x00ff) | ((B << 8) & 0xff00)
#                 if (actor_class, actor_id) not in actors_seen:
#                     actors_seen.append((actor_class, actor_id))
#                     # print("BGRA: ", frame_data[i, j])
#                     # print("actor_class: ", actor_class)
#                     # print("actor_id: ", actor_id)
#                     if actor_class == 14:
#                         print("car!")
#                         print("BGRA: ", frame_data[i, j])
#                         print("actor_class: ", actor_class)
#                         print("car actor_id: ", actor_id)
#                         a = actors.find(int(actor_id)) 
#                         print(a)
#                         try:
#                             print(a.attributes)
#                         except:
#                             pass
#                     if actor_class == 3:
#                         print("buidling!")
#                         print("BGRA: ", frame_data[i, j])
#                         print("actor_class: ", actor_class)
#                         print("buidling actor_id: ", actor_id)
#                         a = actors.find(int(actor_id)) 
#                         print(a)
#                         try:
#                             print(a.attributes)
#                         except:
#                             pass


#                     # if actor.id == self.instance_to_actor[(actor_class, actor_id)]:
#                     #     print("actor: ", actor.attributes)
#         raise ValueError("Stop")

                
