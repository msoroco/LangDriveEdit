import json
import os
from re import match
from collections import Counter
import carla
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


    def track_metadata(self, frame, view, output_dir):
        """
        Track metadata in global files by using the instances of actors visible in the frame.
        :param frame: the output of a camera.instance segmentation sensor.
        :param view: the view of the camera.
        :param output_dir: the directory where the metadata files are stored.
        """
        entry = {}
        entry['frame'] = frame.frame
        entry['view'] = view
        metadata = []
        world_snapshot = self.world.get_snapshot()
        for actor_snapshot in world_snapshot:
            actual_actor = self.world.get_actor(actor_snapshot.id)
            actual_actor_id = actual_actor.id
            actor_metadata = {
                'id': actual_actor_id,
                'type': actual_actor.type_id,
                'semantic_class': actual_actor.semantic_tags,
                'position': {
                    'x': actor_snapshot.get_transform().location.x,
                    'y': actor_snapshot.get_transform().location.y,
                    'z': actor_snapshot.get_transform().location.z
                },
                'rotation': {
                    'pitch': actor_snapshot.get_transform().rotation.pitch,
                    'yaw': actor_snapshot.get_transform().rotation.yaw,
                    'roll': actor_snapshot.get_transform().rotation.roll
                },
                'velocity': {
                    'x': actor_snapshot.get_velocity().x,
                    'y': actor_snapshot.get_velocity().y,
                    'z': actor_snapshot.get_velocity().z
                },
                'angular_velocity': {
                    'x': actor_snapshot.get_angular_velocity().x,
                    'y': actor_snapshot.get_angular_velocity().y,
                    'z': actor_snapshot.get_angular_velocity().z
                },
                'acceleration': {
                    'x': actor_snapshot.get_acceleration().x,
                    'y': actor_snapshot.get_acceleration().y,
                    'z': actor_snapshot.get_acceleration().z
                }
                # 'control': actor_snapshot.get_control()
            }
            for attribute in actual_actor.attributes:
                actor_metadata[attribute] = actual_actor.attributes[attribute]
            if hasattr(actual_actor, 'get_state'):
                actor_metadata['actor_state'] = actual_actor.get_state()
            metadata.append(actor_metadata)
        entry['metadata'] = metadata
        self.metadata.append(entry)

        # actors = self.world.get_actors()
        # for actor in actors:
            # metadata = actor.attributes
            # # add positon to metadata
            # metadata['position'] = actor.get_transform().location
            # # add rotation to metadata
            # metadata['rotation'] = actor.get_transform().rotation
            # # add bounding box to metadata
            # metadata['bounding_box'] = actor.bounding_box
            # # add velocity to metadata
            # metadata['velocity'] = actor.get_velocity()
            # # add acceleration to metadata
            # metadata['acceleration'] = actor.get_acceleration()
            # # add angular velocity to metadata
            # metadata['angular_velocity'] = actor.get_angular_velocity()
            # # add angular acceleration to metadata
            # metadata['angular_acceleration'] = actor.get_angular_acceleration()
            # # add control to metadata
            # metadata['control'] = actor.get_control()
            # # add type to metadata
            # metadata['type'] = actor.type_id
            # # add semantic class
            # metadata['semantic_class'] = actor.semantic_tags
    def save_metadata(self):
        """
        Save the (outstanding) metadata to a json file.
        """
        with open(self.metadata_file, 'w') as f:
            for entry in self.metadata:
                json.dump(entry, f, indent=4)
                f.write('\n')






class MetadataTracker():
    def __init__(self, world, run_output_dir):
        """
        Class tracks metadata in global files by using the instances of actors visible in the frame of an instance segmentation sensor.

        The metadata is tracked in two files that describe the whole simulation.
        The first file is a jsonl file that contains the summary metadata of all actors that appear in some frame of the simulation. It also stores the list of the frames where the actor appears.

        The second file is a json file that contains, per-frame, the detailed metadata of all actors including the ego vehicle.
        """
        self.world = world
        self.output_dir = run_output_dir
        self.apperances = {}
        self.frame_metadata = {}
        self.apperances_file = os.path.join(self.output_dir, 'apperances.jsonl')
        self.frame_metadata_file = os.path.join(self.output_dir, 'frame_metadata.json')

        self.instance_to_actor = self.create_mapping_from_instance_id_to_actor_id()

        


    def track_metadata_TODO_BB(self, frame, view, output_dir, camera):
        """
        Track metadata in global files by using the instances of actors visible in the frame.
        :param frame: the output of a camera.instance segmentation sensor.
        :param view: the view of the camera.
        :param output_dir: the directory where the metadata files are stored.
        """
        frame_number = frame.frame
        key = f'{frame_number}.{view}'
        # for each pixel in the frame, the R value is a semantic tag, and the G & B values are the instance ID.
        frame_data = frame.raw_data # Flattened array of pixel data, use reshape to create an image array.
        frame_data = np.reshape(np.copy(frame.raw_data), (frame.height, frame.width, 4))
        print(view)
        print("frame_data: ", frame_data.shape)
        actors = self.world.get_actors()
        for actor in actors:
            print("actor: ", actor.id)

        visible_actors = get_visible_actors(camera, self.world, actors)

        for actor in visible_actors:
            print(f"Visible actor ID: {actor.id}")
            print(f"Visible actor type: {actor.type_id}")
        
        raise ValueError("Stop")

    def create_mapping_from_instance_id_to_actor_id(self):
        """
        Create a mapping from the instance ID of an actor to the actor ID.
        Spawns five segmentation cameeras around the actor: front, back, left, right, and top.
        Get the instance segmentation ID from each camera at the centre pixel.
        The corresponding instance segmentation ID is the ID appearing in the majority of the cameras.

        An instance segmentation id is composed of a semantic class and an instance id which is only unique within the semantic class.

        NOTE: This function needs to be called before the simulation recordings start because it will tick the world.

        Returns:
            dict: A mapping from the instance ID to the actor ID.        
        """
        def process_image(image, role, votes):
            # Process the instance segmentation image to extract the instance ID
            image_data = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) # BGRA
            # GB -> instance ID
            # R -> semantic tag
            centre_pixel = image_data[image.height//2, image.width//2]
            instance_id = int(str(centre_pixel[1]) + str(centre_pixel[0]))
            semantic_tag = centre_pixel[2]
            votes[role] = (semantic_tag, instance_id)

            image.save_to_disk(f"debugging/{semantic_tag}_{instance_id}_{role}.png")



        actors = self.world.get_actors()
        instance_id_to_actor_id = {}
        for actor in actors:
            votes = {}
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
            for i, role in enumerate(['front', 'back', 'left', 'right', 'top']):
                if role == 'top':
                    x = 0
                    y = 0
                    z = 1.5
                    camera_transform = carla.Transform(
                        actor.get_transform().location + carla.Location(x=x, y=y, z=z),
                        carla.Rotation(
                            pitch=actor.get_transform().rotation.pitch + -90,
                            yaw=actor.get_transform().rotation.yaw,
                            roll=actor.get_transform().rotation.roll
                        ))
                else:
                    x = math.cos(math.radians(90 *i))
                    y = math.sin(math.radians(90 *i))
                    camera_transform = carla.Transform(
                        actor.get_transform().location + carla.Location(x=x, y=y),
                        carla.Rotation(
                            pitch=actor.get_transform().rotation.pitch,
                            yaw=actor.get_transform().rotation.yaw + 90*i,
                            roll=actor.get_transform().rotation.roll
                        ))
                    
                camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=actor)
                camera.listen(lambda image, role=role, votes=votes: process_image(image, role, votes))
                c = 0
                while len(votes) < i+1:
                    self.world.tick()
                    c = c+1
                    if c > 12: # patience
                        break
                camera.destroy()
            if len(votes) != 5: raise Warning(f"Actor {actor.id} did not have 5 votes")
            # once all votes are in, get the tuple that appears the most.
            winner = Counter(votes.values()).most_common(1)[0]
            instance_id = winner[0][1]
            semantic_tag = winner[0][0]
            print(f"Actor {actor.id} has instance ID {(semantic_tag, instance_id)}")
            print(actor.attributes)
            if (semantic_tag, instance_id) in instance_id_to_actor_id:
                if instance_id_to_actor_id[(semantic_tag, instance_id)] != actor.id:
                    print(actor.attributes)
                    a = self.world.get_actor(instance_id_to_actor_id[(semantic_tag, instance_id)])
                    print(a.attributes)
                    raise ValueError(f"Instance ID {(semantic_tag, instance_id)} already associated with actor {instance_id_to_actor_id[(semantic_tag, instance_id)]}")
            instance_id_to_actor_id[(semantic_tag, instance_id)] = actor.id
            self.world.tick() # destroy camera
        return instance_id_to_actor_id



    def track_metadata_TODO(self, frame, view, output_dir):
        """
        Track metadata in global files by using the instances of actors visible in the frame.
        :param frame: the output of a camera.instance segmentation sensor.
        :param view: the view of the camera.
        :param output_dir: the directory where the metadata files are stored.
        """
        frame_number = frame.frame
        key = f'{frame_number}.{view}'
        # for each pixel in the frame, the R value is a semantic tag, and the G & B values are the instance ID.
        frame_data = frame.raw_data # Flattened array of pixel data, use reshape to create an image array.
        frame_data = np.reshape(np.copy(frame.raw_data), (frame.height, frame.width, 4))
        print(view)
        print("frame_data: ", frame_data.shape)
        actors = self.world.get_actors()
        actors_seen = []
        for i in range(frame_data.shape[0]):
            for j in range(frame_data.shape[1]):
                actor_class = frame_data[i, j, 2]
                G = frame_data[i, j, 1] 
                B =frame_data[i, j, 0]
                actor_id = int(str(G) + str(B))
                if (actor_class, actor_id) not in actors_seen:
                    actors_seen.append((actor_class, actor_id))
                    print("BGRA: ", frame_data[i, j])
                    print("actor_class: ", actor_class)
                    print("actor_id: ", actor_id)
                    if actor_class == 14:
                        print("car!")
                        # print("BGRA: ", frame_data[i, j])
                        # print("actor_class: ", actor_class)
                        # print("car actor_id: ", actor_id)
                for actor in actors:
                    if actor.id == self.instance_to_actor[(actor_class, actor_id)]:
                        print("actor: ", actor.attributes)
        raise ValueError("Stop")

                
