import os
import imageio
import numpy as np
import math
import random

def create_video_from_images(image_folder, output_dir, fps=30, savetype='mp4', levels=3):
    """
    Create a
    :param image_folder: folder containing images
    :param output_file: output video file
    :param fps: frames per second
    :return: savepath
    """

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    if len(images) == 0:
        print(f"No images found in {image_folder}")
        return
    images.sort() # sort images by name
    frames = [imageio.imread(os.path.join(image_folder, img)) for img in images]
    # get the names of each folder for the last `levels` folders
    parts = []
    for _ in range(levels):
        image_folder, tail = os.path.split(image_folder)
        if tail:
            parts.append(tail)
        else:
            break
    name = ".".join(reversed(parts))

    if savetype == "gif":
        savepath = os.path.join(
            output_dir, f"{name}.gif"
        )
        imageio.mimsave(
            savepath,
            frames,
            format="GIF",
            duration=1 / fps,
            loop=0,
        )
    elif savetype == "mp4":
        savepath = os.path.join(
            output_dir, f"{name}.mp4"
        )
        with imageio.get_writer(savepath, fps=fps, codec='libx264') as writer:
            for frame in frames:
                writer.append_data(frame)
    return savepath


def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """
    Create a projection matrix to project 3D coordinates to 2D camera plane.
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    """
    Calculate 2D projection of 3D coordinate
    :param loc: 3D coordinate (carla.Position object)
    :param K: Camera matrix
    :param w2c: World to camera transformation matrix
    :return: 2D projection of 3D coordinate
    """
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def get_actor_bounding_boxes(world, actors):
    """Get the bounding boxes of the actors in the world."""
    bounding_boxes = []
    for actor in actors:
        bounding_box = actor.bounding_box
        bounding_box.location = actor.get_transform().transform(bounding_box.location)
        bounding_boxes.append(bounding_box)
    return bounding_boxes

def project_bounding_boxes(camera, bounding_boxes):
    """Project the bounding boxes into the camera view."""
    camera_transform = camera.get_transform()
    camera_location = camera_transform.location
    camera_rotation = camera_transform.rotation

    projected_boxes = []
    for bounding_box in bounding_boxes:
        # Transform the bounding box to the camera's coordinate system
        bounding_box.location = bounding_box.location - camera_location
        bounding_box.location = bounding_box.location.rotate(camera_rotation)

        # Project the bounding box into the camera view
        projected_box = camera.get_world_to_screen(bounding_box.location)
        projected_boxes.append(projected_box)
    return projected_boxes

def is_bounding_box_in_view(camera, bounding_box):
    """Check if the bounding box is within the camera's field of view."""
    camera_fov = camera.attributes['fov']
    camera_aspect_ratio = camera.attributes['aspect_ratio']

    # Get the camera's field of view in radians
    camera_fov_rad = math.radians(camera_fov)
    camera_aspect_ratio = float(camera_aspect_ratio)

    # Get the bounding box's coordinates in the camera's coordinate system
    bounding_box_location = bounding_box.location

    # Check if the bounding box is within the camera's field of view
    if abs(bounding_box_location.x) <= camera_fov_rad / 2 and abs(bounding_box_location.y) <= camera_fov_rad / 2 * camera_aspect_ratio:
        return True
    return False


def get_visible_actors(camera, world, actors):
    """Get the actors that are visible in the camera view."""
    bounding_boxes = get_actor_bounding_boxes(world, actors)
    projected_boxes = project_bounding_boxes(camera, bounding_boxes)

    visible_actors = []
    for actor, bounding_box in zip(actors, projected_boxes):
        if is_bounding_box_in_view(camera, bounding_box):
            visible_actors.append(actor)
    return visible_actors



def get_neighborhood(image, x, y, window_size):
    half_window = window_size // 2
    neighborhood = image[max(0, y-half_window):y+half_window+1, max(0, x-half_window):x+half_window+1]
    return neighborhood

def ssd(neighborhood1, neighborhood2):
    """ Return the sum of squared differences between two neighborhoods. """
    return np.sum((neighborhood1 - neighborhood2) ** 2)

def find_best_matches(input_image, neighborhood, window_size, tolerance=0.1, sample_fraction=0.1):
    """Find the best matches for a neighborhood in an input image.
        The candidate neighbourhoods are non-overlapping.
    """
    input_height, input_width = input_image.shape[:2]
    half_window = window_size // 2
    best_matches = []
    min_ssd = float('inf')

    # Generate a list of all possible neighborhood coordinates
    all_coordinates = [(x, y) for y in range(half_window, input_height - half_window)
                       for x in range(half_window, input_width - half_window)]

    # Randomly sample a subset of coordinates
    # sample_size = int(len(all_coordinates) * sample_fraction)
    sample_size=5
    sampled_coordinates = random.sample(all_coordinates, sample_size)

    for x, y in sampled_coordinates:
        candidate_neighborhood = get_neighborhood(input_image, x, y, window_size)
        candidate_ssd = ssd(neighborhood, candidate_neighborhood)
        if candidate_ssd < min_ssd:
            min_ssd = candidate_ssd
            best_matches = [(x, y)]
        elif candidate_ssd <= min_ssd * (1 + tolerance):
            best_matches.append((x, y))

    return best_matches

def synthesize_texture(input_image, output_size, window_size):
    """Based on the Efros and Leung algorithm for texture synthesis."""
    input_image = np.array(input_image)
    output_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    half_window = window_size // 2

    # Initialize the output image with a small seed region from the input image
    seed_x = random.randint(0, input_image.shape[1] - window_size)
    seed_y = random.randint(0, input_image.shape[0] - window_size)
    output_image[:window_size, :window_size] = input_image[seed_y:seed_y+window_size, seed_x:seed_x+window_size]

    print("output size: ", output_size, "half window: ", half_window)
    print("num iterations in both total: ", (output_size[1] - half_window) * (output_size[0] - half_window))
    for y in range(half_window, output_size[1] - half_window):
        for x in range(half_window, output_size[0] - half_window):
            neighborhood = get_neighborhood(output_image, x, y, window_size)
            best_matches = find_best_matches(input_image, neighborhood, window_size)
            best_match = random.choice(best_matches)
            output_image[y, x] = input_image[best_match[1], best_match[0]]

    return output_image