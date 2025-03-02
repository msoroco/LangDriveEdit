import os
import imageio
import numpy as np
import math

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