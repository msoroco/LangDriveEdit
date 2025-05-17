# %%
import os
import glob
import json

import numpy as np
from IPython.display import display
from tqdm import tqdm
import cv2
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from datasets import Dataset, Features, Image as DatasetImage, Value, load_dataset, concatenate_datasets
from PIL import Image, ImageChops
from io import BytesIO
import time
# %%
parser = argparse.ArgumentParser(description="Process the parent folder path.")
parser.add_argument("parent_folder", type=str, help="Path to the parent folder")
parser.add_argument("--threshold", type=float, default=0.95, help="Threshold for SSIM similarity")
parser.add_argument("--filter_similar_images", action="store_true", help="Filter similar images")

args = parser.parse_args()

parent_folder = args.parent_folder

print(f"Parent folder: {parent_folder}")
print(f"Filter similar images: {args.filter_similar_images}")
if args.filter_similar_images:
    print(f"Similarity threshold: {args.threshold}")


# %%
def read_first_line_as_json(filepath):
    """Read the first line of a file and parse it as JSON."""
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
        return json.loads(first_line)


def read_metadata_lines(filepath):
    """Read all lines except the first as JSON dicts."""
    with open(filepath, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        return [json.loads(line.strip()) for line in lines if line.strip()]

# def get_caption_for_image(metadata_list, world_frame, view):
#     """Find the metadata dict matching world_frame and view."""
#     for entry in metadata_list:
#         if entry.get("world_frame") == world_frame and entry.get("view") == view:
#             return entry
#     return None

def build_metadata_index(metadata_list, metadata_by_frame_number=True):
    """Build an index for faster lookups of metadata by world_frame and view."""
    if metadata_by_frame_number:
        return {(entry["frame_number"], entry["view"]): entry for entry in metadata_list}
    return {(entry["world_frame"], entry["view"]): entry for entry in metadata_list}

def get_caption_for_image(metadata_index, world_frame, view, frame_number=None):
    result = metadata_index.get((world_frame, view))
    # if result is None:
    #     print(f"Warning: No metadata found for world_frame {world_frame} and view {view}.")
    #     print("getting caption for image", world_frame, view)
    #     # print(metadata_index.keys())
    #     # raise ValueError(f"Metadata not found for world_frame {world_frame} and view {view}.")
    return result


def compute_distance_from_camera(position, camera_position):
    """Compute the distance from the camera to the object."""
    pos = np.array([position["x"], position["y"], position["z"]])
    cam = np.array([camera_position["x"], camera_position["y"], camera_position["z"]])
    return np.linalg.norm(pos - cam)


def filter_captions(caption: dict, view: str):
    """filter the caption to produce outputs similar to ifinder"""
    if caption is None:
        return None
    # extract the position of any sensor with the role_name == view
    for sensor in caption["sensor_metadata"]:
        if sensor.get("role_name") == view:
            ego_pos = sensor.get("position", {})

    # filter the captions to remove the "sensor_metadata" key
    if "sensor_metadata" in caption:
        del caption["sensor_metadata"]
    # filter the vehicle_metadata
    if "vehicle_metadata" in caption:
        desired_keys = ["id", "type", "base_type", "color", "image_bbox_2d"]
        filtered_vehicle_metadata = []
        for vehicle in caption["vehicle_metadata"]:
            filtered = {k: v for k, v in vehicle.items() if k in desired_keys}
            filtered["distance_from_ego_in_meters"] = compute_distance_from_camera(vehicle["position"], ego_pos)
            # get the bounding box of the vehicle
            if "image_bbox_2d" in vehicle:
                bbox = vehicle["image_bbox_2d"]
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = float(bbox['x0']), float(bbox['y0']), float(bbox['x1']), float(bbox['y1'])
                    area = (x2 - x1) * (y2 - y1)
                    if area < 1000: # 0.0025 * image_area:
                        # delete the vehicle if the bounding box is not valid
                        continue # skip the vehicle if the bounding box is not valid
            filtered_vehicle_metadata.append(filtered)
        caption["vehicle_metadata"] = filtered_vehicle_metadata
    else:
        caption["vehicle_metadata"] = []
        
    if "walker_metadata" in caption:
        desired_keys = ["id", "type", "color", "image_bbox_2d", "gender", "age", "clothing", "use_wheelchair"]
        filtered_walker_metadata = []
        for walker in caption["walker_metadata"]:
            filtered = {k: v for k, v in walker.items() if k in desired_keys}
            filtered["distance_from_ego_in_meters"] = compute_distance_from_camera(walker["position"], ego_pos)
            # get the bounding box of the walker
            if "image_bbox_2d" in walker:
                bbox = walker["image_bbox_2d"]
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = float(bbox['x0']), float(bbox['y0']), float(bbox['x1']), float(bbox['y1'])
                    area = (x2 - x1) * (y2 - y1)
                    if area < 960:
                        continue
            if filtered["distance_from_ego_in_meters"] > 70:
                continue
            filtered_walker_metadata.append(filtered)
        caption["walker_metadata"] = filtered_walker_metadata
    else:
        caption["walker_metadata"] = []

    if "traffic_metadata" in caption:
        desired_keys = ["id", "type", "base_type", "light_colour", "image_bbox_2d"]
        filtered_traffic_light_metadata = []
        for traffic_light in caption["traffic_metadata"]:
            filtered = {k: v for k, v in traffic_light.items() if k in desired_keys}
            filtered["distance_from_ego_in_meters"] = compute_distance_from_camera(traffic_light["position"], ego_pos)
            # get the bounding box of the traffic light
            if "image_bbox_2d" in traffic_light:
                bbox = traffic_light["image_bbox_2d"]
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = float(bbox['x0']), float(bbox['y0']), float(bbox['x1']), float(bbox['y1'])
                    area = (x2 - x1) * (y2 - y1)
                    if area < 960: # 0.0025 * image_area:
                        continue # skip the traffic light if the bounding box is not valid
            filtered_traffic_light_metadata.append(filtered)
        caption["traffic_metadata"] = filtered_traffic_light_metadata
    else:
        caption["traffic_metadata"] = []

    return caption


road_textures = {
    "black_asphalt.jpeg": "Smooth, dark black asphalt with a uniform surface and subtle speckled pattern",
    "dark_asphalt.jpg": "Dark charcoal asphalt with a fine texture and sublte surface variations",
    "finer_rock.jpg": "Grey asphalt surface with densely packed multicolored stone featuring possible light flecks",
    "grey_asphalt.jpeg": "Medium gray asphalt with a slightly rough surface and visible small aggregate pieces",
    "grey_cement.jpeg": "Flat, smooth rock surface with a consistent light gray color smooth aggregate pieces",
    "grey_cracked.jpeg": "fine-grained grey pavement with visible cracks and signs of aging across the surface",
    "grey_rocks.jpg": "Coarse gravel surface composed of small-sized angular light grey rocks",
    "light_dirt.jpeg": "Pale brown road with a possibly powdery appearance",
    "sharp_cobble.jpeg": "Grey cobblestone pattern with irregular slate-gray stones with angular edges tightly fitted together",
}



def add_metadata_to_caption(caption: dict, global_metadata: dict, edit: str = None):
    """Add global metadata to the caption."""
    # weather metadata
    for key, value in global_metadata.items():
        if 'weather' in key:
            caption[key] = value
        # road textures
        if edit == 'road_texture':
            if 'road_texture' == key:
                texture_file = os.path.basename(value)
                caption["road_texture_file"] = texture_file
                caption["road_texture"] = road_textures.get(texture_file)
                assert caption["road_texture"] is not None, f"Unknown road texture: {texture_key}"
        if edit == 'traffic_light_state':
            # failsafe in case individual traffic light metadata is not present
            if "traffic_light_mapping" == key:
                caption["traffic_light_mapping"] = global_metadata["traffic_light_mapping"]
    return caption


def load_image_with_retry(image_path, retries=3, delay=1):
    """Attempt to load an image with retries in case of failure."""
    for attempt in range(retries):
        try:
            return Image.open(image_path)
        except OSError as e:
            print(f"Error loading image {image_path}. Attempt {attempt + 1} of {retries}. Error: {e}")
            time.sleep(delay)  # Wait before retrying
    print(f"Failed to load image {image_path} after {retries} attempts. Skipping.")
    return None


def read_image_with_retry(image_path, retries=3, delay=1):
    """Attempt to read an image into grayscale array with retries in case of failure."""
    for attempt in range(retries):
        try:
            return np.array(Image.open(image_path).convert("L"))  # Convert to grayscale
        except OSError as e:
            print(f"Error reading {image_path}. Attempt {attempt + 1} of {retries}.")
            time.sleep(delay)  # Wait before retrying
    print(f"Failed to read {image_path} after {retries} attempts. Skipping.")
    return None

def are_images_similar(img1, img2, threshold=0.95):
    """Check if two images are too similar based on SSIM."""
    # img1 = read_image_with_retry(img1_path)
    # img2 = read_image_with_retry(img2_path)
    if img1 is None or img2 is None:
        return False  # Treat as not similar if either image cannot be read
    similarity, _ = ssim(img1, img2, full=True)
    return similarity >= threshold


def process_run_dir(run_dir, filter_similar_images, similarity_threshold, metadata_by_frame_number=True):
    """Process a single run_x folder."""
    data = []
    images_added_count = 0
    total_images_count = 0

    try:
        # Read metadata
        edited_metadata_file = os.path.join(run_dir, "1", "frame_metadata.json")
        edited_global_metadata = read_first_line_as_json(edited_metadata_file)
        editing_type = edited_global_metadata["edit"]

        if editing_type != "building_texture":

            fps = float(edited_global_metadata["fps"])

            source_metadata_file = os.path.join(run_dir, "0", "frame_metadata.json")
            source_global_metadata = read_first_line_as_json(source_metadata_file)

            # per-frame metadata
            source_metadata_lines = read_metadata_lines(source_metadata_file)
            edited_metadata_lines = read_metadata_lines(edited_metadata_file)

            source_metadata_index = build_metadata_index(source_metadata_lines, metadata_by_frame_number)
            edited_metadata_index = build_metadata_index(edited_metadata_lines, metadata_by_frame_number)

            temp_source_metadata_index = None
            temp_edited_metadata_index = None
    except:
        return data, total_images_count, images_added_count

    # For each rgb subfolder like "front", "front_left", etc.
    rgb_view_subfolders = sorted(glob.glob(os.path.join(run_dir, "0", "sensor.camera.rgb", "*")))

    # Initialize indices to None
    filtered_indices = [] if filter_similar_images else None

    for rgb_subfolder0 in rgb_view_subfolders:
        view = os.path.basename(rgb_subfolder0)
        rgb_subfolder1 = os.path.join(run_dir, "1", "sensor.camera.rgb", view)
        semantic_subfolder0 = os.path.join(run_dir, "0", "sensor.camera.semantic_segmentation", view)
        semantic_subfolder1 = os.path.join(run_dir, "1", "sensor.camera.semantic_segmentation", view)

        # Collect all images in 0
        source_rgb_imgs = sorted(glob.glob(os.path.join(rgb_subfolder0, "*")))
        source_semantic_imgs = sorted(glob.glob(os.path.join(semantic_subfolder0, "*")))
        # Collect all images in 1
        edited_rgb_imgs = sorted(glob.glob(os.path.join(rgb_subfolder1, "*")))
        edited_semantic_imgs = sorted(glob.glob(os.path.join(semantic_subfolder1, "*")))

        # Ensure the folders have the same number of images
        assert len(source_rgb_imgs) == len(edited_rgb_imgs), f"source: {len(source_rgb_imgs)}, edited: {len(edited_rgb_imgs)}, folder: {rgb_subfolder1}"
        assert len(source_semantic_imgs) == len(edited_semantic_imgs), f"source: {len(source_semantic_imgs)}, edited: {len(edited_semantic_imgs)}, folder: {semantic_subfolder1}"
        assert len(source_rgb_imgs) == len(source_semantic_imgs), f"source: {len(source_rgb_imgs)}, semantic: {len(source_semantic_imgs)}, folder: {semantic_subfolder0}"

        total_images_count += len(source_rgb_imgs)

        image_cache = {}
        def get_cached_image(image_path):
            """Load an image and cache it to avoid repeated reads."""
            if image_path not in image_cache:
                image_cache[image_path] = read_image_with_retry(image_path)
            return image_cache[image_path]

        if filter_similar_images:
            # print("filtering similar images")
            # start_time = time.time()
            if not filtered_indices:  # empty list or none
                # Compute filtered indices in source_rgb_imgs for the first camera view (front)
                for i, img_path in enumerate(source_rgb_imgs):
                    img = get_cached_image(img_path)
                    if img is not None:  # Check if the image is valid
                        filtered_indices.append(i)  # Add the first valid image index
                        break
                # Iterate through the rest of the images
                for i in range(filtered_indices[0] + 1, len(source_rgb_imgs)):
                    img1 = get_cached_image(source_rgb_imgs[i])
                    img2 = get_cached_image(source_rgb_imgs[filtered_indices[-1]])
                    if img1 is None or img2 is None:
                        if img2 is None:
                            print(f"!!!!!!!!!!Failed to load image: {source_rgb_imgs[filtered_indices[-1]]}")
                        continue  # Skip comparison if either image failed to load
                    if not are_images_similar(img1, img2, threshold=similarity_threshold):
                        filtered_indices.append(i)

            # Adjust other lists to match the filtered source_rgb_imgs
            # Use the same indices for all views (front-left, rear, etc.)
            filtered_source_rgb_imgs = [source_rgb_imgs[i] for i in filtered_indices]
            filtered_source_semantic_imgs = [source_semantic_imgs[i] for i in filtered_indices]
            filtered_edited_rgb_imgs = [edited_rgb_imgs[i] for i in filtered_indices]
            filtered_edited_semantic_imgs = [edited_semantic_imgs[i] for i in filtered_indices]
            # print(f"Filtered {len(source_rgb_imgs) - len(filtered_source_rgb_imgs)} images in {time.time() - start_time:.2f} seconds.")
        else:
            filtered_source_rgb_imgs = source_rgb_imgs
            filtered_source_semantic_imgs = source_semantic_imgs
            filtered_edited_rgb_imgs = edited_rgb_imgs
            filtered_edited_semantic_imgs = edited_semantic_imgs


        if editing_type == "building_texture" and ("testing" in run_dir or "validation" in run_dir):
            print("WORKAROUND FOR USING OLD BUILDING METADATA")
            for s, e, s_sem, e_sem in zip(filtered_source_rgb_imgs, filtered_edited_rgb_imgs, filtered_source_semantic_imgs, filtered_edited_semantic_imgs):
                images_added_count += 1
                data.append({
                    "source_image": s,
                    "edited_image": e,
                    "source_semantic": s_sem,
                    "edited_semantic": e_sem,
                    "source_caption": {'metadata': None},
                    "edited_caption": {'metadata': None},
                    "source_caption_unfiltered": {'metadata': None},
                    "edited_caption_unfiltered": {'metadata': None},
                    "edit": editing_type,
                    "source_overlay": "",
                    "edited_overlay": "",
                })

        else:
            # Pair up
            # print(f"Pairing images in {view} view")
            for i, s, e, s_sem, e_sem in zip(filtered_indices, filtered_source_rgb_imgs, filtered_edited_rgb_imgs, filtered_source_semantic_imgs, filtered_edited_semantic_imgs):
                
                # print(f"Processing image {i} in {view} view")
                # start_time = time.time()
                s_world_frame_number = int(i * (20 / fps))
                e_world_frame_number = int(i * (20 / fps))
                # Extract world_frame from filename (assumes basename is like 'sensor.camera.rgb_front_000637.png')
                s_world_frame = int(os.path.splitext(os.path.basename(s))[0].split('_')[-1])
                e_world_frame = int(os.path.splitext(os.path.basename(e))[0].split('_')[-1])

                # create a dictionary for faster lookups
                # time_building = time.time()

                # print(f"Building metadata index took {(time.time() - time_building) * 1000:.2f} ms.")

                # Get captions (metadata dicts) for source and edited images
                s_frame_number = s_world_frame if not metadata_by_frame_number else s_world_frame_number
                e_frame_number = e_world_frame if not metadata_by_frame_number else e_world_frame_number

                # time_getting = time.time()
                s_cap = get_caption_for_image(source_metadata_index, s_frame_number, view)
                if s_cap is None:
                    temp_s_world_frame = s_world_frame if metadata_by_frame_number else s_world_frame_number
                    if temp_source_metadata_index is None:  # cache for future directory views which may have the issue
                        temp_source_metadata_index = build_metadata_index(source_metadata_lines, not metadata_by_frame_number)
                    s_cap = get_caption_for_image(temp_source_metadata_index, temp_s_world_frame, view)
                    if s_cap is None:
                        print(f"Missing metadata for {s}. Skipping.")
                        continue
                e_cap = get_caption_for_image(edited_metadata_index, e_frame_number, view)
                if e_cap is None:
                    temp_e_frame_number = e_world_frame if metadata_by_frame_number else e_world_frame_number
                    if temp_edited_metadata_index is None: # cache for future directory views which may have the issue
                        temp_edited_metadata_index = build_metadata_index(edited_metadata_lines, not metadata_by_frame_number)
                    e_cap = get_caption_for_image(temp_edited_metadata_index, temp_e_frame_number, view)
                    if e_cap is None:
                        print(f"Missing metadata for {e}. Skipping.")
                        continue

                if s_cap is None or e_cap is None:
                    print(s_frame_number, e_frame_number, i)
                    print("s_cap", s_cap)
                    print("e_cap", e_cap)

                    print(f"Missing metadata for {s} or {e}. Skipping.")
                    print(s_world_frame, e_world_frame, i)
                    print(view, editing_type)
                    print("run_dir", run_dir)
                    if s_cap is None:
                        print("s_cap", s_cap)
                        raise ValueError(f"Missing metadata for {s}. Skipping.")
                    if e_cap is None:
                        print("e_cap", e_cap)
                        raise ValueError(f"Missing metadata for {e}. Skipping.")

                # print(f"Getting captions took {(time.time() - time_getting) * 1000:.2f} ms.")

                # print("source caption", s_cap)
                # filter the captions to produce ifinder-like outputs
                # time_filtering = time.time()
                s_cap_simple = filter_captions(s_cap, view)
                e_cap_simple = filter_captions(e_cap, view)
                # print(f"Filtering captions took {(time.time() - time_filtering) * 1000:.2f} ms.")

                # print("simple source caption", s_cap_simple)

                # add the global metadata
                # time_adding = time.time()
                s_cap_simple = add_metadata_to_caption(s_cap_simple, source_global_metadata, editing_type)
                e_cap_simple = add_metadata_to_caption(e_cap_simple, edited_global_metadata, editing_type)
                # print(f"Adding global metadata took {(time.time() - time_adding) * 1000:.2f} ms.")


                overlay_source_path = os.path.join(
                    run_dir, "0", "sensor.camera.instance_segmentation", view, "overlays",
                    f"{s_world_frame:06d}_overlay.png"
                )
                edited_overlay_path = os.path.join(
                    run_dir, "1", "sensor.camera.instance_segmentation", view, "overlays",
                    f"{e_world_frame:06d}_overlay.png"
                )
                # print(f"Processing image {i} in {view} view took {(time.time() - start_time) * 1000:.2f} ms.")

                images_added_count += 1
                data.append({
                    "source_image": s,
                    "edited_image": e,
                    "source_semantic": s_sem,
                    "edited_semantic": e_sem,
                    "source_caption": s_cap_simple,
                    "edited_caption": e_cap_simple,
                    "source_caption_unfiltered": s_cap,
                    "edited_caption_unfiltered": e_cap,
                    "edit": editing_type,
                    "source_overlay": overlay_source_path,
                    "edited_overlay": edited_overlay_path,
                })

    return data, total_images_count, images_added_count


def pair_rgb_semantic_images(root_dir, filter_similar_images=True, similarity_threshold=0.97):
    data = []
    total_images_count = 0
    images_added_count = 0

    # Get all run_x folders
    run_dirs = sorted(glob.glob(os.path.join(root_dir, "run_*")))

    # Use ProcessPoolExecutor to parallelize the processing of run_x folders
    with ProcessPoolExecutor(max_workers=8) as executor:
        process_func = partial(process_run_dir, filter_similar_images=filter_similar_images, similarity_threshold=similarity_threshold, metadata_by_frame_number=True)
        results = list(tqdm(executor.map(process_func, run_dirs), total=len(run_dirs), desc="Processing run_x folders"))

    # Combine results from all processes
    for result in results:
        run_data, run_total_images_count, run_images_added_count = result
        data.extend(run_data)
        total_images_count += run_total_images_count
        images_added_count += run_images_added_count

    print(f"Total images tuples considered: {total_images_count} - Filtered out: {total_images_count - images_added_count} = Added images: {images_added_count}")
    return Dataset.from_list(data)

# %%

my_dataset = pair_rgb_semantic_images(parent_folder, filter_similar_images=args.filter_similar_images, similarity_threshold=args.threshold)
# raise ValueError("remember to make max_workers=4 as original^^^")

# %%
simple_edit_prompts = {
    'time_of_day': 'Change the time of day.',
    'weather': 'Change the weather.',
    'weather_and_time_of_day': 'Change both the weather and time of day.',
    'building_texture': 'Change the texture of the buildings.',
    'vehicle_color': 'Change the color of the vehicles.',
    'vehicle_replacement': 'Replace the vehicles with vehicles of a different type or model.',
    'vehicle_deletion': 'Remove the vehicles.',
    'walker_color': 'Change the clothing of the pedestrians.',
    'walker_replacement': 'Replace the pedestrians with different pedestrians.',
    'walker_deletion': 'Remove the pedestrians.',
    'road_texture': 'Change the texture of the road.',
    'traffic_light_state': 'Change the state of the traffic lights.'
}

simple_reverse_edit_prompts = {
    'time_of_day': 'Change the time of day.',
    'weather': 'Change the weather.',
    'weather_and_time_of_day': 'Change both the weather and time of day.',
    'building_texture': 'Change the texture of the buildings.',
    'vehicle_color': 'Change the color of the vehicles.',
    'vehicle_replacement': 'Replace the vehicles with vehicles of a different type or model.',
    'vehicle_deletion': 'Add vehicles to the scene.',
    'walker_color': 'Change the clothing of the pedestrians.',
    'walker_replacement': 'Replace the pedestrians with different pedestrians.',
    'walker_deletion': 'Add pedestrians to the scene.',
    'road_texture': 'Change the texture of the road.',
    'traffic_light_state': 'Change the state of the traffic lights.'
}

def add_prompt_column(dataset, simple_edit_prompts):
    dataset = dataset.map(lambda x: {**x, "edit_prompt": simple_edit_prompts[x["edit"]]})
    return dataset

def add_reverse_prompt_column(dataset, simple_reverse_edit_prompts):
    dataset = dataset.map(lambda x: {**x, "reverse_edit_prompt": simple_reverse_edit_prompts[x["edit"]]})
    return dataset

# %%

# https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
semantic_classes = {
    0: "RoadLines",
    1: "Roads",
    2: "Sidewalks",
    3: "Buildings",
    4: "Walls",
    5: "Fences",
    6: "Poles",
    7: "TrafficLights",
    8: "TrafficSigns",
    9: "Trees",
    10: "Ground",
    11: "Sky",
    12: "Pedestrians",
    13: "Drivers",
    14: "Vehicles",
    15: "Trucks",
    16: "Busses",
    17: "Train",
    18: "Motorcycles",
    19: "Bicycle", 
    20: "Static", # trash cans, benches, billboards etc.
    21: "Dynamic",  # I think 24 is RoadLines and 21 is tables and chairs
    22: "Terrain"
}

def get_semantic_class_name(class_id):
    return semantic_classes[class_id]

def get_semantic_class_id(class_name):
    return {v: k for k, v in semantic_classes.items()}[class_name]


relevant_semantic_classes =  {
    'time_of_day': None,
    'weather': None,
    'weather_and_time_of_day': None,
    'building_texture': ["Buildings"],
    'vehicle_color': ["Vehicles", "Trucks", 'Busses', "Motorcycles", "Drivers"],
    'vehicle_replacement': ["Vehicles", "Trucks", 'Busses', "Motorcycles", "Drivers"],
    'vehicle_deletion': ["Vehicles", "Trucks", 'Busses', "Motorcycles", "Drivers"],
    'walker_color': ["Pedestrians"],
    'walker_replacement': ["Pedestrians"],
    'walker_deletion': ["Pedestrians"],
    'road_texture': ["Roads"],
    'traffic_light_state': ["TrafficLights"]
}

def get_relevant_semantic_classes(edit):
    return relevant_semantic_classes[edit]



# %%
def compute_class_pervasiveness(r_image, class_names):
    # compute the percentage of pixels in the image that contain the class_id
    ps = {}
    for class_name in class_names:
        class_id = get_semantic_class_id(class_name)
        p = np.mean(np.array(r_image) == class_id)
        ps[class_name] = p
    return ps


# %%


def compare_histograms(image1, image2, mask1, mask2):
    """
    Returns correlation in the range of -1 to 1.
    """
    # Convert boolean masks to binary masks (uint8)
    mask1 = (mask1.astype(np.uint8)) * 255
    mask2 = (mask2.astype(np.uint8)) * 255

    hist1 = cv2.calcHist([np.array(image1)], [0], mask1, [256], [0, 256])
    hist2 = cv2.calcHist([np.array(image2)], [0], mask2, [256], [0, 256])
    # normalize the histograms for independence between the number of pixels
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation

def compute_colour_difference(source_rgb, source_sem, edited_rgb, edited_sem, relevant_class="Buildings", threshold=0):
    """ Returns True if the edited image has a different colour correlation in histograms (above threshold) than the source image """
    # # find all the pixels where both semantic images are buildings
    source_sem_np = source_sem.split()[0]
    source_sem_np = np.array(source_sem_np) 
    edited_sem_np = edited_sem.split()[0]
    edited_sem_np = np.array(edited_sem_np)
    building_mask_source = source_sem_np == get_semantic_class_id(relevant_class)
    building_mask_edit = edited_sem_np == get_semantic_class_id(relevant_class)

    correlation = compare_histograms(source_rgb, edited_rgb, building_mask_source, building_mask_edit)


    if correlation > threshold:
        # not enough colour change to justify the building textures being editted
        return False
    return True


# %%
def filter_image_pairs(dataset, threshold=0.1):
    """ For each image pair, filter out images that don't appear to have been editted (ie too similar) """
    data_to_keep = []

    progress_bar = tqdm(dataset, desc="Filtering for edited pairs (kept: 0)")
    for i, data in enumerate(progress_bar):
        edit = data["edit"]
        relevant_sem_classes = get_relevant_semantic_classes(edit)
        
        source_caption = data['source_caption']
        # consider the size of the bounding boxes in the source image.
        image = load_image_with_retry(data["source_image"])
        if image is None:
            continue  # Skip this data point if the image cannot be loaded
        image_area = image.size[0] * image.size[1]

        if relevant_sem_classes is None:
            # print(f"Edit type '{edit}' is global. Keeping the image pair.")
            # This is a valid edit because it is a global edit
            data_to_keep.append(data)
            progress_bar.set_description(f"Filtering for edited pairs (kept: {len(data_to_keep)})")
            continue

        if edit in ['building_texture', 'road_texture']:
            # Consider the semantic source image and compute how much of the class of interest is present
            source_semantic_img = load_image_with_retry(data["source_semantic"])
            if source_semantic_img is None:
                continue  # Skip this data point if the image cannot be loaded
            # The semantic information is stored in the R channel
            source_semantic_img = source_semantic_img.split()[0]
            source_semantic_img = np.array(source_semantic_img)
            source_class_pervasiveness = compute_class_pervasiveness(source_semantic_img, relevant_sem_classes)

            # If the sum of the pervasiveness' is less than the threshold, then the image is not edited
            if sum(source_class_pervasiveness.values()) < threshold:
                continue

            # Before keeping building edits, check that the edited image has a different building texture
            if edit == "building_texture":
                source_image = load_image_with_retry(data["source_image"])
                edited_image = load_image_with_retry(data["edited_image"])
                source_semantic = load_image_with_retry(data["source_semantic"])
                edited_semantic = load_image_with_retry(data["edited_semantic"])

                if None in (source_image, edited_image, source_semantic, edited_semantic):
                    continue  # Skip if any image fails to load

                different = compute_colour_difference(
                    source_image, source_semantic, edited_image, edited_semantic,
                    relevant_class="Buildings", threshold=0.5
                )
                if not different:
                    continue

            data_to_keep.append(data)
            progress_bar.set_description(f"Filtering for edited pairs (kept: {len(data_to_keep)})")
            continue

        elif edit in ['vehicle_color', 'vehicle_replacement', 'vehicle_deletion']:
            minimum_bbox_area_ratio = 0.012
            vehicle_metadata = source_caption.get("vehicle_metadata", [])
            if vehicle_metadata is None:
                continue   # no vehicles in the image, do not keep the image
            for vehicle in vehicle_metadata:
                if "image_bbox_2d" in vehicle:
                    bbox = vehicle["image_bbox_2d"]
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = float(bbox['x0']), float(bbox['y0']), float(bbox['x1']), float(bbox['y1'])
                        area = (x2 - x1) * (y2 - y1)
                        # print("vehicle area", area, "image area", image_area, "ratio", area / image_area, "min ratio", minimum_bbox_area_ratio)
                        if area / image_area > minimum_bbox_area_ratio:
                            # this is a valid edit because the bounding box is large enough
                            data_to_keep.append(data)
                            progress_bar.set_description(f"Filtering for edited pairs (kept: {len(data_to_keep)})")
                            break
            continue # no vehicles in the image, do not keep the image

        elif edit in ['walker_color', 'walker_replacement', 'walker_deletion']:
            minimum_bbox_area_ratio = 0.007
            walker_metadata = source_caption.get("walker_metadata", [])
            if walker_metadata is None:
                continue  # no walkers in the image, do not keep the image
            for walker in walker_metadata:
                if "image_bbox_2d" in walker:
                    bbox = walker["image_bbox_2d"]
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = float(bbox['x0']), float(bbox['y0']), float(bbox['x1']), float(bbox['y1'])
                        area = (x2 - x1) * (y2 - y1)
                        # print("walker area", area, "image area", image_area, "ratio", area / image_area, "min ratio", minimum_bbox_area_ratio)
                        if area / image_area > minimum_bbox_area_ratio:
                            # this is a valid edit because the bounding box is large enough
                            data_to_keep.append(data)
                            progress_bar.set_description(f"Filtering for edited pairs (kept: {len(data_to_keep)})")
                            break
            continue # no walkers in the image, do not keep the image
        elif edit in ["traffic_light_state"]:
            minimum_bbox_area_ratio = 0.003
            # Check if the traffic light state is present in the metadata
            traffic_light_metadata = source_caption.get("traffic_metadata", [])
            if traffic_light_metadata is None:
                continue # no traffic lights in the image, do not keep the image
            for traffic_light in traffic_light_metadata:
                if "image_bbox_2d" in traffic_light:
                    bbox = traffic_light["image_bbox_2d"]
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = float(bbox['x0']), float(bbox['y0']), float(bbox['x1']), float(bbox['y1'])
                        area = (x2 - x1) * (y2 - y1)
                        # print("traffic light area", area, "image area", image_area, "ratio", area / image_area, "min ratio", minimum_bbox_area_ratio)
                        if area / image_area > minimum_bbox_area_ratio:
                            # this is a valid edit because the bounding box is large enough
                            data_to_keep.append(data)
                            progress_bar.set_description(f"Filtering for edited pairs (kept: {len(data_to_keep)})")
                            break
            continue # no traffic lights in the image, do not keep the image
        else:
            print(f"Edit type '{edit}' not recognized. Skipping this data point.")
            continue  # Skip if the edit type is not recognized

        raise ValueError(f"Edit type '{edit}' not recognized. Skipping this data point.")
        
        
    return Dataset.from_list(data_to_keep)

def change_to_relative_paths(dataset, parent_folder):
    for i, data in enumerate(dataset):
        source_image = data["source_image"]
        edited_image = data["edited_image"]
        source_semantic = data["source_semantic"]
        edited_semantic = data["edited_semantic"]

        source_image = os.path.relpath(source_image, parent_folder)
        edited_image = os.path.relpath(edited_image, parent_folder)
        source_semantic = os.path.relpath(source_semantic, parent_folder)
        edited_semantic = os.path.relpath(edited_semantic, parent_folder)

        dataset[i]["source_image"] = source_image
        dataset[i]["edited_image"] = edited_image
        dataset[i]["source_semantic"] = source_semantic
        dataset[i]["edited_semantic"] = edited_semantic

    return dataset

# %%

def create_blank_image(width, height):
    """Create a blank (black) image."""
    blank_image = Image.fromarray(np.ones((height, width), dtype=np.uint8) * 255)  # White image
    buffer = BytesIO()
    blank_image.save(buffer, format="PNG")
    return {'bytes': buffer.getvalue(), 'path': None}  # Return as bytes with metadata


def add_image_mask(dataset):
    def process_row(data, add_reverse_mask=True):
        edit = data["edit"]
        relevant_sem_classes = get_relevant_semantic_classes(edit)

        # Consider the semantic source image and compute how much of the class of interest is present
        source_semantic_img = Image.open(data["source_semantic"])
        # The semantic information is stored in the R channel
        source_semantic_img = source_semantic_img.split()[0]
        source_semantic_img = np.array(source_semantic_img)

        target_semantic_img = Image.open(data["edited_semantic"])
        target_semantic_img = target_semantic_img.split()[0]
        target_semantic_img = np.array(target_semantic_img)

        if relevant_sem_classes is None:
            # This is a global edit
            height, width = source_semantic_img.shape
            blank_mask = create_blank_image(width, height) 
            data["mask_image"] = blank_mask
            if add_reverse_mask:
                data["reverse_mask_image"] = blank_mask
            return data

        semantic_ids = [get_semantic_class_id(class_name) for class_name in relevant_sem_classes]
        # Create a binary mask which is 1 where the class is present and 0 otherwise
        image_mask = np.isin(source_semantic_img, semantic_ids).astype(np.uint8) * 255  # Convert to 0 and 255
        mask_image = Image.fromarray(image_mask)  # Convert to PIL Image
        # Convert the PIL Image to bytes
        buffer = BytesIO()
        mask_image.save(buffer, format="PNG")
        data["mask_image"] = {'bytes': buffer.getvalue(), 'path': None}  # Store as bytes with metadata
        # Add the reverse mask image
        if add_reverse_mask:
            reverse_mask_image = np.isin(target_semantic_img, semantic_ids).astype(np.uint8) * 255
            reverse_mask_image = Image.fromarray(reverse_mask_image)
            buffer = BytesIO()
            reverse_mask_image.save(buffer, format="PNG")
            data["reverse_mask_image"] = {'bytes': buffer.getvalue(), 'path': None}
        return data

    # Use the map function to apply the process_row function to each row in the dataset
    dataset = dataset.map(process_row)
    return dataset
    

# %%

# Define this at the module level (outside any function)
def process_chunk_for_filter(chunk, threshold=0.1):
    return filter_image_pairs(chunk, threshold=threshold)

def filter_image_pairs_parallel(dataset, threshold=0.1, num_workers=4):
    """Parallelize the image pair filtering process across multiple workers."""
    # Split dataset into chunks
    chunk_size = max(1, len(dataset) // num_workers)
    chunks = [dataset.select(range(i, min(i + chunk_size, len(dataset)))) 
              for i in range(0, len(dataset), chunk_size)]
    
    print(f"Split dataset into {len(chunks)} chunks of approximately {chunk_size} items each")
    
    # Process each chunk in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress
        results = list(tqdm(
            executor.map(process_chunk, chunks),
            total=len(chunks),
            desc="Processing chunks in parallel"
        ))
    
    if not results:
        return Dataset.from_list([])
    # Concatenate all filtered dataset chunks
    combined_dataset = concatenate_datasets(results)
    print(f"Filtered from {len(dataset)} to {len(combined_dataset)} items")
    
    return combined_dataset


# filtered_dataset = filter_image_pairs_parallel(my_dataset, threshold=0.1, num_workers=4)

# # filter out images that don't appear to have been editted
filtered_dataset = filter_image_pairs(my_dataset)


# add the binary mask for the edit region
filtered_dataset = add_image_mask(filtered_dataset)
# add the prompt column
filtered_dataset = add_prompt_column(filtered_dataset, simple_edit_prompts)
filtered_dataset = add_reverse_prompt_column(filtered_dataset, simple_reverse_edit_prompts)
# filtered_dataset = change_to_relative_paths(filtered_dataset, parent_folder)
# filtered_dataset = change_base_paths(filtered_dataset, simple_reverse_edit_prompts)



# %%
print(filtered_dataset)

# %%
# # save the dataset:
parent_folder_basename = os.path.basename(parent_folder)
# filtered_dataset.save_to_disk(f"./{parent_folder_basename}_paired_dataset")
output_path = f"./{parent_folder_basename}_paired_dataset_new_metadata_uncasted_fixed8_buildings.parquet"
# Save the filtered dataset as Parquet
filtered_dataset.to_parquet(output_path)

# Define the features with the 'image' type for the image columns and 'string' type for captions
features = Features({
    'source_image': DatasetImage(),  # Convert to actual images
    'edited_image': DatasetImage(),
    'source_semantic': DatasetImage(),
    'edited_semantic': DatasetImage(),
    'source_caption': Value('string'),
    'edited_caption': Value('string'),
    'source_caption_unfiltered': Value('string'),
    'edited_caption_unfiltered': Value('string'),
    'edit': Value('string'),
    'source_overlay': Value('string'),
    'edited_overlay': Value('string'),
    'edit_prompt': Value('string'),
    'reverse_edit_prompt': Value('string'),
    'mask_image': DatasetImage(),
    'reverse_mask_image': DatasetImage(),
})

# Convert complex objects to strings before casting
def convert_complex_to_json(example):
    # Convert complex dictionary objects to JSON strings
    if 'source_caption' in example and example['source_caption'] is not None:
        example['source_caption'] = json.dumps(example['source_caption'])
    if 'edited_caption' in example and example['edited_caption'] is not None:
        example['edited_caption'] = json.dumps(example['edited_caption'])
    if 'source_caption_unfiltered' in example and example['source_caption_unfiltered'] is not None:
        example['source_caption_unfiltered'] = json.dumps(example['source_caption_unfiltered'])
    if 'edited_caption_unfiltered' in example and example['edited_caption_unfiltered'] is not None:
        example['edited_caption_unfiltered'] = json.dumps(example['edited_caption_unfiltered'])
    return example

# Apply the conversion before casting
filtered_dataset = filtered_dataset.map(convert_complex_to_json)

# Cast the dataset to use the new features
filtered_dataset = filtered_dataset.cast(features)

output_path = f"./{parent_folder_basename}_paired_dataset_new_metadata_fixed_buildings8.parquet"
# Save the filtered dataset as Parquet
filtered_dataset.to_parquet(output_path)

print(f"Dataset saved to {output_path}")
# time.sleep(5)

# %%

# new_base_dir = '/net/acadia8a/data/msoroco/code/projects/carla/ImageEditing'
# # Function to load images and update dataset structure
# def convert_to_images(example):
#     def update_path(image_obj):
#         # print(type(image_obj))
#         if isinstance(image_obj, dict) and 'path' in image_obj:
#             if image_obj['path'] is None:
#                 # If the path is None, return the object as is
#                 return image_obj
#             if image_obj['path'].startswith('/home/mai/msoroco'):
#                 # Extract the original path
#                 original_path = image_obj['path']
#                 # print(f"Original path: {original_path}")
                
#                 # Compute the relative path from the original base directory
#                 relative_path = os.path.relpath(original_path, start='/home/mai/msoroco/carla/ImageEditing')
#                 # print(f"Relative path: {relative_path}")
                
#                 # Compute the absolute path relative to the new base directory
#                 updated_path = os.path.abspath(os.path.join(new_base_dir, relative_path))
#                 # print(f"Updated path: {updated_path}")
                
#                 # Update the path in the DatasetImage object
#                 image_obj['path'] = updated_path
#         return image_obj

#     example['source_image'] = update_path(example['source_image'])
#     example['edited_image'] = update_path(example['edited_image'])
#     example['source_semantic'] = update_path(example['source_semantic'])
#     example['edited_semantic'] = update_path(example['edited_semantic'])
#     example['mask_image'] = update_path(example['mask_image'])
#     example['reverse_mask_image'] = update_path(example['reverse_mask_image'])
#     return example

# # Define the new features for the dataset
# features_undo = Features({
#     'source_image': DatasetImage(decode=False),
#     'edited_image': DatasetImage(decode=False),
#     'source_semantic': DatasetImage(decode=False),
#     'edited_semantic': DatasetImage(decode=False),
#     'edit': Value('string'),
#     'edit_prompt': Value('string'),
#     'reverse_edit_prompt': Value('string'),
#     'mask_image': DatasetImage(decode=False),
#     'reverse_mask_image': DatasetImage(decode=False),
# })

# # Define the new features for the dataset
# features_redo = Features({
#     'source_image': DatasetImage(),
#     'edited_image': DatasetImage(),
#     'source_semantic': DatasetImage(),
#     'edited_semantic': DatasetImage(),
#     'edit': Value('string'),
#     'edit_prompt': Value('string'),
#     'reverse_edit_prompt': Value('string'),
#     'mask_image': DatasetImage(),
#     'reverse_mask_image': DatasetImage(),
# })

# # Cast the dataset to the new features
# dataset_rebased = filtered_dataset.cast(features_undo)

# # Remove the softlinks from the dataset images
# dataset_rebased = dataset_rebased.map(convert_to_images)

# # Cast the dataset to the new features
# dataset_rebased = dataset_rebased.cast(features_redo)

# parent_folder_basename = os.path.basename(parent_folder)
# output_path = f"./{parent_folder_basename}_paired_dataset2.parquet"
# # Save the filtered dataset as Parquet
# dataset_rebased.to_parquet(output_path)

# print(f"Dataset saved to {output_path}")