# %%


# %% [markdown]
# [documentation](https://github.com/utiasASRL/pyboreas/blob/master/DATA_REFERENCE.md)

# %%
"""
boreas-YYYY-MM-DD-HH-MM
	applanix
		camera_poses.csv
		gps_post_process.csv
		lidar_poses.csv
		radar_poses.csv
	calib
		camera0_intrinsics.yaml
		P_camera.txt
		T_applanix_lidar.txt
		T_camera_lidar.txt
		T_radar_lidar.txt
	camera
		<timestamp>.png
	lidar
		<timestamp>.bin
	radar
		<timestamp>.png
	route.html
	video.mp4
"""



# %%
# camera image metadata should be in /net/acadia1a/data/datasets/boreas/boreas-2021-11-28-09-18/applanix/<sensor>_poses.csv

# %%
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datasets import Dataset, Features, Image as DatasetImage, Value, load_dataset, concatenate_datasets, load_from_disk
from PIL import Image, ImageChops
from io import BytesIO
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random

# %%
glenShields_snow_path = "/net/acadia1a/data/datasets/boreas/boreas-2021-11-28-09-18/camera"
glenShields_snow_path = "/net/acadia1a/data/datasets/boreas/boreas-2021-11-23-14-27/camera"

# Define the test split size
test_split_size = 0.3  # 30% of the data for the test split
random.seed(42)  # Set a seed for reproducibility

# %%

def get_directory_day_time(directory):  
    simulation_dates = directory.split("boreas-")[1]
    # boreas-2021-11-28-09-18
    # boreas-2021-11-23-14-27
    # boreas-YYYY-MM-DD-HH-MM
    splits = simulation_dates.split("-")
    year = splits[0]
    month = splits[1]
    day = splits[2]
    hours = splits[3]
    minutes = splits[4]

    return {
        "year": year,
        "month": month,
        "day": day,
        "hours": hours,
        "minutes": minutes
    }

# %%
glen_sheilds_path = "/net/acadia1a/data/datasets/boreas/"

# %%


def get_camera_pose(image_name, df):
    
    # Filter the row where GPSTime matches the image_name
    df['GPSTime'] = df['GPSTime'].astype(str)
    row = df.loc[df['GPSTime'] == str(image_name)]

    if row.empty:
        print("csv_path", csv_path)
        raise ValueError(f"No matching GPSTime found for image_name: {image_name}")
    if len(row) > 1:
        # try to see why this is the case. If its something reasonable just take the first image.
        raise ValueError(f"Multiple matching GPSTime found for image_name: {image_name}")

    # Extract the values from the row
    easting = row['easting'].iloc[0]
    northing = row['northing'].iloc[0]
    altitude = row['altitude'].iloc[0]
    roll = row['roll'].iloc[0]
    pitch = row['pitch'].iloc[0]
    heading = row['heading'].iloc[0]

    return {
        "easting": easting,  # x
        "northing": northing,  # y
        "altitude": altitude,  # z
        "roll": roll,
        "pitch": pitch,
        "heading": heading  # yaw
    }

# %%
def calculate_distance(pose1, pose2):
    x1 = pose1['easting']
    y1 = pose1['northing']
    z1 = pose1['altitude']
    x2 = pose2['easting']
    y2 = pose2['northing']
    z2 = pose2['altitude']
    # Euclidean distance
    euclidean_distance = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5

    # check orientation
    roll1 = pose1['roll']
    pitch1 = pose1['pitch']
    heading1 = pose1['heading']
    roll2 = pose2['roll']
    pitch2 = pose2['pitch']
    heading2 = pose2['heading']

    # for now assuming orientations are similar (vs quaternions which can be similar: q = -q)
    roll_diff = abs(roll1 - roll2)
    pitch_diff = abs(pitch1 - pitch2)
    heading_diff = abs(heading1 - heading2)

    angular_distance = roll_diff + pitch_diff + heading_diff
    ## alternatively, use boreas utils.py
    # yawPitchRollToRot(y, p, r)
    # rotToQuaternion(C)

    distance = 1 * euclidean_distance + angular_distance
    return distance


def find_closest_image(pose, later_simulation_path, index=None, radius=45):
    csv_path = os.path.join(later_simulation_path, "applanix", "camera_poses.csv")
    df = pd.read_csv(csv_path)
    df['GPSTime'] = df['GPSTime'].astype(str)
    # preload all poses into a dictionary for faster lookup
    poses = {
        row['GPSTime']: {
            "easting": row['easting'],
            "northing": row['northing'],
            "altitude": row['altitude'],
            "roll": row['roll'],
            "pitch": row['pitch'],
            "heading": row['heading']
        }
        for _, row in df.iterrows()
    }

    second_image_path = os.path.join(later_simulation_path, "camera")
    second_images = os.listdir(second_image_path)
    
    best_distance = float("inf")
    best_image = None
    processed_indices = set()  # Keep track of already processed indices
    
    while len(processed_indices) < len(second_images):
        if index is not None:
            # Determine the range to search within
            start_idx = max(0, index - radius)
            end_idx = min(len(second_images), index + radius + 1)
        else:
            # Search the entire list
            start_idx = 0
            end_idx = len(second_images)

        with tqdm(total=end_idx - start_idx, desc=f"Finding closest image. Dist: {best_distance}", leave=False, disable=True) as pbar:
            for i in range(start_idx, end_idx):
                if i in processed_indices:
                    continue  # Skip already processed indices

                processed_indices.add(i)
                image = second_images[i]
                image_name = image.split(".png")[0]

                if image_name not in poses:
                    continue
                # image_pose = get_camera_pose(image_name, df)
                image_pose = poses[image_name]

                distance = calculate_distance(pose, image_pose)

                if distance < best_distance:
                    best_distance = distance
                    best_image = image

                pbar.update(1)
                pbar.set_description(f"Finding closest image. Dist: {best_distance}")

        # If the best distance is acceptable, stop searching
        if best_distance <= 0.75:
            break

        # If the best distance is not acceptable, expand the radius for the next iteration
        radius += radius


    # double check that the best image is not very far from the original pose
    if best_distance > 1:
        # raise ValueError(f"Best image is too far from the original pose: {best_distance}")
        return None
    # Return the image name with the smallest distance
    best_image_path = os.path.join(second_image_path, f"{best_image}")
    return best_image_path

# %%
path_to_weather_road_season_time_mapping = {
    "2020-11-26-13-58": ("overcast", "dry", "autumn", "noon"),
    "2020-12-01-13-26": ("overcast", "snowy", "winter", "noon"),
    "2020-12-04-14-00": ("overcast", "melting snow", "winter", "afternoon"),
    "2020-12-18-13-44": ("clear and sunny", "dry", "winter", "noon"),
    "2021-01-15-12-17": ("partly cloudy", "melting snow", "winter", "noon"),
    "2021-01-19-15-08": ("cloudy", "dry", "winter", "afternoon"),
    "2021-01-26-10-59": ("snowing", "snowy", "winter", "morning"),
    "2021-01-26-11-22": ("snowing", "snowy", "winter", "morning"),
    "2021-02-02-14-07": ("overcast", "melting snow", "winter", "afternoon"),
    "2021-02-09-12-55": ("sunny and partly cloudy", "dry", "winter", "noon"),
    "2021-03-02-13-38": ("mostly clear", "dry", "winter", "noon"),
    "2021-03-09-14-23": ("clear", "dry", "winter", "afternoon"),
    "2021-03-23-12-43": ("cloudy", "dry", "winter", "noon"),
    "2021-03-30-14-23": ("sunny", "dry", "winter", "afternoon"),
    "2021-04-08-12-44": ("clear", "dry", "spring", "noon"),
    "2021-04-13-14-49": ("sunny and partly cloudy", "dry", "spring", "afternoon"),
    "2021-04-15-18-55": ("overcast", "dry", "spring", "evening"),
    "2021-04-20-14-11": ("cloudy", "dry", "spring", "afternoon"),
    "2021-04-22-15-00": ("cloudy", "dry", "spring", "afternoon"),
    "2021-04-29-15-55": ("rainy", "wet", "spring", "afternoon"),
    "2021-05-06-13-19": ("sunny and partly cloudy", "dry", "spring", "noon"),
    "2021-05-13-16-11": ("sunny", "dry", "spring", "afternoon"),
    "2021-06-03-16-00": ("sunny", "dry", "spring", "afternoon"),
    "2021-06-17-17-52": ("sunny", "dry", "summer", "afternoon"),
    "2021-06-29-18-53": ("rainy", "wet", "summer", "evening"),
    "2021-06-29-20-43": ("cloudy", "wet", "summer", "evening"),
    "2021-07-20-17-33": ("rainy", "wet", "summer", "afternoon"),
    "2021-07-27-14-43": ("cloudy", "dry", "summer", "afternoon"),
    "2021-08-05-13-34": ("sunny", "dry", "summer", "noon"),
    "2021-09-02-11-42": ("sunny", "dry", "summer", "morning"),
    "2021-09-07-09-35": ("sunny", "dry", "summer", "morning"),
    "2021-09-08-21-00": ("clear", "dry", "autumn", "night"),
    "2021-09-09-15-28": ("sunny and partly cloudy", "dry", "autumn", "afternoon"),
    "2021-09-14-20-00": ("clear", "dry", "autumn", "night"),
    "2021-10-05-15-35": ("overcast", "dry", "autumn", "afternoon"),
    "2021-10-15-12-35": ("cloudy", "dry", "autumn", "noon"),
    "2021-10-22-11-36": ("overcast", "dry", "autumn", "morning"),
    "2021-10-26-12-35": ("rainy", "wet", "autumn", "noon"),
    "2021-11-02-11-16": ("sunny and partly cloudy", "dry", "autumn", "morning"),
    "2021-11-06-18-55": ("clear", "dry", "autumn", "night"),
    "2021-11-14-09-47": ("cloudy", "dry", "autumn", "morning"),
    "2021-11-16-14-10": ("cloudy", "dry", "autumn", "afternoon"),
    "2021-11-23-14-27": "2021-11-23-14-27",
    "2021-11-28-09-18": ("snowing", "snowy", "winter", "morning"),
}



def find_image_pairs(simulation_path, later_simulation_path, use_progress_bar=True, skip_every=5, add_prompt=True):
    first_images_path = os.path.join(simulation_path, "camera")

    csv_path = os.path.join(simulation_path, "applanix", "camera_poses.csv")

    # check if csv_path exists
    if not os.path.exists(csv_path):
        # print(f"csv_path does not exist: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df['GPSTime'] = df['GPSTime'].astype(str)
    # preload all poses into a dictionary for faster lookup
    poses = {
        row['GPSTime']: {
            "easting": row['easting'],
            "northing": row['northing'],
            "altitude": row['altitude'],
            "roll": row['roll'],
            "pitch": row['pitch'],
            "heading": row['heading']
        }
        for _, row in df.iterrows()
    }

    if not os.path.exists(first_images_path):
        return None

    first_images = os.listdir(first_images_path)

    if add_prompt:
        

    pairs = []
    with tqdm(total=len(first_images) // skip_every, desc="Processing images", disable=not use_progress_bar) as pbar:
        alternate = True  # Flag to alternate between source and edited roles
        for idx, image in enumerate(first_images):
            # Process only every `skip_every`-th image
            if idx % skip_every != 0:
                continue
            image_name = image.split(".png")[0]

            # Search in the camera_poses.csv for the camera pose
            # pose = get_camera_pose(image_name, df)
            pose = poses[image_name]
            second_image_path = find_closest_image(pose, later_simulation_path, idx)

            if second_image_path is not None:
                first_image_path = os.path.join(first_images_path, f"{image}")
                
                # Alternate the roles of source and edited images
                if alternate:
                    pairs.append({
                        "source_image": first_image_path,
                        "edited_image": second_image_path,
                    })
                else:
                    pairs.append({
                        "source_image": second_image_path,
                        "edited_image": first_image_path,
                    })
                
                # Toggle the alternate flag
                alternate = not alternate

            # Update the progress bar description with the current number of pairs
            pbar.set_description(f"Processing images (Pairs: {len(pairs)})")
            pbar.update(1)

    return pairs
            


# %%
### --------------------- code for pairng boreas images together ----------------------------- ###
directories = os.listdir(glen_sheilds_path)
# remove all directories that are not of the form boreas-YYYY-MM-DD-HH-MM
directories = [directory for directory in directories if "boreas-202" in directory]

valid_directories = []
for directory in directories:
    # Check if the camera poses CSV file exists
    csv_path = os.path.join(glen_sheilds_path, directory, "applanix", "camera_poses.csv")
    if not os.path.exists(csv_path):
        print(f"csv_path does not exist: {csv_path}")
        continue  # Skip this directory

    # Check if the camera folder exists
    camera_path = os.path.join(glen_sheilds_path, directory, "camera")
    if not os.path.exists(camera_path):
        print(f"camera_path does not exist: {camera_path}")
        continue  # Skip this directory

    # Add the directory to the valid list
    valid_directories.append(directory)

# Replace the original list with the filtered list
directories = valid_directories
print("number of directories found: ", len(directories))



def process_simulation_pair(args):
    simulation, later_simulation = args
    simulation_path = os.path.join(glen_sheilds_path, simulation)
    later_simulation_path = os.path.join(glen_sheilds_path, later_simulation)
    return find_image_pairs(simulation_path, later_simulation_path, use_progress_bar=False)

# Prepare pairs of simulations
simulation_pairs = [
    (directories[i], directories[j])
    for i in range(len(directories))
    for j in range(i + 1, len(directories))
]





random.shuffle(simulation_pairs)
test_size = int(len(simulation_pairs) * test_split_size)

# Split the list into test and train subsets
test_pairs = simulation_pairs[:test_size]
train_pairs = simulation_pairs[test_size:]

print(f"Total pairs: {len(simulation_pairs)}")
print(f"Train pairs: {len(train_pairs)}")
print(f"Test pairs: {len(test_pairs)}")



train_data = []
with ProcessPoolExecutor() as executor:
    # Wrap the executor.map with tqdm for progress tracking
    for result in tqdm(executor.map(process_simulation_pair, train_pairs), 
                       total=len(train_pairs), 
                       desc="Processing simulation pairs"):
        if result is not None:
            train_data = train_data + result
train_dataset = Dataset.from_list(train_data)


test_data = []
with ProcessPoolExecutor() as executor:
    # Wrap the executor.map with tqdm for progress tracking
    for result in tqdm(executor.map(process_simulation_pair, test_pairs), 
                       total=len(test_pairs), 
                       desc="Processing simulation pairs"):
        if result is not None:
            test_data = test_data + result
test_dataset = Dataset.from_list(test_data)


# %%
features = Features({
    'source_image': DatasetImage(),  # Convert to actual images
    'edited_image': DatasetImage(),
    'source_path': Value('string'),
    'edited_path': Value('string'),
})

def add_relative_paths(dataset, parent_folder):
    def process_row(data):
        source_image = data["source_image"]
        edited_image = data["edited_image"]

        source_image_p = os.path.relpath(source_image, parent_folder)
        edited_image_p = os.path.relpath(edited_image, parent_folder)

        # Add new columns
        data["source_path"] = source_image_p
        data["edited_path"] = edited_image_p
        return data

    # Use the map function to apply the changes
    dataset = dataset.map(process_row)
    return dataset
    
train_dataset = add_relative_paths(train_dataset, glen_sheilds_path)
train_dataset = train_dataset.cast(features)
train_dataset.to_parquet(f"./Boreas_with_paths_paired_train.parquet")

test_dataset = add_relative_paths(test_dataset, glen_sheilds_path)
test_dataset = test_dataset.cast(features)
test_dataset.to_parquet(f"./Boreas_with_paths_paired_test.parquet")

# # %%
# features = Features({
#     'source_image': DatasetImage(),  # Convert to actual images
#     'edited_image': DatasetImage(),
# })

# filtered_dataset1 = my_dataset.cast(features)
# filtered_dataset1.to_parquet(f"./Boreas_paired_dataset.parquet")



