from dataclasses import dataclass
from typing import Tuple, List
import argparse
import numpy as np
import os
from datasets import Dataset, Features, Image as DatasetImage, Value, load_dataset, concatenate_datasets, load_from_disk, DatasetDict
import json
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import concurrent.futures
import io




def process_and_collect(subdir_path):
    mask, filtered_descriptions_dictionary, annotated_img_path, distant_traffic_flag = process_image(subdir_path, args.verbose)
    return (
        mask,
        filtered_descriptions_dictionary,
        os.path.basename(subdir_path),
        annotated_img_path,
        distant_traffic_flag
    )

def process_image(subdir_path: str, verbose=True) -> Tuple[np.ndarray, dict]:
    """Process image data from JSON and create masks and descriptions.
    Args:
        json_data: Dictionary containing image annotations
        
    Returns:
        Tuple containing:
        - Binary mask as numpy array
        - dictionary with global descriptions and objects filtered out by distance.
    """
    subdir_name = os.path.basename(subdir_path)
    json_file = os.path.join(subdir_path, f'all_info_{subdir_name}.json')

    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    img_path = os.path.join(subdir_path, f'{subdir_name}_annotated.jpg')
    img = np.array(Image.open(img_path))
    img_height, img_width = img.shape[:2]
    
    # Create empty binary mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    filtered_object_info = []
    # Gather and sort vehicle objects by distance (closest first)
    vehicle_objs = []
    small_far_count = 0  # Counter for small bboxes > 50m
    for obj in json_data['object_info']:
        if obj['class'].lower() in ['car', 'truck', 'bus', 'van', 'ambulance', 'fire truck']:
            x1, y1, x2, y2 = obj['bbox']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            distance = float(obj['distance_from_ego_vehicle'].split()[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            vehicle_objs.append({
                "obj": obj,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "distance": distance,
                "center": (center_x, center_y),
                "width": width,
                "height": height
            })
            # Count small and far vehicles
            if distance > 50 and area < 13500:
                small_far_count += 1
        elif obj['class'].lower() in ['person', 'bicycle', 'motorcycle', 'wheelchair']:
            x1, y1, x2, y2 = obj['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            distance = float(obj['distance_from_ego_vehicle'].split()[0])
            if distance > 50:
                continue
            mask[y1:y2, x1:x2] = 255
            filtered_object_info.append(obj)
           
        elif obj['class'].lower() == 'building':
            # For non-vehicle objects, just keep them as-is
            filtered_object_info.append(obj)
        else: # road signs, traffic lights.
            distance = float(obj['distance_from_ego_vehicle'].split()[0])
            if distance > 50:
                continue
            filtered_object_info.append(obj)

    vehicle_objs.sort(key=lambda v: v["distance"])

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    accepted_bboxes = []
    for v in vehicle_objs:
        obj = v["obj"]
        x1, y1, x2, y2 = v["bbox"]
        area = v["area"]
        distance = v["distance"]
        center_x, center_y = v["center"]

        # ego vehicle filter: check if the object is within 4 meters and its center is in the bottom fourth of the image and center x is approximately in the middle of the image
        if distance < 4 and center_y >  3 * (img_height / 4) and abs(center_x - img_width // 2) < img_width // 10:
            if verbose: print(f"Skipping ego vehicle at distance {distance}m, center_y {center_y}, center_x {center_x}")
            continue
        # far/small vehicle filter: remove vehicles > 50m away unless the bbox is "big"
        if obj['class'].lower() in ['car', 'truck', 'bus', 'van']:
            if distance > 50 and area < 13500:  # TODO: tune this value.
                if verbose: print(f"Skipping vehicle at distance {distance}m, area {area}")
                continue
        # # Overlap filter: only compare with already accepted vehicles (closer ones)
        # significant_overlap = any(iou(v["bbox"], prev_bbox) > 0.6 for prev_bbox in accepted_bboxes)
        # if significant_overlap:
        #     print(f"Skipping vehicle due to significant overlap with vehicles closer to ego vehicle")
        #     continue

        # vehicles that are within the far vehicle filter but are too small means they are likely truncated or obscured
        too_small_likely_truncated = area < 2800 and distance < 50
        if too_small_likely_truncated:
            if verbose: print(f"Skipping vehicle due to being too small and likely truncated")
            continue

        

        # Accept this vehicle
        accepted_bboxes.append(v["bbox"])
        mask[y1:y2, x1:x2] = 255
        filtered_object_info.append(obj)

        if args.debug:
            print(f"Vehicle: {obj['class']}, Distance: {distance}m, Area: {area}, Location: ({center_x}, {center_y}), Size: ({v['width']}, {v['height']})")
            print(f"Mask shape: {mask.shape}, Mask dtype: {mask.dtype}")
            print(f"Mask min: {mask.min()}, Mask max: {mask.max()}")


        
    distant_traffic = small_far_count >= 5  # You can adjust the threshold as needed

    # Build the filtered dictionary to return
    filtered_descriptions_dictionary = {
        "surrounding_info": json_data["surrounding_info"],
        "distant_traffic": distant_traffic,
        "object_info": filtered_object_info
    }

    mask_img = Image.fromarray(mask)
    mask_bytes_io = io.BytesIO()
    mask_img.save(mask_bytes_io, format='PNG')
    mask_bytes = mask_bytes_io.getvalue()
    
    return mask_bytes, filtered_descriptions_dictionary, img_path, distant_traffic


## works
# def merge_datasets(new_dataset: Dataset, existing_dataset: Dataset) -> Dataset:
#     """Merge datasets using index lookups to minimize memory usage."""
#     import gc
#     from tqdm import tqdm
    
#     # Create a mapping from image path to index in new_dataset
#     index_map = {}
    
#     # Process new_dataset to build the index map
#     print("Building index map...")
#     for i in tqdm(range(len(new_dataset)), desc="Indexing new dataset"):
#         path = new_dataset[i]['img_path']
#         key = os.path.splitext(os.path.basename(path))[0]
#         index_map[key] = i
    
#     print(f"Index map created with {len(index_map)} entries.")
    
#     def add_mask_descriptions(example):
#         """Map function to add mask and description columns to each example."""
#         # Get source image info
#         source_path = example['source_path']
#         source_key = os.path.splitext(os.path.basename(source_path))[0].split('/')[-1].split('.')[0]
        
#         # Get source data directly from new_dataset using the index
#         source_idx = index_map.get(source_key)
#         if source_idx is not None:
#             source_item = new_dataset[source_idx]
#             example['source_mask'] = source_item['mask']
#             example['source_filtered_caption'] = source_item['filtered_descriptions']
#             example['source_annotated_img_path'] = source_item['annotated_img_path']
#         else:
#             print(f"Warning: No data found for source_path: {source_key}")
#             raise ValueError(f"Source path {source_key} not found in new_dataset.")
#             # example['source_mask'] = None
#             # example['source_filtered_caption'] = None
#             # example['source_annotated_img_path'] = None
        
#         # Get edited image info
#         edited_path = example['edited_path']
#         edited_key = os.path.splitext(os.path.basename(edited_path))[0].split('/')[-1].split('.')[0]
        
#         # Get edited data directly from new_dataset using the index
#         edited_idx = index_map.get(edited_key)
#         if edited_idx is not None:
#             edited_item = new_dataset[edited_idx]
#             example['edited_mask'] = edited_item['mask']
#             example['edited_filtered_caption'] = edited_item['filtered_descriptions']
#             example['edited_annotated_img_path'] = edited_item['annotated_img_path']
#         else:
#             print(f"Warning: No data found for edited_path: {edited_key}")
#             raise ValueError(f"Edited path {edited_key} not found in new_dataset.")
#             # example['edited_mask'] = None
#             # example['edited_filtered_descriptions'] = None
#             # example['edited_annotated_img_path'] = None
        
#         return example
    
#     # Map over the existing dataset
#     if args.debug:
#         existing_dataset = existing_dataset.select(range(args.debug))
    
#     # Reduce batch_size to avoid excessive memory usage
#     merged_dataset = existing_dataset.map(
#         add_mask_descriptions,
#         desc="Merging datasets",
#         batch_size=100,      # Smaller batch size
#         num_proc=4,         # Parallelize the process
#         load_from_cache_file=False,
#     )
    
#     # Free memory
#     index_map.clear()
#     gc.collect()
    
#     return merged_dataset


def merge_datasets(new_dataset: Dataset, existing_dataset: Dataset) -> Dataset:
    """Merge datasets using index lookups to minimize memory usage, removing rows with missing data."""
    import gc
    from tqdm import tqdm

    def extract_keys(examples):
        """Extract keys from paths in a batch."""
        keys = [os.path.splitext(os.path.basename(path))[0] for path in examples['img_path']]
        return {"key": keys}

    # Add keys column to the dataset
    print("Adding keys column...")
    new_dataset = new_dataset.map(
        extract_keys, 
        batched=True,
        batch_size=1000,
        desc="Extracting keys"
    )

    # Build index map more efficiently
    print("Building index map...")
    index_map = {}
    for i, (key, path) in enumerate(zip(new_dataset['key'], new_dataset['img_path'])):
        index_map[key] = i
    
    # First, filter out examples with missing keys
    print("Filtering dataset to remove entries with missing keys...")
    def is_valid_example(example):
        """Check if an example has valid source and edited keys in our index map."""
        source_path = example['source_path']
        source_key = os.path.splitext(os.path.basename(source_path))[0].split('/')[-1].split('.')[0]
        
        edited_path = example['edited_path']
        edited_key = os.path.splitext(os.path.basename(edited_path))[0].split('/')[-1].split('.')[0]
        
        # Check if both keys exist in our index map
        valid = source_key in index_map and edited_key in index_map
        
        # Optionally log missing keys
        if not valid and args.debug:
            if source_key not in index_map:
                print(f"Removing row: Missing source_key: {source_key}")
            if edited_key not in index_map:
                print(f"Removing row: Missing edited_key: {edited_key}")
        
        return valid

#################################
    # def process_batch(batch_info):
    #     batch_idx, start_idx, end_idx = batch_info
    #     # Get the batch
    #     batch = existing_dataset.select(range(start_idx, end_idx))
        
    #     # Filter the batch
    #     filtered_batch = batch.filter(
    #         is_valid_example,
    #         desc=f"Filtering batch {batch_idx}",
    #         num_proc=1  # Keep this as 1 since we're parallelizing at the batch level
    #     )
        
    #     # Return None if empty
    #     if len(filtered_batch) == 0:
    #         return None
        
    #     return filtered_batch

    # batch_size = 1000  # Adjust based on memory constraints
    # all_filtered_batches = []
    # batch_infos = []

    # # Create batch information
    # for i in range(0, len(existing_dataset), batch_size):
    #     end_idx = min(i + batch_size, len(existing_dataset))
    #     batch_infos.append((i // batch_size, i, end_idx))

    # # Process batches in parallel with ThreadPoolExecutor
    # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #     # Submit all batch processing tasks
    #     future_to_batch = {
    #         executor.submit(process_batch, batch_info): batch_info[0]
    #         for batch_info in batch_infos
    #     }
        
    #     # Process results as they complete
    #     for future in tqdm(
    #         concurrent.futures.as_completed(future_to_batch),
    #         total=len(batch_infos),
    #         desc="Filtering batches in parallel"
    #     ):
    #         batch_idx = future_to_batch[future]
    #         try:
    #             filtered_batch = future.result()
    #             if filtered_batch is not None:
    #                 all_filtered_batches.append(filtered_batch)
    #         except Exception as e:
    #             print(f"Batch {batch_idx} generated an exception: {e}")
            
    #         # Force garbage collection after each batch completes
    #         gc.collect()
#####################################
    batch_size = 4000  # Adjust based on memory constraints
    all_filtered_batches = []

    for i in tqdm(range(0, len(existing_dataset), batch_size), desc="Filtering in batches"):
        batch = existing_dataset.select(range(i, min(i + batch_size, len(existing_dataset))))
        
        # Filter this batch (using single process)
        filtered_batch = batch.filter(
            is_valid_example,
            desc=f"Filtering batch {i//batch_size}",
            num_proc=12
        )
        
        # Save or accumulate the filtered batch
        if len(filtered_batch) > 0:
            all_filtered_batches.append(filtered_batch)
        
        # Force garbage collection
        gc.collect()
################################
    # Combine all filtered batches
    if all_filtered_batches:
        filtered_dataset = concatenate_datasets(all_filtered_batches)
        print(f"Final filtered dataset has {len(filtered_dataset)} examples")
    else:
        filtered_dataset = Dataset.from_dict({})
        print("No valid examples found")
    
    def add_mask_descriptions(example):
        """Map function to add mask and description columns to each example."""
        # Get source image info
        source_path = example['source_path']
        source_key = os.path.splitext(os.path.basename(source_path))[0].split('/')[-1].split('.')[0]
        
        # Get source data directly from new_dataset using the index
        source_idx = index_map.get(source_key)
        source_item = new_dataset[source_idx]  # This is safe now because we've filtered the dataset
        example['source_mask'] = source_item['mask']
        example['source_filtered_caption'] = source_item['filtered_descriptions']
        example['source_annotated_img_path'] = source_item['annotated_img_path']
        
        # Get edited image info
        edited_path = example['edited_path']
        edited_key = os.path.splitext(os.path.basename(edited_path))[0].split('/')[-1].split('.')[0]
        
        # Get edited data directly from new_dataset using the index
        edited_idx = index_map.get(edited_key)
        edited_item = new_dataset[edited_idx]  # This is safe now because we've filtered the dataset
        example['edited_mask'] = edited_item['mask']
        example['edited_filtered_caption'] = edited_item['filtered_descriptions']
        example['edited_annotated_img_path'] = edited_item['annotated_img_path']
        
        return example
    
    # Apply debug limit if needed
    if args.debug:
        filtered_dataset = filtered_dataset.select(range(min(args.debug, len(filtered_dataset))))
    
    # Map over the filtered dataset
    merged_dataset = filtered_dataset.map(
        add_mask_descriptions,
        desc="Merging datasets",
        batch_size=100,      # Smaller batch size
        num_proc=4,         # Parallelize the process
        load_from_cache_file=False,
    )
    
    # Free memory
    index_map.clear()
    gc.collect()
    
    return merged_dataset



def process_and_write_batches(temp_dataset, existing_dataset, batch_size, output_dir, existing_dataset_name):
    num_rows = len(existing_dataset)
    output_files = []
    for batch_idx in tqdm(range(0, num_rows, batch_size), desc="Merging batches"):
        end_idx = min(batch_idx + batch_size, num_rows)
        batch_ds = existing_dataset.select(range(batch_idx, end_idx))
        merged_batch = merge_datasets(temp_dataset, batch_ds)
        out_path = os.path.join(output_dir, f"{existing_dataset_name}_with_masks_batch{batch_idx // batch_size}.parquet")
        merged_batch.to_parquet(out_path)
        output_files.append(out_path)
    return output_files


def main(args):

    
    
    ### --------------------------------------------- main function --------------------------------------------- ###
    if not args.temp_dataset_input_path:
        masks = []
        filtered_descriptions_dictionary_list = []
        img_path = []
        annotated_img_paths = [] # mainly for debugging later
        distance_traffic_flags = []

        subdir_paths = []
        for subdir in os.listdir(args.boreas_parent_dir):
            subdir_path = os.path.join(args.boreas_parent_dir, subdir)
            if os.path.isdir(subdir_path):
                path = os.path.join(args.boreas_parent_dir, subdir)
                json_file = os.path.join(path, f'all_info_{subdir}.json')
                if os.path.exists(json_file):
                    subdir_paths.append(subdir_path)

        print(f"Found {len(subdir_paths)} subdirectories with JSON files.")
        # if args.debug:
        #     print(f"Found {len(subdir_paths)} subdirectories with JSON files.")
        #     subdir_paths = subdir_paths[:args.debug]  # Limit to 10 for debugging

        # with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        #     results = list(tqdm(executor.map(process_and_collect, subdir_paths), total=len(subdir_paths), desc="Processing images"), disable=args.verbose)

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_and_collect, path) for path in subdir_paths]
            results = []
            with tqdm(total=len(subdir_paths), desc="Processing images", disable=args.verbose) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)



        for mask, filtered_descriptions_dictionary, path, annotated_img_path, distant_traffic_flag in results:
            masks.append(mask)
            filtered_descriptions_dictionary_list.append(filtered_descriptions_dictionary)
            img_path.append(path)
            annotated_img_paths.append(annotated_img_path)
            distance_traffic_flags.append(distant_traffic_flag)



        filtered_descriptions = [json.dumps(g) for g in filtered_descriptions_dictionary_list]


        # save the masks and descriptions to a parquet file (intermediate step)
        temp_dataset = Dataset.from_dict({
            'mask': masks,
            'filtered_descriptions': filtered_descriptions,
            'distant_traffic_flag': distance_traffic_flags,
            'img_path': img_path,
            'annotated_img_path': annotated_img_paths
        })

        features = Features({
            'mask': DatasetImage(),
            'filtered_descriptions': Value('string'),
            'distant_traffic_flag': Value('bool'),
            'img_path': Value('string'),
            'annotated_img_path': Value('string')
        })
        temp_dataset = temp_dataset.cast(features)

        print(f"Temp dataset created with {len(temp_dataset)} rows.")

        temp_dataset.to_parquet(os.path.join(args.output_dir, 'temp_masks_large.parquet'))

    else:
        temp_dataset = load_dataset("parquet", data_files=args.temp_dataset_input_path)['train']

    if args.merge_dataset:
        print(f"Merging with existing dataset: {args.merge_dataset}")
        # load existing dataset
        existing_dataset = load_dataset("parquet", data_files=args.merge_dataset)['train']
        
        merged_dataset = merge_datasets(temp_dataset, existing_dataset)

        # get the name of the existing dataset
        existing_dataset_name = os.path.basename(args.merge_dataset).split('.')[0]
        new_name = existing_dataset_name + "_with_masks_large"

        if not args.overwrite:
            # check if the merged dataset already exists
            if os.path.exists(os.path.join(args.output_dir, f'{new_name}.parquet')):
                print(f"The merged dataset {new_name}.parquet already exists. Use --overwrite to overwrite it.")
                new_name = new_name + "_new"
                
        # save the merged dataset
        merged_dataset.to_parquet(os.path.join(args.output_dir, f'{new_name}.parquet'))
        print(f"Merged dataset saved to {os.path.join(args.output_dir, f'{new_name}.parquet')}")

    # if args.merge_dataset:
    #     print(f"Merging with existing dataset: {args.merge_dataset}")
    #     existing_dataset_stream = load_dataset("parquet", data_files=args.merge_dataset)['train']
    #     existing_dataset_name = os.path.basename(args.merge_dataset).split('.')[0]
    #     batch_size = 5000  # Adjust as needed for your RAM

    #     print(f"Processing {len(existing_dataset_stream)} rows in batches of {batch_size}...")
    #     output_files = process_and_write_batches(temp_dataset, existing_dataset_stream, batch_size, args.output_dir, existing_dataset_name)

    #     # Concatenate all batch files into one final dataset
    #     merged_datasets = [load_dataset("parquet", data_files=f)['train'] for f in output_files]
    #     final_merged = concatenate_datasets(merged_datasets)
    #     final_path = os.path.join(args.output_dir, f'{existing_dataset_name}_with_masks.parquet')
    #     final_merged.to_parquet(final_path)
    #     print(f"Merged dataset saved to {final_path}")

















if __name__ == "__main__":
    
    # argparse for command line arguments
    parser = argparse.ArgumentParser(description="Generate vehicle mask descriptions.")
    parser.add_argument('--boreas_parent_dir',
                        type=str,
                        required=True,
                        help=(
                            'Path to the parent directory of the Boreas dataset'
                            'This directory contains subdirectories for each image ifinder labelled.'
                        )
    )
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Path to the output directory. This directory will contain a Dataset parquet file with the masks.'
    )
    parser.add_argument('--temp_dataset_input_path',
                        type=str,
                        default=None,
                        help='Path to the input temp dataset.'
    )
    parser.add_argument('--merge_dataset',
                        type=str,
                        required=False,
                        default=None,
                        help=(
                            'Path to an existing Boreas dataset parquet file.'
                            'If this argument is provided, the script will merge the masks with existing dataset and save the result as a new dataset in `--output_dir`.'
                        )
    )
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Overwrite the existing dataset if it exists.'
    )
    parser.add_argument('--debug',
                        type=int,
                        default=0,
                        help='Enable debug mode for verbose output. `debug` is also the number of subdirectories to process.'
    )
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='Enable debug mode for verbose output. `debug` is also the number of subdirectories to process.'
    )
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose output.'
    )

    args = parser.parse_args()

    main(args)
