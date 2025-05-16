import os
import time

import numpy as np
import openai
from datasets import load_dataset
from tqdm import tqdm
import uuid
import json
import re

# # Configure OpenAI API
# openai.api_key = os.environ.get("OPENAI_API_KEY")  # Set your API key as an environment variable for security
# client = openai.OpenAI()

def save_batch_id(batch_id):
    with open("/net/acadia8a/data/msoroco/code/projects/ultraedit/chatGPT/batch_tracking.json", "a+") as f:
        f.seek(0)
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"batch_ids": []}
        
        if batch_id not in data["batch_ids"]:
            data["batch_ids"].append(batch_id)
        
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2)

def process_captions_with_chatgpt_batch(captions_0, caption_1, prompt_template, key_differences=None, specific_instructions=None, batch_size='inf', max_tokens=1000, dataset=None, output_dir=None, old_dataset_name=None):
    """
    Process captions using ChatGPT-4o-mini with the OpenAI Batch API
    
    Args:
        captions_0: List of first caption strings from the dataset
        caption_1: List of second caption strings from the dataset
        prompt_template: Template for prompting ChatGPT (will be formatted with each caption)
        key_differences: List of key differences between image pairs
        specific_instructions: List of specific instructions for each image pair
        batch_size: Number of captions to process in each batch file
        max_tokens: Maximum tokens for the response
        
    Returns:
        List of responses from ChatGPT
    """
    # all_responses = []
    all_responses = [None] * len(captions_0)  # Pre-allocate for easier updating
    assert len(captions_0) == len(caption_1), "Captions lists must be of the same length, got {} and {}".format(len(captions_0), len(caption_1))
    
    # Create temporary directory for batch files
    batch_dir = "/net/acadia8a/data/msoroco/code/projects/ultraedit/chatGPT/batch_inputs"
    batch_dir_out = "/net/acadia8a/data/msoroco/code/projects/ultraedit/chatGPT/batch_outputs"
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(batch_dir_out, exist_ok=True)
    
    # Process in batches
    batch_ids = []
    batch_custom_ids = []
    if batch_size == 'inf':
        batch_size = len(captions_0)
    
    for i in range(0, len(captions_0), batch_size):
        batch_0 = captions_0[i:i+batch_size]
        batch_1 = caption_1[i:i+batch_size]
        batch_key_differences = key_differences[i:i+batch_size]
        batch_specific_instructions = specific_instructions[i:i+batch_size]
        
        # Create batch input file
        batch_input_lines = []
        current_batch_custom_ids = []
        
        for c0, c1, kd, si in zip(batch_0, batch_1, batch_key_differences, batch_specific_instructions):
            custom_id = str(uuid.uuid4())
            current_batch_custom_ids.append(custom_id)
            
            # Format request for batch API
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are an expert in autonomous driving and image editing prompts."},
                        {"role": "user", "content": prompt_template.format(
                            caption_0=c0, 
                            caption_1=c1, 
                            key_difference=kd, 
                            specific_instructions=si
                        )}
                    ],
                    "max_tokens": max_tokens
                }
            }
            batch_input_lines.append(json.dumps(request))
        
        # Write batch input to file
        timestamp = int(time.time())
        batch_filename = f"{batch_dir}/batch_input_{i}_{timestamp}.jsonl"
        with open(batch_filename, "w") as f:
            f.write("\n".join(batch_input_lines))
        
        # Upload the file to OpenAI
        uploaded_file = client.files.create(
            file=open(batch_filename, "rb"),
            purpose="batch"
        )
        
        # Create batch job
        try:
            batch = client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            save_batch_id(batch.id)
            batch_ids.append(batch.id)
            batch_custom_ids.extend(current_batch_custom_ids)
            print(f"Submitted batch {batch.id} with {len(batch_input_lines)} requests")
            
        except Exception as e:
            print(f"Error in batch submission: {e}")
    
    # Wait for all batches to complete
    print("Waiting for batches to complete...")
    responses_by_custom_id = {}
    
    failed_batches = []
    completed_batches = 0
    remaining_batches = set(batch_ids)


    last_save_time = time.time()
    save_interval = 3600  # seconds (1 hour)

    while remaining_batches:
        for batch_id in list(remaining_batches):
            status = client.batches.retrieve(batch_id)
            if status.status == "completed":
                # Download output file
                output_file = client.files.retrieve(status.output_file_id)
                output_content = client.files.content(status.output_file_id).text

                # save the output content to a file
                output_filename = f"{batch_dir_out}/batch_output_{batch_id}.jsonl"
                with open(output_filename, "w") as f:
                    f.write(output_content)
                print(f"Batch {batch_id} completed. Output saved to {output_filename}")

                # Parse output lines
                output_lines = output_content.strip().split('\n')
                for line in output_lines:
                    result = json.loads(line)
                    custom_id = result.get('custom_id')

                    response_data = result.get('response', {})
                    idx = batch_custom_ids.index(custom_id)


                    if isinstance(response_data.get('body'), dict):
                        response_body = response_data.get('body', {})
                    else:
                        response_body = json.loads(response_data.get('body', '{}'))

                    content = response_body.get('choices', [{}])[0].get('message', {}).get('content', '')
                    responses_by_custom_id[custom_id] = content
                    all_responses[idx] = content

                completed_batches += 1
                remaining_batches.remove(batch_id)
            elif status.status in ["failed", "expired", "cancelled"]:
                print(f"Batch {batch_id} failed with status: {status.status}")
                failed_batches.append(batch_id)
                remaining_batches.remove(batch_id)
            else:
                print(f"Batch {batch_id} status: {status.status}. Waiting... completed: {completed_batches}/{len(batch_ids)}")
                # --- NEW: Try to fetch partial results for in_progress batches ---
                try:
                    output_file = client.files.retrieve(status.output_file_id)
                    output_content = client.files.content(status.output_file_id).text
                    output_filename = f"{batch_dir_out}/batch_output_{batch_id}_partial.jsonl"
                    with open(output_filename, "w") as f:
                        f.write(output_content)
                    print(f"Batch {batch_id} in progress. Partial output saved to {output_filename}")

                    output_lines = output_content.strip().split('\n')
                    for line in output_lines:
                        result = json.loads(line)
                        custom_id = result.get('custom_id')
                        response_data = result.get('response', {})
                        idx = batch_custom_ids.index(custom_id)
                        if isinstance(response_data.get('body'), dict):
                            response_body = response_data.get('body', {})
                        else:
                            response_body = json.loads(response_data.get('body', '{}'))
                        content = response_body.get('choices', [{}])[0].get('message', {}).get('content', '')
                        responses_by_custom_id[custom_id] = content
                        all_responses[idx] = content
                except Exception as e:
                    print(f"Could not fetch partial results for batch {batch_id}: {e}")


        # Save progress every hour
        if dataset is not None and output_dir is not None and old_dataset_name is not None:
            if time.time() - last_save_time > save_interval:
                # Save partial results
                temp_dataset = dataset.add_column("gpt_response", all_responses)
                os.makedirs(output_dir, exist_ok=True)
                temp_dataset.to_parquet(f"{output_dir}/{old_dataset_name}_with_responses_partial.parquet")
                print(f"Checkpoint: Saved partial dataset with {sum(r is not None for r in all_responses)} responses.")
                print("saved to", f"{output_dir}/{old_dataset_name}_with_responses_partial.parquet")
                last_save_time = time.time()   
        if remaining_batches:
            time.sleep(30)  # Check every 30 seconds
        
    # # Order responses according to the original request order
    # for custom_id in batch_custom_ids:
    #     response = responses_by_custom_id.get(custom_id)
    #     all_responses.append(response)
    
    return all_responses, failed_batches



def parse_prompts(example):
    response = example['gpt_response']
    try:
        # Split into forward/backward sections
        parts = response.split("### Prompt to edit Image-1 into Image-2:")
        if len(parts) < 2:
            print("Error: Expected 2 parts, got:", len(parts))
            print("Full response was:\n", response)
            return {"forward_full_prompts": [], "backward_full_prompts": []}
        
        forward_blk, rest = parts[1].split("### Prompt to edit Image-2 into Image-1:")
        backward_blk = rest

        forward_versions = [
            m.group(1).strip()
            for m in re.finditer(r"version_\d+\s*:\s*(.+)", forward_blk)
        ]
        backward_versions = [
            m.group(1).strip()
            for m in re.finditer(r"version_\d+\s*:\s*(.+)", backward_blk)
        ]

        return {
            "forward_full_prompts": forward_versions,
            "backward_full_prompts": backward_versions,
        }
    except Exception as e:
        print("Error parsing gpt_response:", e)
        print("Full response was:\n", response)
        # returning None tells `dataset.map` to skip this row
        return {"forward_full_prompts": [], "backward_full_prompts": []}




def main(example=None):
    if example is not None:
        print(example)
        captions_0 = [example["source_caption"]]
        captions_1 = [example["edited_caption"]]
        edit = [example['edit']]
    else:

        carla_train_path = '/net/acadia8a/data/msoroco/code/projects/carla/ImageEditing/output_training_new_metadata_paired_dataset_new_metadata4_fixed6.parquet'

        # ## validation set
        # carla_train_path = "/net/acadia8a/data/msoroco/code/projects/carla/ImageEditing/output_testing_new_metadata_paired_dataset_new_metadata_fixed_buildings8.parquet"


        dataset = load_dataset('parquet', data_files=carla_train_path, split='train') #.shuffle(seed=38).select(range(200))

        
        # Extract captions from the dataset
        captions_0 = dataset["source_caption"]
        captions_1 = dataset["edited_caption"]

        edits = dataset['edit']

    # Create custom key_differences and specific_instructions based on edit types
    key_differences = []
    specific_instructions = []

    for edit in tqdm(edits, desc="Processing edit types"):
        if edit == "time_of_day":
            key_difference = "the time of day is changed from image 1 to image 2."
            instruction = "Focus on describing the time of day. Do not mention specific numbers but rather describe qualitatively the general conditions."
        elif edit == "weather":
            key_difference = "the weather is changed from image 1 to image 2."
            instruction = "Focus on describing the weather conditions. Do not mention specific numbers but rather describe qualitatively the general conditions."
        elif edit == "weather_and_time_of_day":
            key_difference = "the time of day and weather are changed from image 1 to image 2."
            instruction = "Focus on describing the global image characteristics such as weather and time. Do not mention specific numbers but rather describe qualitatively the general conditions."

        elif edit == "building_texture":
            key_difference = "the textures and colors of the buildings are changed in image 1 to image 2, but the underlying building architecture is the same."
            instruction = "Simply prompt the AI image editor to 'change the textures and colors of certain buildings'. Assume that it has no knowledge of the original building textures and colors. Do not refer the `original` textures and colors nor the `original image`."

        elif edit == "vehicle_color":
            key_difference = "the vehicle colours are changed in image 1 to image 2."
            instruction = "You MUST describe the type and color of each vehicle to EDIT from left to right.\n**Do NOT use the order in which vehicles are listed in the captions.**\nFor each vehicle, extract the `x0` value from its `image_bbox_2d` field (e.g., `image_bbox_2d`: {\"x0\": 393, ...}).\nSort all vehicles by their `x0` value from smallest to largest to determine the left-to-right order in the image.\nDescribe the vehicles in this sorted order, from left (smallest x0) to right (largest x0).\nDo NOT use the order in the caption text.\nFor example, if three vehicles have `x0` values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right).\nThe AI image editor will be provided a mask image indicating each vehicle to change following your prompt. Specify the base_type and type (brand) of each vehicle."
        elif edit == "vehicle_replacement":
            key_difference = "the vehicles are replaced with different vehicles in image 1 to image 2."
            instruction = "You must describe the type and color of each vehicle to REMOVE from left to right.\n**Do NOT use the order in which vehicles are listed in the captions.**\nFor each vehicle, extract the `x0` value from its `image_bbox_2d` field (e.g., `image_bbox_2d`: {\"x0\": 393, ...}).\nSort all vehicles by their `x0` value from smallest to largest to determine the left-to-right order in the image.\nDescribe the vehicles in this sorted order, from left (smallest x0) to right (largest x0).\nDo NOT use the order in the caption text.\nFor example, if three vehicles have `x0` values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right).\nThe AI image editor will be provided a mask image indicating each vehicle to remove following your prompt.\nYou must also describe each the type and color of each vehicle to ADD from left to right.\n**Do NOT use the order in which vehicles are listed in the captions.**\nFor each vehicle, extract the `x0` value from its `image_bbox_2d` field (e.g., `image_bbox_2d`: {\"x0\": 393, ...}).\nSort all vehicles by their `x0` value from smallest to largest to determine the left-to-right order in the image.\nDescribe the vehicles in this sorted order, from left (smallest x0) to right (largest x0).\nDo NOT use the order in the caption text.\nFor example, if three vehicles have `x0` values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right).\n The AI image editor will be provided a second mask image indicating each vehicle to add following your prompt. Specify the base_type and type (brand) of each vehicle."
            # instruction = "You must describe the type and color of each vehicle to REMOVE from left to right. Ignore the CAPTION'S ORDER of vehicles. Use the bounding box information (image_bbox_2d) to determine the vehicle order from left to right. The AI image editor will be provided a mask image indicating each vehicle to remove following your prompt.\nYou must also describe each the type and color of each vehicle to ADD from left to right. The captions may describe the vehicles in any order so use the bounding box information (image_bbox_2d) to determine the vehicle order from left to right. The AI image editor will be provided a second mask image indicating each vehicle to add following your prompt. Try to specify the type of vehicle (e.g., car, truck, bus, police car)."
        elif edit == "vehicle_deletion":
            key_difference = "the vehicles are removed from image 1 to image 2."
            instruction = "You must describe the type and color of each vehicle to REMOVE (or ADD) from left to right.\n**Do NOT use the order in which vehicles are listed in the captions.**\nFor each vehicle, extract the `x0` value from its `image_bbox_2d` field (e.g., `image_bbox_2d`: {\"x0\": 393, ...}).\nSort all vehicles by their `x0` value from smallest to largest to determine the left-to-right order in the image.\nDescribe the vehicles in this sorted order, from left (smallest x0) to right (largest x0).\nDo NOT use the order in the caption text.\nFor example, if three vehicles have `x0` values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right).\n The AI image editor will be provided a mask image indicating each vehicle to remove (or add) following your prompt. Specify the base_type and type (brand) of each vehicle."
            # instruction = "You must describe the type and color of each vehicle to REMOVE (or ADD) from left to right. The captions may describe the vehicles in any order so use the bounding box information (image_bbox_2d) to determine the vehicle order from left to right. The AI image editor will be provided a mask image indicating each vehicle to remove (or add) following your prompt. Try to specify the type of vehicle (e.g., car, truck, bus, police car)."


        elif edit == "walker_color":
            key_difference = "the pedestrian clothings are changed in image 1 to image 2."
            instruction = (
                "You must describe the type and clothing color of each pedestrian to EDIT from left to right.\n"
                "**Do NOT use the order in which pedestrians are listed in the captions.**\n"
                "For each pedestrian, extract the `x0` value from their `image_bbox_2d` field (e.g., `image_bbox_2d`: {\"x0\": 393, ...}).\n"
                "Sort all pedestrians by their `x0` value from smallest to largest to determine the left-to-right order in the image.\n"
                "Describe the pedestrians in this sorted order, from left (smallest x0) to right (largest x0).\n"
                "Do NOT use the order in the caption text.\n"
                "For example, if three pedestrians have `x0` values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right).\n"
                "The AI image editor will be provided a mask image indicating each pedestrian to change following your prompt."
            )
        elif edit == "walker_replacement":
            key_difference = "the pedestrians are replaced with different pedestrians in image 1 to image 2."
            instruction = (
                "You must describe the type and clothing color of each pedestrian to REMOVE from left to right.\n"
                "**Do NOT use the order in which pedestrians are listed in the captions.**\n"
                "For each pedestrian, extract the `x0` value from their `image_bbox_2d` field (e.g., `image_bbox_2d`: {\"x0\": 393, ...}).\n"
                "Sort all pedestrians by their `x0` value from smallest to largest to determine the left-to-right order in the image.\n"
                "Describe the pedestrians in this sorted order, from left (smallest x0) to right (largest x0).\n"
                "Do NOT use the order in the caption text.\n"
                "For example, if three pedestrians have `x0` values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right).\n"
                "The AI image editor will be provided a mask image indicating each pedestrian to remove following your prompt.\n"
                "You must also describe the type and clothing color of each pedestrian to ADD from left to right.\n"
                "**Do NOT use the order in which pedestrians are listed in the captions.**\n"
                "For each pedestrian, extract the `x0` value from their `image_bbox_2d` field (e.g., `image_bbox_2d`: {\"x0\": 393, ...}).\n"
                "Sort all pedestrians by their `x0` value from smallest to largest to determine the left-to-right order in the image.\n"
                "Describe the pedestrians in this sorted order, from left (smallest x0) to right (largest x0).\n"
                "Do NOT use the order in the caption text.\n"
                "For example, if three pedestrians have `x0` values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right).\n"
                "The AI image editor will be provided a second mask image indicating each pedestrian to add following your prompt."
            )
        elif edit == "walker_deletion":
            key_difference = "the pedestrians from image 1 are removed in image 2."
            instruction = (
                "You must describe the type and clothing color of each pedestrian to REMOVE (or ADD) from left to right.\n"
                "**Do NOT use the order in which pedestrians are listed in the captions.**\n"
                "For each pedestrian, extract the `x0` value from their `image_bbox_2d` field (e.g., `image_bbox_2d`: {\"x0\": 393, ...}).\n"
                "Sort all pedestrians by their `x0` value from smallest to largest to determine the left-to-right order in the image.\n"
                "Describe the pedestrians in this sorted order, from left (smallest x0) to right (largest x0).\n"
                "Do NOT use the order in the caption text.\n"
                "For example, if three pedestrians have `x0` values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right).\n"
                "The AI image editor will be provided a mask image indicating each pedestrian to remove (or add) following your prompt."
            )


        elif edit == "road_texture":
            key_difference = "the road texture and colour is changed from image 1 to image 2."
            instruction = "Note that the road layout is the same in both images. Focus on describing the change in road texture and color. If you cannot describe the resulting road texture, ONLY prompt the AI image editor to 'change the road textures and color' and do not refer to the `original` nor `previous` road textures and colors nor to the previous nor original image."
        elif edit == "traffic_light_state":
            key_difference = "the traffic light state is changed in image 1 to image 2. The traffic light colours were permuted in image 1 to image 2."
            instruction = (
                "You must describe the color of each traffic light to CHANGE from left to right.\n"
                "**Do NOT use the order in which traffic lights are listed in the captions.**\n"
                "For each traffic light, extract the `x0` value from its `image_bbox_2d` field (e.g., `image_bbox_2d`: {\"x0\": 393, ...}).\n"
                "Sort all traffic lights by their `x0` value from smallest to largest to determine the left-to-right order in the image.\n"
                "Describe the traffic lights in this sorted order, from left (smallest x0) to right (largest x0).\n"
                "Do NOT use the order in the caption text.\n"
                "For example, if three traffic lights have `x0` values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right).\n"
                "The AI image editor will be provided a mask image indicating each traffic light to change following your prompt."
            )
        else:
            raise ValueError(f"Unknown edit type: {edit}")
        
        key_differences.append(key_difference)
        specific_instructions.append(instruction)


    
    # Define your prompt template
    # Use {caption} as a placeholder where each caption will be inserted

    #     Your task is to produce a prompt that an AI image editor could use to edit the first image into the second. Produce four versions (paraphrasings) of that prompt.
    # Then produce a prompt that an AI image editor could use to edit the second image into the first. Produce four versions (paraphrasings) of that prompt. Be concise. 

    #     You must describe each the type and color of each vehicle or person present from left to right. The AI image editor will be provided a mask image indicating where each vehicle should go and your prompt.
    prompt_template = """
    You are an expert in autonomous driving, specializing in analyzing traffic scenes. You receive text descriptions of two traffic images from the perspective of an autonomous vehicle's camera. Both images depict the same location, but there is a key difference between the images.

    THE KEY DIFFERENCE IS THAT: {key_difference}

    Your task is to produce two sets of concise prompts that an AI image editor could use:
    1. First, create four versions (paraphrasings) of a prompt that an AI image editor could use to edit Image-1 into Image-2
    2. Then, create four versions (paraphrasings) of a prompt that an AI image editor could use to edit Image-2 into Image-1. These should be independent of the first set of prompts.

    IMPORTANT: Treat each direction as a completely independent editing task. Do NOT use phrases like "add back," "restore," "return to original," or "as it was before" that reference previous editing. Each prompt should stand alone as if the other image didn't exist. Be concise. 


    If bounding box dimensions (image_bbox_2d) are provided, do NOT mention them in any prompt. However you should use the bounding boxes to determine the relative positions of the objects (e.g., left, right) in the images.
    Strictly ignore and do not mention any descriptions that are not directly related to the key difference between the two images. Ignore objects that are further than 80 meters from the ego vehicle.

    {specific_instructions}

    The prompt should be in the format below where each version describes the same contents but with different wording. Be concise.

    ### Prompt to edit Image-1 into Image-2:

    version_1: {{prompt_1}}
    version_2: {{prompt_2}}
    version_3: {{prompt_3}}
    version_4: {{prompt_4}}

    ### Prompt to edit Image-2 into Image-1:
    version_5: {{prompt_5}}
    version_6: {{prompt_6}}
    version_7: {{prompt_7}}
    version_8: {{prompt_8}}

    'Image-1': {caption_0}

    'Image-2': {caption_1}
    """
    
    # Save the updated dataset with responses
    output_dir = "/net/acadia8a/data/msoroco/code/projects/ultraedit/chatGPT/gpt_output_prompts_carla2"
    os.makedirs(output_dir, exist_ok=True)

    old_dataset_name = os.path.basename(carla_train_path).split('.')[0]

    # Process captions with ChatGPT
    responses, failed = process_captions_with_chatgpt_batch(
        captions_0=captions_0,
        caption_1=captions_1,
        prompt_template=prompt_template,
        key_differences=key_differences,
        specific_instructions=specific_instructions,
        batch_size=10000,  # Adjust batch size based on your needs and API limits
        dataset=dataset,
        output_dir=output_dir,
        old_dataset_name=old_dataset_name
    )

    if example is not None:
        print(responses)
        return
    
    # Add responses back to the dataset
    output_dataset = dataset.add_column("gpt_response", responses)
    output_dataset.to_parquet(f"{output_dir}/{old_dataset_name}_with_responses.parquet")

    print(f"Processed {len(responses)} captions and saved the updated dataset.")
    print("saved to", f"{output_dir}/{old_dataset_name}_with_responses.parquet")

    # parse the responses into two lists, forward_full_prompts and backward_full_prompts
    num_proc = 4 #multiprocessing.cpu_count()
    output_dataset = output_dataset.map(parse_prompts, num_proc=num_proc)

    # Filter out rows where parsing failed (empty forward_full_prompts)
    forward_prompts = output_dataset["forward_full_prompts"]
    mask = np.array([len(x) > 0 for x in forward_prompts])
    indices = np.where(mask)[0].tolist()
    output_dataset = output_dataset.select(indices)


    output_dataset.to_parquet(f"{output_dir}/{old_dataset_name}_with_responses_parsed.parquet")
    print(f"Processed {len(responses)} captions and saved the updated dataset with parsed prompts.")
    print("saved to", f"{output_dir}/{old_dataset_name}_with_responses_parsed.parquet")



if __name__ == "__main__":
    example = {
        'source_filtered_caption': """{"surrounding_info": {"weather": "sunny", "road_layout": "straight road", "environment": "city street", "sun_visibility_conditions": "clear", "road_condition": "normal", "surface_type": "asphalt", "surface_color": "light grey", "time_of_the_day": "morning", "precipitation_intensity": "none", "precipitation_visibility_impact": "none", "cloud_cover": "clear"}, "distant_traffic": true, "object_info": [{"class": "car", "bbox": [64, 1228, 121, 1385], "object_id": 18, "distance_from_ego_vehicle": "6.82 meters", "attributes": "White color, Not a police car."}, {"class": "car", "bbox": [59, 1078, 192, 1266], "object_id": 17, "distance_from_ego_vehicle": "13.11 meters", "attributes": "Black color, Not a police car."}, {"class": "car", "bbox": [232, 1056, 571, 1220], "object_id": 15, "distance_from_ego_vehicle": "17.66 meters", "attributes": "Blue color, Not a police car."}, {"class": "car", "bbox": [571, 1057, 754, 1163], "object_id": 16, "distance_from_ego_vehicle": "26.10 meters", "attributes": "Black color, Not a police car."}, {"class": "car", "bbox": [187, 1065, 333, 1129], "object_id": 10, "distance_from_ego_vehicle": "37.72 meters", "attributes": "Black color, Not a police car."}]}""",
        'edited_filtered_caption': """{"surrounding_info": {"weather": "cloudy", "road_layout": "straight road", "environment": "city street", "sun_visibility_conditions": "low visibility", "road_condition": "normal", "surface_type": "asphalt", "surface_color": "dark grey", "time_of_the_day": "morning", "precipitation_intensity": "none", "precipitation_visibility_impact": "none", "cloud_cover": "heavy"}, "distant_traffic": true, "object_info": [{"class": "traffic light", "bbox": [1011, 945, 1027, 966], "object_id": 1, "distance_from_ego_vehicle": "21.56 meters", "attributes": "Red light"}, {"class": "building", "bbox": [1158, 970, 1238, 1064], "object_id": 3, "distance_from_ego_vehicle": "87.66 meters", "attributes": "Brown color, Rectangular texture, Residential type"}, {"class": "building", "bbox": [1296, 926, 1354, 1059], "object_id": 4, "distance_from_ego_vehicle": "85.87 meters", "attributes": "Red color, Rectangular texture, Residential type"}, {"class": "person", "bbox": [1449, 1050, 1469, 1094], "object_id": 26, "distance_from_ego_vehicle": "38.94 meters", "attributes": "wearing Red clothes, age: adult, Not a policeman."}, {"class": "car", "bbox": [803, 1056, 949, 1138], "object_id": 28, "distance_from_ego_vehicle": "27.25 meters", "attributes": "White color, not a police car"}, {"class": "car", "bbox": [2, 1023, 577, 1079], "object_id": 27, "distance_from_ego_vehicle": "52.81 meters", "attributes": "White color, Not a police car."}]}"""
    }
    # main(example=example)
    main()