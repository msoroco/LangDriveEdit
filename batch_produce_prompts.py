import os
import time
import openai
from datasets import load_dataset
from tqdm import tqdm
import uuid
import json
import re
import numpy as np

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

def process_captions_with_chatgpt_batch(
    captions_0, caption_1, prompt_template, batch_size='inf', max_tokens=1000,
    dataset=None, output_dir=None, old_dataset_name=None
):
    """
    Process captions using ChatGPT-4o-mini with the OpenAI Batch API
    
    Args:
        captions_0: List of first caption strings from the dataset
        caption_1: List of second caption strings from the dataset
        prompt_template: Template for prompting ChatGPT (will be formatted with each caption)
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
        
        # Create batch input file
        batch_input_lines = []
        current_batch_custom_ids = []
        
        for c0, c1 in zip(batch_0, batch_1):
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
                        {"role": "user", "content": prompt_template.format(caption_0=c0, caption_1=c1)}
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
        captions_0 = [example["source_filtered_caption"]]
        captions_1 = [example["edited_filtered_caption"]]
    else:
        # carla_train_path = "/net/acadia8a/data/msoroco/code/projects/ultraedit/chatGPT_masks/output_full/Boreas_with_paths_paired_train_simple_prompts_dataset_55k_with_masks.parquet"

        # carla_train_path = "/net/acadia8a/data/msoroco/code/projects/ultraedit/chatGPT_masks/output_full_2/Boreas_with_paths_paired_train_simple_prompts_dataset_with_masks_large.parquet"

        # ## validation set
        carla_train_path = "/net/acadia8a/data/msoroco/code/projects/ultraedit/chatGPT_masks/output_full/Boreas_with_paths_paired_test_simple_prompts_dataset_24k_with_masks.parquet"


        dataset = load_dataset('parquet', data_files=carla_train_path, split='train')

        
        # Extract captions from the dataset
        captions_0 = dataset["source_filtered_caption"]
        captions_1 = dataset["edited_filtered_caption"]


    
    # Define your prompt template
    # Use {caption} as a placeholder where each caption will be inserted

    #     You must describe each the type and color of each vehicle or person present from left to right. The AI image editor will be provided a mask image indicating where each vehicle should go and your prompt.
    prompt_template = """
    You are an expert in autonomous driving, specializing in analyzing traffic scenes. You receive text captions of two traffic images from the perspective of an autonomous vehicle's camera. Both images depict the same location, but there is a key difference between the images.

    Your task is to produce a prompt that an AI image editor could use to edit the first image into the second. Produce four versions (paraphrasings) of that prompt.
    Then produce a prompt that an AI image editor could use to edit the second image into the first. Produce four versions (paraphrasings) of that prompt. Be concise.

    If bounding box dimensions are provided, do NOT mention them in any prompt.
    Strictly ignore and do NOT mention any buildings. ignore and do NOT mention the road layout. Do not mention the environment.
    Do NOT mention traffic lights unless they are present in both images and have different colors.
    You must mention other surrounding info such as weather, time, road color if there are differences.
    DO NOT invent or add any vehicles, people, or objects that aren't explicitly mentioned in the captions. Only describe vehicles that are actually present in the object_info lists of the captions.

    You must describe each the type and color of each vehicle or person to REMOVE from left to right. **Do NOT use the order in which vehicles are listed in the captions.**
    For each vehicle, extract the first value (x0) from its `bbox` field (e.g., `bbox`: [774, 918, 1020, 1099] where 774 is the x0 value).
    Sort all vehicles by their x0 value from smallest to largest to determine the left-to-right order in the image.
    Describe the vehicles in this sorted order, from left (smallest x0) to right (largest x0).
    Do NOT use the order in the caption text.
    For example, if three vehicles have x0 values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right).
    The AI image editor will be provided a mask image indicating each vehicle to remove following your prompt.
    You must also describe each the type and color of each vehicle or person to ADD from left to right.  **Do NOT use the order in which vehicles are listed in the captions.**
    For each vehicle, extract the first value (x0) from its `bbox` field (e.g., `bbox`: [774, 918, 1020, 1099] where 774 is the x0 value).
    Sort all vehicles by their x0 value from smallest to largest to determine the left-to-right order in the image.
    Describe the vehicles in this sorted order, from left (smallest x0) to right (largest x0).
    Do NOT use the order in the caption text.
    For example, if three vehicles have x0 values of 100, 300, and 200, describe them in the order: 100, 200, 300 (left to right). The AI image editor will be provided a second mask image indicating each vehicle to add following your prompt.

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
    output_dir = "/net/acadia8a/data/msoroco/code/projects/ultraedit/chatGPT/gpt_output_prompts_NEW"
    os.makedirs(output_dir, exist_ok=True)

    # Save the updated dataset with responses
    old_dataset_name = os.path.basename(carla_train_path).split('.')[0]

    
    # Process captions with ChatGPT
    responses, failed = process_captions_with_chatgpt_batch(
        captions_0=captions_0,
        caption_1=captions_1,
        prompt_template=prompt_template,
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

    print(f"Processed {len(responses)} captions and saved the updated dataset with unparsed prompts.")
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