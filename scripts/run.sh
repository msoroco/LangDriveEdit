#!/bin/bash
world_size=8
cpus_per_task=$((2 * $world_size))
launch_cmd="sbatch --ntasks=1 --constrain=GPU48GB \
  --cpus-per-task=${cpus_per_task} --mem=$((40 * $world_size))G --gres=gpu:${world_size}"

export PYTHONFAULTHANDLER=1

pretrained_model_name_or_path=BleachNick/SD3_UltraEdit_w_mask
ori_model_name_or_path=BleachNick/SD3_UltraEdit_w_mask

deepspeed="deepspeed/ds_s2_no_offload.json"

dataset_name=(
  path_to_real_dataset.parquet
  path_to_synthetic_dataset.parquet
)
resolution_w=608
resolution_h=368
train_batch_size=32
gradient_accumulation_steps=1
checkpointing_steps=1000
validation_step=2000
visualization_step=1000

epoch=3
lr=5e-05
output_dir="path_to_output_dir/run_name"
dataloader_num_workers=$(($cpus_per_task / $world_size))

set -x

${launch_cmd} accelerate launch --mixed_precision="bf16" --multi_gpu --main_process_port=29555 training/train_sd3_pix2pix_cycle_full.py \
 --pretrained_model_name_or_path ${pretrained_model_name_or_path} \
 --output_dir ${output_dir} \
 --resolution ${resolution_w} ${resolution_h} \
 --max_train_steps 10000 \
 --gradient_checkpointing \
 --train_batch_size ${train_batch_size} \
 --gradient_accumulation_steps ${gradient_accumulation_steps} \
 --checkpointing_steps ${checkpointing_steps}  \
 --checkpoints_total_limit 2 \
 --learning_rate ${lr} \
 --lr_warmup_steps 500 \
 --conditioning_dropout_prob 0.05 \
 --mixed_precision bf16 \
 --seed 3345 \
 --num_validation_images 4 \
 --validation_step ${validation_step} \
 --dataloader_num_workers $dataloader_num_workers \
 --lr_scheduler cosine \
 --report_to wandb \
 --visualization_step ${visualization_step} \
 --ori_model_name_or_path ${ori_model_name_or_path} \
 --max_sequence_length 256 \
 --dataset_name "${dataset_name[@]}" \
 --reverse_edit_prompt_column "backward_full_prompts" \
 --edit_prompt_column "forward_full_prompts" \
 --ds_config_path ${deepspeed} \
 --do_mask \
 --remove_mask_column "source_mask" \
 --add_mask_column "edited_mask" \
 --project_name "ultraedit_sd3" \
 --cycle_reverse_threshold 0 \
 --resume_from_checkpoint "latest" \


set +x