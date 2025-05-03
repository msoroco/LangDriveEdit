#!/bin/bash

gpu=3                               # cuda_visible_device number
version=training_new_metadata            
script=./main.py
venv=/net/acadia8a/data/msoroco/code/projects/carla/venv38/bin/activate
output=/net/acadia8a/data/msoroco/code/projects/carla/ImageEditing/output_${version}
server=carla-server_${gpu}

source $venv

host_port1=2${gpu}20
host_port2=2${gpu}21
host_port3=2${gpu}22
host_port_tm=8${gpu}20

# srun --pty --job-name=uedit --ntasks=1 --cpus-per-task=2 --exclude ma-gpu05,ma-gpu07 --mem=32G --gres=gpu:1

# --exclude ma-gpu05,ma-gpu07,ma-gpu28

bash /net/acadia8a/data/msoroco/code/projects/carla/carla_dev/Dist/CARLA_Shipping_0.9.15-278-g6292059fd-dirty/LinuxNoEditor/CarlaUE4.sh \
    -quality-level=Epic -RenderOffScreen -nosound \
    -carla-port=$host_port1 \
    -world-port=$host_port1 \
    -carla-streaming-port=$host_port2 \
    --seed 10 \
    --seed_edit 10
    -carla-rpc-port=$host_port_tm \