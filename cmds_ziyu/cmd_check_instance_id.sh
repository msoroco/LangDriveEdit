#!/bin/bash

gpu=0                               # cuda_visible_device number
version=output_dir_name             
script=./main.py
venv=/net/acadia8a/data/msoroco/code/projects/carla/ImageEditing/../venv0_9_15/bin/activate
output=./output_${version}
server=carla-server_${gpu}


source $venv

edit_array=(
    # 'time_of_day'
    # 'weather'
    # 'weather_and_time_of_day'
    # 'building_texture'
    # 'vehicle_color'
    # 'vehicle_replacement'
    # 'vehicle_deletion'
    # 'walker_color'
    'walker_replacement'
    # 'walker_deletion'
    # 'road_texture'
    # 'traffic_light_state'
)

# docker run --net=host --runtime=nvidia --name carla_server_0 --gpus "device=0" --env=NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.9.15 /bin/bash CarlaUE4.sh -quality-level=Epic -RenderOffScreen -nosound

# docker run --net=host --runtime=nvidia --name carla_server_1 --gpus "device=1" --env=NVIDIA_VISIBLE_DEVICES=1 carlasim/carla:0.9.15 /bin/bash CarlaUE4.sh -quality-level=Epic -RenderOffScreen -nosound

host_port1=2${gpu}00
host_port2=2${gpu}01
host_port3=2${gpu}02
host_port_tm=8${gpu}00

while true
do
    echo "Press [CTRL+C] to stop.."
    docker stop $server
    docker rm $server
    
    ## Start the container with the specified GPU
    docker run -d --runtime=nvidia \
        --name $server \
        --net=host \
        --gpus "device=${gpu}" \
        --env=NVIDIA_VISIBLE_DEVICES=$gpu \
        carlasim/carla:0.9.15 /bin/bash CarlaUE4.sh \
        -quality-level=Epic -RenderOffScreen -nosound \
        -carla-port=$host_port1 \
        -world-port=$host_port1 \
        -carla-streaming-port=$host_port2 \
        --seed 10 \
        --seed_edit 10
        # -carla-rpc-port=$host_port_tm \
        
        # --net=host \

    # docker start $server
    sleep 15

    for ((i=0; i<${#edit_array[@]}; i++))
    do
        edit=$(shuf -n1 -e "${edit_array[@]}")
        echo =====================================================================
        echo "Editing $edit"
        if ! python3 $script --edit $edit --length 30 --fps 10 --port $host_port1 --tm_port $host_port_tm --output $output; then
            echo "Error occurred while editing $edit"
            continue 2  # Exit the inner for loop and start a new iteration of the while loop
        fi
        sleep 5
    done

    docker stop $server
    sleep 20
    # restart server in the next iteration
done


