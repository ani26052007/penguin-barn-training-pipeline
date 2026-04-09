#!/bin/bash
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export QT_QPA_PLATFORM=offscreen
export BUFFER_PATH=$(pwd)/local_buffer

echo "Starting 20 actors..."
for i in $(seq 0 19); do
    ./singularity_run_cpu.sh local_buffer/nav_benchmark.sif \
        python3 actor.py --id $i --num_actors 20 \
        > logs/actor_$i.log 2>&1 &
    echo "  Actor $i started (PID $!)"
    sleep 2   # stagger starts so they don't all hit ROS master at once
done

echo "All actors launched. Starting learner..."
python train.py --config configs/reinflow.yaml \
    --checkpoint checkpoints/iql_model_node_ready.pt \
    --buffer_path $(pwd)/local_buffer
