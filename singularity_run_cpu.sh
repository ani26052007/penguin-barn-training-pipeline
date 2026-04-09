#!/bin/bash
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
buffer_path=${BUFFER_PATH:-${HOME}/local_buffer}
singularity exec -i -n --network=none -p \
  -B `pwd`:/jackal_ws/src/ros_jackal \
  -B ${buffer_path}:${buffer_path} \
  --env LIBGL_ALWAYS_SOFTWARE=1 \
  --env GALLIUM_DRIVER=llvmpipe \
  --env QT_QPA_PLATFORM=offscreen \
  --env DISPLAY=:99 \
  --env BUFFER_PATH=${buffer_path} \
  ${1} /bin/bash /jackal_ws/src/ros_jackal/entrypoint.sh ${@:2}
