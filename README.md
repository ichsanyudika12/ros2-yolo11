### Prerequisites

- ROS 2 Humble

- OpenCV 

- ONNX Runtime (C++ API)

- cmake, colcon, rosdep

### Model Setup

- models/yolov11.onnx

- models/coconames.txt

### Build & Run

Clone

    git clone https://github.com/ichsanyudika/ros2-yolo11.git
    cd ros2-yolo11

Build

    colcon build --symlink-install

Source

    source install/setup.bash

Run

    ros2 run yolo_ws main

### Result

![](img/img.png)
