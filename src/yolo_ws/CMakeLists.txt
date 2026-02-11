cmake_minimum_required(VERSION 3.5)
project(yolo_ws)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime root directory")

if(NOT ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Please set ONNXRUNTIME_DIR. Example:
  colcon build --cmake-args -DONNXRUNTIME_DIR=/path/to/onnxruntime")
endif()

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)

if(NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR} OR NOT EXISTS ${ONNXRUNTIME_LIB})
  message(FATAL_ERROR "ONNX Runtime not found in ${ONNXRUNTIME_DIR}")
endif()

include_directories(
  include
  include/yolo_ws
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)

add_executable(main
  src/main.cpp
  src/yolov8Predictor.cpp
  src/utils.cpp
)

target_link_libraries(main
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIB}
)

ament_target_dependencies(main
  rclcpp
  geometry_msgs
)

install(TARGETS main
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()