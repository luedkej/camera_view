cmake_minimum_required(VERSION 3.5)
project(tf_static_pubsub)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Build
add_executable(tf_static_pubsub src/tf_static_pubsub_node.py)
ament_target_dependencies(tf_static_pubsub rclpy std_msgs geometry_msgs)

# Install
install(TARGETS tf_static_pubsub
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
