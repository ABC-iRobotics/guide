cmake_minimum_required(VERSION 3.8)
project(guide_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages

  # Services
  "srv/Attribute.srv"
  "srv/CheckSuccess.srv"
  "srv/Collision.srv"
  "srv/Demonstration.srv"
  "srv/Dimensions.srv"
  "srv/Pose.srv"
  "srv/Prompt.srv"
  "srv/Randomize.srv"
  "srv/RegisterScene.srv"
  "srv/SetJoint.srv"
  "srv/StartRecording.srv"
  "srv/StopRecording.srv"
  "srv/FinalizeRecording.srv"

  # Actions
  "action/Demonstration.action"

  # Dependencies
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
