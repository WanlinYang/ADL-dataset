cmake_minimum_required(VERSION 2.8.3)
project(collector)

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES sub
#  CATKIN_DEPENDS cv_bridge image_transport libpcl-all-dev pcl_conversions pcl_ros roscpp sensor_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_definitions(
  ${PCL_DEFINITIONS}
)

add_executable(data_collect src/data_collect.cpp)
target_link_libraries(data_collect
  ${catkin_LIBRARIES}
  ${PCL_COMMON_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(data_color src/data_color.cpp)
target_link_libraries(data_color
  ${catkin_LIBRARIES}
  ${PCL_COMMON_LIBRARIES}
  ${OpenCV_LIBS}
)


add_executable(ellipse_test src/ellipse_test.cpp)
target_link_libraries(ellipse_test
  ${catkin_LIBRARIES}
  ${PCL_COMMON_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(pcl_test src/pcl_test.cpp)
target_link_libraries(pcl_test
  ${catkin_LIBRARIES}
  ${PCL_COMMON_LIBRARIES}
  ${OpenCV_LIBS}
)
