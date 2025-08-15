cmake_minimum_required(VERSION 3.0)
project(r3live_tools)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    pcl_ros
    pcl_conversions
)

find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization filesystem)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp pcl_ros pcl_conversions
  DEPENDS PCL Boost OpenCV
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}

  ${CMAKE_CURRENT_SOURCE_DIR}/../r3live/r3live/src/rgb_map

  # R3LIVE tools 下的头
  ${CMAKE_CURRENT_SOURCE_DIR}/../r3live/r3live/src/tools
)

# R3LIVE rgb_map 源文件
set(R3LIVE_RGB_MAP_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/../r3live/r3live/src/rgb_map/offline_map_recorder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../r3live/r3live/src/rgb_map/image_frame.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../r3live/r3live/src/rgb_map/pointcloud_rgbd.cpp
)

# 你的提取脚本
set(OUR_SRCS
  src/extract_frame.cpp
  # 还要把下面这个 “stub” 文件也加进来（稍后创建）
  src/r3live_tools_stubs.cpp
)

add_executable(extract_frame
  ${OUR_SRCS}
  ${R3LIVE_RGB_MAP_SRCS}
)

target_link_libraries(extract_frame
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
)

