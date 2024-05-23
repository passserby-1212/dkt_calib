cmake_minimum_required(VERSION 3.22)
project(Calibration)

find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加apriltag库
find_package(apriltag REQUIRED)
include_directories(${APRILTAG_INCLUDE_DIRS})

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARIES_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

file(GLOB_RECURSE PARSER_PATH src/*.cpp)
#list(REMOVE_ITEM PARSER_PATH ${PROJECT_SOURCE_DIR}/src/run_intrinsic_calibration.cpp)

add_library(${PROJECT_NAME} STATIC ${PARSER_PATH})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} apriltag)

# add_executable(camera_calibration src/run_intrinsic_calib.cpp )
# target_link_libraries(camera_calibration ${PROJECT_NAME})

# add_executable(single_lidar_cam_calib src/run_single_extrinsic_calib.cpp )
# target_link_libraries(single_lidar_cam_calib ${PROJECT_NAME})

add_executable(group_lidar_cam_calib src/run_group_extrinsic_calib.cpp )
target_link_libraries(group_lidar_cam_calib ${PROJECT_NAME})

# add_executable(lidars_calib src/run_lidars_calib.cpp )
# target_link_libraries(lidars_calib ${PROJECT_NAME})
