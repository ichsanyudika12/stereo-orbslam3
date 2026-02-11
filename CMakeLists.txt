cmake_minimum_required(VERSION 3.0)
project(vo_orb_slam3)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# Find packages
find_package(OpenCV REQUIRED)
find_package(ZED REQUIRED)
find_package(Pangolin REQUIRED)

# CUDA
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# ORB_SLAM3 paths
set(ORB_SLAM3_DIR "/path/to/your/ORB_SLAM3")

include_directories(
    ${ORB_SLAM3_DIR}
    ${ORB_SLAM3_DIR}/include
    ${ORB_SLAM3_DIR}/include/CameraModels
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/include
    ${ORB_SLAM3_DIR}/Thirdparty/g2o
    ${ORB_SLAM3_DIR}/Thirdparty/g2o/include
    ${ORB_SLAM3_DIR}/Thirdparty/Sophus
)

link_directories(
    ${ORB_SLAM3_DIR}/lib
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib
    ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib
)

# File source
file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/*.h
)

add_executable(main ${SOURCES})
set_target_properties(main PROPERTIES
    INSTALL_RPATH "${ORB_SLAM3_DIR}/lib;${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib;${ORB_SLAM3_DIR}/Thirdparty/g2o/lib"
)


# Include directories
target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    /usr/local/cuda/include
)

# Link libraries
target_link_libraries(main
    ${OpenCV_LIBS}
    ${ZED_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib/libg2o.so
    pthread
    dl
    cuda
    cudart
)
