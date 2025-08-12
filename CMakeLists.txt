cmake_minimum_required(VERSION 4.0)
project(LSD_SLAM_from_scratch)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic")

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(g2o REQUIRED)

add_executable(LSD_SLAM_from_scratch lsd-slam/v1.cc)
target_include_directories(LSD_SLAM_from_scratch PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(LSD_SLAM_from_scratch PRIVATE ${OpenCV_LIBS} g2o_core Eigen3::Eigen)
