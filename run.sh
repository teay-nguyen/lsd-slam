#!/bin/bash
rm -r build
cppcheck . --enable=all
cmake -B build
cmake --build build
cp lsd-slam/media/car_pov.mp4 build
cd build
# valgrind --leak-check=yes ./LSD_SLAM_from_scratch
./LSD_SLAM_from_scratch
