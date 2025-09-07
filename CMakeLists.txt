cmake_minimum_required(VERSION 4.0)
project(slam)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic")

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(g2o REQUIRED)

add_executable(slam
  slam/main.cc
  slam/settings.cc
  slam/util/sophus_util.cc
  slam/util/fxn.cc
  slam/global_mapping/g2o_type_sim3_sophus.cc
  slam/global_mapping/key_frame_graph.cc
  slam/model/frame_memory.cc
  slam/model/frame_pose_struct.cc
  slam/model/frame.cc
  slam/depth_estimation/depth_map_pixel_hypothesis.cc
  slam/io_wrapper/image_display.cc
  slam/tracking/reference_tracker.cc
)

target_compile_features(slam PRIVATE cxx_std_17)

set(HAVE_G2O_IMPORTED_TARGETS OFF)
if(NOT DEFINED g2o_DIR)

endif()

find_package(g2o REQUIRED COMPONENTS core stuff types_sba csparse)
if(TARGET g2o::core AND TARGET g2o::stuff)
  set(HAVE_G2O_IMPORTED_TARGETS ON)
endif()

target_include_directories(slam PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(slam PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

if(HAVE_G2O_IMPORTED_TARGETS)
  # Modern: transitive usage requirements handled by g2o targets
  target_link_libraries(slam PRIVATE
    g2o::core
    g2o::stuff
    g2o::types_sba
    g2o::solver_csparse
  )

else()
  # Fallback: plain library names (common on Ubuntu/Debian)
  # NOTE: Link order matters with non-imported libs.
  target_link_libraries(slam PRIVATE
    g2o_core
    g2o_stuff
    g2o_types_sba
    g2o_solver_csparse
  )
  
  # If linker is aggressively dropping “unused” libs, uncomment:
  # target_link_options(slam PRIVATE -Wl,--no-as-needed)
endif()