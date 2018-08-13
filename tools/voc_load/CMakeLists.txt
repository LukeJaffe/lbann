project(voc_load)
cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0054 NEW)

set(COMPILER "gnu")
#set(COMPILER "clang")
#set(CLUSTER "catalyst")
#set(CLUSTER "surface")
#set(CLUSTER "quartz")
set(CLUSTER "pascal")

set(LBANN_DIR ../..)
set(LBANN_INSTALL_DIR ${LBANN_DIR}/build/${COMPILER}.Release.${CLUSTER}.llnl.gov/install)
include(${LBANN_DIR}/cmake/modules/SetupMPI.cmake)
include(${LBANN_DIR}/cmake/modules/SetupOpenMP.cmake)
# NEW
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(VOC_LOAD_EXE voc_load)
set(VOC_LOAD_SRCS voc_load.cpp)
set(WITH_OPENCL OFF)

add_definitions(-Wall)
add_definitions(-O2)
add_definitions(-fopenmp)
add_definitions(-g)
add_definitions(-std=c++11)
add_definitions(-D_JAG_OFFLINE_TOOL_MODE_)
add_definitions(-DLBANN_HAS_OPENCV)


list(APPEND OpenCV_DIR /usr/local/tools/opencv-3.0.0)
list(APPEND OpenCV_DIR /usr)
find_package(OpenCV QUIET HINTS ${OpenCV_DIR})
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")

if(NOT OpenCV_FOUND)
  set(OpenCV_DIR ${LBANN_INSTALL_DIR})
  set(OpenCV_LIBS "libopencv_highgui.so;libopencv_imgproc.so;libopencv_imgcodecs.so;libopencv_core.so")
  set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")
  set(OpenCV_LIB_DIR "${OpenCV_DIR}/lib")
  message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
endif()

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})


find_package(MPI REQUIRED)
message(STATUS "Found MPI: ${MPI_CXX_COMPILER} ${MPI_C_COMPILER} ${MPI_Fortran_COMPILER}")
include_directories(${MPI_CXX_INCLUDE_PATH})


file(GLOB VOC_LOAD_DEPEND_SRCS
     data_reader_csv.cpp
     data_reader_voc.cpp
     lbann/utils/random.cpp)

 add_executable(${VOC_LOAD_EXE} ${VOC_LOAD_SRCS} ${VOC_LOAD_DEPEND_SRCS})
 target_link_libraries(${VOC_LOAD_EXE} ${OpenCV_LIBS} ${CONDUIT_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
