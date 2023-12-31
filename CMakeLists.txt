cmake_minimum_required(VERSION 3.0.2)
project(bevdet)

find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
message(${YAML_CPP_LIBRARIES})
find_package(Eigen3 REQUIRED)

message("yaml -> "${YAML_CPP_INCLUDE_DIRS})

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_C_COMPILER /usr/bin/gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

set(CMAKE_BUILD_TYPE "RELEASE")

set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")

add_compile_options(-W)
add_compile_options(-std=c++14)

set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 86 87)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -ccbin ${CMAKE_CXX_COMPILER}
)

if(${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()


if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")       # orin
  set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
  set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(TENSORRT_INCLUDE_DIRS /root/TensorRT-8.5.2.2/include)  # your tensorrt path
  set(TENSORRT_LIBRARY_DIRS /root/TensorRT-8.5.2.2/lib)
endif()


set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/lib)
# find_library(NVJPEG_LIBRARY nvjpeg ${CUDA_LIB_DIRS})
# if(NVJPEG_LIBRARY)
#   add_definitions(-D__HAVE_NVJPEG__)
#   link_libraries(${NVJPEG_LIBRARY})
#   message(STATUS ${NVJPEG_LIBRARY})
# endif()


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
  message_generation  # 确保包含消息生成的依赖
)



find_package(OpenCV REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

link_directories ( ${OpenCV_LIB_DIR} )
MESSAGE(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")

include_directories ( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")



include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src/bevdet-tensorrt-cpp/include
  ${EIGEN3_INCLUDE_DIRS}
)

link_directories(
	${TENSORRT_LIBRARY_DIRS}
)


catkin_package(
  CATKIN_DEPENDS roscpp message_runtime
)

cuda_add_executable(bevdemo 
    src/bevdet-tensorrt-cpp/demo_bevdet.cpp 
    src/bevdet-tensorrt-cpp/src/bevdet.cpp 
    src/bevdet-tensorrt-cpp/src/preprocess.cu 
    src/bevdet-tensorrt-cpp/src/bevpool.cu 
    src/bevdet-tensorrt-cpp/src/iou3d_nms.cu
    src/bevdet-tensorrt-cpp/src/postprocess.cu
    src/bevdet-tensorrt-cpp/src/data.cpp
    src/bevdet-tensorrt-cpp/src/grid_sampler.cu
    src/bevdet-tensorrt-cpp/src/cpu_jpegdecoder.cpp
    src/bevdet-tensorrt-cpp/src/nvjpegdecoder.cpp
)

# find_package(JPEG REQUIRED)

# if(JPEG_FOUND)
#   message("JPEG library found   " ${JPEG_INCLUDE_DIR} " "  ${JPEG_LIBRARIES})
#   include_directories("/home/orin_uestc_1/anaconda3/include")
#   target_link_libraries(bevdemo "/home/orin_uestc_1/anaconda3/lib/libjpeg.so")
# else()
#   message(FATAL_ERROR "JPEG library not found")
# endif()



target_link_libraries(bevdemo 
    yaml-cpp 
    libnvinfer.so
    libjpeg.so
    ${OpenCV_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES}
)








# cuda_add_library(bevdemo_lib 
#     # src/bevdet-tensorrt-cpp/demo_bevdet.cpp 
#     src/bevdet-tensorrt-cpp/src/bevdet.cpp 
#     src/bevdet-tensorrt-cpp/src/preprocess.cu 
#     src/bevdet-tensorrt-cpp/src/bevpool.cu 
#     src/bevdet-tensorrt-cpp/src/iou3d_nms.cu
#     src/bevdet-tensorrt-cpp/src/postprocess.cu
#     src/bevdet-tensorrt-cpp/src/data.cpp
#     src/bevdet-tensorrt-cpp/src/grid_sampler.cu
#     src/bevdet-tensorrt-cpp/src/cpu_jpegdecoder.cpp
#     src/bevdet-tensorrt-cpp/src/nvjpegdecoder.cpp
# )
# target_link_libraries(bevdemo_lib 
#     yaml-cpp 
#     libnvinfer.so
#     libjpeg.so
#     ${OpenCV_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES}
# )
# # find_package(JPEG REQUIRED)

# # if(JPEG_FOUND)
# #   message("JPEG library found   " ${JPEG_INCLUDE_DIR} " "  ${JPEG_LIBRARIES})
# #   include_directories("/home/orin_uestc_1/anaconda3/include")
# #   target_link_libraries(bevdemo "/home/orin_uestc_1/anaconda3/lib/libjpeg.so")
# # else()
# #   message(FATAL_ERROR "JPEG library not found")
# # endif()

# add_executable(bevdemo src/bevdet-tensorrt-cpp/demo_bevdet.cpp)
# target_link_libraries(bevdemo bevdemo_lib)