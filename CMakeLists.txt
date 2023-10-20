cmake_minimum_required(VERSION 3.18.0) # for CUDA complaint about CUDA compute capability
project(cudaMinors VERSION 0.1.0 LANGUAGES C CXX CUDA)

# for concepts 
set(CMAKE_CXX_STANDARD 14)

include_directories(src/include)

add_library(cudaKernels 
    src/lib/kernels.cu
    src/lib/device_minors.cu
)
add_executable(takeMinors cudaMinors.cu)

add_subdirectory(test)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_link_libraries(takeMinors cudaKernels)


include(CTest)
enable_testing()
add_test(NAME test_simple_dets COMMAND test_simple_dets)
