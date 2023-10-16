cmake_minimum_required(VERSION 3.18.0) # for CUDA complaint about CUDA compute capability
project(cmakeLibQuickStart VERSION 0.1.0 LANGUAGES C CXX CUDA)

# for concepts 
set(CMAKE_CXX_STANDARD 20)

include(CTest)
enable_testing()

include_directories(src/include)

add_library(cudaKernels src/lib/kernels.cu)
add_executable(quickStart cmakeLibQuickStart.cu)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_link_libraries(quickStart cudaKernels)
