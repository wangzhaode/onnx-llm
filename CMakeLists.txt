cmake_minimum_required(VERSION 3.5)
project(onnx-llm)

option(BUILD_FOR_ANDROID "Build for android whith mini memory mode." OFF)
option(LLM_SUPPORT_VISION "Llm model support vision input." OFF)
option(DUMP_PROFILE_INFO "Dump profile info when chat." OFF)
option(BUILD_JNI "Build JNI for android app." OFF)

if (DUMP_PROFILE_INFO)
    add_definitions(-DDUMP_PROFILE_INFO)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()


set(ONNXRUNTIME_PATH ${CMAKE_SOURCE_DIR}/3rd_party/onnxruntime)

link_directories(${ONNXRUNTIME_PATH}/lib)
include_directories(${ONNXRUNTIME_PATH}/include
                    ${CMAKE_SOURCE_DIR}/include)


FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

add_library(llm STATIC ${SRCS})
target_link_libraries(llm onnxruntime)

add_executable(cli_demo ${CMAKE_SOURCE_DIR}/demo/cli_demo.cpp)
target_link_libraries(cli_demo llm)