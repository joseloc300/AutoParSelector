cmake_minimum_required(VERSION 3.5)
project(ep_w)

# If Clava CMake plugin is not installed (e.g., sudo clava-update), the path to the plugin
# can be manually specified by setting the variable Clava_DIR
# The Clava CMake plugin can be found here: https://github.com/specs-feup/clava/tree/master/CMake
#set(Clava_DIR <CMAKE_CLAVA_PLUGIN>)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -O3 -mcmodel=medium")

set(SOURCE_FILES
	NAS_EP.c
)


add_executable(ep_w ${SOURCE_FILES})

target_link_libraries(ep_w m)
	
# Clava section

find_package(Clava REQUIRED)

# Call AutoPar
clava_enable_openmp()
clava_weave(ep_w ../NasAutopar.lara)

clava_weave(ep_w ../CheckOpenMPPragmas.lara ARGS "expectedCodeFile:'${CMAKE_CURRENT_SOURCE_DIR}/expected_output.c'")

# Test

add_custom_command(
     TARGET ep_w
     COMMENT "Running compiled application"
     POST_BUILD 
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
     COMMAND ep_w
)