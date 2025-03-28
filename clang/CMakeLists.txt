# CMakeLists.txt for Vulkan GPU Computation Project

# Required CMake version
cmake_minimum_required(VERSION 3.12)

# Project name
project(VulkanGPUComputation C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find Vulkan package
find_package(Vulkan REQUIRED)

# Find glslc path
find_program(GLSLC_EXECUTABLE glslc)
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc が見つかりません。パスを通してください。")
endif()

# setup shader source and output path （ex: shaders/compute.comp → build/shaders/compute.spv）
set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute.comp)
set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/compute.spv)

# create shader directory if not exists
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

# compile shader with glslc
add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SRC} -o ${SHADER_SPV}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling compute shader with glslc"
)

# add compile target for shader
add_custom_target(shaders ALL DEPENDS ${SHADER_SPV})

# Add executable
add_executable(gpu_computation 
    src/main.c
    # Add other source files here if needed
)

# Include Vulkan headers
target_include_directories(gpu_computation PRIVATE 
    ${Vulkan_INCLUDE_DIRS}
)

# Link Vulkan libraries
target_link_libraries(gpu_computation 
    ${Vulkan_LIBRARIES}
)

# Optional: Add compiler-specific flags
if(MSVC)
    # Microsoft Visual C++ specific flags
    target_compile_options(gpu_computation PRIVATE 
        /W4     # Enable high warning level
        /WX     # Treat warnings as errors
    )
else()
    # GCC/Clang specific flags
    target_compile_options(gpu_computation PRIVATE 
        -Wall 
        -Wextra 
        -Werror
    )
endif()

# Optional: Specify debug/release configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory" FORCE)

# Print out Vulkan information for debugging
message(STATUS "Vulkan_FOUND: ${Vulkan_FOUND}")
message(STATUS "Vulkan_INCLUDE_DIRS: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan_LIBRARIES: ${Vulkan_LIBRARIES}")

add_dependencies(gpu_computation shaders)
