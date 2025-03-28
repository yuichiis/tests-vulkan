# CMakeLists.txt for Vulkan GPU Computation Project (C++ version)

# Required CMake version
cmake_minimum_required(VERSION 3.12)

# Project name and language
project(VulkanGPUComputation CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan package
find_package(Vulkan REQUIRED)

# Find glslc path
find_program(GLSLC_EXECUTABLE glslc)
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found, please put it in your path.")
endif()

# Setup shader source and output path （例: shaders/compute.comp → build/shaders/compute.spv）
set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute.comp)
set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/compute.spv)

# Create shader directory if not exists
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

# Compile shader with glslc
add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SRC} -o ${SHADER_SPV}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling compute shader with glslc"
)

# Add custom target for shader
add_custom_target(shaders ALL DEPENDS ${SHADER_SPV})

# Add executable
add_executable(gpu_computation 
    src/main.cpp
    # If you have other source files, add them here
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

# Ensure shaders are built before the executable
add_dependencies(gpu_computation shaders)

# Optional: Specify debug/release configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory" FORCE)

# Print out Vulkan information for debugging
message(STATUS "Vulkan_FOUND: ${Vulkan_FOUND}")
message(STATUS "Vulkan_INCLUDE_DIRS: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan_LIBRARIES: ${Vulkan_LIBRARIES}")

