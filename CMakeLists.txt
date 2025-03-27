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
