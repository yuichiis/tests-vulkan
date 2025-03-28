#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Structure for array computation
typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t compute_queue_family_index;
    
    // For buffer management
    VkBuffer input_buffer;
    VkBuffer output_buffer;
    VkDeviceMemory input_memory;
    VkDeviceMemory output_memory;
    
    // Shader related
    VkShaderModule compute_shader;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    
    // Command related
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
} VulkanArrayCompute;

// // Macro for error checking
#define VK_CHECK(call) \
    do { \
        VkResult result = call; \
        if (result != VK_SUCCESS) { \
            printf("Vulkan error: %d at %s:%d\n", result, __FILE__, __LINE__); \
            return VK_FALSE; \
        } \
    } while (0)

// Compute shader template
const char* compute_shader_template = "(\n"
"\n"
"#version 450\n"
"\n"
"// Binding definitions\n"
"layout(set = 0, binding = 0) buffer InputBuffer {\n"
"    float input_data[];\n"
"};\n"
"\n"
"layout(set = 0, binding = 1) buffer OutputBuffer {\n"
"    float output_data[];\n"
"};\n"
"\n"
"// Compute shader local workgroup size\n"
"layout(local_size_x = 64) in;\n"
"\n"
"// Custom operation function\n"
"void custom_operation(uint index) {\n"
"    // Implement the operation on the input data\n"
"    // Example: Simple element-wise operation\n"
"    output_data[index] = input_data[index] * 2.0f;\n"
"}\n"
"\n"
"void main() {\n"
"    // Get the global work item index\n"
"    uint index = gl_GlobalInvocationID.x;\n"
"    \n"
"    // Execute the custom operation\n"
"    custom_operation(index);\n"
"}\n"
")";

// Create Vulkan Instance
VkBool32 create_vulkan_instance(VulkanArrayCompute* context) {
    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo){
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "ML Array Computation",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "Array Compute Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_2
        }
    };

    VK_CHECK(vkCreateInstance(&instance_info, NULL, &context->instance));
    return VK_TRUE;
}

// Select physical device
VkBool32 select_physical_device(VulkanArrayCompute* context) {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(context->instance, &device_count, NULL);
    
    if (device_count == 0) {
        printf("No supported GPU found\n");
        return VK_FALSE;
    }

    VkPhysicalDevice* devices = malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(context->instance, &device_count, devices);
    
    // Select the first physical device (proper device selection is required in a production environment)
    context->physical_device = devices[0];
    
    // Get the compute queue family index
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context->physical_device, &queue_family_count, NULL);
    
    VkQueueFamilyProperties* queue_families = malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(context->physical_device, &queue_family_count, queue_families);
    
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            context->compute_queue_family_index = i;
            break;
        }
    }
    
    free(devices);
    free(queue_families);
    
    return VK_TRUE;
}

// create_logical_device_and_queue
VkBool32 create_logical_device_and_queue(VulkanArrayCompute* context) {
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = context->compute_queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };

    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info
    };

    VK_CHECK(vkCreateDevice(context->physical_device, &device_info, NULL, &context->device));
    
    // GetDeviceQueue
    vkGetDeviceQueue(context->device, context->compute_queue_family_index, 0, &context->compute_queue);
    
    return VK_TRUE;
}

// Function for loading shader binaries
char* read_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open shader file: %s\n", filename);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    rewind(file);
    char* buffer = malloc(*size);
    if (!buffer) {
        fclose(file);
        return NULL;
    }
    fread(buffer, 1, *size, file);
    fclose(file);
    return buffer;
}

// create_shader_module
VkBool32 create_shader_module(VulkanArrayCompute* context) {
    size_t shader_size = 0;
    char* shader_code = read_file("../shaders/compute.spv", &shader_size);
    if (!shader_code) return VK_FALSE;

    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shader_size,
        .pCode = (const uint32_t*)shader_code
    };

    VkResult result = vkCreateShaderModule(context->device, &create_info, NULL, &context->compute_shader);
    free(shader_code);

    if (result != VK_SUCCESS) {
        printf("Failed to create shader module: %d\n", result);
        return VK_FALSE;
    }
    return VK_TRUE;
}

// Memory type selection
uint32_t find_memory_type(
    VkPhysicalDevice physicalDevice,
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties,
    int *type_error
) {
    *type_error = 0;

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    // 
    printf("Failed to find suitable memory type!\n");
    *type_error = -1;
    return 0;
}

// create_buffers
VkBool32 create_buffers(VulkanArrayCompute* context, size_t buffer_size) {
    // create input buffers
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = buffer_size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VK_CHECK(vkCreateBuffer(context->device, &buffer_info, NULL, &context->input_buffer));
    VK_CHECK(vkCreateBuffer(context->device, &buffer_info, NULL, &context->output_buffer));

    // Get MemoryRequirements
    VkMemoryRequirements input_mem_req, output_mem_req;
    vkGetBufferMemoryRequirements(context->device, context->input_buffer, &input_mem_req);
    vkGetBufferMemoryRequirements(context->device, context->output_buffer, &output_mem_req);

    // Memory type selection (proper memory type selection is required in a production environment)
    int type_error;
    uint32_t memory_type = find_memory_type(context->physical_device,
        input_mem_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &type_error
    );
    if(type_error!=0) {
        return VK_FALSE;
    }
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = input_mem_req.size,
        .memoryTypeIndex = memory_type // Selecting appropriate indexes for your application
    };
    
    VK_CHECK(vkAllocateMemory(context->device, &alloc_info, NULL, &context->input_memory));
    VK_CHECK(vkAllocateMemory(context->device, &alloc_info, NULL, &context->output_memory));

    // Bind buffer and memory
    VK_CHECK(vkBindBufferMemory(context->device, context->input_buffer, context->input_memory, 0));
    VK_CHECK(vkBindBufferMemory(context->device, context->output_buffer, context->output_memory, 0));

    return VK_TRUE;
}

// create_descriptor_set_layout
VkBool32 create_descriptor_set_layout(VulkanArrayCompute* context) {
    VkDescriptorSetLayoutBinding bindings[2] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        }
    };

    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = bindings
    };

    VK_CHECK(vkCreateDescriptorSetLayout(context->device, &layout_info, NULL, &context->descriptor_set_layout));
    return VK_TRUE;
}

// create_pipeline_layout
VkBool32 create_pipeline_layout(VulkanArrayCompute* context) {
    VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &context->descriptor_set_layout
    };

    VK_CHECK(vkCreatePipelineLayout(context->device, &layout_info, NULL, &context->pipeline_layout));
    return VK_TRUE;
}

// create_compute_pipeline
VkBool32 create_compute_pipeline(VulkanArrayCompute* context) {
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = context->compute_shader,
            .pName = "main"
        },
        .layout = context->pipeline_layout
    };

    VK_CHECK(vkCreateComputePipelines(context->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &context->pipeline));
    return VK_TRUE;
}

// create_command_pool_and_buffer
VkBool32 create_command_pool_and_buffer(VulkanArrayCompute* context) {
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = context->compute_queue_family_index,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };

    VK_CHECK(vkCreateCommandPool(context->device, &pool_info, NULL, &context->command_pool));

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = context->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VK_CHECK(vkAllocateCommandBuffers(context->device, &alloc_info, &context->command_buffer));
    return VK_TRUE;
}

// transfer_data_to_device
VkBool32 transfer_data_to_device(VulkanArrayCompute* context, void* input_data, size_t buffer_size) {
    void* mapped_memory;
    VK_CHECK(vkMapMemory(context->device, context->input_memory, 0, buffer_size, 0, &mapped_memory));
    memcpy(mapped_memory, input_data, buffer_size);
    vkUnmapMemory(context->device, context->input_memory);
    return VK_TRUE;
}

// execute_compute_shader
VkBool32 execute_compute_shader(VulkanArrayCompute* context, uint32_t work_group_count) {
    // Start recording the command buffer
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VK_CHECK(vkBeginCommandBuffer(context->command_buffer, &begin_info));

    // Pipeline bind
    vkCmdBindPipeline(context->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, context->pipeline);

    vkCmdBindDescriptorSets(context->command_buffer, 
        VK_PIPELINE_BIND_POINT_COMPUTE, 
        context->pipeline_layout, 
        0, 1, 
        &context->descriptor_set, 
        0, NULL);

    // Dispatch
    vkCmdDispatch(context->command_buffer, work_group_count, 1, 1);

    // End recording the command buffer
    VK_CHECK(vkEndCommandBuffer(context->command_buffer));

    // Execute the command
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &context->command_buffer
    };
    VK_CHECK(vkQueueSubmit(context->compute_queue, 1, &submit_info, VK_NULL_HANDLE));

    // Wait for execution to complete
    VK_CHECK(vkQueueWaitIdle(context->compute_queue));

    return VK_TRUE;
}

// Read the result
VkBool32 read_result(VulkanArrayCompute* context, void* output_data, size_t buffer_size) {
    void* mapped_memory;
    VK_CHECK(vkMapMemory(context->device, context->output_memory, 0, buffer_size, 0, &mapped_memory));
    memcpy(output_data, mapped_memory, buffer_size);
    vkUnmapMemory(context->device, context->output_memory);
    return VK_TRUE;
}

// Release resources
void cleanup_vulkan_context(VulkanArrayCompute* context) {
    // Release all Vulkan resources
    vkDestroyCommandPool(context->device, context->command_pool, NULL);
    vkDestroyPipeline(context->device, context->pipeline, NULL);
    vkDestroyPipelineLayout(context->device, context->pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(context->device, context->descriptor_set_layout, NULL);

    vkDestroyDescriptorPool(context->device, context->descriptor_pool, NULL);

    vkDestroyShaderModule(context->device, context->compute_shader, NULL);
    
    vkFreeMemory(context->device, context->input_memory, NULL);
    vkFreeMemory(context->device, context->output_memory, NULL);
    
    vkDestroyBuffer(context->device, context->input_buffer, NULL);
    vkDestroyBuffer(context->device, context->output_buffer, NULL);
    
    vkDestroyDevice(context->device, NULL);
    vkDestroyInstance(context->instance, NULL);
}

// --- create_descriptor_pool ---
VkBool32 create_descriptor_pool(VulkanArrayCompute* context) {
    VkDescriptorPoolSize poolSize = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 2
    };

    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
        .maxSets = 1
    };

    VK_CHECK(vkCreateDescriptorPool(context->device, &poolInfo, NULL, &context->descriptor_pool));
    return VK_TRUE;
}

// --- allocate_descriptor_set ---
VkBool32 allocate_descriptor_set(VulkanArrayCompute* context) {
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = context->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &context->descriptor_set_layout
    };

    VK_CHECK(vkAllocateDescriptorSets(context->device, &allocInfo, &context->descriptor_set));
    return VK_TRUE;
}

// --- update_descriptor_set ---
VkBool32 update_descriptor_set(VulkanArrayCompute* context, VkDeviceSize bufferSize) {
    VkDescriptorBufferInfo bufferInfos[2];
    bufferInfos[0].buffer = context->input_buffer;
    bufferInfos[0].offset = 0;
    bufferInfos[0].range = bufferSize;

    bufferInfos[1].buffer = context->output_buffer;
    bufferInfos[1].offset = 0;
    bufferInfos[1].range = bufferSize;

    VkWriteDescriptorSet descriptorWrites[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = context->descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &bufferInfos[0]
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = context->descriptor_set,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &bufferInfos[1]
        }
    };

    vkUpdateDescriptorSets(context->device, 2, descriptorWrites, 0, NULL);
    return VK_TRUE;
}

// Usage Example
int main() {
    VulkanArrayCompute context = {0};
    
    // Vulkan setup
    printf("create_vulkan_instance\n");
    if (!create_vulkan_instance(&context)) return -1;
    printf("select_physical_device\n");
    if (!select_physical_device(&context)) return -1;
    printf("create_logical_device_and_queue\n");
    if (!create_logical_device_and_queue(&context)) return -1;
    
    // Buffer size and work group settings
    const size_t BUFFER_SIZE = 1024 * sizeof(float);
    const uint32_t WORK_GROUP_COUNT = 16;  // Adjustments required
    
    // Prepare input data
    float* input_data = malloc(BUFFER_SIZE);
    float* output_data = malloc(BUFFER_SIZE);
    
    for (size_t i = 0; i < BUFFER_SIZE / sizeof(float); i++) {
        input_data[i] = (float)i;
    }
    
    // Create Vulkan resources
    printf("create_buffers\n");
    if (!create_buffers(&context, BUFFER_SIZE)) return -1;
    printf("create_shader_module\n");
    if (!create_shader_module(&context)) return -1;
    printf("create_descriptor_set_layout\n");
    if (!create_descriptor_set_layout(&context)) return -1;
    printf("create_pipeline_layout\n");
    if (!create_pipeline_layout(&context)) return -1;
    printf("create_compute_pipeline\n");
    if (!create_compute_pipeline(&context)) return -1;
    printf("create_command_pool_and_buffer\n");
    if (!create_command_pool_and_buffer(&context)) return -1;

    printf("create_descriptor_pool\n");
    if (!create_descriptor_pool(&context)) return -1;
    printf("allocate_descriptor_set\n");
    if (!allocate_descriptor_set(&context)) return -1;
    printf("update_descriptor_set\n");
    if (!update_descriptor_set(&context, BUFFER_SIZE)) return -1;    
    
    // Data transfer
    printf("transfer_data_to_device\n");
    if (!transfer_data_to_device(&context, input_data, BUFFER_SIZE)) return -1;
    
    // Execute compute shader
    printf("execute_compute_shader\n");
    if (!execute_compute_shader(&context, WORK_GROUP_COUNT)) return -1;
    
    // Read the result
    printf("read_result\n");
    if (!read_result(&context, output_data, BUFFER_SIZE)) return -1;
    
    // Display the result (for debugging)
    for (size_t i = 0; i < 10; i++) {
        printf("Output[%zu]: %f\n", i, output_data[i]);
    }
    
    // Releasing resources
    printf("cleanup_vulkan_context\n");
    cleanup_vulkan_context(&context);
    
    free(input_data);
    free(output_data);
    
    return 0;
}
