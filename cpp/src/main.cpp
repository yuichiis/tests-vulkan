#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <stdexcept>

class VulkanArrayCompute {
private:
    // Vulkan resources
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue computeQueue;
    uint32_t computeQueueFamilyIndex;

    // Buffers and memory
    vk::Buffer inputBuffer;
    vk::Buffer outputBuffer;
    vk::DeviceMemory inputMemory;
    vk::DeviceMemory outputMemory;

    // Shader and pipeline resources
    vk::ShaderModule computeShader;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSet descriptorSet;

    // Command resources
    vk::CommandPool commandPool;
    vk::CommandBuffer commandBuffer;

    // Helper method to read shader file
    std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open shader file: " + filename);
        }

        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    // Find appropriate memory type
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && 
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type!");
    }

public:
    // Constructor and Destructor
    VulkanArrayCompute() = default;
    ~VulkanArrayCompute() {
        cleanup();
    }

    // Initialize Vulkan resources
    void initialize(size_t bufferSize) {
        createInstance();
        selectPhysicalDevice();
        createLogicalDeviceAndQueue();
        createBuffers(bufferSize);
        createShaderModule();
        createDescriptorSetLayout();
        createPipelineLayout();
        createComputePipeline();
        createCommandPoolAndBuffer();
        createDescriptorPool();
        allocateDescriptorSet();
    }

    // Detailed initialization methods
    void createInstance() {
        vk::ApplicationInfo appInfo{
            "ML Array Computation",
            VK_MAKE_VERSION(1, 0, 0),
            "Array Compute Engine",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_2
        };

        instance = vk::createInstance({
            vk::InstanceCreateFlags(),
            &appInfo
        });
    }

    void selectPhysicalDevice() {
        std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        
        if (devices.empty()) {
            throw std::runtime_error("No supported GPU found");
        }

        // Select first device (in production, use more sophisticated selection)
        physicalDevice = devices[0];

        // Find compute queue family
        auto queueFamilies = physicalDevice.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) {
                computeQueueFamilyIndex = i;
                break;
            }
        }
    }

    void createLogicalDeviceAndQueue() {
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueInfo{
            vk::DeviceQueueCreateFlags(),
            computeQueueFamilyIndex,
            1,
            &queuePriority
        };

        device = physicalDevice.createDevice({
            vk::DeviceCreateFlags(),
            1,
            &queueInfo
        });

        computeQueue = device.getQueue(computeQueueFamilyIndex, 0);
    }

    void createBuffers(size_t bufferSize) {
        vk::BufferCreateInfo bufferInfo{
            vk::BufferCreateFlags(),
            bufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::SharingMode::eExclusive
        };

        inputBuffer = device.createBuffer(bufferInfo);
        outputBuffer = device.createBuffer(bufferInfo);

        // Allocate memory
        vk::MemoryRequirements inputMemReq = device.getBufferMemoryRequirements(inputBuffer);
        vk::MemoryAllocateInfo allocInfo{
            inputMemReq.size,
            findMemoryType(
                inputMemReq.memoryTypeBits, 
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            )
        };

        inputMemory = device.allocateMemory(allocInfo);
        outputMemory = device.allocateMemory(allocInfo);

        // Bind memory to buffers
        device.bindBufferMemory(inputBuffer, inputMemory, 0);
        device.bindBufferMemory(outputBuffer, outputMemory, 0);
    }

    void createShaderModule() {
        auto shaderCode = readFile("../shaders/compute.spv");

        vk::ShaderModuleCreateInfo createInfo{
            vk::ShaderModuleCreateFlags(),
            shaderCode.size(),
            reinterpret_cast<const uint32_t*>(shaderCode.data())
        };

        computeShader = device.createShaderModule(createInfo);
    }

    void createDescriptorSetLayout() {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
        }};

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            vk::DescriptorSetLayoutCreateFlags(),
            static_cast<uint32_t>(bindings.size()),
            bindings.data()
        };

        descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
    }

    void createPipelineLayout() {
        vk::PipelineLayoutCreateInfo layoutInfo{
            vk::PipelineLayoutCreateFlags(),
            1,
            &descriptorSetLayout
        };

        pipelineLayout = device.createPipelineLayout(layoutInfo);
    }

    void createComputePipeline() {
        vk::PipelineShaderStageCreateInfo shaderStageInfo{
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            computeShader,
            "main"
        };

        vk::ComputePipelineCreateInfo pipelineInfo{
            vk::PipelineCreateFlags(),
            shaderStageInfo,
            pipelineLayout
        };

        pipeline = device.createComputePipeline(nullptr, pipelineInfo).value;
    }

    void createCommandPoolAndBuffer() {
        vk::CommandPoolCreateInfo poolInfo{
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            computeQueueFamilyIndex
        };

        commandPool = device.createCommandPool(poolInfo);

        vk::CommandBufferAllocateInfo allocInfo{
            commandPool,
            vk::CommandBufferLevel::ePrimary,
            1
        };

        commandBuffer = device.allocateCommandBuffers(allocInfo)[0];
    }

    void createDescriptorPool() {
        vk::DescriptorPoolSize poolSize{
            vk::DescriptorType::eStorageBuffer,
            2
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            vk::DescriptorPoolCreateFlags(),
            1,
            1,
            &poolSize
        };

        descriptorPool = device.createDescriptorPool(poolInfo);
    }

    void allocateDescriptorSet() {
        vk::DescriptorSetAllocateInfo allocInfo{
            descriptorPool,
            1,
            &descriptorSetLayout
        };

        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];
    }

    void updateDescriptorSet(vk::DeviceSize bufferSize) {
        std::array<vk::DescriptorBufferInfo, 2> bufferInfos{{
            {inputBuffer, 0, bufferSize},
            {outputBuffer, 0, bufferSize}
        }};

        std::array<vk::WriteDescriptorSet, 2> descriptorWrites{{
            {
                descriptorSet,
                0,
                0,
                1,
                vk::DescriptorType::eStorageBuffer,
                nullptr,
                &bufferInfos[0]
            },
            {
                descriptorSet,
                1,
                0,
                1,
                vk::DescriptorType::eStorageBuffer,
                nullptr,
                &bufferInfos[1]
            }
        }};

        device.updateDescriptorSets(descriptorWrites, {});
    }

    void transferDataToDevice(void* inputData, size_t bufferSize) {
        void* mappedMemory = device.mapMemory(inputMemory, 0, bufferSize);
        memcpy(mappedMemory, inputData, bufferSize);
        device.unmapMemory(inputMemory);
    }

    void executeComputeShader(uint32_t workGroupCount) {
        // Begin command buffer
        vk::CommandBufferBeginInfo beginInfo{
            vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        commandBuffer.begin(beginInfo);

        // Bind pipeline and descriptor sets
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, 
            pipelineLayout, 
            0, 
            {descriptorSet}, 
            {}
        );

        // Dispatch compute shader
        commandBuffer.dispatch(workGroupCount, 1, 1);

        // End command buffer
        commandBuffer.end();

        // Submit to queue
        vk::SubmitInfo submitInfo{
            0, nullptr, nullptr,
            1, &commandBuffer,
            0, nullptr
        };
        computeQueue.submit({submitInfo}, nullptr);

        // Wait for completion
        computeQueue.waitIdle();
    }

    void readResult(void* outputData, size_t bufferSize) {
        void* mappedMemory = device.mapMemory(outputMemory, 0, bufferSize);
        memcpy(outputData, mappedMemory, bufferSize);
        device.unmapMemory(outputMemory);
    }

    // Cleanup method
    void cleanup() {
        if (device) {
            device.destroyCommandPool(commandPool);
            device.destroyPipeline(pipeline);
            device.destroyPipelineLayout(pipelineLayout);
            device.destroyDescriptorSetLayout(descriptorSetLayout);
            device.destroyDescriptorPool(descriptorPool);
            device.destroyShaderModule(computeShader);
            device.freeMemory(inputMemory);
            device.freeMemory(outputMemory);
            device.destroyBuffer(inputBuffer);
            device.destroyBuffer(outputBuffer);
            device.destroy();
        }
        
        if (instance) {
            instance.destroy();
        }
    }
};

int main() {
    try {
        // Buffer size and work group settings
        const size_t BUFFER_SIZE = 1024 * sizeof(float);
        const uint32_t WORK_GROUP_COUNT = 16;  // Adjustments may be required

        // Prepare input data
        std::vector<float> inputData(BUFFER_SIZE / sizeof(float));
        std::vector<float> outputData(BUFFER_SIZE / sizeof(float));

        // Initialize input data
        for (size_t i = 0; i < inputData.size(); ++i) {
            inputData[i] = static_cast<float>(i);
        }

        // Create Vulkan compute context
        VulkanArrayCompute vulkanCompute;
        
        // Initialize Vulkan resources
        vulkanCompute.initialize(BUFFER_SIZE);

        // Update descriptor set
        vulkanCompute.updateDescriptorSet(BUFFER_SIZE);

        // Transfer input data to device
        vulkanCompute.transferDataToDevice(inputData.data(), BUFFER_SIZE);

        // Execute compute shader
        vulkanCompute.executeComputeShader(WORK_GROUP_COUNT);

        // Read result back
        vulkanCompute.readResult(outputData.data(), BUFFER_SIZE);

        // Display first 10 results (for debugging)
        for (size_t i = 0; i < 10; ++i) {
            std::cout << "Output[" << i << "]: " << outputData[i] << std::endl;
        }

    } catch (const vk::SystemError& e) {
        std::cerr << "Vulkan error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return -1;
    }

    return 0;
}