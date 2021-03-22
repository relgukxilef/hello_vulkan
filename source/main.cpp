#include <iostream>
#include <memory>
#include <cstring>
#include <iterator>
#include <algorithm>
#include <fstream>

#define GLFW_INCLUDE_VULKAN
#define GLFW_VULKAN_STATIC
#include <GLFW/glfw3.h>

#include "ge1/shader_module.h"
#include "ge1/span.h"

using namespace std;

extern char _binary_shaders_triangle_vertex_glsl_spv_start;
extern char _binary_shaders_triangle_vertex_glsl_spv_end;
extern char _binary_shaders_black_fragment_glsl_spv_start;
extern char _binary_shaders_black_fragment_glsl_spv_end;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void*
) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        //throw runtime_error("vulkan error");
        // can't throw because Nsight causes errors I don't care about
        cerr << "validation layer error: " << callbackData->pMessage << endl;
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        cout << "validation layer warning: " << callbackData->pMessage << endl;
    }

    return VK_FALSE;
}

template<class T>
struct unique_range {
    unique_ptr<T[]> begin;
    T* end;
};

unique_range<char> read_file(const char* filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw runtime_error("failed to open file");
    }

    auto size = file.tellg();
    auto data = make_unique<char[]>(size);

    file.seekg(0);
    file.read(data.get(), size);

    return {std::move(data), data.get() + size};
}

struct swapchain_frame {
    VkImage image;
    VkImageView view;
    VkFramebuffer framebuffer;
    VkCommandBuffer command_buffer;
};

struct frame {
    VkSemaphore image_available_semaphore, render_finished_semaphore;
    VkFence ready_fence;
};

int main() {
    glfwInit();

    unsigned windowWidth = 1280, windowHeight = 720;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window =
        glfwCreateWindow(windowWidth, windowHeight, "Vulkan", nullptr, nullptr);

    // set up error handling
    VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
        .pUserData = nullptr
    };

    VkApplicationInfo applicationInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    // look up extensions needed by GLFW
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    // loop up supported extensions
    uint32_t supportedExtensionCount = 0;
    vkEnumerateInstanceExtensionProperties(
        nullptr, &supportedExtensionCount, nullptr
    );
    auto supportedExtensions =
        make_unique<VkExtensionProperties[]>(supportedExtensionCount);
    vkEnumerateInstanceExtensionProperties(
        nullptr, &supportedExtensionCount, supportedExtensions.get()
    );

    // check support for layers
    const char* enabledLayers[] = {
        "VK_LAYER_KHRONOS_validation",
    };

    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    auto layers = make_unique<VkLayerProperties[]>(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.get());
    for (const auto& enabledLayer : enabledLayers) {
        bool supported = false;
        for (auto i = 0u; i < layerCount; i++) {
            if (strcmp(enabledLayer, layers[i].layerName) == 0) {
                supported = true;
                break;
            }
        }
        if (!supported) {
            throw runtime_error("enabled layer not supported");
        }
    }

    // create instance
    const char *requiredExtensions[]{
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    auto extensionCount = size(requiredExtensions) + glfwExtensionCount;
    auto extensions = make_unique<const char*[]>(extensionCount);
    std::copy(
        glfwExtensions, glfwExtensions + glfwExtensionCount, extensions.get()
    );
    std::copy(
        requiredExtensions, requiredExtensions + size(requiredExtensions),
        extensions.get() + glfwExtensionCount
    );
    VkInstance instance;
    {
        VkInstanceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = &debugUtilsMessengerCreateInfo,
            .pApplicationInfo = &applicationInfo,
            .enabledLayerCount = size(enabledLayers),
            .ppEnabledLayerNames = enabledLayers,
            .enabledExtensionCount = static_cast<uint32_t>(extensionCount),
            .ppEnabledExtensionNames = extensions.get(),
        };
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw runtime_error("failed to create instance");
        }
    }

    // create debug utils messenger
    auto vkCreateDebugUtilsMessengerEXT =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT"
    );
    auto vkDestroyDebugUtilsMessengerEXT =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT"
    );
    VkDebugUtilsMessengerEXT debugUtilsMessenger;
    {
        VkDebugUtilsMessengerCreateInfoEXT createInfo{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
        };
        if (
            vkCreateDebugUtilsMessengerEXT(
                instance, &createInfo, nullptr, &debugUtilsMessenger
            ) != VK_SUCCESS
        ) {
            throw runtime_error("failed to create debug utils messenger");
        }
    }

    // create surface
    VkSurfaceKHR surface;
    if (
        glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS
    ) {
        throw runtime_error("failed to create window surface");
    }

    // look for available devices
    VkPhysicalDevice physicalDevice;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw runtime_error("no Vulkan capable GPU found");
    }
    {
        auto devices = make_unique<VkPhysicalDevice[]>(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.get());
        // TODO: check for VK_KHR_swapchain support
        physicalDevice = devices[0]; // just pick the first one for now
    }

    // look for available queue families
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice, &queueFamilyCount, nullptr
    );
    auto queueFamilies =
        make_unique<VkQueueFamilyProperties[]>(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice, &queueFamilyCount, queueFamilies.get()
    );

    uint32_t graphicsQueueFamily = -1u, presentQueueFamily = -1u;
    for (auto i = 0u; i < queueFamilyCount; i++) {
        const auto& queueFamily = queueFamilies[i];
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphicsQueueFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(
            physicalDevice, i, surface, &presentSupport
        );
        if (presentSupport) {
            presentQueueFamily = i;
        }
    }
    if (graphicsQueueFamily == -1u) {
        throw runtime_error("no suitable queue found");
    }

    // create queues and logical device
    VkDevice device;
    {
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfos[]{
            {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = graphicsQueueFamily,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            }, {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = presentQueueFamily,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            }
        };

        const char* enabledExtensionNames[] = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };

        VkPhysicalDeviceFeatures deviceFeatures{};
        VkDeviceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = size(queueCreateInfos),
            .pQueueCreateInfos = queueCreateInfos,
            .enabledExtensionCount = size(enabledExtensionNames),
            .ppEnabledExtensionNames = enabledExtensionNames,
            .pEnabledFeatures = &deviceFeatures
        };

        if (
            vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
            VK_SUCCESS
        ) {
            throw runtime_error("failed to create logical device");
        }
    }

    // retreive queues
    VkQueue graphicsQueue, presentQueue;
    vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, presentQueueFamily, 0, &presentQueue);

    // create swap chains
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        physicalDevice, surface, &capabilities
    );

    uint32_t formatCount = 0, presentModeCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(
        physicalDevice, surface, &formatCount, nullptr
    );
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice, surface, &presentModeCount, nullptr
    );
    if (formatCount == 0) {
        throw runtime_error("no surface formats supported");
    }
    if (presentModeCount == 0) {
        throw runtime_error("no surface present modes supported");
    }
    auto formats = make_unique<VkSurfaceFormatKHR[]>(formatCount);
    auto presentModes = make_unique<VkPresentModeKHR[]>(presentModeCount);

    vkGetPhysicalDeviceSurfaceFormatsKHR(
        physicalDevice, surface, &formatCount, formats.get()
    );
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice, surface, &presentModeCount, presentModes.get()
    );

    auto surfaceFormat = formats[0];
    for (auto i = 0u; i < formatCount; i++) {
        auto format = formats[i];
        if (
            format.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        ) {
            surfaceFormat = format;
        }
    }

    auto presentMode = VK_PRESENT_MODE_FIFO_KHR;
    int framebufferWidth, framebufferHeight;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    VkExtent2D extent{
        max(
            min<uint32_t>(
                framebufferWidth, capabilities.maxImageExtent.width
            ),
            capabilities.minImageExtent.width
        ),
        max(
            min<uint32_t>(
                framebufferHeight, capabilities.maxImageExtent.height
            ),
            capabilities.maxImageExtent.height
        )
    };

    uint32_t imageCount = capabilities.minImageCount;
    VkSwapchainKHR swapchain;
    {
        uint32_t queueFamilyIndices[]{
            graphicsQueueFamily, presentQueueFamily
        };
        VkSwapchainCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = imageCount,
            .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = VK_SHARING_MODE_CONCURRENT,
            .queueFamilyIndexCount = size(queueFamilyIndices),
            .pQueueFamilyIndices = queueFamilyIndices,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = presentMode,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE,
        };
        vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain);
    }

    // viewport
    VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(extent.width),
        .height = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    VkRect2D scissors{
        .offset = {0, 0},
        .extent = extent,
    };

    // load shaders
    VkShaderModule
        vertexShaderModule = ge1::create_shader_module(device, {
            &_binary_shaders_triangle_vertex_glsl_spv_start,
            &_binary_shaders_triangle_vertex_glsl_spv_end
        }),
        fragmentShaderModule = ge1::create_shader_module(device, {
            &_binary_shaders_black_fragment_glsl_spv_start,
            &_binary_shaders_black_fragment_glsl_spv_end
        });

    // create pipeline
    VkPipelineLayout pipelineLayout;
    VkRenderPass renderPass;
    VkPipeline pipeline;
    {
        VkPipelineShaderStageCreateInfo stageCreateInfos[]{
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertexShaderModule,
                .pName = "main",
            }, {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragmentShaderModule,
                .pName = "main",
            }
        };

        VkPipelineVertexInputStateCreateInfo inputStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 0,
            .vertexAttributeDescriptionCount = 0,
        };
        VkPipelineInputAssemblyStateCreateInfo assemblyStateCreateInfo{
            .sType =
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };
        VkPipelineViewportStateCreateInfo viewportStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissors,
        };
        VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };
        VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };
        VkPipelineColorBlendAttachmentState colorBlendAttachmentState{
            .blendEnable = VK_FALSE,
            .colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachmentState,
        };
        /*VkDynamicState dynamicState[]{
            VK_DYNAMIC_STATE_VIEWPORT,
        };
        VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 1,
            .pDynamicStates = dynamicState,
        };*/
        VkPipelineLayoutCreateInfo layoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        };
        if (
            vkCreatePipelineLayout(
                device, &layoutCreateInfo, nullptr, &pipelineLayout
            ) != VK_SUCCESS
        ) {
            throw runtime_error("failed to create pipeline layout");
        }
        VkAttachmentDescription colorAttachment{
            .format = surfaceFormat.format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };
        VkAttachmentReference colorAttachmentReference{
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentReference,
        };
        VkSubpassDependency dependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };
        VkRenderPassCreateInfo renderPassCreateInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };
        if (
            vkCreateRenderPass(
                device, &renderPassCreateInfo, nullptr, &renderPass
            ) != VK_SUCCESS
        ) {
            throw runtime_error("failed to create render pass");
        }

        VkGraphicsPipelineCreateInfo pipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = size(stageCreateInfos),
            .pStages = stageCreateInfos,
            .pVertexInputState = &inputStateCreateInfo,
            .pInputAssemblyState = &assemblyStateCreateInfo,
            .pViewportState = &viewportStateCreateInfo,
            .pRasterizationState = &rasterizationStateCreateInfo,
            .pMultisampleState = &multisampleStateCreateInfo,
            .pColorBlendState = &colorBlendStateCreateInfo,
            .layout = pipelineLayout,
            .renderPass = renderPass,
            .subpass = 0,
        };
        if (
            vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr,
                &pipeline
            ) != VK_SUCCESS
        ) {
            throw runtime_error("failed to create pipeline");
        }
    }

    // create command pool
    VkCommandPool commandPool;
    {
        VkCommandPoolCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = graphicsQueueFamily,
        };
        if (
            vkCreateCommandPool(device, &createInfo, nullptr, &commandPool) !=
            VK_SUCCESS
        ) {
            throw runtime_error("failed to create command pool");
        }
    }

    // create swapchain frame data
    ge1::unique_span<swapchain_frame> swapchain_frames(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    {
        // command buffers
        auto commandBuffers = make_unique<VkCommandBuffer[]>(imageCount);
        {
            VkCommandBufferAllocateInfo allocateInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = commandPool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = imageCount,
            };
            if (
                vkAllocateCommandBuffers(
                    device, &allocateInfo, commandBuffers.get()
                ) != VK_SUCCESS
            ) {
                throw runtime_error("failed to allocate command buffers");
            }
        }

        ge1::unique_span<VkImage> images(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.begin());
        for (auto i = 0u; i < swapchain_frames.size(); i++) {
            auto& swapchain_frame = swapchain_frames[i];
            swapchain_frame.image = images[i];
            swapchain_frame.command_buffer = commandBuffers[i];

            // views
            {
                VkImageViewCreateInfo createInfo{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = swapchain_frame.image,
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = surfaceFormat.format,
                    .subresourceRange{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    }
                };
                vkCreateImageView(
                    device, &createInfo, nullptr, &swapchain_frame.view
                );
            }

            // framebuffers
            VkImageView attachments[] = {swapchain_frame.view};
            {
                VkFramebufferCreateInfo createInfo{
                    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    .renderPass = renderPass,
                    .attachmentCount = 1,
                    .pAttachments = attachments,
                    .width = extent.width,
                    .height = extent.height,
                    .layers = 1,
                };

                if (
                    vkCreateFramebuffer(
                        device, &createInfo, nullptr,
                        &swapchain_frame.framebuffer
                    ) != VK_SUCCESS
                ) {
                    throw runtime_error("failed to create framebuffer");
                }
            }

            // write commands
            VkCommandBufferBeginInfo bufferBeginInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            };
            if (
                vkBeginCommandBuffer(
                    swapchain_frame.command_buffer, &bufferBeginInfo
                ) != VK_SUCCESS
            ) {
                throw runtime_error("failed to begin recording command buffer");
            }
            VkClearValue clearValue{{{1.0f, 1.0f, 1.0f, 1.0f}}};
            VkRenderPassBeginInfo renderPassBeginInfo{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = renderPass,
                .framebuffer = swapchain_frame.framebuffer,
                .renderArea{
                    .offset = {0, 0},
                    .extent = extent,
                },
                .clearValueCount = 1,
                .pClearValues = &clearValue,
            };
            vkCmdBeginRenderPass(
                swapchain_frame.command_buffer, &renderPassBeginInfo,
                VK_SUBPASS_CONTENTS_INLINE
            );

            vkCmdBindPipeline(
                swapchain_frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline
            );
            vkCmdDraw(swapchain_frame.command_buffer, 3, 1, 0, 0);
            vkCmdEndRenderPass(swapchain_frame.command_buffer);

            if (
                vkEndCommandBuffer(swapchain_frame.command_buffer) != VK_SUCCESS
            ) {
                throw runtime_error("failed to record command buffer");
            }
        }
    }

    // create frame data
    unsigned frames_in_flight = 2;
    ge1::unique_span<frame> frames(frames_in_flight);
    for (auto i = 0u; i < frames.size(); i++) {
        auto& frame = frames[i];

        // create semaphores
        VkSemaphoreCreateInfo semaphore_create_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        VkFenceCreateInfo fence_create_info{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        if (
            vkCreateSemaphore(
                device, &semaphore_create_info, nullptr,
                &frame.image_available_semaphore
            ) != VK_SUCCESS ||
            vkCreateSemaphore(
                device, &semaphore_create_info, nullptr,
                &frame.render_finished_semaphore
            ) != VK_SUCCESS ||
            vkCreateFence(
                device, &fence_create_info, nullptr, &frame.ready_fence
            )
        ) {
            throw runtime_error("failed to create synchronisation objects");
        }
    }

    unsigned frame_index = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        auto& frame = frames[frame_index];

        vkWaitForFences(device, 1, &frame.ready_fence, VK_TRUE, -1ul);
        vkResetFences(device, 1, &frame.ready_fence);

        // get next image from swapchain
        uint32_t image_index;
        vkAcquireNextImageKHR(
            device, swapchain, -1ul, frame.image_available_semaphore,
            VK_NULL_HANDLE,
            &image_index
        );
        auto& swapchain_frame = swapchain_frames[image_index];

        // submit command buffer
        VkSemaphore waitSemaphores[]{frame.image_available_semaphore};
        VkPipelineStageFlags waitStages[]{
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        };
        VkSemaphore signalSemaphores[]{
            frame.render_finished_semaphore
        };
        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = waitSemaphores,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &swapchain_frame.command_buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = signalSemaphores,
        };
        if (
            vkQueueSubmit(graphicsQueue, 1, &submitInfo, frame.ready_fence) !=
            VK_SUCCESS
        ) {
            throw runtime_error("failed to submit draw command buffer");
        }

        // present image
        VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = signalSemaphores,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_index,
        };
        vkQueuePresentKHR(presentQueue, &presentInfo);

        frame_index = (frame_index + 1) % frames.size();

        // TODO: swapchain doesn't necessarily sync with current monitor
        // use VK_KHR_display to wait for vsync of current display
    }


    for (auto& frame : frames) {
        vkWaitForFences(device, 1, &frame.ready_fence, VK_TRUE, -1ul);
        vkDestroySemaphore(device, frame.image_available_semaphore, nullptr);
        vkDestroySemaphore(device, frame.render_finished_semaphore, nullptr);
        vkDestroyFence(device, frame.ready_fence, nullptr);
    }

    for (auto& swapchain_frame : swapchain_frames) {
        vkDestroyFramebuffer(device, swapchain_frame.framebuffer, nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    vkDestroyShaderModule(device, vertexShaderModule, nullptr);
    vkDestroyShaderModule(device, fragmentShaderModule, nullptr);

    for (auto i = 0u; i < imageCount; i++) {
        vkDestroyImageView(device, swapchain_frames[i].view, nullptr);
    }

    vkDestroySwapchainKHR(device, swapchain, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
}
