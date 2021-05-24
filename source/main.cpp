#include <iostream>
#include <memory>
#include <cstring>
#include <iterator>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <array>

#define GLFW_INCLUDE_VULKAN
#define GLFW_VULKAN_STATIC
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "ge1/shader_module.h"
#include "ge1/span.h"
#include "ge1/memory.h"

using namespace std;

extern char _binary_shaders_solid_vertex_glsl_spv_start;
extern char _binary_shaders_solid_vertex_glsl_spv_end;
extern char _binary_shaders_solid_fragment_glsl_spv_start;
extern char _binary_shaders_solid_fragment_glsl_spv_end;

extern float _binary_models_miku_vertices_vbo_start;
extern float _binary_models_miku_vertices_vbo_end;
extern unsigned _binary_models_miku_faces_vbo_start;
extern unsigned _binary_models_miku_faces_vbo_end;


static float matrices[]{
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
};

enum binding : uint32_t {
    vertices, instances
};

static VkVertexInputBindingDescription vertex_binding_descriptions[] {
    {
        // vertices
        .binding = vertices,
        .stride = sizeof(float) * 8,
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    }, {
        // instances
        .binding = instances,
        .stride = sizeof(float) * 16,
        .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE,
    },
};

enum attribute : uint32_t {
    position, normal,
    model_0, model_1, model_2, model_3
};

static VkVertexInputAttributeDescription vertex_attribute_descriptions[]{
    {
        .location = position,
        .binding = vertices,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = 0,
    }, {
        .location = normal,
        .binding = vertices,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = sizeof(float) * 3,
    }, {
        .location = model_0,
        .binding = instances,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = 0,
    }, {
        .location = model_1,
        .binding = instances,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = sizeof(float) * 4,
    }, {
        .location = model_2,
        .binding = instances,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = sizeof(float) * 8,
    }, {
        .location = model_3,
        .binding = instances,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = sizeof(float) * 12,
    },
};

VkSampleCountFlagBits max_sample_count;

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
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

struct swapchain_frame {
    // can't have image_available_semaphore here because calls to
    // vkAcquireNextImageKHR require a semaphore, but at that point it's not
    // known which image will be used
    VkImageView view;
    // TODO: for number of in-flight frames to be independent of number of
    // swapchain images, there would have to be one framebuffer and command
    // buffer per combination of swapchain_frame and render frame
    VkFramebuffer framebuffer;
    VkCommandBuffer command_buffer;
    VkImage color_image;
    VkImageView color_image_view;
    VkDeviceMemory color_memory;
    VkImage depth_image;
    VkImageView depth_image_view;
    VkDeviceMemory depth_memory;
};

struct frame_semaphores {
    // number of swapchain_frames is determined by GPU,
    // number of in-flight frames is not
    // there can't be more in-flight frames than swapchain_frames
    VkSemaphore image_available_semaphore, render_finished_semaphore;
    VkFence ready_fence;
};

struct scene {
    VkBuffer static_buffer;

    uint64_t vertex_offset, face_offset, instance_offset;
    uint32_t index_count;
};

struct display_size {
    // TODO: find better name
    VkSurfaceCapabilitiesKHR capabilities;
    ge1::unique_span<swapchain_frame> swapchain_frames;
    VkExtent2D extent;
    VkSwapchainKHR swapchain;
    VkViewport viewport;
    VkRect2D scissors;
};

void create_display_size(
    int framebuffer_width, int framebuffer_height,
    VkDevice device,
    VkPhysicalDevice physical_device,
    uint32_t graphics_queue_family, uint32_t present_queue_family,
    VkSurfaceKHR surface, VkSurfaceFormatKHR surface_format,
    VkCommandPool command_pool, scene scene,
    VkRenderPass render_pass, VkPipeline pipeline,
    display_size& display_size
) {
    // NOTE: capabilities change with window size
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        physical_device, surface, &display_size.capabilities
    );

    display_size.swapchain_frames = ge1::unique_span<swapchain_frame>(
        display_size.capabilities.minImageCount
    );

    auto present_mode = VK_PRESENT_MODE_FIFO_KHR;
    display_size.extent = {
        max(
            min<uint32_t>(
                framebuffer_width,
                display_size.capabilities.maxImageExtent.width
            ),
            display_size.capabilities.minImageExtent.width
        ),
        max(
            min<uint32_t>(
                framebuffer_height,
                display_size.capabilities.maxImageExtent.height
            ),
            display_size.capabilities.minImageExtent.height
        )
    };

    {
        uint32_t queue_family_indices[]{
            graphics_queue_family, present_queue_family
        };
        VkSwapchainCreateInfoKHR create_info{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = display_size.swapchain_frames.size(),
            .imageFormat = surface_format.format,
            .imageColorSpace = surface_format.colorSpace,
            .imageExtent = display_size.extent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = VK_SHARING_MODE_CONCURRENT,
            .queueFamilyIndexCount = size(queue_family_indices),
            .pQueueFamilyIndices = queue_family_indices,
            .preTransform = display_size.capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = present_mode,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE,
        };
        vkCreateSwapchainKHR(
            device, &create_info, nullptr, &display_size.swapchain
        );
    }

    // viewport
    display_size.viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(display_size.extent.width),
        .height = static_cast<float>(display_size.extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    display_size.scissors = {
        .offset = {0, 0},
        .extent = display_size.extent,
    };

    // create swapchain frame data
    {
        uint32_t image_count;
        vkGetSwapchainImagesKHR(
            device, display_size.swapchain, &image_count, nullptr
        );
        // TODO: create swapchain after actual image count is known
        assert(image_count == display_size.swapchain_frames.size());

        // command buffers
        auto commandBuffers = make_unique<VkCommandBuffer[]>(
            display_size.swapchain_frames.size()
        );
        {
            VkCommandBufferAllocateInfo allocate_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = command_pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = display_size.swapchain_frames.size(),
            };
            if (
                vkAllocateCommandBuffers(
                    device, &allocate_info, commandBuffers.get()
                ) != VK_SUCCESS
            ) {
                throw runtime_error("failed to allocate command buffers");
            }
        }

        ge1::unique_span<VkImage> images(display_size.swapchain_frames.size());
        vkGetSwapchainImagesKHR(
            device, display_size.swapchain, &image_count, images.begin()
        );
        for (auto i = 0u; i < display_size.swapchain_frames.size(); i++) {
            auto& swapchain_frame = display_size.swapchain_frames[i];
            auto image = images[i];
            swapchain_frame.command_buffer = commandBuffers[i];

            VkImageCreateInfo color_image_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = surface_format.format,
                .extent = {
                    display_size.extent.width, display_size.extent.height, 1
                },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = max_sample_count,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage =
                    VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };

            if (
                vkCreateImage(
                    device, &color_image_info, nullptr,
                    &swapchain_frame.color_image
                ) != VK_SUCCESS
            ) {
                throw std::runtime_error("failed to create image");
            }

            VkMemoryRequirements memory_requirements;
            vkGetImageMemoryRequirements(
                device, swapchain_frame.color_image, &memory_requirements
            );

            swapchain_frame.color_memory = ge1::allocate_memory(
                device, physical_device, memory_requirements,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );

            vkBindImageMemory(
                device, swapchain_frame.color_image,
                swapchain_frame.color_memory, 0
            );

            // views
            {
                VkImageViewCreateInfo create_info{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = image,
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = surface_format.format,
                    .subresourceRange{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    }
                };
                vkCreateImageView(
                    device, &create_info, nullptr, &swapchain_frame.view
                );
            }

            {
                VkImageViewCreateInfo create_info{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = swapchain_frame.color_image,
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = surface_format.format,
                    .subresourceRange{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    }
                };
                vkCreateImageView(
                    device, &create_info, nullptr,
                    &swapchain_frame.color_image_view
                );
            }

            // framebuffers
            VkImageView attachments[] = {
                swapchain_frame.color_image_view, swapchain_frame.view
            };
            {
                VkFramebufferCreateInfo create_info{
                    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    .renderPass = render_pass,
                    .attachmentCount = std::size(attachments),
                    .pAttachments = attachments,
                    .width = display_size.extent.width,
                    .height = display_size.extent.height,
                    .layers = 1,
                };

                if (
                    vkCreateFramebuffer(
                        device, &create_info, nullptr,
                        &swapchain_frame.framebuffer
                    ) != VK_SUCCESS
                ) {
                    throw runtime_error("failed to create framebuffer");
                }
            }

            // write commands
            auto& command_buffer = swapchain_frame.command_buffer;
            VkCommandBufferBeginInfo buffer_begin_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            };
            if (
                vkBeginCommandBuffer(
                    command_buffer, &buffer_begin_info
                ) != VK_SUCCESS
            ) {
                throw runtime_error("failed to begin recording command buffer");
            }
            VkClearValue clearValue{{{1.0f, 1.0f, 1.0f, 1.0f}}};
            VkRenderPassBeginInfo render_pass_begin_info{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = render_pass,
                .framebuffer = swapchain_frame.framebuffer,
                .renderArea{
                    .offset = {0, 0},
                    .extent = display_size.extent,
                },
                .clearValueCount = 1,
                .pClearValues = &clearValue,
            };
            vkCmdBeginRenderPass(
                command_buffer, &render_pass_begin_info,
                VK_SUBPASS_CONTENTS_INLINE
            );

            vkCmdBindPipeline(
                command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline
            );

            vkCmdSetViewport(command_buffer, 0, 1, &display_size.viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &display_size.scissors);
            VkBuffer vertex_buffers[] = {
                scene.static_buffer, scene.static_buffer,
            };
            VkDeviceSize offsets[] = {
                scene.vertex_offset, scene.instance_offset
            };
            vkCmdBindVertexBuffers(
                command_buffer, 0,
                size(vertex_buffers), vertex_buffers, offsets
            );
            vkCmdBindIndexBuffer(
                command_buffer, scene.static_buffer, scene.face_offset,
                VK_INDEX_TYPE_UINT32
            );

            vkCmdDrawIndexed(
                command_buffer, scene.index_count, 1, 0, 0, 0
            );
            vkCmdEndRenderPass(command_buffer);

            if (
                vkEndCommandBuffer(command_buffer) != VK_SUCCESS
            ) {
                throw runtime_error("failed to record command buffer");
            }
        }
    }
}

void destroy_display_size(
    VkDevice device,
    const display_size& display_size
) {
    for (auto& swapchain_frame : display_size.swapchain_frames) {
        vkDestroyFramebuffer(device, swapchain_frame.framebuffer, nullptr);
    }

    for (auto& swapchain_frame : display_size.swapchain_frames) {
        vkDestroyImageView(device, swapchain_frame.view, nullptr);
        vkDestroyImageView(device, swapchain_frame.color_image_view, nullptr);
        vkDestroyImage(device, swapchain_frame.color_image, nullptr);
        vkFreeMemory(device, swapchain_frame.color_memory, nullptr);
    }

    vkDestroySwapchainKHR(device, display_size.swapchain, nullptr);
}

int main() {
    glfwInit();

    unsigned windowWidth = 1280, windowHeight = 720;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
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
        .pfnUserCallback = debug_callback,
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
    uint32_t glfw_extension_count = 0;
    auto glfw_extensions =
        glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    // loop up supported extensions
    uint32_t supported_extension_count = 0;
    vkEnumerateInstanceExtensionProperties(
        nullptr, &supported_extension_count, nullptr
    );
    auto supported_extensions =
        make_unique<VkExtensionProperties[]>(supported_extension_count);
    vkEnumerateInstanceExtensionProperties(
        nullptr, &supported_extension_count, supported_extensions.get()
    );

    // check support for layers
    const char* enabled_layers[]{
        "VK_LAYER_KHRONOS_validation",
    };

    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    auto layers = make_unique<VkLayerProperties[]>(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layers.get());
    for (const char* enabled_layer : enabled_layers) {
        bool supported = false;
        for (auto i = 0u; i < layer_count; i++) {
            auto equal = strcmp(enabled_layer, layers[i].layerName);
            if (equal == 0) {
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
    auto extensionCount = size(requiredExtensions) + glfw_extension_count;
    auto extensions = make_unique<const char*[]>(extensionCount);
    std::copy(
        glfw_extensions, glfw_extensions + glfw_extension_count,
        extensions.get()
    );
    std::copy(
        requiredExtensions, requiredExtensions + size(requiredExtensions),
        extensions.get() + glfw_extension_count
    );
    VkInstance instance;
    {
        VkInstanceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = &debugUtilsMessengerCreateInfo,
            .pApplicationInfo = &applicationInfo,
            .enabledLayerCount = std::size(enabled_layers),
            .ppEnabledLayerNames = enabled_layers,
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
            .pfnUserCallback = debug_callback,
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
    VkPhysicalDevice physical_device;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw runtime_error("no Vulkan capable GPU found");
    }
    {
        auto devices = make_unique<VkPhysicalDevice[]>(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.get());
        // TODO: check for VK_KHR_swapchain support
        physical_device = devices[0]; // just pick the first one for now
    }

    // get properties of physical device
    max_sample_count = VK_SAMPLE_COUNT_1_BIT;
    {
        VkPhysicalDeviceProperties physical_device_properties;
        vkGetPhysicalDeviceProperties(
            physical_device, &physical_device_properties
        );

        VkSampleCountFlags sample_count_falgs =
            physical_device_properties.limits.framebufferColorSampleCounts &
            physical_device_properties.limits.framebufferDepthSampleCounts &
            physical_device_properties.limits.framebufferStencilSampleCounts;

        for (auto bit : {
            VK_SAMPLE_COUNT_64_BIT, VK_SAMPLE_COUNT_32_BIT,
            VK_SAMPLE_COUNT_16_BIT, VK_SAMPLE_COUNT_8_BIT,
            VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_2_BIT,
        }) {
            if (sample_count_falgs & bit) {
                max_sample_count = bit;
                break;
            }
        }
    }
    assert(max_sample_count != VK_SAMPLE_COUNT_1_BIT);
    std::cout << max_sample_count << std::endl;

    // look for available queue families
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device, &queueFamilyCount, nullptr
    );
    auto queueFamilies =
        make_unique<VkQueueFamilyProperties[]>(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device, &queueFamilyCount, queueFamilies.get()
    );

    uint32_t graphicsQueueFamily = -1u, presentQueueFamily = -1u;
    for (auto i = 0u; i < queueFamilyCount; i++) {
        const auto& queueFamily = queueFamilies[i];
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphicsQueueFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(
            physical_device, i, surface, &presentSupport
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
            vkCreateDevice(physical_device, &createInfo, nullptr, &device) !=
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
    uint32_t formatCount = 0, presentModeCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(
        physical_device, surface, &formatCount, nullptr
    );
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device, surface, &presentModeCount, nullptr
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
        physical_device, surface, &formatCount, formats.get()
    );
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device, surface, &presentModeCount, presentModes.get()
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

    // load shaders
    VkShaderModule
        vertex_shader_module = ge1::create_shader_module(device, {
            &_binary_shaders_solid_vertex_glsl_spv_start,
            &_binary_shaders_solid_vertex_glsl_spv_end
        }),
        fragment_shader_module = ge1::create_shader_module(device, {
            &_binary_shaders_solid_fragment_glsl_spv_start,
            &_binary_shaders_solid_fragment_glsl_spv_end
        });

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

    // camera
    {
        auto matrix = glm::perspectiveFov<float>(
            30.f, windowWidth, windowHeight, 0.1, 100
        );
        matrix = matrix * glm::lookAt(
            glm::vec3{0, -1, 1}, {0, 0, 1}, {0, 0, 1}
        );
        copy(
            glm::value_ptr(matrix), glm::value_ptr(matrix) + 16,
            matrices
        );
    }

    // create buffers for geometry
    scene scene;
    auto vertex_buffer_size =
        sizeof(float) * (
            &_binary_models_miku_vertices_vbo_end -
            &_binary_models_miku_vertices_vbo_start
        ) + sizeof(unsigned) * (
            &_binary_models_miku_faces_vbo_end -
            &_binary_models_miku_faces_vbo_start
        ) + sizeof(matrices);
    VkBuffer vertex_buffer;
    {
        VkBufferCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = vertex_buffer_size,
            .usage =
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        if (
            vkCreateBuffer(device, &create_info, nullptr, &vertex_buffer) !=
            VK_SUCCESS
        ) {
            throw runtime_error("failed to create vertex buffer");
        }
    }

    VkDeviceMemory vertex_memory = ge1::allocate_memory(
        device, physical_device, vertex_buffer,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    vkBindBufferMemory(device, vertex_buffer, vertex_memory, 0);

    {
        void* data;
        vkMapMemory(device, vertex_memory, 0, vertex_buffer_size, 0, &data);
        void* iterator = data;
        scene.vertex_offset = 0;
        iterator = std::copy(
            &_binary_models_miku_vertices_vbo_start,
            &_binary_models_miku_vertices_vbo_end, (float*)iterator
        );
        scene.face_offset = (char*)iterator - (char*)data;
        iterator = std::copy(
            &_binary_models_miku_faces_vbo_start,
            &_binary_models_miku_faces_vbo_end, (unsigned*)iterator
        );
        scene.instance_offset = (char*)iterator - (char*)data;
        memcpy(
            (char*)iterator,
            matrices, sizeof(matrices)
        );
        vkUnmapMemory(device, vertex_memory);
    }
    scene.static_buffer = vertex_buffer;
    scene.index_count =
        &_binary_models_miku_faces_vbo_end - &_binary_models_miku_faces_vbo_start;

    // create pipeline
    VkRenderPass render_pass;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    {
        VkPipelineShaderStageCreateInfo stage_create_infos[]{
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertex_shader_module,
                .pName = "main",
            }, {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragment_shader_module,
                .pName = "main",
            }
        };

        VkPipelineVertexInputStateCreateInfo input_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = size(vertex_binding_descriptions),
            .pVertexBindingDescriptions = vertex_binding_descriptions,
            .vertexAttributeDescriptionCount =
                size(vertex_attribute_descriptions),
            .pVertexAttributeDescriptions = vertex_attribute_descriptions,
        };
        VkPipelineInputAssemblyStateCreateInfo assembly_state_create_info{
            .sType =
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };
        VkPipelineViewportStateCreateInfo viewport_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            // number of viewports and scissors is still relevant
            .viewportCount = 1,
            .pViewports = nullptr, // dynamic
            .scissorCount = 1,
            .pScissors = nullptr, // dynamic
        };
        VkPipelineRasterizationStateCreateInfo rasterization_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };
        VkPipelineMultisampleStateCreateInfo multisample_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = max_sample_count,
            .sampleShadingEnable = VK_FALSE,
        };
        VkPipelineColorBlendAttachmentState color_blend_attachment_state{
            .blendEnable = VK_FALSE,
            .colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        VkPipelineColorBlendStateCreateInfo color_blend_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment_state,
        };
        VkDynamicState dynamic_state[]{
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        VkPipelineDynamicStateCreateInfo dynamic_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = size(dynamic_state),
            .pDynamicStates = dynamic_state,
        };
        VkPipelineLayoutCreateInfo layout_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        };
        if (
            vkCreatePipelineLayout(
                device, &layout_create_info, nullptr,
                &pipeline_layout
            ) != VK_SUCCESS
        ) {
            throw runtime_error("failed to create pipeline layout");
        }
        VkAttachmentDescription color_attachment{
            .format = surfaceFormat.format,
            .samples = max_sample_count,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        VkAttachmentReference color_attachment_reference{
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        VkAttachmentDescription resolve_attachment{
            .format = surfaceFormat.format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };
        VkAttachmentReference resolve_attachment_reference{
            .attachment = 1,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        };

        VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment_reference,
            .pResolveAttachments = &resolve_attachment_reference,
        };
        VkSubpassDependency dependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask =
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .dstStageMask =
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };
        VkAttachmentDescription attachments[] = {
            color_attachment, resolve_attachment
        };
        VkRenderPassCreateInfo render_pass_create_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = std::size(attachments),
            .pAttachments = attachments,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };
        if (
            vkCreateRenderPass(
                device, &render_pass_create_info, nullptr,
                &render_pass
            ) != VK_SUCCESS
        ) {
            throw runtime_error("failed to create render pass");
        }

        VkGraphicsPipelineCreateInfo pipeline_create_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = size(stage_create_infos),
            .pStages = stage_create_infos,
            .pVertexInputState = &input_state_create_info,
            .pInputAssemblyState = &assembly_state_create_info,
            .pViewportState = &viewport_state_create_info,
            .pRasterizationState = &rasterization_state_create_info,
            .pMultisampleState = &multisample_state_create_info,
            .pColorBlendState = &color_blend_state_create_info,
            .pDynamicState = &dynamic_state_create_info,
            .layout = pipeline_layout,
            .renderPass = render_pass,
            .subpass = 0,
        };
        if (
            vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr,
                &pipeline
            ) != VK_SUCCESS
        ) {
            throw runtime_error("failed to create pipeline");
        }
    }

    // create swapchain
    display_size display_size;

    int framebuffer_width, framebuffer_height;
    glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
    create_display_size(
        framebuffer_width, framebuffer_width, device, physical_device,
        graphicsQueueFamily, presentQueueFamily, surface, surfaceFormat,
        commandPool, scene, render_pass, pipeline,
        display_size
    );

    // create frame data
    unsigned frames_in_flight = 2;
    ge1::unique_span<frame_semaphores> frames(frames_in_flight);
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

        vkWaitForFences(
            device, 1, &frames[frame_index].ready_fence, VK_TRUE, -1ul
        );

        // get next image from swapchain
        uint32_t image_index;
        auto result = vkAcquireNextImageKHR(
            device, display_size.swapchain, -1ul,
            frames[frame_index].image_available_semaphore,
            VK_NULL_HANDLE,
            &image_index
        );
        if (result == VK_SUCCESS) {
            vkResetFences(device, 1, &frames[frame_index].ready_fence);
            auto& swapchain_frame = display_size.swapchain_frames[image_index];

            // submit command buffer
            VkSemaphore waitSemaphores[]{
                frames[frame_index].image_available_semaphore
            };
            VkPipelineStageFlags waitStages[]{
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            };
            VkSemaphore signalSemaphores[]{
                frames[frame_index].render_finished_semaphore
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
                vkQueueSubmit(
                    graphicsQueue, 1, &submitInfo,
                    frames[frame_index].ready_fence
                ) != VK_SUCCESS
            ) {
                throw runtime_error("failed to submit draw command buffer");
            }

            // present image
            VkPresentInfoKHR presentInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = signalSemaphores,
                .swapchainCount = 1,
                .pSwapchains = &display_size.swapchain,
                .pImageIndices = &image_index,
            };
            vkQueuePresentKHR(presentQueue, &presentInfo);

            frame_index = (frame_index + 1) % frames.size();

        } else if (
            result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR
        ) {
            for (auto frame : frames) {
                vkWaitForFences(device, 1, &frame.ready_fence, VK_TRUE, -1ul);
            }

            int framebuffer_width, framebuffer_height;
            glfwGetFramebufferSize(
                window, &framebuffer_width, &framebuffer_height
            );
            if (framebuffer_height > 0 && framebuffer_width > 0) {
                destroy_display_size(device, display_size);

                create_display_size(
                    framebuffer_width, framebuffer_width,
                    device, physical_device,
                    graphicsQueueFamily, presentQueueFamily,
                    surface, surfaceFormat,
                    commandPool, scene, render_pass, pipeline,
                    display_size
                );
            }

        } else {
            throw runtime_error("failed to acquire sawp chain image");
        }

        // TODO: swapchain doesn't necessarily sync with current monitor
        // use VK_KHR_display to wait for vsync of current display
    }


    for (auto& frame : frames) {
        vkWaitForFences(device, 1, &frame.ready_fence, VK_TRUE, -1ul);
        vkDestroySemaphore(device, frame.image_available_semaphore, nullptr);
        vkDestroySemaphore(device, frame.render_finished_semaphore, nullptr);
        vkDestroyFence(device, frame.ready_fence, nullptr);
    }

    destroy_display_size(device, display_size);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyRenderPass(device, render_pass, nullptr);

    vkDestroyBuffer(device, vertex_buffer, nullptr);
    vkFreeMemory(device, vertex_memory, nullptr);

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    vkDestroyShaderModule(device, fragment_shader_module, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
}
