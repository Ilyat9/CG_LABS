#pragma once

#include <vulkan/vulkan_core.h>

#include <vk_mem_alloc.h>

namespace graphics {

struct Buffer {
	VkBuffer handle;
	VmaAllocation allocation;
};

bool createBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage);
void destroyBuffer(Buffer& buffer);
void writeBuffer(const Buffer& buffer, const void* data, VkDeviceSize size);

VkShaderModule loadShader(const char* path);

}
