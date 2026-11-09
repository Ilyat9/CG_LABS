#include "graphics.hpp"

#include <fstream>
#include <iostream>
#include <vector>

#include "graphics_internal.hpp"

namespace graphics {

bool createBuffer(Buffer &buffer, VkDeviceSize size, VkBufferUsageFlags usage) {
    const VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    const VmaAllocationCreateInfo allocation_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    if (vmaCreateBuffer(internal::context.allocator, &buffer_info, &allocation_info, &buffer.handle,
                        &buffer.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan buffer\n";
        return false;
    }

    return true;
}

void destroyBuffer(Buffer &buffer) {
    vmaDestroyBuffer(internal::context.allocator, buffer.handle, buffer.allocation);
    buffer = {};
}

void writeBuffer(const Buffer &buffer, const void *data, VkDeviceSize size) {
    vmaCopyMemoryToAllocation(internal::context.allocator, data, buffer.allocation, 0, size);
}

VkShaderModule loadShader(const char *path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open shader file " << path << '\n';
        return VK_NULL_HANDLE;
    }

    const size_t size = size_t(file.tellg());
    if (size < 20 || size % sizeof(uint32_t) != 0) {
        std::cerr << "Invalid SPIR-V file " << path << '\n';
        return VK_NULL_HANDLE;
    }
    std::vector<uint32_t> code(size / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(code.data()), size) || code[0] != 0x07230203) {
        std::cerr << "Failed to read SPIR-V file " << path << '\n';
        return VK_NULL_HANDLE;
    }

    const VkShaderModuleCreateInfo module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = code.data(),
    };

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(internal::context.device, &module_info, nullptr, &module) !=
        VK_SUCCESS) {
        std::cerr << "Failed to create shader module " << path << '\n';
        return VK_NULL_HANDLE;
    }

    return module;
}

}
