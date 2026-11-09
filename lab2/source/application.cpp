#include "application.hpp"
#include "graphics.hpp"
#include "lighting_math.hpp"
#include <cstddef>
#include <imgui.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace application {
namespace {
constexpr float pi = 3.14159265359f, rad = pi / 180.f;

struct Vertex {
    float position[3];
    float normal[3];
};

struct alignas(16) V4 {
    float x, y, z, w;
};

struct Scene {
    Mat4 vp;
    V4 eye, ambient, direction, directional;
};

struct Uniform {
    Mat4 model;
    V4 albedo, specular;
};

struct Point {
    V4 position;
    V4 color;
};

struct Spot {
    V4 position, color, direction, angles;
};

static_assert(sizeof(Scene) == 128 && sizeof(Uniform) == 96 && sizeof(Point) == 32 &&
              sizeof(Spot) == 64);

struct Object {
    const char *name;
    Vec3 pos, rot, size;
    V4 albedo, specular;
    graphics::Buffer buffer{};
    VkDescriptorSet set{};
    bool is_line = false;
    Vec3 line_end{};
};

std::vector<Object> objects = {
    {"Parallelepiped 1", {-1, 0, 0}, {15, 30, 0}, {1, 1, 1}, {.25, .65, .95, 0}, {1, 1, 1, 64}},
    {"Parallelepiped 2",
     {1.3, -.1, -.6},
     {0, -20, 0},
     {.7, 1.4, .7},
     {.95, .35, .15, 0},
     {1, .85, .65, 96}},
    {"Floor", {0, -.8, 0}, {0, 0, 0}, {7, .1, 7}, {.25, .29, .35, 0}, {.25, .25, .25, 32}}};
std::vector<Point> points = {{{-2, 2, 2, 1}, {.2, .55, 1, 12}}};
std::vector<Spot> spots = {{{2, 3, 2, 1}, {1, .65, .25, 25}, {-.5, -1, -.6, 0}, {15, 28, 0, 0}}};
Vec3 eye{0, 2, 7};
float yaw = 0, pitch = -12, fov = 55, ortho = 7;
bool perspective_on = true, markers = true, animate = false;
V4 ambient{.12, .14, .18, 1}, direction{-.5, -1, -.4, 0}, sun{1, .94, .82, 1.1f};
float delta = 0;
double last = 0;
graphics::Buffer vertex_buffer{}, index_buffer{}, scene_buffer{}, point_buffer{}, spot_buffer{};
size_t point_capacity = 0, spot_capacity = 0;
VkDescriptorSetLayout scene_layout{}, model_layout{};
VkDescriptorPool descriptor_pool{};
VkDescriptorSet scene_set{};
VkPipelineLayout pipeline_layout{};
VkPipeline pipeline{};
uint32_t index_count = 36;

std::vector<Object> helpers;

void check(VkResult r) {
    if (r != VK_SUCCESS)
        throw std::runtime_error("Vulkan operation failed: " + std::to_string(r));
}

void buffer(graphics::Buffer &b, size_t bytes, VkBufferUsageFlags usage) {
    if (!graphics::createBuffer(b, bytes, usage))
        throw std::runtime_error("Cannot allocate GPU buffer");
}

void bindBuffer(VkDescriptorSet set, uint32_t binding, const graphics::Buffer &b, size_t bytes,
                VkDescriptorType type) {
    VkDescriptorBufferInfo info{b.handle, 0, bytes};
    VkWriteDescriptorSet w{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                           .dstSet = set,
                           .dstBinding = binding,
                           .descriptorCount = 1,
                           .descriptorType = type,
                           .pBufferInfo = &info};
    vkUpdateDescriptorSets(graphics::internal::context.device, 1, &w, 0, nullptr);
}

void createMesh() {
    const Vec3 c[] = {{-.8f, -.5f, -.6f}, {.8f, -.5f, -.6f}, {.8f, .5f, -.6f}, {-.8f, .5f, -.6f},
                      {-.8f, -.5f, .6f},  {.8f, -.5f, .6f},  {.8f, .5f, .6f},  {-.8f, .5f, .6f}};
    const int faces[][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 4, 7, 3},
                            {1, 2, 6, 5}, {3, 7, 6, 2}, {0, 1, 5, 4}};
    std::vector<Vertex> v;
    std::vector<uint16_t> idx;
    for (auto &face : faces) {
        Vec3 n = unit(cross(c[face[1]] - c[face[0]], c[face[2]] - c[face[0]]));
        uint16_t base = v.size();
        for (int k : face)
            v.push_back({{c[k].x, c[k].y, c[k].z}, {n.x, n.y, n.z}});
        for (int k : {0, 1, 2, 0, 2, 3})
            idx.push_back(base + k);
    }
    buffer(vertex_buffer, v.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    buffer(index_buffer, idx.size() * 2, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    graphics::writeBuffer(vertex_buffer, v.data(), v.size() * sizeof(Vertex));
    graphics::writeBuffer(index_buffer, idx.data(), idx.size() * 2);
}

void allocateModel(Object &o) {
    buffer(o.buffer, sizeof(Uniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    VkDescriptorSetAllocateInfo a{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                  .descriptorPool = descriptor_pool,
                                  .descriptorSetCount = 1,
                                  .pSetLayouts = &model_layout};
    check(vkAllocateDescriptorSets(graphics::internal::context.device, &a, &o.set));
    bindBuffer(o.set, 0, o.buffer, sizeof(Uniform), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
}

void createDescriptors() {
    auto d = graphics::internal::context.device;
    VkDescriptorSetLayoutBinding b[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo l{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                      .bindingCount = 3,
                                      .pBindings = b};
    check(vkCreateDescriptorSetLayout(d, &l, nullptr, &scene_layout));
    l.bindingCount = 1;
    check(vkCreateDescriptorSetLayout(d, &l, nullptr, &model_layout));
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096},
                                    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
    VkDescriptorPoolCreateInfo pool{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                    .maxSets = 4096,
                                    .poolSizeCount = 2,
                                    .pPoolSizes = sizes};
    check(vkCreateDescriptorPool(d, &pool, nullptr, &descriptor_pool));
    VkDescriptorSetAllocateInfo a{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                  .descriptorPool = descriptor_pool,
                                  .descriptorSetCount = 1,
                                  .pSetLayouts = &scene_layout};
    check(vkAllocateDescriptorSets(d, &a, &scene_set));
    buffer(scene_buffer, sizeof(Scene), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    bindBuffer(scene_set, 0, scene_buffer, sizeof(Scene), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    for (auto &o : objects)
        allocateModel(o);
}

bool createPipeline() {
    auto &context = graphics::internal::context;

    const VkDescriptorSetLayout layouts[] = {scene_layout, model_layout};
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = layouts,
    };

    if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) !=
        VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    VkShaderModule vert = graphics::loadShader("shaders/shape.vert.spv");
    VkShaderModule frag = graphics::loadShader("shaders/shape.frag.spv");
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
        vkDestroyShaderModule(context.device, vert, nullptr);
        vkDestroyShaderModule(context.device, frag, nullptr);
        return false;
    }

    const VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag,
            .pName = "main",
        },
    };

    const VkVertexInputBindingDescription vertex_binding = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    const VkVertexInputAttributeDescription vertex_attributes[] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, normal),
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_binding,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = vertex_attributes,
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    const VkPipelineViewportStateCreateInfo viewport = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    const VkPipelineColorBlendAttachmentState blend_attachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
    };

    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states,
    };

    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic,
        .layout = pipeline_layout,
        .renderPass = context.render_pass,
        .subpass = 0,
    };

    const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
                                                      &pipeline_info, nullptr, &pipeline);

    vkDestroyShaderModule(context.device, vert, nullptr);
    vkDestroyShaderModule(context.device, frag, nullptr);

    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        return false;
    }

    return true;
}

Mat4 model(const Object &o) {
    return multiply(
        translate(o.pos.x, o.pos.y, o.pos.z),
        multiply(rotateZ(o.rot.z * rad),
                 multiply(rotateY(o.rot.y * rad),
                          multiply(rotateX(o.rot.x * rad), scale(o.size.x, o.size.y, o.size.z)))));
}

void colorIntensity(const char *label, V4 &c) {
    ImGui::ColorEdit3(label, &c.x);
    ImGui::SliderFloat("Intensity", &c.w, 0, 80);
}

void presets(int mode) {
    points.clear();
    spots.clear();
    sun.w = mode == 0 ? 1.4f : 0;
    ambient = {.04, .04, .05, 1};
    if (mode == 1)
        points = {{{-2, 2, 2, 1}, {.2, .55, 1, 16}}, {{2, 1.5, 1, 1}, {1, .25, .1, 10}}};
    if (mode == 2)
        spots = {{{0, 3, 2, 1}, {1, .8, .4, 35}, {0, -1, -.65, 0}, {12, 28, 0, 0}}};
}

void controls() {
    ImGui::SetNextWindowPos({14, 14}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({365, 680}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 2 | Camera and lighting");
    ImGui::Text("Variant 3 | Parallelepiped");
    ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
    if (ImGui::Button("Directional only"))
        presets(0);
    ImGui::SameLine();
    if (ImGui::Button("Points"))
        presets(1);
    ImGui::SameLine();
    if (ImGui::Button("Spot"))
        presets(2);
    ImGui::Checkbox("Show light markers and cones", &markers);
    ImGui::Checkbox("Animate lights", &animate);
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Eye position", &eye.x, .05f, -30, 30);
        ImGui::SliderFloat("Yaw", &yaw, -180, 180);
        ImGui::SliderFloat("Pitch", &pitch, -89, 89);
        ImGui::SliderFloat("FOV", &fov, 20, 110);
        ImGui::Checkbox("Perspective", &perspective_on);
        if (!perspective_on)
            ImGui::SliderFloat("Ortho height", &ortho, 2, 20);
        ImGui::TextWrapped("Move: W A S D / Q E outside UI. Right-drag: look.");
        if (ImGui::Button("Reset camera")) {
            eye = {0, 2, 7};
            yaw = 0;
            pitch = -12;
            fov = 55;
            ortho = 7;
            perspective_on = true;
        }
    }
    if (ImGui::CollapsingHeader("Ambient and directional")) {
        ImGui::ColorEdit3("Ambient", &ambient.x);
        ImGui::DragFloat3("Sun direction", &direction.x, .02f, -1, 1);
        colorIntensity("Sun color", sun);
    }
    if (ImGui::CollapsingHeader("Materials and transforms"))
        for (auto &o : objects) {
            ImGui::PushID(o.name);
            if (ImGui::TreeNode(o.name)) {
                ImGui::DragFloat3("Position", &o.pos.x, .03f);
                ImGui::SliderFloat3("Rotation", &o.rot.x, -180, 180);
                ImGui::SliderFloat3("Scale", &o.size.x, .05f, 8);
                ImGui::ColorEdit3("Albedo", &o.albedo.x);
                ImGui::ColorEdit3("Specular color", &o.specular.x);
                ImGui::SliderFloat("Shininess", &o.specular.w, 1, 256);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    if (ImGui::CollapsingHeader("Point lights")) {
        if (ImGui::Button("Add point"))
            points.push_back({{0, 2, 1, 1}, {1, 1, 1, 10}});
        for (size_t i = 0; i < points.size();) {
            ImGui::PushID(int(i));
            auto &l = points[i];
            ImGui::Separator();
            ImGui::Text("Point %zu", i + 1);
            ImGui::DragFloat3("Position", &l.position.x, .03f);
            colorIntensity("Color", l.color);
            bool remove = ImGui::Button("Delete point");
            ImGui::PopID();
            if (remove)
                points.erase(points.begin() + i);
            else
                ++i;
        }
    }
    if (ImGui::CollapsingHeader("Spot lights")) {
        if (ImGui::Button("Add spot"))
            spots.push_back({{0, 3, 2, 1}, {1, 1, 1, 25}, {0, -1, -1, 0}, {15, 28, 0, 0}});
        for (size_t i = 0; i < spots.size();) {
            ImGui::PushID(int(i) + 10000);
            auto &l = spots[i];
            ImGui::Separator();
            ImGui::Text("Spot %zu", i + 1);
            ImGui::DragFloat3("Position", &l.position.x, .03f);
            ImGui::DragFloat3("Direction", &l.direction.x, .02f, -1, 1);
            colorIntensity("Color", l.color);
            ImGui::SliderFloat("Inner angle", &l.angles.x, 0, l.angles.y);
            ImGui::SliderFloat("Outer angle", &l.angles.y, l.angles.x, 80);
            bool remove = ImGui::Button("Delete spot");
            ImGui::PopID();
            if (remove)
                spots.erase(spots.begin() + i);
            else
                ++i;
        }
    }
    ImGui::End();
}

template <class T>
void upload(std::vector<T> &data, graphics::Buffer &b, size_t &capacity, uint32_t binding) {
    size_t count = std::max(size_t(1), data.size());
    if (count != capacity) {
        graphics::destroyBuffer(b);
        buffer(b, count * sizeof(T), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        capacity = count;
        bindBuffer(scene_set, binding, b, count * sizeof(T), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }
    T zero{};
    graphics::writeBuffer(b, data.empty() ? &zero : data.data(), count * sizeof(T));
}

void addHelper(Vec3 p, Vec3 size, V4 color) {
    Object o{"Light", p, {0, 0, 0}, size, color, {0, 0, 0, 1}};
    o.albedo.w = 1;
    helpers.push_back(o);
}

void segment(Vec3 start, Vec3 end) {
    Vec3 direction = end - start;
    if (dot(direction, direction) < 1e-10f)
        return;
    Object line{"Cone", start, {0, 0, 0}, {1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 1}};
    line.is_line = true;
    line.line_end = end;
    helpers.push_back(line);
}

Mat4 helperModel(const Object &object) {
    if (object.is_line)
        return lineMatrix(object.pos, object.line_end);
    return model(object);
}

void rebuildHelpers() {
    auto d = graphics::internal::context.device;
    auto previous = std::move(helpers);
    helpers.clear();
    if (markers) {
        for (auto &l : points)
            addHelper({l.position.x, l.position.y, l.position.z}, {.13, .2, .17}, l.color);
        for (auto &l : spots) {
            Vec3 a{l.position.x, l.position.y, l.position.z},
                axis = unit({l.direction.x, l.direction.y, l.direction.z});
            addHelper(a, {.13, .2, .17}, l.color);
            Vec3 x = unit(cross(axis, std::abs(axis.y) > .95f ? Vec3{1, 0, 0} : Vec3{0, 1, 0})),
                 y = cross(axis, x);
            float length = 1.4f;
            Vec3 center = a + axis * length;
            segment(a, center);
            for (float angle : {l.angles.x, l.angles.y}) {
                float radius = std::tan(angle * rad) * length;
                for (int j = 0; j < 24; j++) {
                    auto point = [&](int k) {
                        float t = 2 * pi * k / 24;
                        return center + (x * std::cos(t) + y * std::sin(t)) * radius;
                    };
                    segment(point(j), point(j + 1));
                    if (j % 6 == 0)
                        segment(a, point(j));
                }
            }
        }
    }
    for (size_t i = 0; i < helpers.size(); ++i) {
        if (i < previous.size()) {
            helpers[i].buffer = previous[i].buffer;
            helpers[i].set = previous[i].set;
        } else
            allocateModel(helpers[i]);
    }
    for (size_t i = helpers.size(); i < previous.size(); ++i) {
        graphics::destroyBuffer(previous[i].buffer);
        check(vkFreeDescriptorSets(d, descriptor_pool, 1, &previous[i].set));
    }
}
}

bool initialize() {
    try {
        createMesh();
        createDescriptors();
        return createPipeline();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return false;
    }
}

void shutdown() {
    auto d = graphics::internal::context.device;
    vkDeviceWaitIdle(d);
    for (auto &h : helpers)
        graphics::destroyBuffer(h.buffer);
    for (auto &o : objects)
        graphics::destroyBuffer(o.buffer);
    for (auto *b : {&vertex_buffer, &index_buffer, &scene_buffer, &point_buffer, &spot_buffer})
        graphics::destroyBuffer(*b);
    vkDestroyPipeline(d, pipeline, nullptr);
    vkDestroyPipelineLayout(d, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(d, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(d, scene_layout, nullptr);
    vkDestroyDescriptorSetLayout(d, model_layout, nullptr);
}

void update(double time) {
    delta = last ? std::min(float(time - last), .1f) : 0;
    last = time;
    controls();
    auto &io = ImGui::GetIO();
    if (!io.WantCaptureMouse && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
        yaw += io.MouseDelta.x * .25f;
        pitch = std::clamp(pitch - io.MouseDelta.y * .25f, -89.f, 89.f);
    }
    if (!io.WantCaptureKeyboard) {
        Vec3 f{std::sin(yaw * rad), 0, -std::cos(yaw * rad)},
            r{std::cos(yaw * rad), 0, std::sin(yaw * rad)};
        float s = delta * 3;
        if (ImGui::IsKeyDown(ImGuiKey_W))
            eye = eye + f * s;
        if (ImGui::IsKeyDown(ImGuiKey_S))
            eye = eye - f * s;
        if (ImGui::IsKeyDown(ImGuiKey_D))
            eye = eye + r * s;
        if (ImGui::IsKeyDown(ImGuiKey_A))
            eye = eye - r * s;
        if (ImGui::IsKeyDown(ImGuiKey_E))
            eye.y += s;
        if (ImGui::IsKeyDown(ImGuiKey_Q))
            eye.y -= s;
    }
    if (animate) {
        for (auto &p : points) {
            float x = p.position.x, z = p.position.z;
            p.position.x = x * std::cos(delta) - z * std::sin(delta);
            p.position.z = x * std::sin(delta) + z * std::cos(delta);
        }
    }
}

void render(const graphics::internal::FrameData &fd) {
    auto &c = graphics::internal::context;

    upload(points, point_buffer, point_capacity, 1);
    upload(spots, spot_buffer, spot_capacity, 2);
    rebuildHelpers();
    float aspect = float(c.swapchain_extent.width) / c.swapchain_extent.height;
    Mat4 proj = perspective_on ? perspective(fov * rad, aspect, .1, 100)
                               : orthographic(ortho, aspect, .1, 100);
    Scene scene{multiply(proj, viewMatrix(eye, yaw * rad, pitch * rad)),
                {eye.x, eye.y, eye.z, 1},
                ambient,
                direction,
                sun};
    graphics::writeBuffer(scene_buffer, &scene, sizeof(scene));
    auto write = [](Object &o, bool helper) {
        Uniform u{helper ? helperModel(o) : model(o), o.albedo,
                  helper ? V4{0, 0, 0, 1} : o.specular};
        graphics::writeBuffer(o.buffer, &u, sizeof(u));
    };
    for (auto &o : objects)
        write(o, false);
    for (auto &h : helpers)
        write(h, true);
    check(vkResetCommandBuffer(fd.command_buffer, 0));
    VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    check(vkBeginCommandBuffer(fd.command_buffer, &begin));
    VkClearValue clear[2]{};
    clear[0].color = {{.018, .024, .038, 1}};
    clear[1].depthStencil = {1, 0};
    VkRenderPassBeginInfo pass{.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                               .renderPass = c.render_pass,
                               .framebuffer = fd.framebuffer,
                               .renderArea = {.extent = c.swapchain_extent},
                               .clearValueCount = 2,
                               .pClearValues = clear};
    vkCmdBeginRenderPass(fd.command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkViewport vp{0, 0, float(c.swapchain_extent.width), float(c.swapchain_extent.height), 0, 1};
    VkRect2D sc{.extent = c.swapchain_extent};
    vkCmdSetViewport(fd.command_buffer, 0, 1, &vp);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &sc);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.handle, &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0,
                            1, &scene_set, 0, nullptr);
    auto draw = [&](Object &o) {
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                1, 1, &o.set, 0, nullptr);
        vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
    };
    for (auto &o : objects)
        draw(o);
    for (auto &h : helpers)
        draw(h);
    vkCmdEndRenderPass(fd.command_buffer);
    check(vkEndCommandBuffer(fd.command_buffer));
}
}
