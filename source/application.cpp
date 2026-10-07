#include "application.hpp"

#include <cmath>
#include <iostream>

#include <imgui.h>

#include "graphics.hpp"
#include "matrix.hpp"

namespace application {

namespace {

constexpr float pi = 3.14159265f;
constexpr float deg_to_rad = pi / 180.0f;

struct Vertex {
	float position[3];
	float color[3];
};

struct Uniform {
	Mat4 mvp;
	float tint[4];
};

struct Object {
	const char* name;

	graphics::Buffer uniform_buffer;
	VkDescriptorSet descriptor_set;

	float position[3];
	float rotation[3];
	float scale[3];
	float color[3];

	bool playing;
	float anim_time;
	float anim_speed;
	float anim_radius;
	float anim_height;
};

constexpr uint32_t object_count = 2;

Object objects[object_count] = {
	{
		.name = "Object 1",
		.position = { 0.0f, 0.0f, 0.0f },
		.rotation = { 20.0f, 30.0f, 0.0f },
		.scale = { 1.0f, 1.0f, 1.0f },
		.color = { 1.0f, 1.0f, 1.0f },
		.playing = false,
		.anim_speed = 1.0f,
		.anim_radius = 2.0f,
		.anim_height = 1.0f,
	},
	{
		.name = "Object 2",
		.position = { -2.5f, 1.0f, -1.0f },
		.rotation = { 0.0f, 0.0f, 0.0f },
		.scale = { 0.5f, 0.5f, 0.5f },
		.color = { 1.0f, 0.6f, 0.3f },
		.playing = true,
		.anim_speed = 1.5f,
		.anim_radius = 1.0f,
		.anim_height = 0.5f,
	},
};

bool use_perspective = true;
float fov = 60.0f;
float ortho_height = 6.0f;
float near_plane = 0.1f;
float far_plane = 100.0f;
float camera_distance = 6.0f;

double last_time = 0.0;

graphics::Buffer vertex_buffer;
graphics::Buffer index_buffer;
uint32_t index_count;

VkDescriptorSetLayout descriptor_set_layout;
VkDescriptorPool descriptor_pool;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

bool createMesh() {
	const float hx = 0.8f;
	const float hy = 0.5f;
	const float hz = 0.6f;

	const float corners[8][3] = {
		{ -hx, -hy, -hz },
		{  hx, -hy, -hz },
		{  hx,  hy, -hz },
		{ -hx,  hy, -hz },
		{ -hx, -hy,  hz },
		{  hx, -hy,  hz },
		{  hx,  hy,  hz },
		{ -hx,  hy,  hz },
	};

	Vertex vertices[8];
	for (int i = 0; i < 8; ++i) {
		vertices[i].position[0] = corners[i][0];
		vertices[i].position[1] = corners[i][1];
		vertices[i].position[2] = corners[i][2];
		vertices[i].color[0] = corners[i][0] / hx * 0.5f + 0.5f;
		vertices[i].color[1] = corners[i][1] / hy * 0.5f + 0.5f;
		vertices[i].color[2] = corners[i][2] / hz * 0.5f + 0.5f;
	}

	const uint16_t indices[] = {
		0, 2, 1, 0, 3, 2,
		4, 5, 6, 4, 6, 7,
		0, 4, 7, 0, 7, 3,
		1, 2, 6, 1, 6, 5,
		3, 7, 6, 3, 6, 2,
		0, 1, 5, 0, 5, 4,
	};
	index_count = sizeof(indices) / sizeof(indices[0]);

	if (!graphics::createBuffer(vertex_buffer, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
		return false;
	}
	if (!graphics::createBuffer(index_buffer, sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) {
		return false;
	}

	graphics::writeBuffer(vertex_buffer, vertices, sizeof(vertices));
	graphics::writeBuffer(index_buffer, indices, sizeof(indices));

	return true;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
	                                &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = object_count,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = object_count,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	for (Object& object : objects) {
		if (!graphics::createBuffer(object.uniform_buffer, sizeof(Uniform),
		                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
			return false;
		}

		const VkDescriptorSetAllocateInfo allocate_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		if (vkAllocateDescriptorSets(context.device, &allocate_info,
		                             &object.descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate descriptor set\n";
			return false;
		}

		const VkDescriptorBufferInfo buffer_info = {
			.buffer = object.uniform_buffer.handle,
			.offset = 0,
			.range = sizeof(Uniform),
		};

		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = object.descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info,
		};

		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	VkShaderModule vert = graphics::loadShader("shaders/shape.vert.spv");
	VkShaderModule frag = graphics::loadShader("shaders/shape.frag.spv");
	if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
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
			.offset = offsetof(Vertex, color),
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

void drawObjectControls(Object& object) {
	if (!ImGui::CollapsingHeader(object.name, ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}

	ImGui::PushID(object.name);

	ImGui::SeparatorText("Transform");
	ImGui::SliderFloat3("Position", object.position, -5.0f, 5.0f);
	ImGui::SliderFloat3("Rotation", object.rotation, -180.0f, 180.0f);
	ImGui::SliderFloat3("Scale", object.scale, 0.1f, 3.0f);

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Tint", object.color);

	ImGui::SeparatorText("Animation");
	if (ImGui::Button(object.playing ? "Pause" : "Play")) {
		object.playing = !object.playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Reset")) {
		object.anim_time = 0.0f;
	}
	ImGui::SliderFloat("Speed", &object.anim_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Radius", &object.anim_radius, 0.0f, 4.0f);
	ImGui::SliderFloat("Height", &object.anim_height, 0.0f, 3.0f);

	ImGui::PopID();
}

Mat4 modelMatrix(const Object& object) {
	const float t = object.anim_time;

	const float x = object.position[0] + object.anim_radius * std::cos(t);
	const float y = object.position[1] + object.anim_height * std::sin(2.0f * t);
	const float z = object.position[2] + object.anim_radius * std::sin(3.0f * t);

	const float rx = object.rotation[0] * deg_to_rad + 0.5f * t;
	const float ry = object.rotation[1] * deg_to_rad + t;
	const float rz = object.rotation[2] * deg_to_rad;

	const Mat4 rotation = multiply(rotateZ(rz), multiply(rotateY(ry), rotateX(rx)));

	return multiply(translate(x, y, z),
	                multiply(rotation, scale(object.scale[0], object.scale[1], object.scale[2])));
}

} // namespace

bool initialize() {
	if (!createMesh()) {
		return false;
	}
	if (!createDescriptors()) {
		return false;
	}
	if (!createPipeline()) {
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);

	for (Object& object : objects) {
		graphics::destroyBuffer(object.uniform_buffer);
	}

	graphics::destroyBuffer(index_buffer);
	graphics::destroyBuffer(vertex_buffer);
}

void update(double time) {
	const float dt = float(time - last_time);
	last_time = time;

	for (Object& object : objects) {
		if (object.playing) {
			object.anim_time += dt * object.anim_speed;
		}
	}

	ImGui::Begin("Controls");

	ImGui::SeparatorText("Projection");
	if (ImGui::RadioButton("Perspective", use_perspective)) {
		use_perspective = true;
	}
	ImGui::SameLine();
	if (ImGui::RadioButton("Orthographic", !use_perspective)) {
		use_perspective = false;
	}
	if (use_perspective) {
		ImGui::SliderFloat("FOV", &fov, 20.0f, 120.0f);
	} else {
		ImGui::SliderFloat("Ortho height", &ortho_height, 1.0f, 20.0f);
	}
	ImGui::SliderFloat("Camera distance", &camera_distance, 2.0f, 20.0f);

	for (Object& object : objects) {
		drawObjectControls(object);
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	const float aspect = float(context.swapchain_extent.width) / float(context.swapchain_extent.height);

	Mat4 projection;
	if (use_perspective) {
		projection = perspective(fov * deg_to_rad, aspect, near_plane, far_plane);
	} else {
		projection = orthographic(ortho_height, aspect, near_plane, far_plane);
	}

	const Mat4 view = translate(0.0f, 0.0f, -camera_distance);
	const Mat4 view_projection = multiply(projection, view);

	for (Object& object : objects) {
		Uniform uniform;
		uniform.mvp = multiply(view_projection, modelMatrix(object));
		uniform.tint[0] = object.color[0];
		uniform.tint[1] = object.color[1];
		uniform.tint[2] = object.color[2];
		uniform.tint[3] = 1.0f;

		graphics::writeBuffer(object.uniform_buffer, &uniform, sizeof(uniform));
	}

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(fd.command_buffer, &begin_info);

	VkClearValue clear_values[2] = {};
	clear_values[0].color = { { 0.08f, 0.08f, 0.1f, 1.0f } };
	clear_values[1].depthStencil = { 1.0f, 0 };

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = { .extent = context.swapchain_extent };

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.handle, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT16);

	for (const Object& object : objects) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        pipeline_layout, 0, 1, &object.descriptor_set, 0, nullptr);
		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
