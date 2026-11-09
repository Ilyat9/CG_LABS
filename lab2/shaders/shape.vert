#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(set = 0, binding = 0, std140) uniform Scene {
    mat4 vp;
    vec4 eye;
    vec4 ambient;
    vec4 direction;
    vec4 directional;
}

scene;

layout(set = 1, binding = 0, std140) uniform Model {
    mat4 model;
    vec4 albedo;
    vec4 specular;
}

object;
layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;

void main() {
    vec4 p = object.model * vec4(position, 1);
    worldPosition = p.xyz;
    worldNormal = transpose(inverse(mat3(object.model))) * normal;
    gl_Position = scene.vp * p;
}
