#version 450
layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 0) out vec4 outColor;

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

struct Point {
    vec4 position;
    vec4 color;
};

struct Spot {
    vec4 position;
    vec4 color;
    vec4 direction;
    vec4 angles;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLights { Point lights[]; }

points;

layout(set = 0, binding = 2, std430) readonly buffer SpotLights { Spot lights[]; }

spots;

vec3 safeUnit(vec3 v) { return dot(v, v) > 1e-12 ? normalize(v) : vec3(0, -1, 0); }

vec3 blinn(vec3 L, vec3 radiance, vec3 N, vec3 V) {
    float diffuse = max(dot(N, L), 0);
    if (diffuse <= 0)
        return vec3(0);
    vec3 H = safeUnit(L + V);
    float spec = pow(max(dot(N, H), 0), object.specular.w);
    return radiance * (object.albedo.rgb * diffuse + object.specular.rgb * spec);
}

void main() {
    if (object.albedo.w > 0.5) {
        outColor = vec4(object.albedo.rgb, 1);
        return;
    }
    vec3 N = normalize(worldNormal), V = safeUnit(scene.eye.xyz - worldPosition);
    vec3 color = scene.ambient.rgb * object.albedo.rgb;
    color +=
        blinn(-safeUnit(scene.direction.xyz), scene.directional.rgb * scene.directional.w, N, V);
    for (int i = 0; i < points.lights.length(); i++) {
        Point l = points.lights[i];
        vec3 d = l.position.xyz - worldPosition;
        float d2 = max(dot(d, d), 0.01);
        color += blinn(safeUnit(d), l.color.rgb * l.color.w / d2, N, V);
    }
    for (int i = 0; i < spots.lights.length(); i++) {
        Spot l = spots.lights[i];
        vec3 d = l.position.xyz - worldPosition;
        float d2 = max(dot(d, d), 0.01);
        float theta = dot(-safeUnit(d), safeUnit(l.direction.xyz));
        float inner = cos(radians(l.angles.x)), outer = cos(radians(l.angles.y));
        float cone = inner - outer < 1e-6 ? step(inner, theta) : smoothstep(outer, inner, theta);
        color += blinn(safeUnit(d), l.color.rgb * l.color.w * cone / d2, N, V);
    }
    outColor = vec4(color, 1);
}
