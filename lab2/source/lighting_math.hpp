#pragma once
#include "matrix.hpp"
#include <algorithm>

struct Vec3 {
    float x, y, z;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline Vec3 unit(Vec3 a) {
    float l = std::sqrt(dot(a, a));
    return l > 1e-6f ? a * (1 / l) : Vec3{0, -1, 0};
}

inline Mat4 viewMatrix(Vec3 eye, float yaw, float pitch) {
    Vec3 f =
        unit({std::cos(pitch) * std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw)});
    Vec3 r = unit(cross(f, {0, 1, 0})), u = cross(r, f);
    Mat4 m = identity();
    m.m[0] = r.x;
    m.m[4] = r.y;
    m.m[8] = r.z;
    m.m[12] = -dot(r, eye);
    m.m[1] = u.x;
    m.m[5] = u.y;
    m.m[9] = u.z;
    m.m[13] = -dot(u, eye);
    m.m[2] = -f.x;
    m.m[6] = -f.y;
    m.m[10] = -f.z;
    m.m[14] = dot(f, eye);
    return m;
}

inline float coneWeight(float cosine, float inner, float outer) {
    float a = std::cos(inner), b = std::cos(outer);
    if (a - b < 1e-6f)
        return cosine >= a ? 1.f : 0.f;
    float t = std::clamp((cosine - b) / (a - b), 0.f, 1.f);
    return t * t * (3 - 2 * t);
}

inline Mat4 lineMatrix(Vec3 start, Vec3 end) {
    Vec3 direction = end - start;
    float length = std::sqrt(dot(direction, direction));
    if (length < 1e-5f)
        return identity();
    Vec3 y = unit(direction);
    Vec3 up = std::abs(y.y) > 0.95f ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 x = unit(cross(y, up));
    Vec3 z = cross(x, y);
    Vec3 center = (start + end) * 0.5f;
    Mat4 result = identity();
    result.m[0] = x.x * 0.012f;
    result.m[1] = x.y * 0.012f;
    result.m[2] = x.z * 0.012f;
    result.m[4] = y.x * length;
    result.m[5] = y.y * length;
    result.m[6] = y.z * length;
    result.m[8] = z.x * 0.012f;
    result.m[9] = z.y * 0.012f;
    result.m[10] = z.z * 0.012f;
    result.m[12] = center.x;
    result.m[13] = center.y;
    result.m[14] = center.z;
    return result;
}
