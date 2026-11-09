#include "../source/lighting_math.hpp"
#include <cassert>
#include <iostream>

bool close(float a, float b) { return std::abs(a - b) < 1e-5; }

int main() {
    auto v = viewMatrix({0, 0, 6}, 0, 0);
    assert(close(v.m[14], -6));
    Vec3 eye{2, 3, 4};
    v = viewMatrix(eye, .6f, -.3f);
    for (int row = 0; row < 3; row++)
        assert(close(v.m[row] * eye.x + v.m[4 + row] * eye.y + v.m[8 + row] * eye.z + v.m[12 + row],
                     0));
    auto n = unit(cross({1, 0, 0}, {0, 1, 0}));
    assert(close(n.z, 1));
    assert(close(dot(unit({0, 0, 0}), unit({0, 0, 0})), 1));
    assert(close(coneWeight(1, .2, .5), 1));
    assert(close(coneWeight(0, .2, .5), 0));
    assert(close(coneWeight(1, .3, .3), 1));
    assert(close(coneWeight(0, .3, .3), 0));
    float mid = (std::cos(.2f) + std::cos(.5f)) / 2;
    assert(close(coneWeight(mid, .2, .5), .5));
    auto p = perspective(1, 1, .1, 100);
    auto depth = [&](float z) { return (p.m[10] * z + p.m[14]) / (-z); };
    assert(close(depth(-.1), 0));
    assert(close(depth(-100), 1));
    for (Vec3 end : {Vec3{2, 3, 4}, Vec3{0, 2, 0}, Vec3{0, -2, 0}}) {
        Vec3 start{0, 0, 0};
        auto line = lineMatrix(start, end);
        assert(close(line.m[12] - line.m[4] * 0.5f, start.x));
        assert(close(line.m[13] - line.m[5] * 0.5f, start.y));
        assert(close(line.m[14] - line.m[6] * 0.5f, start.z));
        assert(close(line.m[12] + line.m[4] * 0.5f, end.x));
        assert(close(line.m[13] + line.m[5] * 0.5f, end.y));
        assert(close(line.m[14] + line.m[6] * 0.5f, end.z));
    }
    auto zero_line = lineMatrix({0, 0, 0}, {0, 0, 0});
    for (float value : zero_line.m)
        assert(std::isfinite(value));
    std::cout << "Camera, normals, zero direction, soft/hard cone and Vulkan "
                 "depth checks passed\n";
}
