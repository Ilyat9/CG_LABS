#pragma once

#include <cmath>

struct Mat4 {
	float m[16];
};

inline Mat4 identity() {
	Mat4 r = {};
	r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
	return r;
}

inline Mat4 multiply(const Mat4& a, const Mat4& b) {
	Mat4 r = {};
	for (int col = 0; col < 4; ++col) {
		for (int row = 0; row < 4; ++row) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.m[k * 4 + row] * b.m[col * 4 + k];
			}
			r.m[col * 4 + row] = sum;
		}
	}
	return r;
}

inline Mat4 translate(float x, float y, float z) {
	Mat4 r = identity();
	r.m[12] = x;
	r.m[13] = y;
	r.m[14] = z;
	return r;
}

inline Mat4 scale(float x, float y, float z) {
	Mat4 r = identity();
	r.m[0] = x;
	r.m[5] = y;
	r.m[10] = z;
	return r;
}

inline Mat4 rotateX(float a) {
	Mat4 r = identity();
	r.m[5] = std::cos(a);
	r.m[6] = std::sin(a);
	r.m[9] = -std::sin(a);
	r.m[10] = std::cos(a);
	return r;
}

inline Mat4 rotateY(float a) {
	Mat4 r = identity();
	r.m[0] = std::cos(a);
	r.m[2] = -std::sin(a);
	r.m[8] = std::sin(a);
	r.m[10] = std::cos(a);
	return r;
}

inline Mat4 rotateZ(float a) {
	Mat4 r = identity();
	r.m[0] = std::cos(a);
	r.m[1] = std::sin(a);
	r.m[4] = -std::sin(a);
	r.m[5] = std::cos(a);
	return r;
}

inline Mat4 perspective(float fov_y, float aspect, float near_plane, float far_plane) {
	const float f = 1.0f / std::tan(fov_y * 0.5f);
	Mat4 r = {};
	r.m[0] = f / aspect;
	r.m[5] = -f;
	r.m[10] = far_plane / (near_plane - far_plane);
	r.m[11] = -1.0f;
	r.m[14] = near_plane * far_plane / (near_plane - far_plane);
	return r;
}

inline Mat4 orthographic(float height, float aspect, float near_plane, float far_plane) {
	const float half_height = height * 0.5f;
	const float half_width = half_height * aspect;
	Mat4 r = {};
	r.m[0] = 1.0f / half_width;
	r.m[5] = -1.0f / half_height;
	r.m[10] = 1.0f / (near_plane - far_plane);
	r.m[14] = near_plane / (near_plane - far_plane);
	r.m[15] = 1.0f;
	return r;
}
