#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <vector>
using namespace std;

__device__ __constant__ float EPSILON = 0.000001f;
__device__ __constant__ float PI = 3.141592;

inline __host__ __device__ float4 f4(float x, float y, float z, float w = 0.0f) {
    return make_float4(x, y, z, w);
}

inline __host__ __device__ float4 f4() {return make_float4(0,0,0,0);}

inline __host__ __device__ float4 f4(float a) {return make_float4(a,a,a,0);}

inline __host__ __device__ float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f);
}

inline __host__ __device__ float4 operator-(const float4 &a, const float4 &b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.0f);
}

inline __host__ __device__ float4 operator*(const float4 &a, float t) {
    return make_float4(a.x * t, a.y * t, a.z * t, 0.0f);
}

inline __host__ __device__ float4 operator*(float t, const float4 &a) {
    return a * t;
}

inline __host__ __device__ float4 operator/(const float4 &a, float t) {
    return make_float4(a.x / t, a.y / t, a.z / t, 0.0f);
}

inline __host__ __device__ float4& operator+=(float4 &a, const float4 &b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

inline __host__ __device__ float4& operator*=(float4 &a, float t) {
    a.x *= t; a.y *= t; a.z *= t;
    return a;
}

inline __host__ __device__ float4& operator*=(float4& a, const float4& b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
    return a;
}

inline __host__ __device__ float4& operator/=(float4 &a, float t) {
    a.x /= t; a.y /= t; a.z /= t;
    return a;
}

inline __host__ __device__ float4 operator*(const float4& a, const float4& b)
{
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, 0.0f);
}

inline __host__ __device__ float4 operator-(const float4& v)
{
    return make_float4(-v.x, -v.y, -v.z, -v.w);
}

inline __host__ __device__ float dot(const float4 &a, const float4 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ float length(const float4 &v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float lengthSquared(const float4 &v) {
    return dot(v, v);
}

inline __host__ __device__ float4 normalize(const float4 &v) {
    float invLen = rsqrtf(dot(v, v)); // 1 / sqrt(dot(v,v))
    return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, 0.0f);
}

inline __host__ __device__ float4 cross3(const float4 &a, const float4 &b) {
    return make_float4(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x,
        0.0f
    );
}

inline __host__ __device__ float clamp(float x, float minVal, float maxVal) {
    if (x < minVal) return minVal;
    if (x > maxVal) return maxVal;
    return x;
}


inline __host__ std::ostream& operator<< (std::ostream& out, const float4& v )
{
    out << "<" << v.x << " " << v.y << " " << v.z << " " << v.w << ">";
    return out; 
}

inline __device__ void toWorld(const float4& wo_local, const float4& n, float4& wo_world)
{
    float4 t, b;
    if (fabs(n.x) > fabs(n.z))
        t = normalize(f4(-n.y, n.x, 0.0f));
    else
        t = normalize(f4(0.0f, -n.z, n.y));

    b = cross3(n, t);
    wo_world = wo_local.x * t + wo_local.y * b + wo_local.z * n;
}

inline __device__ void toLocal(const float4& wo_world, const float4& n, float4& wo_local)
{
    float4 t, b;
    if (fabs(n.x) > fabs(n.z))
        t = normalize(f4(-n.y, n.x, 0.0f));
    else
        t = normalize(f4(0.0f, -n.z, n.y));

    b = cross3(n, t);
    wo_local = f4(dot(wo_world, t), dot(wo_world, b), dot(wo_world, n), 0.0f);
}

// Component-wise min of two float4s
inline __host__ float4 fminf4(const float4 &a, const float4 &b) {
    return f4(
        fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z)
    );
}

// Component-wise max of two float4s
inline __host__ float4 fmaxf4(const float4 &a, const float4 &b) {
    return f4(
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z)
    );
}

inline __host__ __device__ float getFloat4Component(const float4 &v, int i) {
    switch(i) {
        case 0: return v.x;
        case 1: return v.y;
        case 2: return v.z;
        case 3: return v.w;
        default: return 0.0f; // or handle error
    }
}

inline __host__ __device__ void setFloat4Component(float4 &v, int i, float value) {
    switch(i) {
        case 0: v.x = value; break;
        case 1: v.y = value; break;
        case 2: v.z = value; break;
        case 3: v.w = value; break;
        // optionally handle invalid index
    }
}

inline __host__ float surfaceArea(float4 min, float4 max)
{
    return 2.0f * ((max.x-min.x) * (max.y-min.y) +  (max.x-min.x) * (max.z-min.z) + (max.z-min.z) * (max.y-min.y));
}
