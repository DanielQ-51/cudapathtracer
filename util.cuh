#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <curand_kernel.h>
using namespace std;

__device__ __constant__ float EPSILON = 0.00001f;
__device__ __constant__ float RAY_EPSILON = 0.0001f;
__device__ __constant__ float PI = 3.141592f;
__device__ __constant__ float SKY_RADIUS = 100.0f;
__device__ __constant__ float MAX_FIREFLY_LUM = 20.0f;

__device__ __constant__ bool BDPT_LIGHTTRACE = true;
__device__ __constant__ bool BDPT_NEE = false;
__device__ __constant__ bool BDPT_NAIVE = false;
__device__ __constant__ bool BDPT_CONNECTION = false;

__device__ __constant__ bool BDPT_DRAWPATH = false;

inline __host__ __device__ __forceinline__ float4 f4(float x, float y, float z, float w = 0.0f) {
    return make_float4(x, y, z, w);
}

inline __host__ __device__ __forceinline__ float4 f4() {return make_float4(0,0,0,0);}

inline __host__ __device__ __forceinline__ float4 f4(float a) {return make_float4(a,a,a,0);}

inline __host__ __device__ __forceinline__ float2 f2(float x, float y) {return make_float2(x, y);}

inline __host__ __device__ __forceinline__ float2 f2() {return make_float2(0,0);}

inline __host__ __device__ __forceinline__ float2 f2(float a) {return make_float2(a,a);}

inline __host__ __device__ __forceinline__ float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ __forceinline__ float4 operator-(const float4 &a, const float4 &b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator*(const float4 &a, float t) {
    return make_float4(a.x * t, a.y * t, a.z * t, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator*(float t, const float4 &a) {
    return a * t;
}

inline __host__ __device__ __forceinline__ float2 operator*(const float2 &a, float t) {
    return make_float2(a.x * t, a.y * t);
}

inline __host__ __device__ __forceinline__ float2 operator*(float t, const float2 &a) {
    return a * t;
}

inline __host__ __device__ __forceinline__ float4 operator/(const float4 &a, float t) {
    return make_float4(a.x / t, a.y / t, a.z / t, 0.0f);
}

inline __host__ __device__ __forceinline__ float4& operator+=(float4 &a, const float4 &b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

inline __host__ __device__ __forceinline__ float4& operator*=(float4 &a, float t) {
    a.x *= t; a.y *= t; a.z *= t;
    return a;
}

inline __host__ __device__  __forceinline__ float4& operator*=(float4& a, const float4& b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
    return a;
}

inline __host__ __device__ __forceinline__ float4& operator/=(float4 &a, float t) {
    a.x /= t; a.y /= t; a.z /= t;
    return a;
}

inline __host__ __device__ __forceinline__ float4 operator*(const float4& a, const float4& b)
{
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator/(const float4& a, const float4& b)
{
    return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator-(const float4& v)
{
    return make_float4(-v.x, -v.y, -v.z, -v.w);
}

inline __host__ __device__ __forceinline__ float dot(const float4 &a, const float4 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ __forceinline__ float length(const float4 &v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ __forceinline__ float lengthSquared(const float4 &v) {
    return dot(v, v);
}

inline __host__ __device__ __forceinline__ float4 normalize(const float4 &v) {
    float invLen = rsqrtf(dot(v, v)); // 1 / sqrt(dot(v,v))
    return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 cross3(const float4 &a, const float4 &b) {
    return make_float4(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x,
        0.0f
    );
}

inline __host__ __device__ __forceinline__ float clamp(float x, float minVal, float maxVal) {
    if (x < minVal) return minVal;
    if (x > maxVal) return maxVal;
    return x;
}


inline __host__ std::ostream& operator<< (std::ostream& out, const float4& v )
{
    out << "<" << v.x << " " << v.y << " " << v.z << " " << v.w << ">";
    return out; 
}

inline __device__ __forceinline__ void toWorld(const float4& wo_local, const float4& n, float4& wo_world)
{
    float4 t, b;
    if (fabs(n.x) > fabs(n.z))
        t = normalize(f4(-n.y, n.x, 0.0f));
    else
        t = normalize(f4(0.0f, -n.z, n.y));

    b = cross3(n, t);
    wo_world = wo_local.x * t + wo_local.y * b + wo_local.z * n;
}

inline __device__ __forceinline__ void toLocal(const float4& wo_world, const float4& n, float4& wo_local)
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
inline __host__ __device__ __forceinline__ float4 fminf4(const float4 &a, const float4 &b) {
    return f4(
        fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z)
    );
}

// Component-wise max of two float4s
inline __host__ __device__ __forceinline__ float4 fmaxf4(const float4 &a, const float4 &b) {
    return f4(
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z)
    );
}

inline __host__ __device__ __forceinline__ float getFloat4Component(const float4 &v, int i) {
    switch(i) {
        case 0: return v.x;
        case 1: return v.y;
        case 2: return v.z;
        case 3: return v.w;
        default: return 0.0f; // or handle error
    }
}

inline __host__ __device__ __forceinline__ void setFloat4Component(float4 &v, int i, float value) {
    switch(i) {
        case 0: v.x = value; break;
        case 1: v.y = value; break;
        case 2: v.z = value; break;
        case 3: v.w = value; break;
        // optionally handle invalid index
    }
}

inline __host__ float surfaceArea(const float4& min, const float4& max)
{
    float dx = max.x - min.x;
    float dy = max.y - min.y;
    float dz = max.z - min.z;
    return 2.0f * (dx * dy + dx * dz + dy * dz);
}

__host__ __device__ __forceinline__ float4 sqrtf4(const float4& v) {
    return make_float4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), 0.0f);
}

__host__ __device__ __forceinline__ float4 rotateX(const float4& v, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return f4(
        v.x,
        v.y * c - v.z * s,
        v.y * s + v.z * c
    );
}

__host__ __device__ __forceinline__ float4 rotateY(const float4& v, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return f4(
        v.x * c + v.z * s,
        v.y,
        -v.x * s + v.z * c
    );
}

__host__ __device__ __forceinline__ float4 rotateZ(const float4& v, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return f4(
        v.x * c - v.y * s,
        v.x * s + v.y * c,
        v.z
    );
}

__device__ __forceinline__ float4 sampleSphere(curandState& localState, float R)
{
    float u = curand_uniform(&localState);
    float v = curand_uniform(&localState);

    float z = 1.0f - 2.0f * u;          // cos(theta) uniformly distributed
    float r = sqrtf(max(0.f, 1.f - z*z));

    float phi = 2.0f * 3.141592f * v;

    float x = r * cosf(phi);
    float y = r * sinf(phi);

    return f4(R * x, R * y, R * z);
}

__device__ inline float luminance(float4 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}