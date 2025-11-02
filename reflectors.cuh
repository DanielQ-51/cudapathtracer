#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include "util.cuh"
#include "objects.cuh"
#include <curand_kernel.h>

__device__ void cosine_f(const float4& baseColor, float4& newColor)
{
    newColor = baseColor/PI;
}

__device__ void cosine_pdf(const float4& wo_local, float& pdf)
{
    pdf = fmaxf(wo_local.z, EPSILON)/PI;
}


__device__ void cosine_sample_f(curandState& localState, float4& wo, float& pdf)
{
    float u1 = curand_uniform(&localState);
    float u2 = curand_uniform(&localState); 

    float r = sqrt(u1);
    float phi = 2.0f * PI * u2; 

    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(1.0f - u1); 

    wo = f4(x,y,z);
    cosine_pdf(wo, pdf);
}

__device__ void f(Material* materials, int materialID, const float4& wi)
{

}

__device__ void sample_f(Material* materials, int materialID, const float4& wi, float4& wo)
{
    
}

__device__ void pdf(Material* materials, int materialID, const float4& wi, const float4& wo)
{

}

__device__ void f(Material* materials, int materialID, const float4& wi)
{

}

__device__ void sample_f(Material* materials, int materialID, const float4& wi, float4& wo)
{
    Material mat = materials[materialID-1];
}

__device__ void pdf(Material* materials, int materialID, const float4& wi, const float4& wo)
{

}