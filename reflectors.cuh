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


__device__ void cosine_sample_f(curandState& localState, const float4& baseColor, float4& wo, float4& f_val, float& pdf)
{
    float u1 = curand_uniform(&localState);
    float u2 = curand_uniform(&localState); 

    float r = sqrt(u1);
    float phi = 2.0f * PI * u2; 

    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(1.0f - u1); 

    wo = f4(x,y,z);

    cosine_f(baseColor, f_val);
    cosine_pdf(wo, pdf);
}

__device__ float D_GGX(const float4& h, float alpha) 
{
    float cosThetaH = h.z;
    float alpha2 = alpha * alpha;
    float denom = cosThetaH*cosThetaH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * denom * denom);
}

__device__ float G1_Smith(float nDotV, float alpha) 
{
    float a = alpha * sqrtf(1.0f - nDotV*nDotV) / nDotV;
    return 2.0f / (1.0f + sqrtf(1.0f + a*a));
}

__device__ float G1_GGX(const float4& v, const float4& h, float alpha)
{
    float cosTheta = v.z;
    float tanTheta = sqrtf(1.0f - cosTheta * cosTheta) / cosTheta;
    float a = 1.0f / (alpha * tanTheta);
    if (a < 1.6f)
        return (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a);
    else
        return 1.0f;
}

/*
__device__ float G_Smith(float nDotWi, float nDotWo, float alpha) 
{
    return G1_Smith(nDotWi, alpha) * G1_Smith(nDotWo, alpha);
}*/

__device__ float G_Smith(const float4& wi, const float4& wo, const float4& h, float alpha)
{
    return G1_GGX(wi, h, alpha) * G1_GGX(wo, h, alpha);
}

__device__ float4 Fresnel_Conductor(float cosTheta, const float4& eta, const float4& k) {
    float4 cosTheta2 = f4(cosTheta*cosTheta);
    float4 sinTheta2 = f4(1.0f) - cosTheta2;
    
    float4 eta2 = eta * eta;
    float4 k2 = k * k;
    
    float4 t0 = eta2 - k2 - sinTheta2;
    float4 a2plusb2 = sqrtf4(t0*t0 + 4.0f * eta2 * k2);
    float4 t1 = a2plusb2 + cosTheta2;
    float4 a = sqrtf4(0.5f * (a2plusb2 + t0));
    float4 t2 = 2.0f * cosTheta * a;
    
    float4 Rs = (t1 - t2) / (t1 + t2);
    float4 t3 = cosTheta2 * a2plusb2 + sinTheta2 * sinTheta2;
    float4 t4 = t2 * sinTheta2;
    float4 Rp = (t3 - t4) / (t3 + t4);
    return (Rs + Rp) * 0.5f;
    //return (t1 - t2) / (t1 + t2);
}

__device__ void microfacet_f(const float4& eta, const float4& k, float roughness, const float4& wi, const float4& wo, float4& f_val)
{
    if (wi.z <= 0.0f || wo.z <= 0.0f) {
        f_val = f4(0.0f);
        return;
    }
    float nDotWi = wi.z;
    float nDotWo = wo.z;

    float4 h = normalize(wi + wo);

    if (h.z <= 0.0f) h = f4(-h.x, -h.y, -h.z); // flip so h.z > 0
    
    float alpha = roughness * roughness;

    float D = D_GGX(h, alpha);
    float G = G_Smith(wi, wo, h, alpha);

    float4 f = Fresnel_Conductor(dot(wi, h), eta, k);

    f_val = (D * G * f) / fmaxf(4.0f * nDotWi * nDotWo, EPSILON);
}

__device__ void microfacet_pdf(const float& roughness, const float4& wi, const float4& wo, float& pdf)
{

    float4 h = normalize(wi + wo);
    float D = D_GGX(h, roughness *roughness);
    float denom = 4.0f * dot(wo, h);
    //if (denom <= EPSILON) { pdf = 0.0f; return; }
    pdf = (D * h.z) / (denom);
}

__device__ void microfacet_sample_f(curandState& localState, const float4& eta, const float4& k, float roughness, const float4& wi, 
    float4& wo, float4& f_val, float& pdf)
{
    float u1 = curand_uniform(&localState);

    float alpha = roughness * roughness;
    float phi = 2.0f * PI * curand_uniform(&localState);
    float cosTheta = sqrtf((1.0f - u1) / (1.0f + (alpha*alpha - 1.0f) * u1));
    float sinTheta = sqrtf(fmaxf(1.0f - cosTheta*cosTheta, 0.0f));

    float4 h_local;
    h_local.x = sinTheta * cosf(phi);
    h_local.y = sinTheta * sinf(phi);
    h_local.z = cosTheta; // points along local normal

    wo = 2.0f * dot(wi, h_local) * h_local - wi;
    if (wo.z <= 0.0f) wo.z = -wo.z;
    
    microfacet_f(eta, k, roughness, wi, wo, f_val); // f_val set
    microfacet_pdf(roughness, wi, wo, pdf); // pdf set
}

__device__ float4 Fresnel_Schlick(float cosTheta, const float4& F0) 
{
    return F0 + (f4(1.0f) - F0) * powf(1.0f - cosTheta, 5.0f);
}

__device__ void f_eval(curandState& localState, const Material* materials, int materialID, 
    const float4& wi, const float4& wo, float4& f_val)
{
    const Material& mat = materials[materialID];
    if (mat.type == MAT_DIFFUSE)
    {
        cosine_f(mat.albedo, f_val);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_f(mat.eta, mat.k, mat.roughness, -wi, wo, f_val);
    }
}

__device__ void sample_f_eval(curandState& localState, const Material* materials, int materialID, 
    const float4& wi, float4& wo, float4& f_val, float& pdf)
{
    const Material& mat = materials[materialID];
    if (mat.type == MAT_DIFFUSE)
    {
        cosine_sample_f(localState, mat.albedo, wo, f_val, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_sample_f(localState, mat.eta, mat.k, mat.roughness, -wi, wo, f_val, pdf);
    }
}

__device__ void pdf_eval(Material* materials, int materialID, const float4& wi, const float4& wo, float& pdf)
{
    const Material& mat = materials[materialID];
    if (mat.type == MAT_DIFFUSE)
    {
        cosine_pdf(wo, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_pdf(mat.roughness, -wi, wo, pdf);
    }
}