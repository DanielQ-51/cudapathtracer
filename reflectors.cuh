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
    //return (Rs + Rp) * 0.5f;
    return (t1 - t2) / (t1 + t2);
}

__device__ void microfacet_metal_f(const float4& eta, const float4& k, float roughness, const float4& wi, const float4& wo, float4& f_val)
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

__device__ void microfacet_metal_pdf(const float& roughness, const float4& wi, const float4& wo, float& pdf)
{

    float4 h = normalize(wi + wo);
    float D = D_GGX(h, roughness *roughness);
    float denom = 4.0f * dot(wo, h);
    //if (denom <= EPSILON) { pdf = 0.0f; return; }
    pdf = (D * h.z) / (denom);
}

__device__ void microfacet_metal_sample_f(curandState& localState, const float4& eta, const float4& k, float roughness, const float4& wi, 
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
    
    microfacet_metal_f(eta, k, roughness, wi, wo, f_val); // f_val set
    microfacet_metal_pdf(roughness, wi, wo, pdf); // pdf set
}

// fabs is used here for costheta
__device__ inline float schlick_fresnel(float cosTheta, float etaI, float etaT)
{
    float R0 = (etaI - etaT) / (etaI + etaT);
    R0 = R0 * R0;
    return R0 + (1.0f - R0) * powf(1.0f - fabsf(cosTheta), 5.0f);
}

// -----------------------------------------------------------------------------
// EVAL (given wi, wo, etaI, etaT, and reflect boolean)
// -----------------------------------------------------------------------------
__device__ void smooth_dielectric_f(
    const float4& wi, const float4& wo,
    float etaI, float etaT, bool reflect, bool TIR,
    float4& f_val)
{
    printf("never called");
    float cosThetaI = wi.z;
    float cosThetaO = wo.z;
    float F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (reflect) {
        // Perfect specular reflection: wo.z = wi.z
        // BSDF value for delta reflection (handled as Dirac delta)
        if (TIR)
            f_val = f4(1.0f / fabsf(cosThetaO));
        else
            f_val = f4(F / fabsf(cosThetaO));
    } else {
        // Perfect specular refraction
        float eta = etaI / etaT;
        f_val = f4((1.0f - F) * eta * eta / fabsf(cosThetaO));
    }
}

// -----------------------------------------------------------------------------
// SAMPLE_F (importance sample the reflection/refract direction)
// -----------------------------------------------------------------------------
__device__ void smooth_dielectric_sample_f(curandState& localState,
    const float4& wi, float etaI, float etaT, float4& wo, float4& f_val, float& pdf)
{
    float cosThetaI = wi.z; // always pos

    float eta = etaI / etaT;
    float cosThetaT2 = 1.0f - eta * eta * (1.0f - cosThetaI * cosThetaI);

    float F;
    if (cosThetaT2 < 0.0f)
    {
        wo = f4(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;
        f_val = f4(1.0f / fabsf(cosThetaO));
        pdf = 1.0f;
        return;
    }
    
    F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (curand_uniform(&localState) < F) {
        wo = f4(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;
        pdf = F;
        f_val = f4(F / fabsf(cosThetaO));
    } 
    else {
        float eta = etaI / etaT;

        wo = f4(
            -eta * wi.x, 
            -eta * wi.y, 
            - (sqrtf(cosThetaT2))
        );

        float cosThetaO = wo.z;
        // BSDF term
        f_val = f4((1.0f - F) * eta * eta / fabsf(cosThetaO));
        pdf = 1.0f - F;
    }
    
}

// -----------------------------------------------------------------------------
// PDF (for MIS weighting)
// -----------------------------------------------------------------------------
__device__ void smooth_dielectric_pdf(
    const float4& wi, const float4& wo,
    float etaI, float etaT, bool reflect, bool TIR,
    float& pdf)
{
    printf("never called");
    float cosThetaI = wi.z;
    float F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (reflect) {
        if (TIR)
            pdf = 1.0f;
        else
            pdf = F;
    } else {
        pdf = 1.0f - F;
    }
}
// wi is always point away from the surface on the positive z hemisphere. since we flipped the intersect normal if it was a backface before converting to local
__device__ void dumb_smooth_dielectric_sample_f(curandState& localState,
    const float4& wi, float etaSurface, bool backface, float4& wo, float4& f_val, float& pdf)
{
    float etaI, etaT;
    if (backface)
    {
        etaI = etaSurface;
        etaT = 1.0f;
    }
    else
    {
        etaI = 1.0f;
        etaT = etaSurface;
    }

    float cosThetaI = wi.z; // always pos

    float eta = etaI / etaT;
    float cosThetaT2 = 1.0f - eta * eta * (1.0f - cosThetaI * cosThetaI);

    float F;
    if (cosThetaT2 < 0.0f)
    {
        wo = f4(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;
        f_val = f4(1.0f / fabsf(cosThetaO));
        pdf = 1.0f;
        return;
    }
    
    F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (curand_uniform(&localState) < F) {
        wo = f4(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;
        pdf = F;
        f_val = f4(F / fabsf(cosThetaO));
    } 
    else {
        float eta = etaI / etaT;

        wo = f4(
            -eta * wi.x, 
            -eta * wi.y, 
            - (sqrtf(cosThetaT2))
        );

        float cosThetaO = wo.z;
        // BSDF term
        f_val = f4((1.0f - F) * eta * eta / fabsf(cosThetaO));
        pdf = 1.0f - F;
    }
}

// For dielectrics, when this function is called, we know whether or not it refracts, and that etaI and etaT are in fact correct
// wi passed in is facing the surface, so we flip it normally. The shading uses wi as pointing away
__device__ void f_eval(curandState& localState, const Material* materials, int materialID, 
    const float4& wi, const float4& wo, float etaI, float etaT, float4& f_val, bool reflect_dielectric, bool TIR)
{
    const Material& mat = materials[materialID];
    if (mat.type == MAT_DIFFUSE)
    {
        cosine_f(mat.albedo, f_val);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_metal_f(mat.eta, mat.k, mat.roughness, -wi, wo, f_val);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        smooth_dielectric_f(-wi, wo, etaI, etaT, reflect_dielectric, TIR, f_val);
    }
}

// For dielectrics, when this function is called, we know whether or not it refracts, and that etaI and etaT are in fact correct
// wi passed in is facing the surface, so we flip it normally. The shading uses wi as pointing away
__device__ void sample_f_eval(curandState& localState, const Material* materials, int materialID, 
    const float4& wi, float etaI, float etaT, bool backface, float4& wo, float4& f_val, float& pdf)
{
    const Material& mat = materials[materialID];
    if (mat.type == MAT_DIFFUSE)
    {
        cosine_sample_f(localState, mat.albedo, wo, f_val, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_metal_sample_f(localState, mat.eta, mat.k, mat.roughness, -wi, wo, f_val, pdf);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        //dumb_smooth_dielectric_sample_f(localState, -wi, mat.ior, backface , wo, f_val, pdf);
        smooth_dielectric_sample_f(localState, -wi, etaI, etaT, wo, f_val, pdf);
    }
}

// For dielectrics, when this function is called, we know whether or not it refracts, and that etaI and etaT are in fact correct
// wi passed in is facing the surface, so we flip it normally. The shading uses wi as pointing away
__device__ void pdf_eval(Material* materials, int materialID, const float4& wi, const float4& wo, float etaI, float etaT, float& pdf, bool reflect_dielectric, bool TIR)
{
    const Material& mat = materials[materialID];
    if (mat.type == MAT_DIFFUSE)
    {
        cosine_pdf(wo, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_metal_pdf(mat.roughness, -wi, wo, pdf);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        smooth_dielectric_pdf(-wi, wo, etaI, etaT, reflect_dielectric, TIR, pdf);
    }
}