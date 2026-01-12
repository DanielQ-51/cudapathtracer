#pragma once

#include "objects.cuh"

//__global__ void colorPixel (int w, int h, float4* colors);
__host__ void updateConstants(RenderConfig& config);

__host__ void launch_unidirectional(int maxDepth, Camera camera, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors);

__host__ void launch_naive_unidirectional(int maxDepth, Camera camera, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors);

__host__ void launch_bidirectional(int eyeDepth, int lightDepth, Camera camera, PathVertices* eyePath, PathVertices* lightPath, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, int w, int h, float4 sceneCenter, float sceneRadius, float4* colors, float4* overlay, bool postProcess);

__host__ void launch_VCM(int eyeDepth, int lightDepth, Camera camera, VCMPathVertices* lightPath, Photons* photons, Photons* photons_sorted, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, int w, int h, float4 sceneCenter, float sceneRadius, float4 sceneMin, float4* colors, float4* overlay, bool postProcess, float mergeRadiusPower, float initialRadiusMultiplier);