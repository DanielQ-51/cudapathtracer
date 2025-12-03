#include <cuda_runtime.h>
#include <cuda.h>
#include "util.cuh"
#include "objects.cuh"

//__global__ void colorPixel (int w, int h, float4* colors);
__host__ void launch_unidirectional(int maxDepth, Camera camera, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors);

__host__ void launch_bidirectional(int maxDepth, Camera camera, PathVertex* eyePath, PathVertex* lightPath, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, int w, int h, float4 sceneCenter, float sceneRadius, float4* colors);
