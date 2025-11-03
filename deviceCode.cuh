#include <cuda_runtime.h>
#include <cuda.h>
#include "util.cuh"
#include "objects.cuh"

//__global__ void colorPixel (int w, int h, float4* colors);
__host__ void launch(int maxDepth, Material* materials, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors);
