
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include "util.cuh"
#include "objects.cuh"
#include "reflectors.cuh"
#include <curand_kernel.h>


/*__global__ void colorPixel (int w, int h, float4* colors)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int pixelIdx = y*w + x;

    //colors[pixelIdx] = make_float4(1.0f,1.0f,0.0f,0.0f);

    colors[pixelIdx] = make_float4 ((1.0f * x)/w,(1.0f * y)/w, 0.0f, 0.0f);
}*/
__device__ void triangleIntersect(Vertex* verts, Triangle* tri, const Ray& r, float4& barycentric, float& tval)
{
    float4 tria = verts[tri->aInd].position;
    float4 trib = verts[tri->bInd].position;
    float4 tric = verts[tri->cInd].position;
    float4 e1 = trib - tria;
    float4 e2 = tric - tria;

    float4 h = cross3(r.direction, e2);
    float a = dot(h, e1);

    if (fabs(a) < EPSILON)
    {
        barycentric = f4();
        tval = -1.0f;
        return;
    }
    float f = 1.0/a;

    float4 s = r.origin-tria;
    float u = f * dot(s, h);
    float4 q = cross3(s, e1);
    float v = f * dot(r.direction, q);
    float t = f * dot(e2, q);


    if (((u >= 0) && (v >= 0) && (u + v <= 1)) && t > EPSILON)
    {
        barycentric = f4(u, v, 1.0f-u-v);
        tval = t;
        return;
    }
    else
    {
        barycentric = f4();
        tval = -1.0f;
        return;
    }
}

__device__ bool aabbIntersect(const Ray& r, float4 minCorner, float4 maxCorner, float& tmin, float& tmax)
{
    tmin = -1e30f; // initialize to -infinity
    tmax = 1e30f;  // initialize to +infinity

    // Compute inverse ray direction once
    float4 invDir = make_float4(
        1.0f / r.direction.x,
        1.0f / r.direction.y,
        1.0f / r.direction.z,
        0.0f
    );

    // X axis
    float tx1 = (minCorner.x - r.origin.x) * invDir.x;
    float tx2 = (maxCorner.x - r.origin.x) * invDir.x;
    float tx_min = fminf(tx1, tx2);
    float tx_max = fmaxf(tx1, tx2);
    tmin = fmaxf(tmin, tx_min);
    tmax = fminf(tmax, tx_max);

    // Y axis
    float ty1 = (minCorner.y - r.origin.y) * invDir.y;
    float ty2 = (maxCorner.y - r.origin.y) * invDir.y;
    float ty_min = fminf(ty1, ty2);
    float ty_max = fmaxf(ty1, ty2);
    tmin = fmaxf(tmin, ty_min);
    tmax = fminf(tmax, ty_max);

    // Z axis
    float tz1 = (minCorner.z - r.origin.z) * invDir.z;
    float tz2 = (maxCorner.z - r.origin.z) * invDir.z;
    float tz_min = fminf(tz1, tz2);
    float tz_max = fmaxf(tz1, tz2);
    tmin = fmaxf(tmin, tz_min);
    tmax = fminf(tmax, tz_max);
    
    return (tmax >= tmin) && (tmax > 0.0f);
}

__device__ void BVHSceneIntersect(const Ray& r, BVHnode* BVH, int* BVHindices,Vertex* verts, Triangle* scene, Intersection& intersect, float max_t = 999999.0f, bool shortCircuit = false)
{
    intersect.valid = false;
    float min_t = 3.402823466e+38f;

    int nodeStack[64]; // Or 64. A stack depth of 32 is usually fine for a good BVH.
    int stackTop = 0;
    nodeStack[stackTop++] = 0; // Push the root node (index 0)

    while (stackTop > 0)
    {
        // Pop the next node to check
        int currentIndex = nodeStack[--stackTop];
        BVHnode& node = BVH[currentIndex];

        // 2. If it's a leaf node, check its triangles
        if (node.primCount > 0)
        {
            // This is your existing 'for' loop logic, which is correct.
            // Loop through primitives and update min_t and intersect.
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int idx = BVHindices[i];
                Triangle* tri = &scene[idx];
                float4 barycentric;
                float t;
                triangleIntersect(verts, tri, r, barycentric, t);

                // NOTE: Here, max_t is your original max_t, 
                // but min_t is the *current closest hit*
                if (shortCircuit && ((t != -1.0f) && (t < min_t && t < max_t)))
                {
                    intersect.valid = true;
                    return;
                }
                else if ((t != -1.0f) && (t < min_t && t < max_t))
                {
                    min_t = t; // Update the closest-hit distance
                    intersect.point = r.at(t);
                    intersect.color = verts[tri->aInd].color * barycentric.z + verts[tri->bInd].color * barycentric.x + verts[tri->cInd].color * barycentric.y;
                    intersect.normal = normalize(verts[tri->aInd].normal * barycentric.z + verts[tri->bInd].normal * barycentric.x + verts[tri->cInd].normal * barycentric.y);
                    intersect.materialID = tri->materialID;
                    intersect.emission = tri->emission;
                    intersect.valid = true;
                    intersect.tri = *tri;
                }
            }
        }
        // 3. If it's an internal node, push its children onto the stack
        else
        {
            if (node.left >= 0 || node.right >= 0)
            {
                float tminL, tmaxL, tminR, tmaxR;
                bool hitLeft = false, hitRight = false;

                // Test left child if it exists
                if (node.left >= 0)
                    hitLeft = aabbIntersect(r, BVH[node.left].aabbMIN, BVH[node.left].aabbMAX, tminL, tmaxL);

                // Test right child if it exists
                if (node.right >= 0)
                    hitRight = aabbIntersect(r, BVH[node.right].aabbMIN, BVH[node.right].aabbMAX, tminR, tmaxR);

                // If both children were hit, push the farther one first
                if (hitLeft && hitRight)
                {
                    if (tminL < tminR)
                    {
                        nodeStack[stackTop++] = node.right; // farther
                        nodeStack[stackTop++] = node.left;  // nearer
                    }
                    else
                    {
                        nodeStack[stackTop++] = node.left;  // farther
                        nodeStack[stackTop++] = node.right; // nearer
                    }
                }
                else if (hitLeft)
                {
                    nodeStack[stackTop++] = node.left;
                }
                else if (hitRight)
                {
                    nodeStack[stackTop++] = node.right;
                }
            }
        }
    }
}

__device__ void sceneIntersection(const Ray& r, Vertex* verts, Triangle* scene, int triNum, 
    Intersection& intersect , float max_t = 999999.0f, bool shortCircuit = false)
{
    intersect.valid = false;
    float min_t = 3.402823466e+38f;
    
    for (int i = 0; i < triNum; i++)
    {
        Triangle* tri = &scene[i];
        float4 barycentric;
        float t;
        triangleIntersect(verts, tri, r, barycentric, t);
        if (shortCircuit && ((t != -1.0f) && (t < min_t && t < max_t)))
        {
            intersect.valid = true;
            return;
        }
        else if ((t != -1.0f) && (t < min_t && t < max_t))
        {
            min_t = t;
            intersect.point = r.at(t);
            //intersect.normal = verts[tri->aInd].normal;
            intersect.color = verts[tri->aInd].color * barycentric.z + verts[tri->bInd].color * barycentric.x + verts[tri->cInd].color * barycentric.y;
            intersect.normal = normalize(verts[tri->aInd].normal * barycentric.z + verts[tri->bInd].normal * barycentric.x + verts[tri->cInd].normal * barycentric.y);
            intersect.materialID = tri->materialID;
            intersect.emission = tri->emission;
            intersect.valid = true;
            intersect.tri = *tri;
        }
    }
}

__global__ void initRNG(curandState* states, int width, int height, unsigned long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(seed, idx, 0, &states[idx]);  
}

__device__ void nextEventEstimation(curandState& localState, BVHnode* BVH, int* BVHindices, const float4& wo, Vertex* vertices, int vertNum,
    Triangle* scene, int triNum, Triangle* lights, int lightNum, int materialID,const Intersection& intersect, 
    float& light_pdf, float4& contribution, Triangle* light = nullptr, const Intersection* newIntersect = nullptr)
{
    contribution = f4(0.0f,0.0f,0.0f);
    Triangle l;
    float4 apos;
    float4 bpos;
    float4 cpos;
    float u;
    float v;
    float4 p;
    float4 n;

    if (light == nullptr)
    {   
        if (lightNum == 0)
        {
            light_pdf = -1.0f;
            return;
        }
        int index = min(static_cast<int>(curand_uniform(&localState) * lightNum), lightNum - 1);
        l = lights[index];
        apos = vertices[l.aInd].position;
        bpos = vertices[l.bInd].position;
        cpos = vertices[l.cInd].position;
        u = sqrtf(curand_uniform(&localState));
        v = curand_uniform(&localState);

        p = (1.0f - u) * apos + u * (1.0f - v) * bpos + u * v * cpos; // point on light
        n = intersect.normal;
    }
    else 
    {
        l = *light;
        apos = vertices[l.aInd].position;
        bpos = vertices[l.bInd].position;
        cpos = vertices[l.cInd].position;
        p = newIntersect->point;
        n = newIntersect->normal;
    }
    


    float4 surfaceToLight = p-intersect.point;
    
    
    float4 wi = normalize(surfaceToLight);
    Ray r = Ray(intersect.point + n * EPSILON, wi);
    
    float t;
    float4 dummy;
    triangleIntersect(vertices, &l, r, dummy, t);
    
    Intersection sceneIntersect = Intersection();
    //sceneIntersection(r, vertices, scene, triNum, sceneIntersect, t*(0.9999), true);
    BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, sceneIntersect, t*(0.9999), true);
    // following if statement tests for scene intersection (direct light) AND
    // whether the original light intersect was valid
    if (!sceneIntersect.valid && t != -1.0f) // direct LOS from intersection to light
    {
        float distanceSQR = lengthSquared(surfaceToLight);
        float4 lightNormal = vertices[l.aInd].normal;

        float cosThetaLight = fmaxf(dot(lightNormal, -wi), EPSILON);
        float cosThetaSurface = fmaxf(dot(n, wi), EPSILON);

        float G = cosThetaLight * cosThetaSurface/distanceSQR;
        float area = 0.5f * length(cross3(bpos - apos, cpos - apos));
        
        light_pdf = distanceSQR / (cosThetaLight * lightNum * area);
        float4 Le = l.emission;
        float4 f_val;
        cosine_f(intersect.color, f_val);

        contribution = f_val * Le / (light_pdf);
    }
    else {}
}

__global__ void Li (curandState* rngStates, BVHnode* BVH, int* BVHindices, int maxDepth, Vertex* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    float4 colorSum = f4();

    curandState localState = rngStates[pixelIdx];
    float du = curand_uniform(&localState);
    float dv = curand_uniform(&localState);

    float4 a = f4();
    float4 cameraOrigin = f4();
    Ray r = Ray();
    float4 beta = f4();
    float4 Li = f4();
    int materialID = -1;
    int depth = -1;
    float4 f_val = f4();
    float4 wo_local = f4();
    float pdf = EPSILON;
    float4 wo_world = f4();
    float4 wi_world = f4();
    float4 wi_local = f4();
    float light_pdf = EPSILON;
    float4 nee = f4();
    Intersection intersect = Intersection();
    Intersection previousintersect = Intersection();
    float neeWeight = EPSILON; // reused for all MIS interactions
    float bsdfWeight = EPSILON; // reused for all MIS interactions
    float p = EPSILON;// for russian roulette
    float luminance = EPSILON; // for russian roulette

    for (int currSample = 0; currSample < numSample; currSample++)
    {   
        beta = f4(1.0f, 1.0f, 1.0f);
        Li = f4();

        du = curand_uniform(&localState);
        dv = curand_uniform(&localState);
        cameraOrigin = f4(0.0f,0.0f,1.0f);
        a = f4(cameraOrigin.x + (x + 1.0f*du - 0.5f - w/2.0f) * (1.0f / w), 
                        cameraOrigin.y + (y + 1.0f*dv - 0.5f - h/2.0f) * (1.0f / h),
                        cameraOrigin.z-1.0f);
        r.origin = cameraOrigin;
        r.direction = a-cameraOrigin;

        for (depth = 0; depth < 100; depth++)
        {
            intersect.valid = false;
            //sceneIntersection(r, vertices, scene, triNum, intersect);
            BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

            if (!intersect.valid) 
            {
                Li += beta * f4(0.4f,0.4f,0.7f);
                break;
            }
            materialID = intersect.materialID;

            if (lengthSquared(intersect.emission) > EPSILON)
            {
                if (depth == 0)
                {
                    Li += beta * intersect.emission;
                }
                else if (useMIS)
                {
                    nextEventEstimation(localState, BVH, BVHindices, wi_local, vertices, vertNum, scene, 
                        triNum, lights, lightNum, materialID, previousintersect, light_pdf, 
                        nee, &intersect.tri, &previousintersect);
                    
                    if (light_pdf > EPSILON)
                    {
                        bsdfWeight = pdf * pdf / (light_pdf * light_pdf 
                        + pdf * pdf);
                        Li += beta * intersect.emission * bsdfWeight;
                    }
                    
                    
                    
                }
            }

            toLocal(-r.direction, intersect.normal, wi_local);
            wi_world = normalize(-r.direction);

            if (useMIS)
            {
                nextEventEstimation(localState, BVH, BVHindices, wi_local, vertices, vertNum, scene, 
                triNum, lights, lightNum, materialID, intersect, light_pdf, nee);

                if (light_pdf > EPSILON)
                {
                    neeWeight = light_pdf * light_pdf / (fmaxf(wi_local.z, EPSILON)/PI * fmaxf(wi_local.z, EPSILON)/PI + light_pdf * light_pdf);

                    Li += beta * nee * neeWeight;
                }
                
            }

            cosine_f(intersect.color, f_val);
            cosine_sample_f(localState, wo_local, pdf);

            toWorld(wo_local, intersect.normal, wo_world);
            
            if (pdf < EPSILON) 
            break;

            beta *= (f_val * fabs(wo_local.z) / pdf);

            if (depth > maxDepth)
            {
                luminance = dot(beta, f4(0.2126f, 0.7152f, 0.0722f));
                p = clamp(luminance, 0.05f, 0.99f);

                if (curand_uniform(&localState) > p)   // survive with probability p
                    break;

                beta /= p;  // compensate for the survival probability
            }

            r.origin = intersect.point + intersect.normal * EPSILON;
            r.direction = wo_world;
            previousintersect = intersect;        
        }
        colorSum += Li;
    }
    colors[pixelIdx] = colorSum/numSample;
    rngStates[pixelIdx] = localState;
}

__host__ void launch(int maxDepth, BVHnode* BVH, int* BVHindices, Vertex* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors)
{
    dim3 blockSize(16, 16);  
    dim3 gridSize((w+15)/16, (h+15)/16);
    curandState* d_rngStates;
    cudaMalloc(&d_rngStates, w * h * sizeof(curandState));

    unsigned long seed = 103033UL;
    initRNG<<<gridSize, blockSize>>>(d_rngStates, w, h, seed);
    cudaDeviceSynchronize();

    Li<<<gridSize, blockSize>>>(d_rngStates, BVH, BVHindices, maxDepth, vertices, vertNum, scene, triNum, 
        lights, lightNum, numSample, useMIS, w, h, colors);

    cudaDeviceSynchronize();
    cudaFree(d_rngStates);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error code: " << static_cast<int>(err) << std::endl;
        // only call this if the code isn't catastrophic
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    }
    else
        std::cout << "no cuda error" << std::endl;
}