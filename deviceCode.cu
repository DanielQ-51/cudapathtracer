
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include "util.cuh"
#include "objects.cuh"
#include "reflectors.cuh"
#include "imageUtil.cuh"
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
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
__device__ bool triangleIntersect(Vertices* verts, Triangle* tri, const Ray& r, float4& barycentric, float& tval)
{
    float4 tria = verts->positions[tri->aInd];
    float4 trib = verts->positions[tri->bInd];
    float4 tric = verts->positions[tri->cInd];
    float4 e1 = trib - tria;
    float4 e2 = tric - tria;

    float4 h = cross3(r.direction, e2);
    float a = dot(h, e1);

    if (fabsf(a) < 1e-12f) 
        return false;
    
    float f = 1.0/a;

    float4 s = r.origin-tria;
    float u = f * dot(s, h);
    float4 q = cross3(s, e1);
    float v = f * dot(r.direction, q);
    float t = f * dot(e2, q);


    if (((u >= 0) && (v >= 0) && (u + v <= 1)) && t > 0.0f)
    {
        barycentric = f4(u, v, 1.0f-u-v);
        tval = t;
        return true;
    }
    else
    {
        barycentric = f4();
        return false;
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

__device__ void BVHSceneIntersect(const Ray& r, BVHnode* BVH, int* BVHindices, Vertices* verts, Triangle* scene, Intersection& intersect, float max_t = 999999.0f, int skipTri = -1)
{
    intersect.valid = false;
    float min_t = 3.402823466e+38f;

    int nodeStack[128];
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
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int idx = BVHindices[i];
                if (idx == skipTri) continue;
                Triangle* tri = &scene[idx];
                float4 barycentric;
                float t;
                bool hitTri = triangleIntersect(verts, tri, r, barycentric, t);

                if (hitTri && (t < min_t) && (t < max_t))
                {
                    min_t = t; // Update the closest-hit distance
                    intersect.point = r.at(t);
                    intersect.color = verts->colors[tri->aInd] * barycentric.z + 
                                        verts->colors[tri->bInd] * barycentric.x + 
                                        verts->colors[tri->cInd] * barycentric.y;
                    intersect.normal = normalize(verts->normals[tri->naInd] * barycentric.z + 
                                        verts->normals[tri->nbInd] * barycentric.x + 
                                        verts->normals[tri->ncInd] * barycentric.y);

                    intersect.uv = verts->uvs[tri->uvaInd] * barycentric.z + 
                        verts->uvs[tri->uvbInd] * barycentric.x + 
                        verts->uvs[tri->uvcInd] * barycentric.y;
                    if (dot(intersect.normal, r.direction) > 0.0f) 
                    {
                        intersect.normal = -intersect.normal;
                        intersect.backface = true;
                    }
                    else 
                    {
                        intersect.backface = false;
                    }
                        
                    intersect.materialID = tri->materialID;
                    intersect.emission = tri->emission;
                    intersect.valid = true;
                    //intersect.tri = *tri; // could be bad for performance
                    intersect.triIDX = idx;

                    intersect.dist = t;
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

__device__ void BVHShadowRay(const Ray& r, BVHnode* BVH, int* BVHindices, Vertices* verts, Triangle* scene, Material* materials, float4& throughputScale, float max_t, int skip_tri)
{
    int nodeStack[128];
    int stackTop = 0;
    nodeStack[stackTop++] = 0; // Push the root node (index 0)

    throughputScale = f4(1.0f);
    while (stackTop > 0)
    {
        // Pop the next node to check
        int currentIndex = nodeStack[--stackTop];
        BVHnode& node = BVH[currentIndex];

        // 2. If it's a leaf node, check its triangles
        if (node.primCount > 0)
        {
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int idx = BVHindices[i];
                Triangle* tri = &scene[idx];
                float4 barycentric;
                float t;
                bool hitTri = triangleIntersect(verts, tri, r, barycentric, t);

                if (idx == skip_tri)
                    continue;

                if (hitTri && (t < max_t))
                {
                    int matID = tri->materialID;
                    if (materials[matID].type == MAT_LEAF)
                    {
                        // 1. We hit a leaf. Don't stop. Just darken the ray.
                        float4 transColor = materials[matID].albedo;
                        float transmission = materials[matID].transmission;
                        
                        float4 n = verts->normals[tri->naInd] * barycentric.z + 
                        verts->normals[tri->nbInd] * barycentric.x + 
                        verts->normals[tri->ncInd] * barycentric.y;
                        
                        float cosTheta = fabsf(dot(r.direction, normalize(n)));
                        
                        float F = schlick_fresnel(cosTheta, 1.0f, materials[matID].ior);
                        
                        throughputScale *= transColor * transmission * (1.0f - F);

                        if (fmaxf(throughputScale.x, fmaxf(throughputScale.y, throughputScale.z)) < 0.01f) 
                        {
                            throughputScale = f4(0.0f);
                            return;
                        }
                    }
                    else 
                    {
                        throughputScale = f4(0.0f);
                        return;
                    }
                }
            }
        }
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

__device__ void sceneIntersection(const Ray& r, Vertices* verts, Triangle* scene, int triNum, 
    Intersection& intersect)
{
    intersect.valid = false;
    float min_t = 3.402823466e+38f;
    
    for (int i = 0; i < triNum; i++)
    {
        Triangle* tri = &scene[i];
        float4 barycentric;
        float t;
        bool hitTri = triangleIntersect(verts, tri, r, barycentric, t); // returns true if it hits the tri
        if (hitTri && (t < min_t))
        {
            min_t = t; // Update the closest-hit distance
            intersect.point = r.at(t);
            intersect.color = verts->colors[tri->aInd] * barycentric.z + 
                                verts->colors[tri->bInd] * barycentric.x + 
                                verts->colors[tri->cInd] * barycentric.y;
            intersect.normal = normalize(verts->normals[tri->naInd] * barycentric.z + 
                                verts->normals[tri->nbInd] * barycentric.x + 
                                verts->normals[tri->ncInd] * barycentric.y);

            intersect.uv = verts->uvs[tri->uvaInd] * barycentric.z + 
                verts->uvs[tri->uvbInd] * barycentric.x + 
                verts->uvs[tri->uvcInd] * barycentric.y;
            if (dot(intersect.normal, r.direction) > 0.0f) 
            {
                intersect.normal = -intersect.normal;
                intersect.backface = true;
            }
            else 
            {
                intersect.backface = false;
            }
                
            intersect.materialID = tri->materialID;
            intersect.emission = tri->emission;
            intersect.valid = true;
            //intersect.tri = *tri; // could be bad for performance
            intersect.triIDX = i;

            intersect.dist = t;
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

__device__ void neePDF(Vertices* vertices, Triangle* scene, int lightNum, int lightTriInd, const Intersection& intersect, 
    float& light_pdf, float etaI, float etaT, const Intersection* newIntersect)
{
    Triangle l = scene[lightTriInd];
    float4 apos = vertices->positions[l.aInd];
    float4 bpos = vertices->positions[l.bInd];
    float4 cpos = vertices->positions[l.cInd];
    float4 p = newIntersect->point;
    float4 n = newIntersect->normal;

    float4 surfaceToLight = p-intersect.point;
    float4 wi = normalize(surfaceToLight);

    float distanceSQR = lengthSquared(surfaceToLight);
    float4 lightNormal = vertices->normals[l.naInd];

    float cosThetaLight = dot(lightNormal, -wi);
    float cosThetaSurface = fabsf(dot(n, wi));

    float area = 0.5f * length(cross3(bpos - apos, cpos - apos));
    
    light_pdf = distanceSQR / (cosThetaLight * lightNum * area);
}


__device__ void nextEventEstimation(curandState& localState, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices,
    Triangle* scene, Triangle* lights, int lightNum, const Intersection& intersect, const float4& wo, 
    float& light_pdf, float4& contribution, float4& surfaceToLight_local, float etaI, float etaT)
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

    if (lightNum == 0)
    {
        light_pdf = -1.0f;
        return;
    }
    int index = min(static_cast<int>(curand_uniform(&localState) * lightNum), lightNum - 1);
    l = lights[index];
    apos = vertices->positions[l.aInd];
    bpos = vertices->positions[l.bInd];
    cpos = vertices->positions[l.cInd];
    u = sqrtf(curand_uniform(&localState));
    v = curand_uniform(&localState);
    p = (1.0f - u) * apos + u * (1.0f - v) * bpos + u * v * cpos; // point on light
    n = intersect.normal;
    
    float4 surfaceToLight = p-intersect.point;  
    float4 wi = normalize(surfaceToLight);

    Ray r = Ray(intersect.point + wi * EPSILON, wi);
    
    float t;
    float4 dummy;
    triangleIntersect(vertices, &l, r, dummy, t);
    
    Intersection sceneIntersect = Intersection();
    float4 throughputScale = f4(1.0f);
    BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, t*(1.0f - EPSILON), -1);
    // following if statement tests for scene intersection (direct light) AND
    // whether the original light intersect was valid
    //if (!sceneIntersect.valid && t != -1.0f) // direct LOS from intersection to light
    if (lengthSquared(throughputScale) > 0.0f)
    {
        float distanceSQR = lengthSquared(surfaceToLight);
        float4 lightNormal = vertices->normals[l.naInd];

        float cosThetaLight = dot(lightNormal, -wi);
        float cosThetaSurface = fabsf(dot(n, wi));

        //float G = cosThetaLight * cosThetaSurface/distanceSQR;
        float area = 0.5f * length(cross3(bpos - apos, cpos - apos));
        
        light_pdf = distanceSQR / (cosThetaLight * lightNum * area);
        float4 Le = l.emission;
        float4 f_val;
        float4 wi_local;
        toLocal(wi, intersect.normal, wi_local);
        surfaceToLight_local = wi_local;

        // wo is the incoming direction (passed to this function)
        // wi_local is the computed outgoing direction to the light
        f_eval(materials, intersect.materialID, textures, wo, wi_local, etaI, etaT, f_val, intersect.uv);

        contribution = f_val * Le * cosThetaSurface / light_pdf;
        contribution *= throughputScale;
    }
}

__device__ void removeMaterialFromStack(int* stack, int* stackTop, int materialID)
{
    int i_found = -1;
    for (int i = (*stackTop) - 1; i > 0; i--)
    {
        if (stack[i] == materialID)
        {
            i_found = i;
            break;
        }
    }

    if (i_found != -1)
    {
        for (int i = i_found; i < (*stackTop) - 1; i++)
        {
            stack[i] = stack[i + 1];
        }
        (*stackTop)--;
    }
}

__device__ float4 sampleSky(float4 direction)
{
    float4 unit_dir = normalize(direction); 

    float t = 0.5f * (unit_dir.y + 1.0f);

    //float4 c_horizon = 2.2f* f4(1.0f, 0.8f, 0.2f);
    //float4 c_zenith  = f4(0.4f, 0.4f, 0.8f);
    float4 c_horizon = 1.0f * f4(1.0f, 0.4f, 0.2f);
    //float4 c_horizon = 1.0f * f4(1.0f, 1.0f, 0.2f);
    float4 c_zenith  = f4(0.3f, 0.4f, 0.8f);
    //float4 c_zenith  = f4(0.9f, 0.9f, 0.2f);

    float4 sky_color = (1.0f - t) * c_horizon + t * c_zenith;

    float4 sun_dir = normalize(f4(-0.45f, 0.05f, 0.866f)); 
    float sun_focus = 800.0f;
    float sun_intensity = 15.0f;
    float4 sun_base = f4(1.0f, 0.8f, 0.2f);

    float sun_factor = pow(max(0.0f, dot(unit_dir, sun_dir)), sun_focus);
    float4 sun_final = sun_base * sun_intensity * sun_factor;

    return sky_color;
}

__global__ void Li_naive_unidirectional (curandState* rngStates, Camera camera, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, int maxDepth, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    curandState localState = rngStates[pixelIdx];

    Ray r = camera.generateCameraRay(localState, x, y);
    float4 Li = f4();
    float4 beta = f4(1.0f);
    for (int depth = 0; depth < maxDepth; depth++)
    {
        Intersection intersect;
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

        if (!intersect.valid)
        {
            Li += beta * sampleSky(r.direction);
            break;
        }

        

        float4 toSurface_local;
        float4 toNext_local;
        toLocal(r.direction, intersect.normal, toSurface_local);
        if (isnan(toSurface_local.x) || isnan(toSurface_local.y) || isnan(toSurface_local.z)) {
            printf("NaN DETECTED ON toSurfaceLocal: (%f, %f, %f) at depth %d\n local: (%f, %f, %f)\n normal: (%f, %f, %f)\n\n", 
                r.direction.x, r.direction.y, r.direction.z, depth, toSurface_local.x, toSurface_local.y, toSurface_local.z, intersect.normal.x, intersect.normal.y, intersect.normal.z);
        }
        float4 f_val;
        float pdf;
        sample_f_eval(localState, materials, intersect.materialID, textures, toSurface_local, 1.0f, 1.0f, intersect.backface, toNext_local, f_val, pdf, intersect.uv);

        if (pdf <= 0.0f || lengthSquared(f_val) < EPSILON) break;

        Li += intersect.emission * beta;

        beta *= f_val * fabsf(toNext_local.z) / pdf;

        float4 toNext_world;
        toWorld(toNext_local, intersect.normal, toNext_world);

        r.origin = intersect.point + ((toNext_local.z > 0.0f) ? (intersect.normal * RAY_EPSILON) : (-intersect.normal * RAY_EPSILON));
        r.direction = toNext_world;
    }
    colors[pixelIdx] += Li;
    rngStates[pixelIdx] = localState;
}

__host__ void launch_naive_unidirectional(int maxDepth, Camera camera, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors)
{
    dim3 blockSize(16, 16);  
    dim3 gridSize((w+15)/16, (h+15)/16);
    curandState* d_rngStates;
    cudaMalloc(&d_rngStates, w * h * sizeof(curandState));

    unsigned long seed = 103033UL;
    initRNG<<<gridSize, blockSize>>>(d_rngStates, w, h, seed);
    cudaDeviceSynchronize();

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));
    
    auto lastSaveTime = std::chrono::steady_clock::now();
    float saveIntervalSeconds = 5.0f;
    Image image = Image(w, h);

    std::cout << "Running Kernels" << std::endl;
    
    for (int currSample = 0; currSample < numSample; currSample++)
    {
        Li_naive_unidirectional<<<gridSize, blockSize>>>(d_rngStates, camera, materials, textures, BVH, BVHindices, maxDepth, vertices, vertNum, scene, triNum, 
            lights, lightNum, numSample, useMIS, w, h, colors);
        
        cudaDeviceSynchronize();
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastSaveTime).count();

        if (elapsed >= saveIntervalSeconds) 
        {
            std::vector<float4> h_colors(w * h);
            cudaMemcpy(h_colors.data(), colors, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    if (isnan(h_colors[i].x) || isnan(h_colors[i].y) || isnan(h_colors[i].z)) {
                        h_colors[i] = f4(1.0f, 0.0f, 1.0f); // Bright Pink for NaN
                    }
                    if (isinf(h_colors[i].x) || isinf(h_colors[i].y) || isinf(h_colors[i].z)) {
                        h_colors[i] = f4(0.0f, 1.0f, 0.0f); // Bright Green for Inf
                    }
                    if (h_colors[image.toIndex(i, j)].x < 0 || h_colors[image.toIndex(i, j)].y < 0 || h_colors[image.toIndex(i, j)].z < 0)
                        cout << i << ", " << j << " Negative color written: <" << h_colors[image.toIndex(i, j)].x << ", " << h_colors[image.toIndex(i, j)].y << ", " 
                        << h_colors[image.toIndex(i, j)].z << ">"<< endl;
                    
                    image.setColor(i, j, h_colors[image.toIndex(i, j)] / (float)(currSample + 1));
                }
            }
            std::string filename = "render.bmp";
            image.saveImageBMP(filename);
            image.saveImageCSV_MONO(0);
            lastSaveTime = now;
            printf("saved progress at %d samples.\n", currSample);
        }

    }
    
    cudaDeviceSynchronize();
    cudaFree(d_rngStates);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: CUDA Error code: " << static_cast<int>(err) << std::endl;
        // only call this if the code isn't catastrophic
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    }
    else
        std::cout << "Render executed with no CUDA error" << std::endl;
}

__global__ void Li_unidirectional (curandState* rngStates, Camera camera, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, int maxDepth, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;
    
    float4 colorSum = f4();

    curandState localState = rngStates[pixelIdx];
    for (int currSample = 0; currSample < numSample; currSample++)
    {   
        Ray r = Ray();
        float4 beta = f4(1.0f, 1.0f, 1.0f);
        float4 Li = f4();
        float4 wi_local = f4();
        float4 wo_local = f4();
        float4 wi_world = f4();
        Intersection previousintersectREAL = Intersection(); // last REAL intersection (true hit)
        Intersection previousintersectANY = Intersection(); // last any intersection (includes false hits passing thru media)
        previousintersectANY.triIDX = -1;

        // for nested dielectrics
        int mediumStack[16];
        int stackTop = 0;
        mediumStack[stackTop++] = 0; //index of AIR (IOR 1.0f, Priority 99)

        r = camera.generateCameraRay(localState, x, y);

        float pdf = EPSILON;
        float etaI = EPSILON;
        float etaT = EPSILON;

        bool hitFirstnonSpecular = false;

        for (int depth = 0; depth < 100; depth++)
        {   
            
            Intersection intersect = Intersection();
            intersect.valid = false;
            //BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect, 999999.0f, false, previousintersectANY.triIDX);
            BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect, 999999.0f);

            if (!intersect.valid) 
            {
                Li += beta * sampleSky(r.direction);
                break;
            }
            
            int materialID = intersect.materialID;

            //float4 old_wi_local = wi_local;

            toLocal(r.direction, intersect.normal, wi_local); // assign new wi_local
            wi_world = normalize(r.direction);

            bool isSpecular = false;
            if (materials[materialID].isSpecular)
            {
                isSpecular = true;
            }

            bool trueHit = true;

            int minPrior = materials[mediumStack[0]].priority;
            int minPriorID = mediumStack[0];
            for (int i = 1; i < stackTop; i++)
            {
                if (materials[mediumStack[i]].priority < minPrior)
                {
                    minPrior = materials[mediumStack[i]].priority;
                    minPriorID = mediumStack[i];
                }
            }

            float4 absorption_coeff = materials[minPriorID].absorption;
            float distanceTraveled = intersect.dist;

            if (distanceTraveled > EPSILON)
            {
                float4 attenuation = f4(
                    exp(-absorption_coeff.x * distanceTraveled), 
                    exp(-absorption_coeff.y * distanceTraveled), 
                    exp(-absorption_coeff.z * distanceTraveled)
                );
                beta *= attenuation;
            }
            
            if (materials[materialID].boundary) // if this material is a boundary between media
            {
                
                if (materials[materialID].priority <= minPrior) // new material has lower or equal priority to the minimum priority
                {
                    // true hit, continue with shading
                    if (materials[materialID].type == MAT_SMOOTHDIELECTRIC)
                    {
                        etaI = materials[minPriorID].ior; //the dominating current medium
                        if (!intersect.backface) //entering surface
                        {
                            etaT = materials[materialID].ior; //is later added iff we actually refract
                        }
                        else // exiting dominant surface
                        {
                            if (stackTop == 1)
                            {
                                if (materials[mediumStack[0]].priority != 99)
                                    printf("error: single medium in stack that isnt air\n");
                                etaT = 1.0f;
                            }
                            else
                            {
                                // the new material is a true hit, so it must appear somewhere in the stack, since we are exiting it
                                minPrior = 99;
                                int secondLowest = mediumStack[0];
                                for (int i = 0; i < stackTop; i++)
                                {   //printf("%d\n", materials[mediumStack[i]].priority);
                                    if (materials[mediumStack[i]].priority)
                                    {
                                        // checks for the dominant medium in the absence of the one we are exiting, defaults to air
                                        if (minPrior > materials[mediumStack[i]].priority && mediumStack[i] != materialID)
                                        {
                                            secondLowest = mediumStack[i];
                                            minPrior = materials[mediumStack[i]].priority;
                                        }
                                    }
                                }
                                // this is the dominant medium if we pretend like the one we just exited isnt there
                                // we KNOW that we are exiting the DOMINANT medium since it is a true hit
                                etaT = materials[secondLowest].ior;
                            }
                            
                        }
                    }
                }
                else // false hit, ignore intersection
                {
                    trueHit = false;
                    if (!intersect.backface) //entering non dominant surface
                    {
                        mediumStack[stackTop++] = intersect.materialID; //push new medium

                    }
                    else //exiting non dominant surface
                    {
                        removeMaterialFromStack(mediumStack, &stackTop, materialID);
                    }
                }
            }
            else // if this isnt a boundary event. We still need to know the current medium ior for thin walled events
                etaI = materials[minPriorID].ior; 

            if (trueHit)
            {
                if (lengthSquared(intersect.emission) > EPSILON)
                {
                    if (depth == 0 || !hitFirstnonSpecular)
                    {
                        Li += beta * intersect.emission;
                    }
                    else if (useMIS && !isSpecular) // found light using BSDF sampling, weigh against NEE
                    {
                        float light_pdf = EPSILON;
                        
                        neePDF(vertices, scene, lightNum, intersect.triIDX, previousintersectREAL, light_pdf, etaI, etaT, &intersect);
                        if (light_pdf > EPSILON)
                        {
                            float bsdfWeight = pdf * pdf / (light_pdf * light_pdf 
                            + pdf * pdf);
                            Li += beta * intersect.emission * bsdfWeight;
                        }
                    }
                }

                if (useMIS && lengthSquared(intersect.emission) < EPSILON && !isSpecular) // using nee mainly, weigh against BSDF pdf
                {
                    float4 nee;
                    float light_pdf = EPSILON;
                    // we get wo_local, the direction from surface to sampled light, to evaluate the bsdf pdf, 
                    // and store it in wo_local
                    nextEventEstimation(localState, materials, textures, BVH, BVHindices, vertices, scene, lights, lightNum, intersect, 
                        wi_local, light_pdf, nee, wo_local, etaI, etaT);
                    
                    if (light_pdf > EPSILON)
                    {
                        // to calculate the bsdf pdf
                        pdf_eval(materials, materialID, textures, wi_local, wo_local, etaI, etaT, pdf, intersect.uv); // stores the bsdf pdf val in pdf
                        float neeWeight = light_pdf * light_pdf / (pdf * pdf + light_pdf * light_pdf);

                        Li += beta * nee * neeWeight;
                    }

                }
                float4 f_val = f4();
                sample_f_eval(localState, materials, materialID, textures, wi_local, etaI, etaT, intersect.backface, wo_local, f_val, pdf, intersect.uv);

                float4 wo_world= f4();
                toWorld(wo_local, intersect.normal, wo_world);

                pdf = fmaxf(pdf, 0.01);
                
                
                if (trueHit)
                {
                    if (wo_local.z < 0.0f) // refracted
                    {
                        if (!intersect.backface) // entering new surface (is dominant)
                        {
                            mediumStack[stackTop++] = intersect.materialID;
                        }
                        else // exiting dominant surface (materialID garunteed to be dominant)
                        {
                            removeMaterialFromStack(mediumStack, &stackTop, materialID);
                        }
                    }
                }
                

                beta *= (f_val * fabsf(wo_local.z) / pdf);
                //beta = fminf4(beta, f4(10.0f));

                if (wo_local.z > 0) // reflected
                    r.origin = intersect.point + intersect.normal * EPSILON;
                else //refracted
                    r.origin = intersect.point - intersect.normal * EPSILON;

                r.direction = normalize(wo_world);
                previousintersectREAL = intersect;
                
                
            }
            else
            {
                //float4 wo_world = normalize(r.direction);
                toLocal(r.direction, intersect.normal, wo_local);
                //r.origin = intersect.point - intersect.normal * EPSILON * 1.0F; // needs to go through, so offset on other side of normal
                r.origin = intersect.point + r.direction * 0.001f; // needs to go through, so offset on other side of normal
                depth--; // to unbias the russian roulette, which depends on a maxdepth (false hits do not contribute actual depth)
            }
            previousintersectANY = intersect;

            if (depth > maxDepth)
            {
                float luminance = dot(beta, f4(0.2126f, 0.7152f, 0.0722f));
                float p = clamp(luminance, 0.05f, 0.99f);

                if (curand_uniform(&localState) > p)   // survive with probability p
                    break;

                beta /= p;  // compensate for the survival probability
            }

            if (!isSpecular)
            {
                hitFirstnonSpecular = true;
            }
            
        }
        //Li = f4(fmaxf(Li.x, 0.0f), fmaxf(Li.y, 0.0f), fmaxf(Li.z, 0.0f));
        colorSum += Li;
    }
    colors[pixelIdx] = colorSum;
    rngStates[pixelIdx] = localState;
}

__host__ void launch_unidirectional(int maxDepth, Camera camera, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors)
{
    dim3 blockSize(16, 16);  
    dim3 gridSize((w+15)/16, (h+15)/16);
    curandState* d_rngStates;
    cudaMalloc(&d_rngStates, w * h * sizeof(curandState));

    unsigned long seed = 103033UL;
    initRNG<<<gridSize, blockSize>>>(d_rngStates, w, h, seed);
    cudaDeviceSynchronize();

    Li_unidirectional<<<gridSize, blockSize>>>(d_rngStates, camera, materials, textures, BVH, BVHindices, maxDepth, vertices, vertNum, scene, triNum, 
        lights, lightNum, numSample, useMIS, w, h, colors);

    cudaDeviceSynchronize();
    cudaFree(d_rngStates);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: CUDA Error code: " << static_cast<int>(err) << std::endl;
        // only call this if the code isn't catastrophic
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    }
    else
        std::cout << "Render executed with no CUDA error" << std::endl;
}

__device__ bool BDPTnextEventEstimation(curandState& localState, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices,
    Triangle* scene, Triangle* lights, int lightNum, int materialID, float4 shadingPos, const float4 toShadingPos_local, const float4 shadingPos_normal,
    const float2 uv, float& light_pdf, float4& contribution, float4& shadingPos_to_lightPos, int& lightInd, float& cosLight, float& pdf_emit, 
    float etaI, float etaT, float sceneRadius)
{
    int totalLightNum = SAMPLE_ENVIRONMENT ? (lightNum + 1) : lightNum; // +1 for the sky
    lightInd = SAMPLE_ENVIRONMENT ? (min(static_cast<int>(curand_uniform(&localState) * (lightNum + 1)), lightNum) - 1) : 
        (min(static_cast<int>(curand_uniform(&localState) * (lightNum)), lightNum - 1)); 
    
    float pdf_chooseLight = 1.0f / ((float) (SAMPLE_ENVIRONMENT ? (lightNum + 1) : lightNum));

    if (lightInd != -1)
    {
        Triangle l = lights[lightInd];
        float4 apos = vertices->positions[l.aInd];
        float4 bpos = vertices->positions[l.bInd];
        float4 cpos = vertices->positions[l.cInd];

        float4 anorm = vertices->normals[l.naInd];
        float4 bnorm = vertices->normals[l.nbInd];
        float4 cnorm = vertices->normals[l.ncInd];

        float u = sqrtf(curand_uniform(&localState));
        float v = curand_uniform(&localState);

        float w0 = (1.0f - u);
        float w1 = u * (1.0f - v);
        float w2 = u * v;

        float4 p = w0 * apos + w1 * bpos + w2 * cpos;
        float4 lightNormal = normalize(w0 * anorm + w1 * bnorm + w2 * cnorm);

        float4 surfaceToLight = p-shadingPos;  
        shadingPos_to_lightPos = surfaceToLight;
        float4 surfaceToLight_unit = normalize(surfaceToLight);

        float distanceSQR = lengthSquared(surfaceToLight);
        distanceSQR = fmaxf(distanceSQR, RAY_EPSILON);

        Ray r = Ray(shadingPos + shadingPos_normal * RAY_EPSILON, surfaceToLight_unit);

        float distance = sqrtf(distanceSQR);

        float4 throughputScale = f4(1.0f);
        BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, distance - EPSILON, l.triInd);

        if (lengthSquared(throughputScale) > 0.0f)
        {
            float cosThetaLight = dot(lightNormal, -surfaceToLight_unit);
            cosLight = cosThetaLight;

            if (cosThetaLight < EPSILON) {
                contribution = f4(0.0f);
                return true;
            }
            float cosThetaSurface = fabsf(dot(shadingPos_normal, surfaceToLight_unit));

            float G = cosThetaLight * cosThetaSurface / distanceSQR;

            float maxG = 2.0f;
            if (G > maxG) {
                G = maxG; 
            }

            float area = 0.5f * length(cross3(bpos - apos, cpos - apos));

            float pdf_choosePoint_area = 1.0f / area;
            float pdf_chooseLightPoint_area = pdf_choosePoint_area * pdf_chooseLight;
            light_pdf = pdf_chooseLightPoint_area; // nee pdf
            pdf_emit = cosThetaLight / PI; // emit pdf, directional

            //float pdf_chooseLightPoint_solidAngle = pdf_chooseLightPoint_area * distanceSQR / cosThetaLight;
            //light_pdf = pdf_chooseLightPoint_solidAngle;

            float4 towardLight_local;
            toLocal(surfaceToLight_unit, shadingPos_normal, towardLight_local);

            float4 f_val;
            f_eval(materials, materialID, textures, toShadingPos_local, towardLight_local, etaI, etaT, f_val, uv);

            contribution = throughputScale * f_val * l.emission * G / light_pdf; // unweighted
            return false;
        }
    }
    else
    {
        float4 dir_to_sky = sampleSphere(localState, 1.0f); 
        float pdf_dir_solidAngle = 1.0f / (4.0f * PI);
        
        float4 surfaceToLight_unit = dir_to_sky;
        shadingPos_to_lightPos = surfaceToLight_unit;
        Ray r = Ray(shadingPos + shadingPos_normal * RAY_EPSILON, surfaceToLight_unit);

        float4 throughputScale = f4(1.0f);
        
        BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, SKY_RADIUS, -1);

        if (lengthSquared(throughputScale) > 0.0f)
        {
            //printf("Sky visible from <%f,%f,%f> with normal <%f,%f,%f> and direction <%f,%f,%f>\n", 
            //    r.origin.x, r.origin.y, r.origin.z, shadingPos_normal.x, shadingPos_normal.y, shadingPos_normal.z,
            //    surfaceToLight_unit.x, surfaceToLight_unit.y, surfaceToLight_unit.z);
            float cosThetaSurface = fabsf(dot(shadingPos_normal, surfaceToLight_unit));
            cosLight = -69.420; // not used in calculations for sky

            float4 Le = sampleSky(surfaceToLight_unit);

            float pdf_chooseLightPoint_solidAngle = pdf_chooseLight * pdf_dir_solidAngle;
            pdf_emit = pdf_chooseLightPoint_solidAngle; // emit pdf, directional
            //use the same disk sampling used in the light vertex generation
            float pdf_pos_area = 1.0f / (PI * sceneRadius * sceneRadius);
            light_pdf = pdf_chooseLightPoint_solidAngle;

            float4 towardLight_local;
            toLocal(surfaceToLight_unit, shadingPos_normal, towardLight_local);

            float4 f_val;
            f_eval(materials, materialID, textures, toShadingPos_local, towardLight_local, etaI, etaT, f_val, uv);

            // Unweighted Contribution
            // For infinite lights: Le * f_r * cosTheta / pdf_solidAngle
            contribution = throughputScale * f_val * Le * cosThetaSurface / pdf_chooseLightPoint_solidAngle;
            return false;
        }
    }
    return true;
}

// populates the section of eyePath buffer specified by the arguments. Also asigns the length of the path
__device__ void generateEyePath(curandState& localState, Camera camera, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, int maxDepth, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int w, int h, int x, int y, float sceneRadius, PathVertices* eyePath, int& pathLength) 
{
    pathLength = 1;
    Ray r = camera.generateCameraRay(localState, x, y);
    int firstIdx = pathBufferIdx(w, h, x, y, 0);

    float4 currThroughput = f4(1.0f);
    
    eyePath->pt[firstIdx] = r.origin;
    eyePath->n[firstIdx] = camera.getForwardVector();
    eyePath->beta[firstIdx] = currThroughput;
    eyePath->isDelta[firstIdx] = true; // it is delta meaning the probability of a light path hitting it randomly is zero

    eyePath->lightInd[firstIdx] = -51;

    eyePath->misWeight[firstIdx] = 0.0f;
    eyePath->uv[firstIdx] = f2(0.0f);

    float aspect = (float)w / (float)h;
    float halfH = camera.fovScale;
    float halfW = halfH * aspect;
    float sensorArea = (2.0f * halfW) * (2.0f * halfH);

    float cosAtCamera = fabsf(dot(camera.getForwardVector(), r.direction)); // r.direction should be normalized already

    float prevPDF_solidAngle; // outgoing pdf from scattering functions
    float prev_cosine; // the previous cosine between the normal and the outgoing ray

    prevPDF_solidAngle = 1.0f / (sensorArea * cosAtCamera * cosAtCamera * cosAtCamera);
    prev_cosine = cosAtCamera;

    // these shouldnt be needed for the first vertex
    eyePath->d_vc[firstIdx] = 0.0f;
    eyePath->d_vcm[firstIdx] = 0.0f;

    // stores the accumulated mis ratios in the form ratio(1+ratio(1+ratio...))
    float currMIS = 0.0f;

    float prev_d_vcm = -1.0f;
    float prev_d_vc = -1.0f;

    float pdf_onebeforePrevRev_SA = -1.0f;
    bool prevWasDelta = false;

    for (int depth = 1; depth < maxDepth; depth++)
    {
        int currIdx = pathBufferIdx(w, h, x, y, depth);
        int prevIdx = pathBufferIdx(w, h, x, y, depth-1);
        Intersection intersect = Intersection();
        intersect.valid = false;
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

        if (!intersect.valid) // treat this as an endpoint
        {
            /* NOT IMPLEMENTED CORRECTLY FOR VCM STYLE YET
            // 1. Setup Vertex (Endpoint)
            // We place the vertex on the sky sphere for visualization/debugging,
            // but the math below ignores this position and uses the "Virtual Disk" instead.
            float R_env = SKY_RADIUS;
            //eyePath->pt[currIdx] = r.origin + r.direction * R_env;
            eyePath->pt[currIdx] = r.direction * R_env;
            eyePath->wo[currIdx] = f4();
            eyePath->n[currIdx]  = normalize(-r.direction); 
            eyePath->materialID[currIdx]  = -1; 
            eyePath->lightInd[currIdx] = -1; // Mark as sky
            
            // 2. Define the "Virtual Disk" Probability
            // This MUST match the 'pdf_pos' from your Light Path code.
            // It says: "The probability density of picking a point on the scene window."
            float pdf_pos_area = 1.0f / (PI * sceneRadius * sceneRadius);

            // 3. Forward PDF (Camera -> Sky)
            // Convert the BSDF's Solid Angle PDF to Area Measure on the Virtual Disk.
            // Notice: We do NOT use distSQR or Cosine here.
            float pdfFwd_area = prevPDF_solidAngle * pdf_pos_area;

            // 4. Reverse PDF (Sky -> Camera)
            // "What is the prob that the Light Strategy (Step 1) generated this ray?"
            // It picks a direction (1/4PI) and a point on the disk (pdf_pos_area).
            float p_chooseLight = 1.0f / (lightNum + 1.0f);
            float pdfRev_area = (1.0f / (4.0f * PI)) * pdf_pos_area * p_chooseLight;

            // 5. Update MIS Accumulator (Balance Heuristic)
            // This variable tracks the ratio of probabilities for the entire path history.
            float ratio = pdfRev_area / pdfFwd_area;

            float connectionStrategy = eyePath->isDelta[prevIdx] ? 0.0f : 1.0f;

            // 6. Calculate Final Weight for this strategy
            // This is the final weight!
            eyePath->misWeight[currIdx] = 1.0f / (1.0f + ratio * ratio * (connectionStrategy + eyePath->misWeight[prevIdx]));

            // 7. Final Contribution
            float4 Le = sampleSky(r.direction);
            
            // this is the unweighted contribution. We will extract this the connect path function.
            eyePath->beta[currIdx] = currThroughput * Le;

            eyePath->isDelta[currIdx] = false; 
            pathLength++;


            //debugPrintPath(w, h, x, y, pathLength, *eyePath);
            //printf("+1\n");
            */
            return;
        }
        float4 geomN = intersect.normal;
        bool doubleSided = materials[intersect.materialID].type == MAT_SMOOTHDIELECTRIC || materials[intersect.materialID].type == MAT_LEAF; // or check flag
        eyePath->uv[currIdx] = intersect.uv;
        eyePath->beta[currIdx] = currThroughput;

        eyePath->materialID[currIdx] = intersect.materialID;
        eyePath->pt[currIdx] = intersect.point;

        bool currDelta = materials[eyePath->materialID[currIdx]].isSpecular;
        eyePath->isDelta[currIdx] = currDelta;

        if (intersect.backface)
        {
            eyePath->backface[currIdx] = true;
        }
        else
            eyePath->backface[currIdx] = false;
        
        eyePath->n[currIdx] = geomN;

        eyePath->wo[currIdx] = normalize(-r.direction);

        float4 wo_world = intersect.point - eyePath->pt[prevIdx]; // the incoming direction, pointing at the new surface
        float4 wo_local; // the incoming direction to the current path vertex. we use this for the cosine in the pdf conversion
        toLocal(r.direction, intersect.normal, wo_local);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate forward pdf (previous vertex to current)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float distanceSQR = fmaxf(lengthSquared(wo_world), RAY_EPSILON);

        // previous pdf (solid angle) * abs of dot product of current normal with incoming direction into the current surface divided by distance squared
        float pdfFwd_area = prevPDF_solidAngle * fabsf(wo_local.z) / distanceSQR;
        eyePath->pdfFwd[currIdx] = pdfFwd_area;

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Scatter to next vertex
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float pdfFwd_solidAngle;
        float4 f_val;
        float4 wi_local; //okay apparently wi is the outgoing direction now wtf

        float etaI = 1.0f; // TEMPORARY, CHANGE AFTER IMPLEMENTING PRIORITY NESTED DIELECTRICS
        float etaT = 1.0f;

        sample_f_eval(localState, materials, intersect.materialID, textures, wo_local, etaI, etaT, intersect.backface, wi_local, f_val, pdfFwd_solidAngle, intersect.uv);
        
        //radiance is conserved through dielectric boundaries, so we dont need to apply a correction like we did for the light path

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate backwards pdf (current vertex to previous)
        //---------------------------------------------------------------------------------------------------------------------------------------------------
        float4 nextToCurrent_local = -wi_local;
        float4 currentToPrev_local = -wo_local;

        float pdfRev_solidAngle;
        float pdfRev_area;

        pdf_eval(materials, intersect.materialID, textures, nextToCurrent_local, currentToPrev_local, etaI, etaT, pdfRev_solidAngle, intersect.uv);

        // pdfRev is not stored

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Wrapping it up, self explanatory
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        if (depth == 1) {
            float pdf_connect = 1.0f;
            float pdf_trace = 1.0f;
            //float numLightSample = (float) w * (float) h;
            float numLightSample = 1.0f;

            float vcm = (pdf_connect * numLightSample) / (pdf_trace * pdfFwd_area);
            float vc = 0.0f;

            eyePath->d_vcm[currIdx] = vcm;
            eyePath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else if (prevWasDelta) 
        {
            float G = prev_cosine / distanceSQR; // distance to previous vertex
            float vcm = 0.0f;
            float vc = (G / pdfFwd_area) * (pdf_onebeforePrevRev_SA * prev_d_vc);

            eyePath->d_vcm[currIdx] = vcm;
            eyePath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else
        {
            float G = prev_cosine / distanceSQR; // distance to previous vertex
            float vcm = 1.0f / pdfFwd_area;
            float vc = (G / pdfFwd_area) * (prev_d_vcm + pdf_onebeforePrevRev_SA * prev_d_vc);

            eyePath->d_vcm[currIdx] = vcm;
            eyePath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Set up next interaction
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float4 wi_world;
        toWorld(wi_local, intersect.normal, wi_world);

        if (lengthSquared(scene[intersect.triIDX].emission) > EPSILON)
            eyePath->lightInd[currIdx] = scene[intersect.triIDX].lightInd;
        else
            eyePath->lightInd[currIdx] = -51; // -1 is reserved for the sun

        if (pdfFwd_solidAngle < EPSILON)
            break;

        currThroughput = currThroughput * f_val * fabsf(wi_local.z) / pdfFwd_solidAngle;

        bool transmitting = dot(wi_world, eyePath->n[currIdx]) < 0.0f;

        if (transmitting)
            r.origin = intersect.point - eyePath->n[currIdx] * RAY_EPSILON;
        else
            r.origin = intersect.point + eyePath->n[currIdx] * RAY_EPSILON;

        r.origin = intersect.point + (transmitting ? (-eyePath->n[currIdx] * RAY_EPSILON) : (eyePath->n[currIdx] * RAY_EPSILON));
        r.direction = normalize(wi_world);

        pathLength++;

        // update the previous state
        prev_cosine = fabsf(wi_local.z);
        prevWasDelta = currDelta;
        prevPDF_solidAngle = pdfFwd_solidAngle; 
    }
}

__device__ void generateFirstLightPathVertex(curandState& localState, int maxDepth, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int w, int h, int x, int y, float4 sceneCenter, float sceneRadius, PathVertices* lightPath, float& pdf_solidAngle, float& cosine, float4& out_wi) 
{
    int firstIdx = pathBufferIdx(w, h, x, y, 0);

    // the convention is that light index -1 is the environment, and that lightNum doesnt include the environment
    int lightInd = SAMPLE_ENVIRONMENT ? (min(static_cast<int>(curand_uniform(&localState) * (lightNum + 1)), lightNum) - 1) : 
        (min(static_cast<int>(curand_uniform(&localState) * (lightNum)), lightNum - 1)); 
    
    float pdf_chooseLight = 1.0f / ((float) (SAMPLE_ENVIRONMENT ? (lightNum + 1) : lightNum));
    lightPath->uv[firstIdx] = f2(0.0f);
    lightPath->backface[firstIdx] = false;
    
    if (lightInd != -1)
    {
        Triangle l = lights[lightInd];
        float4 apos = vertices->positions[l.aInd];
        float4 bpos = vertices->positions[l.bInd];
        float4 cpos = vertices->positions[l.cInd];

        float4 anorm = vertices->normals[l.naInd];
        float4 bnorm = vertices->normals[l.nbInd];
        float4 cnorm = vertices->normals[l.ncInd];

        float u = sqrtf(curand_uniform(&localState));
        float v = curand_uniform(&localState);

        float w0 = (1.0f - u);
        float w1 = u * (1.0f - v);
        float w2 = u * v;

        lightPath->materialID[firstIdx] = l.materialID;

        lightPath->pt[firstIdx] = w0 * apos + w1 * bpos + w2 * cpos;
        lightPath->n[firstIdx] = normalize(w0 * anorm + w1 * bnorm + w2 * cnorm);

        float4 dummy;
        float firstPDF_solidAngle;
        float4 outOfLight_world;

        float4 outOfLight_local; 
        cosine_sample_f(localState, dummy, outOfLight_local, dummy, firstPDF_solidAngle);
        toWorld(outOfLight_local, lightPath->n[firstIdx] ,outOfLight_world);

        lightPath->wo[firstIdx] = f4(0.0f); // degenerate vector, does not exist
        out_wi = outOfLight_world; // Pass back locally

        float area = 0.5f * length(cross3(bpos - apos, cpos - apos));
        float pdf_samplePoint = 1.0f / area;
        float pdfFwd_val = pdf_chooseLight * pdf_samplePoint;
        lightPath->pdfFwd[firstIdx] = pdfFwd_val;
        // pdfRev is 0.0f, no store needed

        float4 Le = l.emission;
        lightPath->beta[firstIdx] = (Le * PI) / pdfFwd_val;
        //lightPath->beta[firstIdx] = f4(1.0f) / pdfFwd_val;

        lightPath->lightInd[firstIdx] = lightInd;
        lightPath->isDelta[firstIdx] = false;

        lightPath->misWeight[firstIdx] = 0.0f;

        pdf_solidAngle = pdfFwd_val * firstPDF_solidAngle;// spatial times directional pdf forms the complete solid angle pdf
        cosine = fabsf(outOfLight_local.z);
    }
    else
    {
        lightPath->materialID[firstIdx] = -1;
        float4 dir_in = -sampleSphere(localState, 1.0f);
        float pdf_dir = 1.0f / (4.0f * PI);

        // some dark magic idk
        float4 tangent;
        float4 bitangent;

        if (fabsf(dir_in.x) > 0.9f) 
        {
            float4 worldY = f4(0.0f, 1.0f, 0.0f);
            tangent = normalize(cross3(worldY, dir_in));
        } 
        else 
        {
            float4 worldX = f4(1.0f, 0.0f, 0.0f);
            tangent = normalize(cross3(worldX, dir_in));
        }

        bitangent = normalize(cross3(dir_in, tangent));

        float r1 = curand_uniform(&localState);
        float r2 = curand_uniform(&localState);
        
        float disk_r = sceneRadius * sqrtf(r1); 
        float disk_phi = 2.0f * PI * r2;

        float u_offset = disk_r * cosf(disk_phi);
        float v_offset = disk_r * sinf(disk_phi);

        float4 p_disk = sceneCenter + (tangent * u_offset) + (bitangent * v_offset);
        float4 p_world = p_disk - (dir_in * SKY_RADIUS);
        float pdf_pos = 1.0f / (PI * sceneRadius * sceneRadius);

        lightPath->pt[firstIdx] = p_world;
        lightPath->n[firstIdx] = normalize(dir_in);

        lightPath->wo[firstIdx] = f4(0.0f); // degenerate vector, does not exist
        out_wi = dir_in;

        lightPath->lightInd[firstIdx] = -1; // the sky
        lightPath->isDelta[firstIdx] = false; // the sky

        float pdfFwd_full = pdf_chooseLight * pdf_pos * pdf_dir;
        float pdfFwd_val = pdf_chooseLight * pdf_pos;
        lightPath->pdfFwd[firstIdx] = pdfFwd_val;
        
        float4 Le = sampleSky(-dir_in);
        lightPath->beta[firstIdx] = Le / pdfFwd_full;
        //lightPath->beta[firstIdx] = f4(1.0f) / pdfFwd_val;

        lightPath->misWeight[firstIdx] = 0.0f;

        // unused
        pdf_solidAngle = pdf_dir;
        cosine = 1.0f;
    }
}

__device__ void generateLightPath(curandState& localState, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, int maxDepth, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int w, int h, int x, int y, float4 sceneCenter, float sceneRadius, PathVertices* lightPath, int& pathLength) 
{
    pathLength = 1;
    int firstIdx = pathBufferIdx(w, h, x, y, 0);
    Ray r;
    float prevPDF_solidAngle; // outgoing pdf from scattering functions
    float prev_cosine; // the previous cosine between the normal and the outgoing ray
    float4 start_wi;

    float4 currThroughput = f4(1.0f);

    generateFirstLightPathVertex(localState, maxDepth, vertices, vertNum, scene, triNum, lights, lightNum, w, h, x, y, sceneCenter, sceneRadius, lightPath, prevPDF_solidAngle, prev_cosine, start_wi);

    currThroughput = lightPath->beta[firstIdx];

    r.origin = lightPath->pt[firstIdx] + lightPath->n[firstIdx] * RAY_EPSILON;
    r.direction = start_wi;
    
    prev_cosine = fabsf(dot(normalize(start_wi), lightPath->n[firstIdx]));
    prevPDF_solidAngle = prev_cosine / PI; // emission pdf

    // these shouldnt be needed for the first vertex
    lightPath->d_vc[firstIdx] = 0.0f;
    lightPath->d_vcm[firstIdx] = 0.0f;

    // stores the accumulated mis ratios in the form ratio(1+ratio(1+ratio...))
    float currMIS = 0.0f;

    float prev_d_vcm = -1.0f;
    float prev_d_vc = -1.0f;

    float pdf_onebeforePrevRev_SA = -1.0f;
    bool prevWasDelta = false;

    for (int depth = 1; depth < maxDepth; depth++)
    {
        int currIdx = pathBufferIdx(w, h, x, y, depth);
        int prevIdx = pathBufferIdx(w, h, x, y, depth-1);

        Intersection intersect = Intersection();
        intersect.valid = false;
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

        if (!intersect.valid)
        {
            return;
        }
            
        lightPath->uv[currIdx] = intersect.uv;
        lightPath->beta[currIdx] = currThroughput;
        float4 geomN = intersect.normal;

        lightPath->materialID[currIdx] = intersect.materialID;
        lightPath->pt[currIdx] = intersect.point;
        
        bool currDelta = materials[lightPath->materialID[currIdx]].isSpecular;
        lightPath->isDelta[currIdx] = currDelta;

        if (intersect.backface)
        {
            lightPath->backface[currIdx] = true;
        }
        else
            lightPath->backface[currIdx] = false;
        
        lightPath->n[currIdx] = geomN;
        
        lightPath->wo[currIdx] = normalize(-r.direction);

        float4 wo_world = intersect.point - lightPath->pt[prevIdx]; // the incoming direction, pointing at the new surface
        float4 wo_local; // the incoming direction to the current path vertex. we use this for the cosine in the pdf conversion
        toLocal(r.direction, intersect.normal, wo_local);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate forward pdf (previous vertex to current)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float distanceSQR = fmaxf(lengthSquared(wo_world), RAY_EPSILON);

        // previous pdf (solid angle) * abs of dot product of current normal with incoming direction into the current surface divided by distance squared
        float pdfFwd_area; 
        float denom; 

        pdfFwd_area = prevPDF_solidAngle * fabsf(wo_local.z) / distanceSQR;
        lightPath->pdfFwd[currIdx] = pdfFwd_area;
        

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Scatter to next vertex
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float pdfFwd_solidAngle;
        float4 f_val;
        float4 wi_local; //okay apparently wi is the outgoing direction now wtf

        float etaI = 1.0f; // TEMPORARY, CHANGE AFTER IMPLEMENTING PRIORITY NESTED DIELECTRICS
        float etaT = 1.0f;

        sample_f_eval(localState, materials, intersect.materialID, textures, wo_local, etaI, etaT, intersect.backface, wi_local, f_val, pdfFwd_solidAngle, intersect.uv);

        if (materials[intersect.materialID].type == MAT_SMOOTHDIELECTRIC) // If this is glass/water
        {
            bool transmitted = (wi_local.z > 0) != (wo_local.z > 0); 

            if (transmitted) {
                float correction = (etaT * etaT) / (etaI * etaI);
                
                f_val *= correction;
            }
        }
        
        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate backwards pdf (current vertex to previous)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float4 nextToCurrent_local = -wi_local;
        float4 currentToPrev_local = -wo_local;

        float pdfRev_solidAngle;
        float pdfRev_area;

        pdf_eval(materials, intersect.materialID, textures, nextToCurrent_local, currentToPrev_local, etaI, etaT, pdfRev_solidAngle, intersect.uv);

        // pdfRev is not stored

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Wrapping it up, self explanatory
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float4 wi_world;
        toWorld(wi_local, intersect.normal, wi_world);
        // wi is not stored

        // we don't store the light index for a light path

        if (pdfFwd_solidAngle < EPSILON)
            break;

        currThroughput = currThroughput * f_val * fabsf(wi_local.z) / pdfFwd_solidAngle;

        if (depth == 1) {

            /*the spatial probability of picking the starting light. This is equal to both
            the probability of connecting to the light vertex via nee, and also equal to the
            probability of starting a light path at the light vertex. This value is stored
            in the forward pdf of the light vertex
            */
            float pdf_sampleLight = lightPath->pdfFwd[firstIdx];

            // the pdf of connecting to the previous light via NEE
            float pdf_connect = pdf_sampleLight;

            // the pdf of starting a light path at the previous light
            float pdf_trace = pdf_sampleLight;

            // to convert to area density at previous vertex
            float G = prev_cosine / distanceSQR;

            // pdfFwdArea is the emission pdf of the light, converted to area density at the current vertex
            float vcm = (pdf_connect) / (pdf_trace * pdfFwd_area);
            float vc = G / (pdf_trace * pdfFwd_area);

            lightPath->d_vcm[currIdx] = vcm;
            lightPath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else if (prevWasDelta) 
        {
            float G = prev_cosine / distanceSQR; // distance to previous vertex
            float vcm = 0.0f;
            float vc = (G / pdfFwd_area) * (pdf_onebeforePrevRev_SA * prev_d_vc);

            lightPath->d_vcm[currIdx] = vcm;
            lightPath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else
        {
            // to convert to area density at previous vertex
            float G = prev_cosine / distanceSQR;

            float vcm = 1.0f / pdfFwd_area;
            float vc = (G / pdfFwd_area) * (prev_d_vcm + pdf_onebeforePrevRev_SA * prev_d_vc);

            lightPath->d_vcm[currIdx] = vcm;
            lightPath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }


        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Set up next interaction
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        bool transmitting = dot(wi_world, lightPath->n[currIdx]) < 0.0f;

        if (transmitting)
            r.origin = intersect.point - lightPath->n[currIdx] * RAY_EPSILON;
        else
            r.origin = intersect.point + lightPath->n[currIdx] * RAY_EPSILON;
        r.direction = wi_world;

        pathLength++;
        prevPDF_solidAngle = pdfFwd_solidAngle; // update the prev pdf
        prev_cosine = fabsf(wi_local.z); // update the prev cosine
        prevWasDelta = currDelta;
    }
}

// performs the randomwalk from a sampled light, and takes care of the vertex connection sttage where light is made to directly hit the camera lense.
__global__ void lightPathTracing (curandState* rngStates, Camera camera, PathVertices* eyePath, PathVertices* lightPath, int* lightPathLengths, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, 
    int lightDepth, Vertices* vertices, int vertNum, Triangle* scene, int triNum, Triangle* lights, int lightNum, int numSample, int w, int h, float4 sceneCenter, float sceneRadius, float4* colors, float4* overlay) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    curandState localState = rngStates[pixelIdx];

    int lightPathLength;
    generateLightPath(localState, materials, textures, BVH, BVHindices, lightDepth, vertices, vertNum, scene, triNum, lights, lightNum, w, h, x, y, sceneCenter, sceneRadius, lightPath, lightPathLength);

    lightPathLengths[pixelIdx] = lightPathLength;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Perform special case of the connection: what if the light ray just connected straight to the camera?
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    
    for (int s = 1; (s <= lightPathLength) && (BDPT_LIGHTTRACE); s++)
    {
        int lightPathIDX = pathBufferIdx(w, h, x, y, s - 1);

        float2 pixelPos;
        if (!camera.worldToRaster(lightPath->pt[lightPathIDX], pixelPos))
            continue;

        int px = (int)pixelPos.x;
        int py = (int)pixelPos.y;
        int newPixelIndex = py * w + px;

        //if (px == 928 && py == 900)
        //    drawPath(overlay, lightPath, camera, x, y, w, lightPathLength, lightDepth, f4(1.0f, 1.0f, 0.0f));

        if (lightPath->isDelta[lightPathIDX])
            continue;

        float etaI = 1.0f;
        float etaT = 1.0f;

        float4 lightToCamera = camera.cameraOrigin - lightPath->pt[lightPathIDX];
        float4 lightToCamera_unit = normalize(lightToCamera);

        Ray r = Ray(lightPath->pt[lightPathIDX] + lightPath->n[lightPathIDX] * RAY_EPSILON, lightToCamera_unit);
        float4 throughputScale;

        BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, length(lightToCamera) - RAY_EPSILON, -1);

        if (lengthSquared(throughputScale) > EPSILON)
        {
            int prevIdx = pathBufferIdx(w, h, x, y, s - 2); 
            float cosAtLight = dot(lightPath->n[lightPathIDX], lightToCamera_unit);
            float cosAtCamera = fabsf(dot(camera.getForwardVector(), -lightToCamera_unit));

            if (cosAtLight <= EPSILON) continue;

            float4 lightNormal = lightPath->n[lightPathIDX];
            float4 currToPrev_world = lightPath->wo[lightPathIDX];
            float4 currToPrev_local;
            toLocal(currToPrev_world, lightNormal, currToPrev_local);

            float4 lightToCamera_local;
            toLocal(lightToCamera_unit, lightNormal, lightToCamera_local);

            //---------------------------------------------------------------------------------------------------------------------------------------------------
            // Unweighted contribution calculation
            //---------------------------------------------------------------------------------------------------------------------------------------------------

            float4 light_f;
            if (s == 1)
            {
                light_f = f4(1.0f/PI); // since we initialized beta with a pi factor
            }
            else
            {
                f_eval(materials, lightPath->materialID[lightPathIDX], textures, -currToPrev_local, lightToCamera_local, etaI, etaT, light_f, lightPath->uv[lightPathIDX]);
            }

            float aspect = (float)w / (float)h;
            float halfH = camera.fovScale;
            float halfW = halfH * aspect;
            float sensorArea = (2.0f * halfW) * (2.0f * halfH);
            float pixelArea = sensorArea / (float)(w * h);
            
            float We = 1.0f / (pixelArea * cosAtCamera * cosAtCamera * cosAtCamera);

            float distanceSQR = fmaxf(lengthSquared(lightToCamera), RAY_EPSILON);
            float G = (cosAtLight * cosAtCamera) / distanceSQR;

            float4 contribution = lightPath->beta[lightPathIDX] * light_f * G * throughputScale * We; // unweighted
            
            //---------------------------------------------------------------------------------------------------------------------------------------------------
            // MIS Weight Calculation
            //---------------------------------------------------------------------------------------------------------------------------------------------------
            float misWeight;
            
            if (s == 1)
            {
                float pdf_sample_y0 = lightPath->pdfFwd[lightPathIDX];
                float pdf_traceFromCamera = cosAtLight / (distanceSQR * sensorArea * cosAtCamera * cosAtCamera * cosAtCamera);

                float wLight = pdf_traceFromCamera / pdf_sample_y0;
    
                misWeight = 1.0f / (1.0f + wLight);
            }
            else
            {
                // we dont consider randomwalks onto the eye lens
                float wEye = 0.0f;

                // the chance to begin a eye path at the eye vertex
                float pdf_trace = 1.0f;

                // the chance to connect a light vertex to the eye vertex
                float pdf_connect = 1.0f;

                float traceRatio = pdf_trace / pdf_connect;

                // the camera emission pdf of generating the current vertex
                float pdf_currRev_area = cosAtLight / (distanceSQR * sensorArea * cosAtCamera * cosAtCamera * cosAtCamera);

                //float numLightSample = (float) w * (float) h;
                float numLightSample = 1.0f;

                float pdf_oneBeforePrevRev_SA;
                pdf_eval(materials, lightPath->materialID[lightPathIDX], textures, -lightToCamera_local, currToPrev_local, etaI, etaT, 
                        pdf_oneBeforePrevRev_SA, lightPath->uv[lightPathIDX]);
                
                float wLight = traceRatio * (pdf_currRev_area / numLightSample) * 
                    (lightPath->d_vcm[lightPathIDX] + pdf_oneBeforePrevRev_SA * lightPath->d_vc[lightPathIDX]);

                misWeight = 1.0f / (1.0f + wLight + wEye);
            }


            float4 weightedContribution = contribution * misWeight;
            
            if (!BDPT_DOMIS)
                weightedContribution = contribution;
            
            if (BDPT_PAINTWEIGHT)
                weightedContribution = f4(misWeight);
            else
                weightedContribution = weightedContribution / (float)(w * h);

            atomicAdd(&colors[newPixelIndex].x, weightedContribution.x);
            atomicAdd(&colors[newPixelIndex].y, weightedContribution.y);
            atomicAdd(&colors[newPixelIndex].z, weightedContribution.z);
        }
    }
    rngStates[pixelIdx] = localState;
}
/*
 This function is never called with t=1. That is reserved for the first kernel to deal with
 Returns the unweighted contribution and the mis weight 
 
 The accumulated misWeight stored at a given path vertex vi does not know about the location of vi+1,
 so therefore it cannot hold the reverse/fwd pdf ratio at vi, since the reverse pdf at that point requires
 and incident direction corresponding to where vi+1 is (that is not known at path generation time).

 Therefore, we must always complete the partial mis weight by applying a recursive ratio corresponding
 to the reverse/fwd pdf of vi, in addition to any other terms representing other strategies.
 */
__device__ bool connectPath(curandState& localState, int t, int s, int x, int y, int w, int h, Camera camera, int maxEyeDepth, int maxLightDepth, Material* materials,BVHnode* BVH, int* BVHindices, Vertices* vertices, 
    Triangle* scene, Triangle* lights, int lightNum, float4* textures, float sceneRadius, int eyePathLength, int lightPathLength, PathVertices* eyePath, PathVertices* lightPath, float4& contribution, float& misWeight)
{
    int eyePathIDX = pathBufferIdx(w, h, x, y, t - 1);
    int eyePathPREVIDX = pathBufferIdx(w, h, x, y, t - 2);
    int lightPathIDX;
        
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Delta Case
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    if (s > 0)
    {   
        lightPathIDX = pathBufferIdx(w, h, x, y, s - 1);
        if (eyePath->isDelta[eyePathIDX] || lightPath->isDelta[lightPathIDX])
            return true;
    }
    else
    {
        if (eyePath->isDelta[eyePathIDX])
            return true;
    }
    

    float etaI = 1.0f; // placeholders
    float etaT = 1.0f;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // s = k > 1, t = 1: Connect light directly to camera. This is handled in the lightpathtracing kernel in the first pass
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // s = 1, t = k > 1: NEE
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    
    if (s == 1 && t > 1 && BDPT_NEE)
    {
        //float eye_misWeight = eyePath->misWeight[eyePathIDX];
        float4 nee_contribution_unweighted; // assigned in nee
        float pdf_connect; // assigned in nee. in area measure for area light, and in SA for environment
        float4 eyeToLight; // assigned in nee
        int lightInd; // assigned in nee
        float cosLight; // assigned in nee
        float pdf_emit_SA; // the probability that the light was sampled to emit, decoupled from the nee probability

        float4 prevToCurr_local;
        float4 prevTocurr = -eyePath->wo[eyePathIDX];
        float4 prevTocurrUnit = normalize(prevTocurr);
        // shading function expects toShadingPos_local to face towards the surface, wo faces away
        toLocal(prevTocurr, eyePath->n[eyePathIDX], prevToCurr_local);

        // sets eyeToLight, lightPDF_area, lightInd, cosLight, neecontributionunweighted
        bool occluded = BDPTnextEventEstimation(localState, materials, textures, BVH, BVHindices, vertices, scene, lights, lightNum, eyePath->materialID[eyePathIDX], 
            eyePath->pt[eyePathIDX], prevToCurr_local, eyePath->n[eyePathIDX], eyePath->uv[eyePathIDX], pdf_connect, nee_contribution_unweighted, eyeToLight, 
            lightInd, cosLight, pdf_emit_SA, etaI, etaT, sceneRadius);
        if (occluded)
        {
            return true;
        }
        float4 eyeToLight_unit = normalize(eyeToLight);
        float4 eyeToLight_local;
        toLocal(eyeToLight_unit, eyePath->n[eyePathIDX], eyeToLight_local);
        if (lightInd != -1)
        {
            float distanceSQR = fmaxf(lengthSquared(eyeToLight), RAY_EPSILON);

            float wLight;
            float wEye;

            float pdf_eyeToLight_solidAngle;
            
            pdf_eval(materials, eyePath->materialID[eyePathIDX], textures, prevToCurr_local, eyeToLight_local, etaI, etaT, pdf_eyeToLight_solidAngle, eyePath->uv[eyePathIDX]);
            float pdf_bsdf_area = pdf_eyeToLight_solidAngle * fabsf(cosLight) / distanceSQR;

            float bsdfRatio = pdf_bsdf_area / pdf_connect;
            wLight = bsdfRatio;

            // the probability to start a ligth path at the light, in this implementation its equal to nee probability
            float pdf_trace = pdf_connect;

            float traceRatio = pdf_trace / pdf_connect;

            // the probability of a light path emitting to the current vertex, converted to area density at the current vertex
            float pdf_currRev_area = pdf_emit_SA * fabsf(eyeToLight_local.z) / distanceSQR;

            float pdf_oneBeforePrevRev_SA;
            pdf_eval(materials, eyePath->materialID[eyePathIDX], textures, -eyeToLight_local, -prevToCurr_local, etaI, etaT, pdf_oneBeforePrevRev_SA, eyePath->uv[eyePathIDX]);

            wEye = traceRatio * pdf_currRev_area * (eyePath->d_vcm[eyePathIDX] + pdf_oneBeforePrevRev_SA * eyePath->d_vc[eyePathIDX]);

            misWeight = 1.0f / (1.0f + wLight + wEye);

            contribution = nee_contribution_unweighted * eyePath->beta[eyePathIDX];
            //printf("wLight: %f, wEye %f\n", wLight, eyePath->d_vc[eyePathIDX]);
        }
        else // uh im not doing this rn
        {
            // environment lights currently unimplemented
        }

        
        return true;
    }

    
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // s = 0, t = k > 1: eye randomwalk randomly walked onto a light source.
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    
    if (s == 0 && t >= 1) 
    {
        if (!BDPT_NAIVE)
            return true;
        if (t == eyePathLength && (eyePath->lightInd[eyePathIDX] == -1)) // path terminated on the sky. We need to add the sky contribution (stored in beta)
        {
            // environment lights currently unimplemented
            return false;
        }
        else if (eyePath->lightInd[eyePathIDX] != -51 && !eyePath->backface[eyePathIDX]) // ie. we are on a light, and we are on the right side of it
        {
            float4 lightToPrev_unit = normalize(eyePath->wo[eyePathIDX]);
            float4 lightNorm = eyePath->n[eyePathIDX];
            float cosThetaLight = fabsf(dot(lightNorm, lightToPrev_unit));
            float distanceSQR = lengthSquared(eyePath->pt[eyePathIDX] - eyePath->pt[eyePathPREVIDX]);
            if (t == 1)
            {
                float cosAtCamera = fabsf(dot(eyePath->n[eyePathPREVIDX], -lightToPrev_unit));

                float aspect = (float)w / (float)h;
                float halfH = camera.fovScale;
                float halfW = halfH * aspect;
                float sensorArea = (2.0f * halfW) * (2.0f * halfH);

                float4 light_f = f4(1.0f/PI);
                
                float pdf_traceFromCamera = cosThetaLight / (distanceSQR * sensorArea * cosAtCamera * cosAtCamera * cosAtCamera);

                float pdf_chooseLight = 1.0f / (SAMPLE_ENVIRONMENT ? (lightNum + 1.0f) : lightNum);

                Triangle light = lights[eyePath->lightInd[eyePathIDX]];
                float4 apos = vertices->positions[light.aInd];
                float4 bpos = vertices->positions[light.bInd];
                float4 cpos = vertices->positions[light.cInd];

                float area = 0.5f * length(cross3(bpos - apos, cpos - apos));

                float pdf_connect = pdf_chooseLight / area;

                float wEye = pdf_connect / pdf_traceFromCamera;
    
                misWeight = 1.0f / (1.0f + wEye);
            }
            else
            {
                float4 Le = lights[eyePath->lightInd[eyePathIDX]].emission;

                float wEye;
                float wLight = 0.0f;
                float pdf_connect;

                if (eyePath->isDelta[eyePathPREVIDX])
                {
                    pdf_connect = 0.0f;
                }
                else
                {
                    float pdf_chooseLight = 1.0f / (SAMPLE_ENVIRONMENT ? (lightNum + 1.0f) : lightNum);

                    Triangle light = lights[eyePath->lightInd[eyePathIDX]];
                    float4 apos = vertices->positions[light.aInd];
                    float4 bpos = vertices->positions[light.bInd];
                    float4 cpos = vertices->positions[light.cInd];

                    float area = 0.5f * length(cross3(bpos - apos, cpos - apos));

                    pdf_connect = pdf_chooseLight / area;
                }

                // these happen to be equal, since pdf_trace is the spatial prob of starting light path at the current vertex
                float pdf_trace = pdf_connect;
                
                // purely the emission pdf of generating the previous vertex
                float pdf_oneBeforePrevRev_SA = cosThetaLight / PI; 

                wEye = pdf_connect * eyePath->d_vcm[eyePathIDX] + pdf_trace * pdf_oneBeforePrevRev_SA * eyePath->d_vc[eyePathIDX];

                misWeight = 1.0f / (1.0f + wEye + wLight);

                contribution = Le * eyePath->beta[eyePathIDX];
                
                if (eyePath->isDelta[eyePathPREVIDX])
                {
                    float lum = luminance(contribution);
                    if (lum > MAX_FIREFLY_LUM)
                    {
                        contribution *= (MAX_FIREFLY_LUM / lum);
                    }
                }
            }
            
        }
        return true;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // General Case: s > 1, t > 1
    // Connect a vertex from the Eye Path (eyePathIDX) to a vertex from the Light Path (lightPathIDX)
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    
    if ( s > 1 && t > 1 && BDPT_CONNECTION)
    {
        float4 lightPos = lightPath->pt[lightPathIDX];
        float4 eyePos = eyePath->pt[eyePathIDX];
        float4 lightNorm = lightPath->n[lightPathIDX];
        float4 eyeNorm = eyePath->n[eyePathIDX];

        float4 eyeToLight = lightPos - eyePos; 
        float distanceSQR = fmaxf(lengthSquared(eyeToLight), RAY_EPSILON);
        float distance = length(eyeToLight);
        float4 eyetoLight_unit = eyeToLight / distance; // Normalized direction: Eye -> Light
        float4 lightToEye_unit = -eyetoLight_unit; // Normalized direction: Eye -> Light

        if (distanceSQR < RAY_EPSILON)
            return true;

        float cosLight = fabsf(dot(lightNorm, -eyetoLight_unit));
        float cosEye = fabsf(dot(eyeNorm, eyetoLight_unit));

        // If geometry allows connection
        if ((cosLight > EPSILON) && (cosEye > EPSILON)) 
        {
            // currently connections cannot happen through transmissive materials
            Ray r = Ray(eyePos + eyeNorm * RAY_EPSILON, eyetoLight_unit);

            float4 throughputScale;
            BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, distance - RAY_EPSILON, -1);

            if (lengthSquared(throughputScale) > EPSILON)
            {
                //-------------------------------------------------------
                // Calculate reverse pdf at curr eye index (area)
                //-------------------------------------------------------
                float4 lightToEye_localAtLight;
                toLocal(lightToEye_unit, lightNorm, lightToEye_localAtLight);

                float4 toLightFromPrev_localAtLight;
                toLocal(-lightPath->wo[lightPathIDX], lightNorm, toLightFromPrev_localAtLight);

                // bsdf evaluated at the light vertex, of scattering towards the eye vertex
                float pdf_eyeRev_SA;
                pdf_eval(materials, lightPath->materialID[lightPathIDX], textures, toLightFromPrev_localAtLight, 
                    lightToEye_localAtLight, etaI, etaT, pdf_eyeRev_SA, lightPath->uv[lightPathIDX]);
                
                // convert to area density around the eye vertex
                float pdf_eyeRev_area = pdf_eyeRev_SA * cosEye / distanceSQR;

                //-------------------------------------------------------
                // Calculate reverse pdf at prev eye index (SA)
                //-------------------------------------------------------

                float4 lightToEye_localAtEye;
                toLocal(lightToEye_unit, eyeNorm, lightToEye_localAtEye);

                float4 toPrevFromEye_localAtEye;
                toLocal(eyePath->wo[eyePathIDX], eyeNorm, toPrevFromEye_localAtEye);
                
                // pdf of generating the vertex before the eye vertex
                float pdf_oneBeforeEyeRev_SA;
                pdf_eval(materials, eyePath->materialID[eyePathIDX], textures, lightToEye_localAtEye, 
                    toPrevFromEye_localAtEye, etaI, etaT, pdf_oneBeforeEyeRev_SA, eyePath->uv[eyePathIDX]);
                
                //-------------------------------------------------------
                // Calculate reverse pdf at curr light index (area)
                //-------------------------------------------------------
                
                float4 toEyeFromPrev_localAtEye = -toPrevFromEye_localAtEye;
                float4 eyeToLight_localAtEye = -lightToEye_localAtEye;

                float pdf_lightRev_SA;
                pdf_eval(materials, eyePath->materialID[eyePathIDX], textures, toEyeFromPrev_localAtEye, 
                    eyeToLight_localAtEye, etaI, etaT, pdf_lightRev_SA, eyePath->uv[eyePathIDX]);

                float pdf_lightRev_area = pdf_lightRev_SA * cosLight / distanceSQR;

                //-------------------------------------------------------
                // Calculate reverse pdf at prev light index (SA)
                //-------------------------------------------------------

                float4 eyeToLight_localAtLight = -lightToEye_localAtLight;
                float4 toPrevFromLight_localAtLight = -toLightFromPrev_localAtLight;
                
                float pdf_oneBeforeLightRev_SA;
                pdf_eval(materials, lightPath->materialID[lightPathIDX], textures, eyeToLight_localAtLight, 
                    toPrevFromLight_localAtLight, etaI, etaT, pdf_oneBeforeLightRev_SA, lightPath->uv[lightPathIDX]);

                float wEye = pdf_eyeRev_area * (eyePath->d_vcm[eyePathIDX] + pdf_oneBeforeEyeRev_SA * eyePath->d_vc[eyePathIDX]);
                float wLight = pdf_lightRev_area * (lightPath->d_vcm[lightPathIDX] + pdf_oneBeforeLightRev_SA * lightPath->d_vc[lightPathIDX]);

                misWeight = 1.0f / (1.0f + wEye + wLight);
                
                float4 f_eye;
                f_eval(materials, eyePath->materialID[eyePathIDX], textures, lightToEye_localAtEye, 
                    toPrevFromEye_localAtEye, etaI, etaT, f_eye, eyePath->uv[eyePathIDX]);

                float4 f_light;
                f_eval(materials, lightPath->materialID[lightPathIDX], textures, eyeToLight_localAtLight, 
                    toPrevFromLight_localAtLight, etaI, etaT, f_light, lightPath->uv[lightPathIDX]);
                
                float G = fabsf(cosEye * cosLight) / distanceSQR;
                float maxG = 2.0f;
                if (G > maxG) {
                    G = maxG; 
                }

                contribution = eyePath->beta[eyePathIDX] * lightPath->beta[lightPathIDX] * f_eye * f_light * G * throughputScale;

                return true;
            }
        }
    }
    return true;
}


__global__ void Li_bidirectional(curandState* rngStates, Camera camera, PathVertices* eyePath, PathVertices* lightPath, int* lightPathLengths, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, 
    int eyeDepth, int lightDepth, Vertices* vertices, int vertNum, Triangle* scene, int triNum, Triangle* lights, int lightNum, int numSample, int w, int h, float4 sceneCenter, float sceneRadius, float4* colors, float4* overlay) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;
    
    curandState localState = rngStates[pixelIdx];

    int eyePathLength = 0; // measures number of pathvertices, not segments
    int lightPathLength = lightPathLengths[pixelIdx]; // measures number of pathvertices, not segments

    // light path is already computed from the previous kernel
    generateEyePath(localState, camera, materials, textures, BVH, BVHindices, eyeDepth, vertices, vertNum, scene, triNum, lights, 
        lightNum, w, h, x, y, sceneRadius, eyePath, eyePathLength);

    //if (curand_uniform(&localState) < (1.0f / (w * h * 2.0f)))
    if (x == 300 && y == 300)
    {
        //drawPath(overlay, eyePath, camera, x, y, w, eyePathLength, eyeDepth, f4(1.0f, 0.0f, 0.0f));
        //drawPath(overlay, lightPath, camera, x, y, w, lightPathLength, lightDepth, f4(1.0f, 1.0f, 0.0f));
    }
    //if (x == 458 && y == (1000-634))
    //    debugPrintPath(w, x, y, maxDepth, *eyePath);

    float4 fullContribution = f4(0.0f);

    // using bdpt naming conventions with t and s
    for (int t = 2; t <= eyePathLength; t++) 
    {
        for (int s = 0; s <= lightPathLength; s++) 
        {
            float4 unweighted_contribution = f4(0.0f); // set in connect path
            float misWeight = 0.0f; // set in connect path

            //if (t != 2 || s != 2)
            //    continue;

            if (!connectPath(localState, t, s, x, y, w, h, camera, eyeDepth, lightDepth, materials, BVH, BVHindices, vertices, scene, lights, lightNum, 
                textures, sceneRadius, eyePathLength, lightPathLength, eyePath, lightPath, unweighted_contribution, misWeight) && BDPT_DRAWPATH)
            {
                drawPath(overlay, eyePath, camera, x, y, w, eyePathLength, eyeDepth, f4(curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState)));
            }

            if (s == 0 && lengthSquared(unweighted_contribution) > 0.0f && misWeight > 0.2f)
            {
                //printf("Unweighted: <%f, %f, %f> weight: %f Pixel: <%d, %d> \n", unweighted_contribution.x, unweighted_contribution.y, unweighted_contribution.z, misWeight, x, y);
            }
                
                
            float4 weightedContribution = unweighted_contribution * misWeight;
            //fullContribution += unweighted_contribution * misWeight;
            //fullContribution += f4(misWeight);
            //fullContribution += unweighted_contribution;
            if (BDPT_DOMIS)
                fullContribution += weightedContribution;
            else if (BDPT_PAINTWEIGHT)
                fullContribution += f4(misWeight);
            else
                fullContribution += unweighted_contribution;
            
            
            
            //fullContribution += unweighted_contribution;
            if (misWeight > 0.0f || lengthSquared(unweighted_contribution) > 0.0f)
            {
                //printf("mis weight: <%f>\n Full contribution: <%f, %f, %f>\n", misWeight, unweighted_contribution.x, unweighted_contribution.y, unweighted_contribution.z);
            }
                
        }
    }

    colors[pixelIdx] += fullContribution;
    rngStates[pixelIdx] = localState;
}

__global__ void cleanAndFormatImage(
    float4* accumulationBuffer, // Your raw 'colors' buffer (Sum of samples)
    float4* overlayBuffer,      // Your 'overlay' buffer
    float4* outputBuffer,       // A temporary buffer to store the result for saving
    int w, int h, 
    int currentSampleCount) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= w || idy >= h) return;

    int pixelIndex = idy * w + idx;

    // 1. Read the raw accumulated color
    float4 acc = accumulationBuffer[pixelIndex];
    float4 ov = overlayBuffer[pixelIndex];
    float4 finalColor;

    // 2. Check for NaNs/Infs BEFORE normalization
    if (isnan(acc.x) || isnan(acc.y) || isnan(acc.z)) {
        finalColor = make_float4(1.0f, 0.0f, 1.0f, 1.0f); // Bright Pink Error
    } 
    else if (isinf(acc.x) || isinf(acc.y) || isinf(acc.z)) {
        finalColor = make_float4(0.0f, 1.0f, 0.0f, 1.0f); // Bright Green Error
    } 
    else {
        // 3. Normalize (Average the samples)
        float scale = 1.0f / (float)(currentSampleCount + 1);
        finalColor = make_float4(acc.x * scale, acc.y * scale, acc.z * scale, 1.0f);
    }

    // 4. Apply Overlay (if present)
    // Assuming overlay logic: if overlay is NOT black, it overrides the render
    if (ov.x != 0.0f || ov.y != 0.0f || ov.z != 0.0f) {
        finalColor = ov;
    }

    // 5. Write to the output buffer
    outputBuffer[pixelIndex] = finalColor;
}

__host__ void launch_bidirectional(int eyeDepth, int lightDepth, Camera camera, PathVertices* eyePath, PathVertices* lightPath, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, int w, int h, float4 sceneCenter, float sceneRadius, float4* colors, float4* overlay)
{
    // --- SETUP ---
    dim3 blockSize(16, 16);  
    dim3 gridSize((w + 15) / 16, (h + 15) / 16);

    // Create a CUDA Stream (Required for Graphs)
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // RNG Setup
    curandState* d_rngStates;
    cudaMalloc(&d_rngStates, w * h * sizeof(curandState));
    unsigned long seed = 103033UL;
    initRNG<<<gridSize, blockSize, 0, stream>>>(d_rngStates, w, h, seed);
    
    // Path Lengths
    int* d_pathLengths = nullptr;
    cudaMalloc(&d_pathLengths, w * h * sizeof(int));
    cudaMemsetAsync(d_pathLengths, 0, w * h * sizeof(int), stream);

    // Temporary buffer for saving images (holds the normalized, clean result)
    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, w * h * sizeof(float4));

    // Memory Info Print
    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n", freeB / (1024.0 * 1024), totalB / (1024.0 * 1024));
    
    // Image Object (CPU)
    Image image = Image(w, h); // Assuming this exists
    std::vector<float4> h_finalOutput(w * h); // Host buffer for saving

    // Timing
    auto lastSaveTime = std::chrono::steady_clock::now();
    float saveIntervalSeconds = 5.0f;

    // --- CUDA GRAPH SETUP ---
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    bool graphCreated = false;

    std::cout << "Starting Render with CUDA Graphs..." << std::endl;
    
    for (int currSample = 0; currSample < numSample; currSample++)
    {
        // 1. CREATE GRAPH (Only runs on the first iteration)
        if (!graphCreated) {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            // Record the core render kernels
            // Note: We use '0' shared memory and pass the 'stream'
            lightPathTracing<<<gridSize, blockSize, 0, stream>>>(
                d_rngStates, camera, eyePath, lightPath, d_pathLengths, materials, textures, BVH, BVHindices, 
                lightDepth, vertices, vertNum, scene, triNum, lights, lightNum, numSample, w, h, 
                sceneCenter, sceneRadius, colors, overlay
            );

            Li_bidirectional<<<gridSize, blockSize, 0, stream>>>(
                d_rngStates, camera, eyePath, lightPath, d_pathLengths, materials, textures, BVH, BVHindices, 
                eyeDepth, lightDepth, vertices, vertNum, scene, triNum, lights, lightNum, numSample, w, h, 
                sceneCenter, sceneRadius, colors, overlay
            );

            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }

        // 2. LAUNCH GRAPH (Fast!)
        cudaGraphLaunch(instance, stream);

        // 3. CHECK SAVE INTERVAL (Every ~50 samples to save CPU cycles)
        if (DO_PROGRESSIVERENDER && currSample % 25 == 0) {
            cudaStreamSynchronize(stream);

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastSaveTime).count();

            if (elapsed >= saveIntervalSeconds) 
            {
                // Pause the stream to process the image safely
                cudaStreamSynchronize(stream);

                // Run the helper kernel (Handles NaNs, Normalization, Overlay)
                cleanAndFormatImage<<<gridSize, blockSize, 0, stream>>>(
                    colors, overlay, d_finalOutput, w, h, currSample
                );

                // Copy the clean result to Host
                cudaMemcpyAsync(h_finalOutput.data(), d_finalOutput, w * h * sizeof(float4), cudaMemcpyDeviceToHost, stream);
                
                // Wait for copy to finish
                cudaStreamSynchronize(stream);

                // Save to Image object (Now trivial loop)
                #pragma omp parallel for // Optional: OpenMP to speed up this CPU loop
                for (int i = 0; i < w * h; i++) {
                    // Convert 1D index to 2D
                    int x = i % w;
                    int y = i / w;
                    // Just set the color directly, no logic needed here!
                    image.setColor(x, y, h_finalOutput[i]);
                }

                std::string filename = "render.bmp";
                image.saveImageBMP(filename);
                image.saveImageCSV_MONO(0);
                
                lastSaveTime = std::chrono::steady_clock::now();
                printf("Saved progress at %d samples.\n", currSample);

                // Reset overlay if needed (As per your original logic)
                cudaMemsetAsync(overlay, 0, w * h * sizeof(float4), stream);
            }
        }
    }
    
    // Final cleanup
    cudaStreamSynchronize(stream);
    
    // Cleanup resources
    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_pathLengths);
    cudaFree(d_rngStates);
    cudaFree(d_finalOutput);

    // Error Checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Render executed successfully." << std::endl;
    }
}

/*
__host__ void launch_bidirectional(int eyeDepth, int lightDepth, Camera camera, PathVertices* eyePath, PathVertices* lightPath, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, int w, int h, float4 sceneCenter, float sceneRadius, float4* colors, float4* overlay)
{
    dim3 blockSize(16, 16);  
    dim3 gridSize((w+15)/16, (h+15)/16);
    curandState* d_rngStates;
    cudaMalloc(&d_rngStates, w * h * sizeof(curandState));

    unsigned long seed = 103033UL;
    initRNG<<<gridSize, blockSize>>>(d_rngStates, w, h, seed);
    cudaDeviceSynchronize();

    int* d_pathLengths = nullptr;

    cudaMalloc(&d_pathLengths, w * h * sizeof(int));
    cudaMemset(d_pathLengths, 0, w * h * sizeof(int));

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));
    
    auto lastSaveTime = std::chrono::steady_clock::now();
    float saveIntervalSeconds = 5.0f;
    Image image = Image(w, h);

    std::cout << "Running Kernels" << std::endl;
    
    for (int currSample = 0; currSample < numSample; currSample++)
    {

        lightPathTracing<<<gridSize, blockSize>>>(d_rngStates, camera, eyePath, lightPath, d_pathLengths, materials, textures, BVH, BVHindices, lightDepth, vertices, vertNum, scene, triNum, 
                lights, lightNum, numSample, w, h, sceneCenter, sceneRadius, colors, overlay);
        Li_bidirectional<<<gridSize, blockSize>>>(d_rngStates, camera, eyePath, lightPath, d_pathLengths, materials, textures, BVH, BVHindices, eyeDepth, lightDepth, vertices, vertNum, scene, triNum, 
                lights, lightNum, numSample, w, h, sceneCenter, sceneRadius, colors, overlay);
        
        cudaDeviceSynchronize();
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastSaveTime).count();

        if (elapsed >= saveIntervalSeconds) 
        {
            std::vector<float4> h_colors(w * h);
            cudaMemcpy(h_colors.data(), colors, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

            std::vector<float4> h_overlay(w * h);
            cudaMemcpy(h_overlay.data(), overlay, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    if (isnan(h_colors[i].x) || isnan(h_colors[i].y) || isnan(h_colors[i].z)) {
                        h_colors[i] = f4(1.0f, 0.0f, 1.0f); // Bright Pink for NaN
                    }
                    if (isinf(h_colors[i].x) || isinf(h_colors[i].y) || isinf(h_colors[i].z)) {
                        h_colors[i] = f4(0.0f, 1.0f, 0.0f); // Bright Green for Inf
                    }
                    if (h_colors[image.toIndex(i, j)].x < 0 || h_colors[image.toIndex(i, j)].y < 0 || h_colors[image.toIndex(i, j)].z < 0)
                        cout << i << ", " << j << " Negative color written: <" << h_colors[image.toIndex(i, j)].x << ", " << h_colors[image.toIndex(i, j)].y << ", " 
                        << h_colors[image.toIndex(i, j)].z << ">"<< endl;
                    
                    if (h_overlay[image.toIndex(i, j)].x == 0.0f && h_overlay[image.toIndex(i, j)].y == 0.0f && h_overlay[image.toIndex(i, j)].z == 0.0f)
                        image.setColor(i, j, h_colors[image.toIndex(i, j)] / (float)(currSample + 1));
                    else
                        image.setColor(i, j, h_overlay[image.toIndex(i, j)]);
                }
            }
            std::string filename = "render.bmp";
            image.saveImageBMP(filename);
            image.saveImageCSV_MONO(0);
            lastSaveTime = now;
            printf("saved progress at %d samples.\n", currSample);

            cudaMemset(overlay, 0, w * h * sizeof(float4));
        }

    }
    
    cudaDeviceSynchronize();
    cudaFree(d_pathLengths);
    cudaFree(d_rngStates);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: CUDA Error code: " << static_cast<int>(err) << std::endl;
        // only call this if the code isn't catastrophic
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    }
    else
        std::cout << "Render executed with no CUDA error" << std::endl;
}*/