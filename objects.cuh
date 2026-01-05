#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include "util.cuh"
#include <numeric>
#include <algorithm>
#include <curand_kernel.h>

struct BVHnode
{
    float4 aabbMIN;
    float4 aabbMAX;
    int left;
    int right;
    int first;
    int primCount;

    __host__ BVHnode() {}

    __host__ BVHnode(float4 min, float4 max, int l, int r, int f, int ct)
        : aabbMIN(min), aabbMAX(max), left(l), right(r), first(f), primCount(ct) {} 
};

inline void printBVH(const std::vector<BVHnode>& bvh, const std::vector<int>& indices, int level = 0, int nodeIdx = 0)
{
    if (nodeIdx >= bvh.size()) return;

    const BVHnode& node = bvh[nodeIdx];

    // Indent based on tree level
    for (int i = 0; i < level; i++) std::cout << "  ";

    std::cout << "Node " << nodeIdx 
              << " | first=" << node.first 
              << " primCount=" << node.primCount 
              << " left=" << node.left 
              << " right=" << node.right 
              << "\n";

    for (int i = 0; i < level; i++) std::cout << "  ";
    std::cout << "  AABB min=(" << node.aabbMIN.x << "," << node.aabbMIN.y << "," << node.aabbMIN.z << ")"
              << " max=(" << node.aabbMAX.x << "," << node.aabbMAX.y << "," << node.aabbMAX.z << ")\n";

    // If leaf node, print its primitives
    if (node.primCount > 0)
    {
        for (int i = 0; i < level; i++) std::cout << "  ";
        std::cout << "  Primitives: ";
        for (int i = node.first; i < node.first + node.primCount; i++)
            std::cout << indices[i] << " ";
        std::cout << "\n";
    }

    // Recurse into children
    if (node.left != -1) printBVH(bvh, indices, level + 1, node.left);
    if (node.right != -1) printBVH(bvh, indices, level + 1, node.right);
}

inline void printBVHSummary(const std::vector<BVHnode>& bvh, int nodeIdx = 0, int level = 0,
                            int& totalNodes = *(new int(0)),
                            std::vector<int>* leafDepths = nullptr,
                            std::vector<int>* leafSizes = nullptr)
{
    if (!leafDepths) leafDepths = new std::vector<int>();
    if (!leafSizes) leafSizes = new std::vector<int>();

    if (nodeIdx >= bvh.size() || nodeIdx < 0) return;

    const BVHnode& node = bvh[nodeIdx];
    totalNodes++;

    if (node.left == -1 && node.right == -1) // leaf node
    {
        leafDepths->push_back(level);
        leafSizes->push_back(node.primCount);
    }

    if (node.left != -1)
        printBVHSummary(bvh, node.left, level + 1, totalNodes, leafDepths, leafSizes);
    if (node.right != -1)
        printBVHSummary(bvh, node.right, level + 1, totalNodes, leafDepths, leafSizes);

    // Only print at the top level
    if (level == 0)
    {
        int leafCount = leafDepths->size();
        int maxDepth = leafCount > 0 ? *std::max_element(leafDepths->begin(), leafDepths->end()) : 0;
        int largestLeaf = leafCount > 0 ? *std::max_element(leafSizes->begin(), leafSizes->end()) : 0;

        auto mean = [](const std::vector<int>& v) {
            return v.empty() ? 0.0 : static_cast<double>(std::accumulate(v.begin(), v.end(), 0LL)) / v.size();
        };

        auto stddev = [mean](const std::vector<int>& v) {
            if (v.empty()) return 0.0;
            double m = mean(v);
            double sumSq = 0.0;
            for (int x : v) sumSq += (x - m) * (x - m);
            return std::sqrt(sumSq / v.size());
        };

        auto median = [](std::vector<int> v) -> double {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            size_t n = v.size();
            return n % 2 == 0 ? 0.5 * (v[n/2 - 1] + v[n/2]) : v[n/2];
        };

        std::cout << "================= BVH Summary =================\n";
        std::cout << "Total nodes:       " << totalNodes << "\n";
        std::cout << "Leaf nodes:        " << leafCount << "\n";
        std::cout << "Max leaf depth:    " << maxDepth << "\n";
        std::cout << "Average leaf depth:" << mean(*leafDepths) << "\n";
        std::cout << "Median leaf depth: " << median(*leafDepths) << "\n";
        std::cout << "Leaf depth stddev: " << stddev(*leafDepths) << "\n";
        std::cout << "Largest leaf:      " << largestLeaf << " primitives\n";
        std::cout << "Average leaf size: " << mean(*leafSizes) << " primitives\n";
        std::cout << "Median leaf size:  " << median(*leafSizes) << " primitives\n";
        std::cout << "Leaf size stddev:  " << stddev(*leafSizes) << "\n";
        std::cout << "Internal nodes:    " << (totalNodes - leafCount) << "\n";

        // Print top 10 largest leaf sizes
        std::vector<int> sortedLeafSizes = *leafSizes;
        std::sort(sortedLeafSizes.begin(), sortedLeafSizes.end(), std::greater<int>());
        std::cout << "Top 10 largest leaf sizes: ";
        for (size_t i = 0; i < std::min<size_t>(10, sortedLeafSizes.size()); ++i)
            std::cout << sortedLeafSizes[i] << " ";
        std::cout << "\n";

        std::cout << "===============================================\n";

        delete leafDepths;
        delete leafSizes;
        delete &totalNodes;
    }
}

struct Vertices
{
    float4* positions;
    float4* normals;
    float4* colors;
    float2* uvs;
};

struct Triangle
{
    int aInd;
    int bInd;
    int cInd;
    int naInd, nbInd, ncInd; // Normal indices
    //int normInd; // THIS ISNT BEING USED

    int uvaInd, uvbInd, uvcInd;
    int materialID;
    float4 emission;

    int lightInd;
    int triInd;

    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, float4 e)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), emission(e) {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, int uva, int uvb, int uvc, float4 e)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), uvaInd(uva), uvbInd(uvb), uvcInd(uvc), emission(e) {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, int uva, int uvb, int uvc, float4 e, int lind, int tind)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), uvaInd(uva), uvbInd(uvb), uvcInd(uvc), emission(e), lightInd(lind), triInd(tind){}
};

struct Ray
{
    float4 origin;
    float4 direction;

    __host__ __device__ Ray() {}

    __host__ __device__ Ray(const float4& o, const float4& d) : origin(o), direction(d) {}

    __host__ __device__ float4 at(float t) const {return origin + t*direction;}

};

struct Camera
{
    float4 cameraOrigin;
    //float4 direction = f4(0.0f, 0.0f, -1.0f);
    int w;
    int h;

    float xRot; // in radians
    float yRot; // in radians
    float zRot; // in radians

    float aperture;
    float focalDist;
    //float imagePlaneDist; // for FOV
    float fovScale;

    float antiAliasJitterDist;

    float4 forward;
    float4 right;
    float4 up;

    __host__ static Camera Pinhole(const float4& cameraOrigin, int w, int h, float xR, float yR, float zR, float FOV, float aajitter = 2.0f)
    {
        Camera c;

        c.w = w;
        c.h = h;

        c.cameraOrigin = cameraOrigin;
        c.fovScale = tanf((FOV * 0.5f) * (3.141592f / 180.0f));
        c.xRot = xR * (3.14159265f / 180.0f);
        c.yRot = yR * (3.14159265f / 180.0f);
        c.zRot = zR * (3.14159265f / 180.0f);

        c.aperture = 0.000001f;
        c.focalDist = 1.0f/FOV;

        c.antiAliasJitterDist = aajitter;

        c.preCompute();

        return c;
    }

    __host__ static Camera NotPinhole(const float4& cameraOrigin, int w, int h, float xR, float yR, float zR, float FOV, float aperture, float focalDist, float aajitter = 2.0f)
    {
        Camera c;

        c.w = w;
        c.h = h;

        c.cameraOrigin = cameraOrigin;
        c.fovScale = tanf((FOV * 0.5f) * (3.141592f / 180.0f));
        c.xRot = xR * (3.14159265f / 180.0f);
        c.yRot = yR * (3.14159265f / 180.0f);
        c.zRot = zR * (3.14159265f / 180.0f);

        c.aperture = aperture;
        c.focalDist = focalDist;

        c.antiAliasJitterDist = aajitter;

        c.preCompute();
        return c;
    }


    // returns a camera ray with normalized direction
    __device__ Ray generateCameraRay(curandState& localState, int x, int y)
    {
        Ray r;
        float aspect = (float)w / (float)h;

        // 1. Anti-Aliasing & Screen Coords
        float jitterX = (curand_uniform(&localState) - 0.5f) * antiAliasJitterDist;
        float jitterY = (curand_uniform(&localState) - 0.5f) * antiAliasJitterDist;

        float u = (2.0f * ((x + jitterX) / (float)w) - 1.0f) * aspect * fovScale;
        float v = (2.0f * ((y + jitterY) / (float)h) - 1.0f) * fovScale;

        // 2. Calculate Focal Point using Precomputed Basis Vectors
        // This automatically handles X, Y, and Z rotation correctly.
        // Note: 'forward' corresponds to local (0,0,-1), so we move positive along 'forward'
        // to go deeper into the scene.
        float4 focalPoint = cameraOrigin + (right * (u * focalDist)) + (up * (v * focalDist)) + (forward * focalDist);

        // 3. Sample the Lens (Aperture)
        float4 lensOffset = f4(0.0f, 0.0f, 0.0f, 0.0f);
        
        if (aperture > 0.0f)
        {
            float r_rnd = curand_uniform(&localState); 
            float theta = 2.0f * 3.141592f * curand_uniform(&localState);
            
            float radius = aperture * sqrtf(r_rnd);
            float lensU = radius * cosf(theta);
            float lensV = radius * sinf(theta);

            // Use precomputed right/up vectors! 
            lensOffset = (right * lensU) + (up * lensV);
        }

        // 4. Final Ray Construction
        r.origin = cameraOrigin + lensOffset;
        r.direction = normalize(focalPoint - r.origin);

        return r;
    }

    __host__ void preCompute()
    {
        float4 localForward = f4(0.0f, 0.0f, -1.0f, 0.0f);

        float4 worldForward = rotateX(localForward, xRot);
        worldForward = rotateY(worldForward, yRot);
        worldForward = rotateZ(worldForward, zRot);

        forward = normalize(worldForward);

        float4 localRight = f4(1.0f, 0.0f, 0.0f, 0.0f);
        right = normalize(rotateZ(rotateY(rotateX(localRight, xRot), yRot), zRot));

        float4 localUp = f4(0.0f, 1.0f, 0.0f, 0.0f);
        up = normalize(rotateZ(rotateY(rotateX(localUp, xRot), yRot), zRot));
        
    }

    __host__ __device__ float4 getForwardVector() const
    {
        return forward;
    }

    __host__ __device__ float4 getRightVector() const
    {
        return right;
    }

    __host__ __device__ float4 getUpVector() const
    {
        return up;
    }

    // dark magic
    __device__ bool worldToRaster(const float4& pointWorld, float2& pixelPos)
    {
        float4 dir = pointWorld - cameraOrigin;

        float4 fwd = getForwardVector();
        float4 right = getRightVector();
        float4 up = getUpVector();

        float distZ = dot(dir, fwd);

        if (distZ <= 0.001f) return false; 

        float distX = dot(dir, right);
        float distY = dot(dir, up);

        float slopeX = distX / distZ;
        float slopeY = distY / distZ;
        
        float aspect = (float)w / (float)h;

        float ndcX = slopeX / (aspect * fovScale);
        float ndcY = slopeY / fovScale;

        if (ndcX < -1.0f || ndcX > 1.0f || ndcY < -1.0f || ndcY > 1.0f) {
            return false;
        }
        
        pixelPos.x = (ndcX + 1.0f) * 0.5f * (float)w;
        pixelPos.y = (ndcY + 1.0f) * 0.5f * (float)h;

        return true;
    }
};

__device__ __forceinline__ void drawLine(float4* overlay, Camera camera, float4 p1, float4 p2, float4 color)
{
    float nearClip = 0.002f; 
    float4 camPos = camera.cameraOrigin;
    float4 camFwd = camera.forward;

    float d1 = dot(p1 - camPos, camFwd) - nearClip;
    float d2 = dot(p2 - camPos, camFwd) - nearClip;

    if (d1 < 0.0f && d2 < 0.0f) return;

    if (d1 < 0.0f) {
        float t = d1 / (d1 - d2);
        p1 = p1 + (p2 - p1) * t;
    } 
    else if (d2 < 0.0f) {
        float t = d2 / (d2 - d1);
        p2 = p2 + (p1 - p2) * t;
    }

    float2 pxf1, pxf2;

    if (!camera.worldToRaster(p1, pxf1) || !camera.worldToRaster(p2, pxf2))
        return;
    
    int x0 = (int)pxf1.x;
    int y0 = (int)pxf1.y;

    int x1 = (int)pxf2.x;
    int y1 = (int)pxf2.y;

    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy; 

    while (true) {
        if (x0 >= 0 && x0 < camera.w && y0 >= 0 && y0 < camera.h) {
            int pixelIndex = y0 * camera.w + x0;
            
            atomicExch(&overlay[pixelIndex].x, color.x);
            atomicExch(&overlay[pixelIndex].y, color.y);
            atomicExch(&overlay[pixelIndex].z, color.z);
        }

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 >= dy) { 
            err += dy; 
            x0 += sx; 
        }
        if (e2 <= dx) { 
            err += dx; 
            y0 += sy; 
        }
    }
}

struct PathVertices {
    // material ID at this vertex
    int* materialID;
    
    // location of this vertex
    float4* pt;

    // normal at this vertex
    float4* n;

    // direction from this vertex to the previous
    float4* wo;

    // uv coordinates of this vertex
    float2* uv;

    //float4* wi; storing is not neccesary

    // throughput
    float4* beta;

    // pdf of going from previous vertex to this in area measure
    float* pdfFwd;
    
    // pdf of going from current vertex to previous vertex in area measure
    //float* pdfRev;
    
    // accumulated weight up this this vertex
    float* misWeight; 
    
    /*
    Equal to (for intermediate vertices):

    G - Geometry factor to convert a reverse pdf into area measure around the previous vertex (cos at prev, distanceSQR from previous)
    pdfFwd - Forward pdf of generating the current vertex
    PREVpdfRev - solid angle pdf of generating the vertex before the previous vertex
    d_vcm for the previous vertex
    d_vc for the previous vertex
    */
    float* d_vc;

    // stores the forward pdf for the previous vertex (area measure)
    float* d_vcm;
    bool* isDelta;
    
    int* lightInd;
    bool* backface;
};


inline __device__ __forceinline__ int pathBufferIdx(int w, int h, int x, int y, int depth)
{
    //return (y * w + x) * maxDepth + depth;
    return (depth * w * h) + y * w + x;
}

// rasterizes a path defined by the pathvertices. Used for debugging and visualization
__device__ __forceinline__ void drawPath(float4* overlay, PathVertices* path, Camera camera, int x, int y, int w, 
    int depth, int maxDepth, float4 color)
{
    for (int i = 0; i < depth - 1; i++)
    {
        //float ratio = (float)(i+1) / float(depth); 
        int pathIDX1 = pathBufferIdx(w, x, y, i, maxDepth);
        int pathIDX2 = pathBufferIdx(w, x, y, i+1, maxDepth);
        drawLine(overlay, camera, path->pt[pathIDX1], path->pt[pathIDX2], color);
    }
}

static __device__ __forceinline__ void debugPrintPath(
    int w, int h, int x, int y, int maxDepth,
    const PathVertices& PV)
{
    //const int pixelIndex = y * w + x;

    // Print header first
    printf("=== PATH DUMP pixel (%d, %d) Length: %d ===\n", x, y, maxDepth);

    // Print each depth entry with its own single printf
    // (but each line is atomic so it will not mix)
    for (int depth = 0; depth < maxDepth; depth++)
    {
        int idx = pathBufferIdx(w, h, x, y, depth);

        printf(
            "Depth %d\n"
            "  materialID: %d\n"
            "  pt:   (%.3f, %.3f, %.3f)\n"
            "  n:    (%.3f, %.3f, %.3f)\n"
            "  wo:   (%.3f, %.3f, %.3f)\n"
            "  uv:   (%.3f, %.3f)\n"
            "  beta: (%.3f, %.3f, %.3f)\n"
            "  pdfFwd: %.6f\n"
            "  misWeight: %.6f\n"
            "  isDelta: %d\n"
            "  lightInd: %d\n"
            "  backface: %d\n\n",

            depth,
            PV.materialID[idx],
            PV.pt[idx].x, PV.pt[idx].y, PV.pt[idx].z,
            PV.n[idx].x, PV.n[idx].y, PV.n[idx].z,
            PV.wo[idx].x, PV.wo[idx].y, PV.wo[idx].z,
            PV.uv[idx].x, PV.uv[idx].y,
            PV.beta[idx].x, PV.beta[idx].y, PV.beta[idx].z,
            //PV.pdfFwd[idx],
            //PV.misWeight[idx],
            PV.isDelta[idx],
            PV.lightInd[idx],
            PV.backface[idx]
        );
    }
}


struct Intersection
{
    float4 point;
    float4 normal;
    float4 color;
    float4 emission;

    float2 uv;
    //Ray ray;
    //Triangle tri;
    int triIDX;
    int materialID;
    bool valid;
    bool backface;

    float dist;

    __device__ Intersection() {valid = false; uv = f2(-1.0f);};
};

enum IntegratorChoice {
    UNIDIRECTIONAL = 0,
    BIDIRECTIONAL = 1,
    NAIVE_UNIDIRECTIONAL = 2,
};

enum MaterialType {
    MAT_DIFFUSE = 0,
    MAT_METAL = 1,
    MAT_SMOOTHDIELECTRIC = 2,
    MAT_MICROFACETDIELECTRIC = 3,
    MAT_LEAF = 4,
    MAT_FLOWER = 5,
    MAT_DELTAMIRROR = 6
};

struct Material
{
    bool hasTexture;
    int startInd;
    int width;
    int height;

    bool hasTransMap;
    int tstartInd;
    int twidth;
    int theight;


    int type;
    
    float4 albedo;
    float roughness;

    float4 eta;
    float4 k;
    float ior;

    float metallic;
    float specular;
    float transmission; 

    bool isSpecular;
    bool boundary; // for mediums tack calculations

    bool thinWalled;

    float4 absorption;

    int priority; // dielectric priority, for nested dielectrics/medium stack

    __host__ Material()
        : type(MAT_DIFFUSE), albedo(f4(0.8f)),
          roughness(0.5f), eta(f4(0)), k(f4(0)),
          ior(1.5f), metallic(0.0f), specular(1.0f), transmission(0.0f) {}

    __host__ static Material Diffuse(const float4& color) {
        Material m;
        m.type = MAT_DIFFUSE;
        m.hasTexture = false;

        m.albedo = color;
        m.roughness = 1.0f;
        m.boundary = false;
        m.absorption = f4();
        m.thinWalled = false;
        m.isSpecular = false;
        return m;
    }

    __host__ static Material DiffuseTextured(int sInd, int w, int h) {
        Material m;
        m.type = MAT_DIFFUSE;
        m.hasTexture = true;

        m.startInd = sInd;
        m.width = w;
        m.height = h;

        m.roughness = 1.0f;
        m.boundary = false;
        m.absorption = f4();
        m.thinWalled = false;

        m.isSpecular = false;
        return m;
    }

    __host__ static Material Metal(const float4& n, const float4& k, float roughness = 0.1f) {
        Material m;
        m.type = MAT_METAL;
        m.hasTexture = false;

        m.eta = n;
        m.k = k;
        m.roughness = roughness;
        m.albedo = f4(1.0f);  // metals usually reflect via Fresnel, not albedo tint
        m.metallic = 1.0f;
        m.boundary = false;
        m.absorption = f4();
        m.thinWalled = false;

        m.isSpecular = false;
        return m;
    }

    __host__ static Material SmoothDielectric(float ior = 1.5f, const float4& k = f4(), int pri = 0) {
        Material m;
        m.type = MAT_SMOOTHDIELECTRIC;
        m.hasTexture = false;

        m.ior = ior;
        m.albedo = f4(1.0f);

        m.priority = pri;
        m.isSpecular = true;
        m.boundary = true;

        m.absorption = k;
        m.thinWalled = false;
        return m;
    }

    __host__  static Material MicrofacetDielectric(float ior = 1.5f, float roughness = 0.0f, const float4& k = f4()) {
        Material m;
        m.type = MAT_MICROFACETDIELECTRIC;
        m.hasTexture = false;

        m.ior = ior;
        m.k = k;
        m.roughness = roughness;
        m.albedo = f4(1.0f);

        m.thinWalled = false;
        return m;
    }

    __host__ static Material Leaf(int sInd, int w, int h,float ior = 1.5f, float roughness = 0.7, float4 albedo = f4(), float transmission = 0.05f)
    {
        Material m;
        m.type = MAT_LEAF;

        m.hasTexture = true;
        m.hasTransMap = false;

        m.ior = ior;
        m.roughness = roughness;
        m.albedo = albedo;
        m.transmission = transmission;
        m.boundary = false;

        m.startInd = sInd;
        m.width = w;
        m.height = h;

        m.thinWalled = true;

        m.isSpecular = false;

        return m;
    }

    __host__ static Material Leaf(int sInd, int w, int h, int tsInd, int tw, int th, float ior = 1.5f, float roughness = 0.7, float4 albedo = f4(), float transmission = 0.05f)
    {
        Material m;
        m.type = MAT_LEAF;

        m.hasTexture = true;
        m.hasTransMap = true;
        
        m.ior = ior;
        m.roughness = roughness;
        m.albedo = albedo;
        m.transmission = transmission;
        m.boundary = false;

        m.startInd = sInd;
        m.width = w;
        m.height = h;

        m.tstartInd = tsInd;
        m.twidth = tw;
        m.theight = th;

        m.thinWalled = true;

        m.isSpecular = false;

        return m;
    }
    
    __host__ static Material Mirror()
    {
        Material m;
        m.type = MAT_DELTAMIRROR;

        m.hasTexture = false;
        m.hasTransMap = false;

        m.isSpecular = true;

        return m;
    }
};