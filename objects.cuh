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

    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, float4 e)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), emission(e) {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, int uva, int uvb, int uvc, float4 e)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), uvaInd(uva), uvbInd(uvb), uvcInd(uvc), emission(e) {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, int uva, int uvb, int uvc, float4 e, int lind)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), uvaInd(uva), uvbInd(uvb), uvcInd(uvc), emission(e), lightInd(lind){}
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

        return c;
    }

    __device__ Ray generateCameraRay(curandState& localState, int x, int y)
    {
        Ray r;
        float aspect = (float)w / (float)h;

        // 1. Anti-Aliasing Jitter
        // Use the struct variable, not hardcoded 0.5
        float jitterX = (curand_uniform(&localState) - 0.5f) * antiAliasJitterDist;
        float jitterY = (curand_uniform(&localState) - 0.5f) * antiAliasJitterDist;

        // 2. Screen Space Coordinates
        float u = (2.0f * ((x + jitterX) / (float)w) - 1.0f) * aspect * fovScale;
        float v = (2.0f * ((y + jitterY) / (float)h) - 1.0f) * fovScale;

        // 3. Calculate Focal Point on the PLANAR focus plane
        // We create the vector in local space pointing to the focus plane.
        // Since local Z is -1, multiplying by focalDist ensures the Z-depth is exactly focalDist.
        // DO NOT NORMALIZE HERE.
        float4 focalVectorLocal = f4(u * focalDist, v * focalDist, -focalDist);

        // 4. Rotate Focal Vector to World Space
        float4 focalVectorWorld = rotateX(focalVectorLocal, xRot);
        focalVectorWorld = rotateY(focalVectorWorld, yRot);

        // This is the specific point in world space where the ray must end up
        float4 focalPoint = cameraOrigin + focalVectorWorld;

        // 5. Sample the Lens (Aperture)
        float4 lensOffset = f4(0.0f, 0.0f, 0.0f, 0.0f);
        
        if (aperture > 0.0f)
        {
            float r_rnd = curand_uniform(&localState); 
            float theta = 2.0f * 3.141592f * curand_uniform(&localState);
            
            // Uniform disk sampling (sqrt for area preservation)
            float radius = aperture * sqrtf(r_rnd);
            float lensU = radius * cosf(theta);
            float lensV = radius * sinf(theta);

            // Re-calculate Basis vectors (Or better: store them in struct)
            float4 baseRight = f4(1.0f, 0.0f, 0.0f);
            float4 baseUp    = f4(0.0f, 1.0f, 0.0f);
            
            float4 camRight = rotateY(rotateX(baseRight, xRot), yRot);
            float4 camUp    = rotateY(rotateX(baseUp, xRot), yRot);

            lensOffset = (camRight * lensU) + (camUp * lensV);
        }

        // 6. Final Ray Construction
        // Ray starts at the perturbed lens position
        r.origin = cameraOrigin + lensOffset;
        
        // Ray points from perturbed origin towards the fixed focal point
        r.direction = normalize(focalPoint - r.origin);

        return r;
    }

    __host__ __device__ float4 getForwardVector() const
    {
        float4 localForward = f4(0.0f, 0.0f, -1.0f, 0.0f);

        float4 worldForward = rotateX(localForward, xRot);
        worldForward = rotateY(worldForward, yRot);

        return normalize(worldForward);
    }
    
};

struct PathVertex {
    int materialID;
    float4 pt;
    float4 n;
    float4 wo;
    float4 wi;
    float4 beta; 
    float pdfFwd;        
    float pdfRev;        
    bool isDelta;
    bool isLight;
    int lightInd;
};

struct Intersection
{
    float4 point;
    float4 normal;
    float4 color;
    float4 emission;

    float2 uv;
    //Ray ray;
    Triangle tri;
    int triIDX;
    int materialID;
    bool valid;
    bool backface;

    float dist;

    __device__ Intersection() {valid = false;};
};

enum IntegratorChoice {
    UNIDIRECTIONAL = 0,
    BIDIRECTIONAL = 1,
};

enum MaterialType {
    MAT_DIFFUSE = 0,
    MAT_METAL = 1,
    MAT_SMOOTHDIELECTRIC = 2,
    MAT_MICROFACETDIELECTRIC = 3,
    MAT_LEAF = 4,
    MAT_FLOWER = 5
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

        return m;
    }
};