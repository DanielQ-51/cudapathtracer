#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include "util.cuh"
#include <numeric>
#include <algorithm>

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
};

struct Triangle
{
    int aInd;
    int bInd;
    int cInd;
    int naInd, nbInd, ncInd; // Normal indices
    //int normInd; // THIS ISNT BEING USED
    int materialID;
    float4 emission;

    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, float4 e)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), emission(e) {}
};

struct Ray
{
    float4 origin;
    float4 direction;

    __host__ __device__ Ray() {}

    __host__ __device__ Ray(const float4& o, const float4& d) : origin(o), direction(d) {}

    __host__ __device__ float4 at(float t) const {return origin + t*direction;}

};

struct Intersection
{
    float4 point;
    float4 normal;
    float4 color;
    float4 emission;
    //Ray ray;
    Triangle tri;
    int triIDX;
    int materialID;
    bool valid;
    bool backface;

    float dist;

    __device__ Intersection() {valid = false;};
};

enum MaterialType {
    MAT_DIFFUSE = 0,
    MAT_METAL = 1,
    MAT_SMOOTHDIELECTRIC = 2,
    MAT_MICROFACETDIELECTRIC = 3
};

struct Material
{
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
    bool transmissive; // for mediums tack calculations

    float4 absorption;

    int priority; // dielectric priority, for nested dielectrics/medium stack

    __host__ __device__ Material()
        : type(MAT_DIFFUSE), albedo(f4(0.8f)),
          roughness(0.5f), eta(f4(0)), k(f4(0)),
          ior(1.5f), metallic(0.0f), specular(1.0f), transmission(0.0f) {}

    __host__ __device__ static Material Diffuse(const float4& color) {
        Material m;
        m.type = MAT_DIFFUSE;
        m.albedo = color;
        m.roughness = 1.0f;
        m.transmissive = false;
        m.absorption = f4();
        return m;
    }

    __host__ __device__ static Material Metal(const float4& n, const float4& k, float roughness = 0.1f) {
        Material m;
        m.type = MAT_METAL;
        m.eta = n;
        m.k = k;
        m.roughness = roughness;
        m.albedo = f4(1.0f);  // metals usually reflect via Fresnel, not albedo tint
        m.metallic = 1.0f;
        m.transmissive = false;
        m.absorption = f4();
        return m;
    }

    __host__ __device__ static Material SmoothDielectric(float ior = 1.5f, const float4& k = f4(), int pri = 0) {
        Material m;
        m.type = MAT_SMOOTHDIELECTRIC;
        m.ior = ior;
        m.albedo = f4(1.0f);

        m.priority = pri;
        m.isSpecular = true;
        m.transmissive = true;

        m.absorption = k;
        return m;
    }

    __host__ __device__ static Material MicrofacetDielectric(float ior = 1.5f, float roughness = 0.0f, const float4& k = f4()) {
        Material m;
        m.type = MAT_MICROFACETDIELECTRIC;
        m.ior = ior;
        m.k = k;
        m.roughness = roughness;
        m.albedo = f4(1.0f);

        return m;
    }
};