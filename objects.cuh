#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include "util.cuh"

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
                            int& totalNodes = *(new int(0)), int& leafCount = *(new int(0)),
                            int& maxDepth = *(new int(0)), int& largestLeaf = *(new int(0)))
{
    if (nodeIdx >= bvh.size() || nodeIdx < 0) return;

    const BVHnode& node = bvh[nodeIdx];
    totalNodes++;
    maxDepth = std::max(maxDepth, level);

    if (node.left == -1 && node.right == -1)
    {
        leafCount++;
        largestLeaf = std::max(largestLeaf, node.primCount);
    }

    if (node.left != -1)
        printBVHSummary(bvh, node.left, level + 1, totalNodes, leafCount, maxDepth, largestLeaf);
    if (node.right != -1)
        printBVHSummary(bvh, node.right, level + 1, totalNodes, leafCount, maxDepth, largestLeaf);

    // Only print once at the top level
    if (level == 0)
    {
        std::cout << "================= BVH Summary =================\n";
        std::cout << "Total nodes:      " << totalNodes << "\n";
        std::cout << "Leaf nodes:       " << leafCount << "\n";
        std::cout << "Max depth:        " << maxDepth << "\n";
        std::cout << "Largest leaf:     " << largestLeaf << " primitives\n";
        std::cout << "Internal nodes:   " << (totalNodes - leafCount) << "\n";
        std::cout << "===============================================\n";
        delete &totalNodes;
        delete &leafCount;
        delete &maxDepth;
        delete &largestLeaf;
    }
}



struct Vertex 
{
    float4 position;
    float4 normal;
    float4 color;
    //float2 uv;

    __host__ __device__ Vertex (const float4& p,const float4& c, const float4& n) 
        : position(p), normal(n), color(c) {}
};

struct Triangle
{
    int aInd;
    int bInd;
    int cInd;
    int normInd; // THIS ISNT BEING USED
    int materialID;
    float4 emission;

    __host__ __device__ Triangle() {}

    __host__ Triangle(int a, int b, int c, int matID, float4 em)
        : aInd(a), bInd(b), cInd(c), materialID(matID), emission(em) {}
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
    int materialID;
    bool valid;

    __device__ Intersection() {valid = false;};
};