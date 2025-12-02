
#include "imageUtil.cuh"
#include "util.cuh"
#include "deviceCode.cuh"
#include "objects.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <set>

using namespace std;

void readObjSimple(string filename, vector<float4>& points, vector<float4>& normals, vector<float4>& colors, vector<float2>& uvs,vector<Triangle>& mesh, 
    vector<Triangle>& lights, float4 c, float4 e, int materialID);

void computeInfoForBVH(Vertices& vertices, vector<Triangle>& mesh, vector<float4>& centroids, 
    vector<float4>& AABBmins, vector<float4>& AABBmaxes)
{
    for (int i = 0; i < mesh.size(); i++)
    {
        Triangle tri = mesh[i];
        float4 a = vertices.positions[tri.aInd];
        float4 b = vertices.positions[tri.bInd];
        float4 c = vertices.positions[tri.cInd];
        centroids.push_back(f4((a.x+b.x+c.x)/3.0f,(a.y+b.y+c.y)/3.0f,(a.z+b.z+c.z)/3.0f));

        // Compute AABB min
        float4 minPos = f4(
            fminf(fminf(a.x, b.x), c.x),
            fminf(fminf(a.y, b.y), c.y),
            fminf(fminf(a.z, b.z), c.z)
        );
        AABBmins.push_back(minPos);

        // Compute AABB max
        float4 maxPos = f4(
            fmaxf(fmaxf(a.x, b.x), c.x),
            fmaxf(fmaxf(a.y, b.y), c.y),
            fmaxf(fmaxf(a.z, b.z), c.z)
        );
        AABBmaxes.push_back(maxPos);
    }
}

int partitionPrimitives(vector<int>& indices, vector<float4>& centroids, int start, int end, int axis, float splitPos)
{
    int mid = start;
    for (int i = start; i < end; i++)
    {
        float4 c = centroids[indices[i]];
        if (getFloat4Component(c, axis) < splitPos)
        {
            swap(indices[i], indices[mid]);
            mid++;
        }
    }
    return mid;
}

void SAH( vector<int>& indices, vector<float4>& centroids, vector<float4>& AABBmins, vector<float4>& AABBmaxes, int start, 
    int end, int& axis, float4 minBound, float4 maxBound, float& splitPos, float& minCost, int& backup)
{
    const int numBuckets = 30;
    struct Bucket { float4 min, max; int count; };
    Bucket buckets[numBuckets];
    for (int i = 0; i < numBuckets; i++) 
    {
        buckets[i].min = f4(FLT_MAX);
        buckets[i].max = f4(-FLT_MAX);
        buckets[i].count = 0;
    }

    for (int i = start; i < end; i++) 
    {
        int idx = indices[i];
        float c = getFloat4Component(centroids[idx] , axis);
        int b = int(numBuckets * (c - getFloat4Component(minBound , axis)) / (getFloat4Component(maxBound , axis) - getFloat4Component(minBound , axis)));
        b = clamp(b, 0, numBuckets - 1);
        buckets[b].count++;
        buckets[b].min = fminf4(buckets[b].min, AABBmins[idx]);
        buckets[b].max = fmaxf4(buckets[b].max, AABBmaxes[idx]);
    }

    minCost = FLT_MAX;
    int bestSplit = -1;
    
    for (int i = 1; i < numBuckets; i++) 
    {
        float4 leftMin = buckets[0].min;
        float4 leftMax = buckets[0].max;
        int leftCount = buckets[0].count;
        for (int j = 1; j < i; j++) {
            leftMin = fminf4(leftMin, buckets[j].min);
            leftMax = fmaxf4(leftMax, buckets[j].max);
            leftCount += buckets[j].count;
        }

        float4 rightMin = buckets[i].min;
        float4 rightMax = buckets[i].max;
        int rightCount = buckets[i].count;
        for (int j = i; j < numBuckets; j++) {
            rightMin = fminf4(rightMin, buckets[j].min);
            rightMax = fmaxf4(rightMax, buckets[j].max);
            rightCount += buckets[j].count;
        }

        float cost = 1.0f + (leftCount * surfaceArea(leftMin, leftMax) + rightCount * surfaceArea(rightMin, rightMax))
                           / surfaceArea(minBound, maxBound);
        if (cost < minCost && (leftCount > 0 && rightCount > 0)) {
            minCost = cost;
            bestSplit = i;
        }
    }

    if (bestSplit == -1)
    {
        //cout << "no best split" << endl;

        int mid = (start + end) / 2;
        std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end,
            [&](int a, int b) { return getFloat4Component(centroids[a], axis) < getFloat4Component(centroids[b], axis); });
        
        splitPos = getFloat4Component(centroids[indices[mid]], axis);
    }
    else
        splitPos = getFloat4Component(minBound , axis) + (getFloat4Component(maxBound , axis) - getFloat4Component(minBound , axis)) * (float(bestSplit) / float(numBuckets));
}

int buildBVH(vector<BVHnode>& nodes, vector<int>& indices, vector<float4>& centroids, 
    vector<float4>& AABBmins, vector<float4>& AABBmaxes, int start, int end, 
    int maxLeafSize, int& largestLeaf, int& backup)
{
    int nodeIndex = nodes.size();
    nodes.push_back(BVHnode());

    float4 minBound = AABBmins[indices[start]];
    float4 maxBound = AABBmaxes[indices[start]];
    for (int i = start; i < end; i++)
    {
        int idx = indices[i];
        minBound = fminf4(minBound, AABBmins[idx]);
        maxBound = fmaxf4(maxBound, AABBmaxes[idx]);
    }
    nodes[nodeIndex].aabbMIN = minBound;
    nodes[nodeIndex].aabbMAX = maxBound;

    int primCount = end - start;

    if (primCount <= maxLeafSize) {
        nodes[nodeIndex].first = start;
        nodes[nodeIndex].primCount = primCount;
        nodes[nodeIndex].left = nodes[nodeIndex].right = -1;
        largestLeaf = max(primCount, largestLeaf);
        return nodeIndex;
    }
    float xdiff = maxBound.x - minBound.x;
    float ydiff = maxBound.y - minBound.y;
    float zdiff = maxBound.z - minBound.z;
    int axis = 0;
    if (ydiff > xdiff && ydiff > zdiff)
        axis = 1;
    else if (zdiff > xdiff && zdiff > ydiff)
        axis = 2;

    //int bestSplit = 0;
    float splitPos;
    float cost = 0;
    SAH(indices, centroids, AABBmins, AABBmaxes, start, end, axis, minBound, maxBound, splitPos, cost, backup);

    /*if (primCount <= maxLeafSize || cost >= primCount * 1) {
        // Force a leaf
        nodes[nodeIndex].first = start;
        nodes[nodeIndex].primCount = primCount;
        nodes[nodeIndex].left = nodes[nodeIndex].right = -1;
        largestLeaf = max(primCount, largestLeaf);
        return nodeIndex;
    }*/
    int mid;
    
    int numLeft = 0;
    for (int i = start; i < end; i++) {
        if (getFloat4Component(centroids[indices[i]], axis) < splitPos)
            numLeft++;
    }
    //cout << "1numLeft: " << numLeft << endl;
    if (numLeft > 0 && numLeft < (primCount - 1))
        mid = partitionPrimitives(indices, centroids, start, end, axis, splitPos);
    else
    {
        //cout << "midpoint backup" << endl;
        float sum = 0.0f;
        backup++;
        for (int i = start; i < end; i++)
        {
            sum += getFloat4Component(centroids[indices[i]], axis);
        }
        splitPos = sum/primCount;
    }
    //cout << "SECOND failed split at " << splitPos << " on the " << axis << " numbered axis" << endl; 
    
    numLeft = 0;
    for (int i = start; i < end; i++) {
        if (getFloat4Component(centroids[indices[i]], axis) < splitPos)
            numLeft++;
    }
    //cout << "2numLeft: " << numLeft << endl;
    if (numLeft > 0 && numLeft < (primCount - 1))
        mid = partitionPrimitives(indices, centroids, start, end, axis, splitPos);
    else
    {
        nodes[nodeIndex].first = start;
        nodes[nodeIndex].primCount = primCount;
        nodes[nodeIndex].left = nodes[nodeIndex].right = -1;
        largestLeaf = max(primCount, largestLeaf);
        //cout << start << " " << mid << " " << end << endl;
        //cout << "force leaf " << primCount << endl;
        return nodeIndex;
    }
    //cout << start << " " << mid << " " << end << endl;
        //cout << "Split at " << splitPos << " on the " << axis << " numbered axis" << endl; 
    //cout << "HELLLLLLLLLLLLLLLLLOOOOOOOOOOO " << endl;
    nodes[nodeIndex].left  = buildBVH(nodes, indices, centroids, AABBmins, AABBmaxes, start, mid, maxLeafSize, largestLeaf, backup);
    nodes[nodeIndex].right = buildBVH(nodes, indices, centroids, AABBmins, AABBmaxes, mid, end, maxLeafSize, largestLeaf, backup);

    nodes[nodeIndex].primCount = 0;
    nodes[nodeIndex].first = -1;
    
    return nodeIndex;
}

int main ()
{
    auto start = std::chrono::high_resolution_clock::now();
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Render Settings
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    int w = 3840;
    int h = 2160;

    int sampleCount = 3000;
    int maxDepth = 8;
    int maxLeafSize = 2;
    

    Image image = Image(w, h);

    cout << "Rendering at " << w << " by " << h << " pixels, with " << 
        sampleCount << " samples per pixel, and a maximum leaf size of " <<
        maxLeafSize << endl << endl;

    float4* out_colors;
    cudaMalloc(&out_colors, w * h * sizeof(float4));

    Vertices vertices;
    vector<float4> points;
    vector<float4> normals;
    vector<float4> colors; // unused now
    vector<float2> uvs;
    vector<Triangle> mesh;
    vector<Triangle> lightsvec;
    vector<BVHnode> bvhvec;

    vector<float4> centroids;
    vector<float4> minboxes;
    vector<float4> maxboxes;

    vector<Material> mats;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading Textures
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    vector<Image> images;
    vector<float4> pixels;
    vector<int> widths;
    vector<int> heights;
    vector<int> startIndices;
    int currentStartIndex = 0;

    images.push_back(loadBMPToImage("textures/enkidutexture.bmp"));
    images.push_back(loadBMPToImage("textures/enkiduchibitexture.bmp"));
    images.push_back(loadBMPToImage("textures/leaftex2.bmp"));
    images.push_back(loadBMPToImage("textures/leafautumn.bmp"));
    images.push_back(loadBMPToImage("textures/leaftransmission.bmp"));

    for (Image i : images)
    {
        vector<float4> pix = i.data();

        pixels.insert(pixels.end(), pix.begin(), pix.end());

        widths.push_back(i.width);
        heights.push_back(i.height);
        startIndices.push_back(currentStartIndex);
        currentStartIndex += i.width*i.height;
    }

    float4* textures_d;

    cudaMalloc(&textures_d, pixels.size() * sizeof(float4));
    cudaMemcpy(textures_d, pixels.data(), pixels.size() * sizeof(float4), cudaMemcpyHostToDevice);

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Creating Materials
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    Material lambertTextured = Material::DiffuseTextured(startIndices[0], widths[0], heights[0]);
    Material lambert2Textured = Material::DiffuseTextured(startIndices[1], widths[1], heights[1]);

    Material lambertBlue = Material::Diffuse(f4(0.4f,0.4f,0.8f));
    Material lambertWhite = Material::Diffuse(f4(0.9f,0.9f,0.9f));
    Material lambertGreen = Material::Diffuse(f4(0.2f,0.6f,0.6f));
    Material lambertRed = Material::Diffuse(f4(0.99f,0.01f,0.01f));

    float4 eta_steel = f4(0.14f, 0.16f, 0.13f, 1.0f);   // real part (R,G,B,alpha)
    float4 k_steel   = f4(4.1f, 2.3f, 3.1f, 1.0f);     // imaginary part (absorption)


    float4 eta_gold = f4(0.17f, 0.35f, 1.5f);  // real part of refractive index
    float4 k_gold   = f4(3.1f, 2.7f, 1.9f);   // imaginary part, absorption
    float roughness_polished = 0.05f;  
    float roughness_rough = 0.15f;  

    Material gold = Material::Metal(eta_gold, eta_gold, roughness_polished);
    Material steel = Material::Metal(eta_steel, eta_steel, roughness_rough);

    float roughness = 0.05f;
    float ior = 1.5f;

    Material glass = Material::SmoothDielectric(ior, f4(0.0f), 1);

    Material water = Material::SmoothDielectric(1.333f, f4(), 2);
    Material tea = Material::SmoothDielectric(1.333f, 2.5f * f4(0.180f, 1.5f, 2.996f), 2);

    Material ice = Material::SmoothDielectric(1.31f, f4(0.2f), 0);

    Material air = Material::SmoothDielectric(1.0f, f4(0.0f), 99);

    //Material leaf = Material::Leaf(1.5f, 0.6f, f4(0.8f, 0.25f, 0.28f), 0.2f);
    Material leaf = Material::Leaf(startIndices[2], widths[2], heights[2], 1.5f, 0.10f, f4(0.22f, 0.75f, 0.28f), 0.15f);
    Material leafAutumn = Material::Leaf(startIndices[3], widths[3], heights[3], startIndices[5], widths[5], heights[5], 1.5f, 0.7f, f4(0.22f, 0.75f, 0.28f), 0.01f);
    Material canopy = Material::Leaf(startIndices[2], widths[2], heights[2], 1.5f, 0.9f, f4(0.22f, 0.75f, 0.28f), 0.7f);
    Material leafStem = Material::Diffuse(f4(0.90f, 0.9f, 0.83f));
    Material sky = Material::Diffuse(f4(0.4f, 0.4f, 1.00f));

    mats.push_back(air); // index 0

    mats.push_back(lambertBlue); // index 1
    mats.push_back(lambertWhite); // index 2
    mats.push_back(lambertGreen); // index 3
    mats.push_back(gold); // index 4
    mats.push_back(glass); // index 5
    mats.push_back(lambertRed); // index 6
    mats.push_back(steel); // index 7
    mats.push_back(tea); // index 8
    mats.push_back(ice); // index 9
    mats.push_back(water); // index 10
    mats.push_back(lambertTextured); // index 11
    mats.push_back(lambert2Textured); // index 12
    mats.push_back(leaf); // index 13
    mats.push_back(leafStem); // index 14
    mats.push_back(sky); // index 15
    mats.push_back(leafAutumn); // index 16
    //mats.push_back(green); // index 16

    Material* mats_d;

    cudaMalloc(&mats_d, mats.size() * sizeof(Material));
    cudaMemcpy(mats_d, mats.data(), mats.size() * sizeof(Material), cudaMemcpyHostToDevice);

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading Scene Data
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    //readObjSimple("scenedata/smallbox.obj", points, normals, colors, uvs, mesh, lightsvec, f4(0.9f,0.9f,0.9f), f4(), 11);
    //cout << "scene data read. There are " << mesh.size() << " Triangles." << endl;
    //readObjSimple("scenedata/leftwall.obj", points, normals, colors, uvs, mesh, lightsvec, f4(0.4f,0.4f,0.8f), f4(), 1);
    //readObjSimple("scenedata/rightwall.obj", points, normals, colors, uvs, mesh, lightsvec, f4(0.8f,0.2f,0.2f), f4(), 3);
    //readObjSimple("scenedata/rightwall.obj", points, normals, colors, mesh, lightsvec, f4(0.2f,0.6f,0.6f), f4(), 2);
    //readObjSimple("scenedata/leftball.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(0.9f,0.4f,0.4f), f4(), 4);
    //readObjSimple("scenedata/cup3.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(0.9f,0.4f,0.4f), f4(), 12);
    //readObjSimple("scenedata/leaves3.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(0.9f,0.4f,0.4f), f4(), 13);
    //readObjSimple("scenedata/leafstem.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(0.9f,0.4f,0.4f), f4(), 14);
    //readObjSimple("scenedata/Cup.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(0.9f,0.4f,0.4f), f4(), 5);
    //readObjSimple("scenedata/water.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(0.9f,0.4f,0.4f), f4(), 10);
    //readObjSimple("scenedata/icecubes.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(0.9f,0.4f,0.4f), f4(), 9);
    //readObjSimple("scenedata/spoon.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(0.9f,0.4f,0.4f), f4(), 7);
    //readObjSimple("scenedata/swordbetter.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(0.9f,0.1f,0.1f), f4(), 3);//5.0f*f4(3.0f,1.0f,1.0f)
    //readObjSimple("scenedata/character.obj", vertices, mesh, lightsvec, 1.0f*f4(0.9f,0.9f,0.9f), f4(), 1);
    //readObjSimple("scenedata/smallcube.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(0.9f,0.9f,0.9f), f4(), 5);
    //readObjSimple("scenedata/swordbetter.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), f4(), 1);
    //readObjSimple("scenedata/leftlight.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 20.0f*f4(10.0f,1.0f,1.0f), 2);
    //readObjSimple("scenedata/rightlight.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 20.0f*f4(3.0f,3.0f,10.0f), 2);
    //readObjSimple("scenedata/reallysmalllight.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 30.0f*f4(7.0f,7.0f,3.0f), 1);
    //readObjSimple("scenedata/lightforward.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 0.5f*f4(7.0f,7.0f,7.0f), 2);
    //readObjSimple("scenedata/leaflight2.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 2.3f*f4(9.0f,9.0f,7.0f), 2);
    //readObjSimple("scenedata/lightbehindleaf.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 5.0f*f4(9.0f,9.0f,7.0f), 2);
    //readObjSimple("scenedata/backlight.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 1.5f*f4(6.0f,7.0f,7.0f), 2);

    readObjSimple("scenedata/sun2.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 300.0f*f4(10.0f,3.0f,2.0f), 2);
    //readObjSimple("scenedata/sunback.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 0.0f*f4(9.0f,6.0f,2.0f), 2);
    readObjSimple("scenedata/leavescomplex.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 0.0f*f4(9.0f,9.0f,7.0f), 13);
    readObjSimple("scenedata/branches.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 0.0f*f4(9.0f,9.0f,7.0f), 14);
    readObjSimple("scenedata/waterdrops.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 0.0f*f4(9.0f,9.0f,7.0f), 10);
    //readObjSimple("scenedata/wallkinda.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 0.0f*f4(9.0f,9.0f,7.0f), 6);
    //readObjSimple("scenedata/canopy.obj", points, normals, colors, uvs, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 0.0f * f4(0.4f, 0.4f, 1.00f), 16);
        
    Vertices* verts;
    Triangle* scene;
    Triangle* lights;

    cudaMalloc(&verts,  sizeof(Vertices));
    Vertices temp;

    cudaMalloc(&temp.positions, sizeof(float4) * points.size());
    cudaMalloc(&temp.normals, sizeof(float4) * normals.size());
    cudaMalloc(&temp.colors,  sizeof(float4) * colors.size());
    cudaMalloc(&temp.uvs,  sizeof(float2) * uvs.size());

    cudaMemcpy(temp.positions, points.data(), points.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.normals, normals.data(), normals.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.colors, colors.data(), colors.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.uvs, uvs.data(), uvs.size() * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(verts, &temp, sizeof(Vertices), cudaMemcpyHostToDevice);

    vector<int> indvec(mesh.size());
    for (int i = 0; i < mesh.size(); i++) indvec[i] = i;

    if (mesh.size() == 0) {
        cout << "Error: No triangles loaded." << endl;
        return 1;
    }
    cout << "scene data read. There are " << mesh.size() << " Triangles." << endl;

    auto afterRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterRead = afterRead - start;
    std::cout << "Scene construction took: " << elapsed_seconds_afterRead.count() << " seconds" << std::endl << endl;

    
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Computing BVH (on CPU)
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    vertices.positions = points.data();
    vertices.normals   = normals.data();
    vertices.colors    = colors.data();
    vertices.uvs    = uvs.data();
    computeInfoForBVH(vertices, mesh, centroids, minboxes, maxboxes);

    cout << "BVH data computed" << endl;

    int failcount = 0;
    int backupCt = 0;
    int startNode = buildBVH(bvhvec, indvec, centroids, minboxes, maxboxes, 0, mesh.size(), maxLeafSize, failcount, backupCt);

    cout << "BVH built. Largest leaf is size: " << failcount << "." << " Backup was called "<< backupCt << " times." << endl;
    //printBVH(bvhvec, indvec);
    printBVHSummary(bvhvec);

    auto afterBVH = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterBVH = afterBVH - afterRead;
    std::cout << "BVH construction took: " << elapsed_seconds_afterBVH.count() << " seconds" << std::endl << endl;
    
    BVHnode* BVH;
    int* BVHindices;
    
    cudaMalloc(&scene, mesh.size() * sizeof(Triangle));
    cudaMalloc(&lights, lightsvec.size() * sizeof(Triangle));
    cudaMalloc(&BVH, bvhvec.size() * sizeof(BVHnode));
    cudaMalloc(&BVHindices, indvec.size() * sizeof(int));

    cudaMemcpy(scene, mesh.data(), mesh.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(lights, lightsvec.data(), lightsvec.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(BVH, bvhvec.data(), bvhvec.size() * sizeof(BVHnode), cudaMemcpyHostToDevice);
    cudaMemcpy(BVHindices, indvec.data(), indvec.size() * sizeof(int), cudaMemcpyHostToDevice);

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Launch GPU Code - goes to functions in deviceCode.cu
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    

    launch(maxDepth, mats_d, textures_d, BVH, BVHindices, verts, points.size(), scene, mesh.size(), lights, lightsvec.size(), sampleCount, true, w, h, out_colors);

    float4* host_colors = new float4[w * h];
    cudaMemcpy(host_colors, out_colors, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            
            if (host_colors[image.toIndex(i, j)].x < 0 || host_colors[image.toIndex(i, j)].y < 0 || host_colors[image.toIndex(i, j)].z < 0)
                cout << i << ", " << j << " Negative color written: <" << host_colors[image.toIndex(i, j)].x << ", " << host_colors[image.toIndex(i, j)].y << ", " 
                    << host_colors[image.toIndex(i, j)].z << ">"<< endl;
            /*
            if (host_colors[image.toIndex(i, j)].x > 1.0f || host_colors[image.toIndex(i, j)].y > 1.0f || host_colors[image.toIndex(i, j)].z > 1.0f)
                cout << i << ", " << j << " Big color written: <" << host_colors[image.toIndex(i, j)].x << ", " << host_colors[image.toIndex(i, j)].y << ", " 
                    << host_colors[image.toIndex(i, j)].z << ">"<< endl;*/
            image.setColor(i, j, host_colors[image.toIndex(i, j)]);
        }
        
    }
    
    // memory freeing
    cudaFree(out_colors);
    cudaFree(verts);
    cudaFree(scene);
    cudaFree(lights);
    cudaFree(BVH);
    cudaFree(BVHindices);
    cudaFree(mats_d);
    cudaFree(textures_d);
    delete[] host_colors;

    std::string filename = "render.bmp";
    image.saveImageBMP(filename);



    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds_render = end - afterBVH;
    std::cout << "Render took: " << elapsed_seconds_render.count() << " seconds" << std::endl << endl;


    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Total Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Total Elapsed time (ms): " << elapsed_ms.count() << " milliseconds" << std::endl;

    
}


void readObjSimple(string filename, vector<float4>& points, vector<float4>& normals, vector<float4>& colors, vector<float2>& uvs,vector<Triangle>& mesh, 
    vector<Triangle>& lights, float4 c, float4 e, int materialID)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open OBJ file\n";
        return;
    }
    int startIndex = points.size();
    int normalStartIndex = normals.size();
    int uvStartIndex = uvs.size();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line[0] == 's') continue; // skip comments

        std::istringstream iss(line);
        std::string prefix;
        
        iss >> prefix;
        

        if (prefix == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            float4 p = make_float4(x, y, z, 0.0f);
            points.push_back(p);
        }
        else if (prefix == "vt") 
        {
            double u, v;
            iss >> u >> v;

            float2 uv = f2(u,1.0f-v);
            uvs.push_back(uv);
        }
        else if (prefix == "vn") {
            double x, y, z;
            iss >> x >> y >> z;
            float4 n = make_float4(x, y, z, 0.0f);
            normals.push_back(n);
        }
        else if (prefix == "f") {
            vector<string> items;

            string vertinfo;
            vector<int> vertexIndices;
            vector<int> normalIndices;
            vector<int> uvIndices;

            bool hasUV = uvIndices.size() == vertexIndices.size();
            bool hasN  = normalIndices.size() == vertexIndices.size();
            while (iss >> vertinfo) 
            {
                istringstream vss(vertinfo);
                string idx;

                if (getline(vss, idx, '/'))
                {
                    if (!idx.empty())
                        vertexIndices.push_back(stoi(idx) - 1);
                }
                if (getline(vss, idx, '/'))
                {
                    if (!idx.empty())
                        uvIndices.push_back(stoi(idx) - 1);
                }
                if (getline(vss, idx, '/'))
                {
                    if (!idx.empty())
                        normalIndices.push_back(stoi(idx) - 1);
                }
            }
            int n = vertexIndices.size();
            // Triangulate the polygon as a fan from the first vertex
            for (int i = 1; i < n - 1; ++i) {
                int idx0 = vertexIndices[0] + startIndex;
                int idx1 = vertexIndices[i] + startIndex;
                int idx2 = vertexIndices[i + 1] + startIndex;

                /*
                int n_idx0 = normalIndices[0] + normalStartIndex;
                int n_idx1 = normalIndices[i] + normalStartIndex;
                int n_idx2 = normalIndices[i + 1] + normalStartIndex;

                int uv_idx0 = uvIndices[0] + uvStartIndex;
                int uv_idx1 = uvIndices[i] + uvStartIndex;
                int uv_idx2 = uvIndices[i + 1] + uvStartIndex;
                */

                int uv_idx0 = hasUV ? uvIndices[0] + uvStartIndex : -1;
                int uv_idx1 = hasUV ? uvIndices[i] + uvStartIndex : -1;
                int uv_idx2 = hasUV ? uvIndices[i + 1] + uvStartIndex : -1;

                int n_idx0  = hasN ? normalIndices[0] + normalStartIndex : -1;
                int n_idx1  = hasN ? normalIndices[i] + normalStartIndex : -1;
                int n_idx2  = hasN ? normalIndices[i + 1] + normalStartIndex : -1;

                Triangle tri(idx0, idx1, idx2, n_idx0, n_idx1, n_idx2, materialID, uv_idx0, uv_idx1, uv_idx2, e);
                mesh.push_back(tri);

                if (lengthSquared(e) > 0) {
                    lights.push_back(tri);
                }
            }
        }
    }

    file.close();
}

