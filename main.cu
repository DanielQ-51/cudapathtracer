
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

void readObjSimple(string filename, vector<float4>& points, vector<float4>& normals, vector<float4>& colors, 
    vector<Triangle>& mesh, vector<Triangle>& lights, float4 c, float4 e, int materialID);

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
    const int numBuckets = 20;
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
    float cost;
    SAH(indices, centroids, AABBmins, AABBmaxes, start, end, axis, minBound, maxBound, splitPos, cost, backup);
    

    /*if (false)
    {
        nodes[nodeIndex].first = start;
        nodes[nodeIndex].primCount = primCount;
        nodes[nodeIndex].left = nodes[nodeIndex].right = -1;
        largestLeaf = max(primCount, largestLeaf);
        //cout << "force leaf Cost. Primcount: " << primCount << " cost:" << cost << endl;
        return nodeIndex;
    }*/
        

    //splitPos = (getFloat4Component(maxBound , axis) + getFloat4Component(minBound , axis))/2.0f;
    //cout << "FIRST failed split at " << splitPos << " on the " << axis << " numbered axis" << endl; 
    //int numLeft = 0;
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
    // image setup
    int w = 1000;
    int h = 1000;
    Image image = Image(w, h);

    int sampleCount = 1000;
    int maxLeafSize = 40;

    cout << "Rendering at " << w << " by " << h << " pixels, with " << 
        sampleCount << " samples per pixel, and a maximum leaf size of " <<
        maxLeafSize << endl << endl;

    float4* out_colors;
    cudaMalloc(&out_colors, w * h * sizeof(float4));

    Vertices vertices;
    vector<float4> points;
    vector<float4> normals;
    vector<float4> colors;
    vector<Triangle> mesh;
    vector<Triangle> lightsvec;
    vector<BVHnode> bvhvec;

    vector<float4> centroids;
    vector<float4> minboxes;
    vector<float4> maxboxes;

    vector<Material> mats;

    Material lambertBlue = Material::Diffuse(f4(0.4f,0.4f,0.8f));
    Material lambertWhite = Material::Diffuse(f4(0.9f,0.9f,0.9f));
    Material lambertGreen = Material::Diffuse(f4(0.2f,0.6f,0.6f));

    float4 eta_gold = f4(0.17f, 0.35f, 1.5f);  // real part of refractive index
    float4 k_gold   = f4(3.1f, 2.7f, 1.9f);   // imaginary part, absorption
    float roughness_polished = 0.1f;  

    Material ironPolished = Material::Metal(eta_gold, eta_gold, roughness_polished);

    mats.push_back(lambertBlue); // index 0
    mats.push_back(lambertWhite); // index 1
    mats.push_back(lambertGreen); // index 2
    mats.push_back(ironPolished); // index 3

    Material* mats_d;

    cudaMalloc(&mats_d, mats.size() * sizeof(Material));
    cudaMemcpy(mats_d, mats.data(), mats.size() * sizeof(Material), cudaMemcpyHostToDevice);
    
    
    readObjSimple("scenedata/smallbox.obj", points, normals, colors, mesh, lightsvec, f4(0.9f,0.9f,0.9f), f4(), 1);
    //cout << "scene data read. There are " << mesh.size() << " Triangles." << endl;
    readObjSimple("scenedata/leftwall.obj", points, normals, colors, mesh, lightsvec, f4(0.4f,0.4f,0.8f), f4(), 0);
    readObjSimple("scenedata/rightwall.obj", points, normals, colors, mesh, lightsvec, f4(0.2f,0.6f,0.6f), f4(), 2);
    readObjSimple("scenedata/tophalfmiku.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(0.9f,0.4f,0.4f), f4(), 3);
    //readObjSimple("scenedata/character.obj", vertices, mesh, lightsvec, 1.0f*f4(0.9f,0.9f,0.9f), f4(), 1);
    //readObjSimple("scenedata/smallbox.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(0.9f,0.9f,0.9f), f4(), 1);
    //readObjSimple("scenedata/swordbetter.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), f4(), 1);
    //readObjSimple("scenedata/leftlight.obj", vertices, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 20.0f*f4(10.0f,1.0f,1.0f), 1);
    //readObjSimple("scenedata/rightlight.obj", vertices, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 3.0f*f4(3.0f,3.0f,10.0f), 1);
    //readObjSimple("scenedata/reallysmalllight.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 30.0f*f4(7.0f,7.0f,3.0f), 1);
    readObjSimple("scenedata/lightforward.obj", points, normals, colors, mesh, lightsvec, 1.0f*f4(1.0f,1.0f,1.0f), 1.600f*f4(7.0f,7.0f,3.0f), 1);

        
    Vertices* verts;
    Triangle* scene;
    Triangle* lights;

    cudaMalloc(&verts,  sizeof(Vertices));
    Vertices temp;

    cudaMalloc(&temp.positions, sizeof(float4) * points.size());
    cudaMalloc(&temp.normals, sizeof(float4) * normals.size());
    cudaMalloc(&temp.colors,  sizeof(float4) * colors.size());

    cudaMemcpy(temp.positions, points.data(), points.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.normals, normals.data(), normals.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.colors, colors.data(), colors.size() * sizeof(float4), cudaMemcpyHostToDevice);
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

    vertices.positions = points.data();
    vertices.normals   = normals.data();
    vertices.colors    = colors.data();
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
    
    //cudaMalloc(&verts, vertices.size() * sizeof(Vertex));
    cudaMalloc(&scene, mesh.size() * sizeof(Triangle));
    cudaMalloc(&lights, lightsvec.size() * sizeof(Triangle));
    cudaMalloc(&BVH, bvhvec.size() * sizeof(BVHnode));
    cudaMalloc(&BVHindices, indvec.size() * sizeof(int));

    //cudaMemcpy(verts, vertices.data(), vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(scene, mesh.data(), mesh.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(lights, lightsvec.data(), lightsvec.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(BVH, bvhvec.data(), bvhvec.size() * sizeof(BVHnode), cudaMemcpyHostToDevice);
    cudaMemcpy(BVHindices, indvec.data(), indvec.size() * sizeof(int), cudaMemcpyHostToDevice);


    launch(6, mats_d ,BVH, BVHindices, verts, points.size(), scene, mesh.size(), lights, lightsvec.size(), sampleCount, true, w, h, out_colors);

    float4* host_colors = new float4[w * h];
    cudaMemcpy(host_colors, out_colors, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
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

// Simple obj reading. no textures. Meant for flat shaded meshes.
void readObjSimple(string filename, vector<float4>& points, vector<float4>& normals, vector<float4>& colors, vector<Triangle>& mesh, 
    vector<Triangle>& lights, float4 c, float4 e, int materialID)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open OBJ file\n";
        return;
    }
    int startIndex = points.size();
    int normalStartIndex = normals.size(); // <--- ADD THIS

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue; // skip comments

        std::istringstream iss(line);
        std::string prefix;
        
        iss >> prefix;
        

        if (prefix == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            float4 p = make_float4(x, y, z, 0.0f);
            points.push_back(p);
        }
        else if (prefix == "vt") {}
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

                }
                if (getline(vss, idx, '/'))
                {
                    if (!idx.empty())
                        normalIndices.push_back(stoi(idx) - 1);
                }
            }
            int n = vertexIndices.size();
            //int startIndex = points.size();

            // Push all vertices directly into the main vector
            //for (int i = 0; i < n; ++i) {
            //    vertices.push_back(Vertex(points[vertexIndices[i]], c, normals[normalIndices[i]]));
            //}

            // Triangulate the polygon as a fan from the first vertex
            for (int i = 1; i < n - 1; ++i) {
                int idx0 = vertexIndices[0] + startIndex;
                int idx1 = vertexIndices[i] + startIndex;
                int idx2 = vertexIndices[i + 1] + startIndex;

                int n_idx0 = normalIndices[0] + normalStartIndex;
                int n_idx1 = normalIndices[i] + normalStartIndex;
                int n_idx2 = normalIndices[i + 1] + normalStartIndex;

                Triangle tri(idx0, idx1, idx2, n_idx0, n_idx1, n_idx2, materialID, e);
                mesh.push_back(tri);

                if (lengthSquared(e) > 0) {
                    lights.push_back(tri);
                }
            }
        }
    }

    for (int i = startIndex; i < points.size(); i++)
    {
        colors.push_back(c);
    }

    file.close();
}

