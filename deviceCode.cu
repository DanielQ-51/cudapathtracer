
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
__device__ void triangleIntersect(Vertices* verts, Triangle* tri, const Ray& r, float4& barycentric, float& tval)
{
    float4 tria = verts->positions[tri->aInd];
    float4 trib = verts->positions[tri->bInd];
    float4 tric = verts->positions[tri->cInd];
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

__device__ void BVHSceneIntersect(const Ray& r, BVHnode* BVH, int* BVHindices, Vertices* verts, Triangle* scene, Intersection& intersect, float max_t = 999999.0f, bool shortCircuit = false, int skipTri = -1)
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
                triangleIntersect(verts, tri, r, barycentric, t);

                if (shortCircuit && ((t > EPSILON) && (t < min_t) && (t < max_t)))
                {
                    intersect.valid = true;
                    return;
                }
                else if ((t > EPSILON) && (t < min_t) && (t < max_t))
                {
                    min_t = t; // Update the closest-hit distance
                    intersect.point = r.at(t);
                    intersect.color = verts->colors[tri->aInd] * barycentric.z + 
                                        verts->colors[tri->bInd] * barycentric.x + 
                                        verts->colors[tri->cInd] * barycentric.y;
                    intersect.normal = normalize(verts->normals[tri->naInd] * barycentric.z + 
                                        verts->normals[tri->nbInd] * barycentric.x + 
                                        verts->normals[tri->ncInd] * barycentric.y);
                    if (dot(intersect.normal, r.direction) > 0.0f) 
                    {
                        intersect.normal = -intersect.normal;
                        intersect.backface = true;
                    }
                    else intersect.backface = false;
                        
                    intersect.materialID = tri->materialID;
                    intersect.emission = tri->emission;
                    intersect.valid = true;
                    intersect.tri = *tri;

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

__device__ void sceneIntersection(const Ray& r, Vertices* verts, Triangle* scene, int triNum, 
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
            intersect.color = verts->colors[tri->aInd] * barycentric.z + 
                                        verts->colors[tri->bInd] * barycentric.x + 
                                        verts->colors[tri->cInd] * barycentric.y;
            intersect.normal = normalize(verts->normals[tri->naInd] * barycentric.z + 
                                        verts->normals[tri->nbInd] * barycentric.x + 
                                        verts->normals[tri->ncInd] * barycentric.y);
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


__device__ void nextEventEstimation(curandState& localState, Material* materials, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum,
    Triangle* scene, int triNum, Triangle* lights, int lightNum, int materialID, const Intersection& intersect, const float4& wo, 
    float& light_pdf, float4& contribution, float4& surfaceToLight_local, float etaI, float etaT, bool reflect, bool TIR, Triangle* light = nullptr, const Intersection* newIntersect = nullptr)
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
        apos = vertices->positions[l.aInd];
        bpos = vertices->positions[l.bInd];
        cpos = vertices->positions[l.cInd];
        u = sqrtf(curand_uniform(&localState));
        v = curand_uniform(&localState);

        p = (1.0f - u) * apos + u * (1.0f - v) * bpos + u * v * cpos; // point on light
        n = intersect.normal;
    }
    else 
    {
        l = *light;
        apos = vertices->positions[l.aInd];
        bpos = vertices->positions[l.bInd];
        cpos = vertices->positions[l.cInd];
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
        float4 lightNormal = vertices->normals[l.naInd];

        float cosThetaLight = dot(lightNormal, -wi);
        float cosThetaSurface = dot(n, wi);

        float G = cosThetaLight * cosThetaSurface/distanceSQR;
        float area = 0.5f * length(cross3(bpos - apos, cpos - apos));
        
        light_pdf = distanceSQR / (cosThetaLight * lightNum * area);
        float4 Le = l.emission;
        float4 f_val;
        float4 wi_local;
        toLocal(wi, intersect.normal, wi_local);
        surfaceToLight_local = wi_local;

        // wo is the incoming direction (passed to this function)
        // wi_local is the computed outgoing direction to the light
        f_eval(localState, materials, materialID, wo, wi_local, etaI, etaT, f_val, reflect, TIR);

        contribution = f_val * Le * cosThetaSurface / light_pdf;
    }
    else {}
}

__device__ void removeMaterialFromStack(int* stack, int* stackTop, int materialID)
{
    // 1. Find the index of the material to remove.
    //    We search backwards from the top (*stackTop - 1) down to index 1.
    //    We NEVER check index 0, which is our permanent 'AIR' medium.
    int i_found = -1;
    for (int i = (*stackTop) - 1; i > 0; i--)
    {
        if (stack[i] == materialID)
        {
            i_found = i;
            break;
        }
    }

    // 2. If we found the material...
    if (i_found != -1)
    {
        // 3. Shift all elements above it down by one to "fill the gap".
        //    We start at the found index and overwrite it with the next element,
        //    and continue this until we've moved the last element down.
        for (int i = i_found; i < (*stackTop) - 1; i++)
        {
            stack[i] = stack[i + 1];
        }

        // 4. Decrement the stack top to reflect the removal.
        (*stackTop)--;
    }
    
    // If i_found == -1, the material wasn't in the stack (or was only at index 0).
    // In a correct simulation, this shouldn't happen when exiting, but this
    // function is robust against it and will simply do nothing.
}

__global__ void Li (curandState* rngStates, Material* materials, BVHnode* BVH, int* BVHindices, int maxDepth, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
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

        float pitch = -15.0f * (3.14159265f / 180.0f);

        //float4 cameraOrigin = f4(-0.10f,0.25f,-0.0f);
        float4 cameraOrigin = f4(-0.07f,0.232f,0.8f);
        float4 a = f4(cameraOrigin.x + (x + 2.0f*curand_uniform(&localState) - 1.0f - w/2.0f) * (1.0f / w), 
                        cameraOrigin.y + (y + 2.0f*curand_uniform(&localState) - 1.0f - h/2.0f) * (1.0f / h),
                        cameraOrigin.z-2.0f);
        r.origin = cameraOrigin;
        r.direction = rotateX(a-cameraOrigin, pitch);

        r.direction = normalize(r.direction);


        float pdf = EPSILON;
        float etaI = EPSILON;
        float etaT = EPSILON;

        bool hitFirstnonSpecular = false;

        for (int depth = 0; depth < 100; depth++)
        {   
            
            Intersection intersect = Intersection();
            intersect.valid = false;
            //sceneIntersection(r, vertices, scene, triNum, intersect);
            //BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect, 999999.0f, false, -1);
            BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect, 999999.0f, false, previousintersectANY.triIDX);

            if (!intersect.valid) 
            {
                Li += beta * f4(0.4f,0.6f,0.6f);
                break;
            }
            
            int materialID = intersect.materialID;

            float4 old_wi_local = wi_local;

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
            
            if (materials[materialID].transmissive)
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
                        float4 nee;
                        float light_pdf = EPSILON;
                        float4 dummy;

                        // to calculate the light sampling pdf (a little inefficient)
                        nextEventEstimation(localState, materials, BVH, BVHindices, vertices, vertNum, scene, 
                            triNum, lights, lightNum, materialID, previousintersectREAL, old_wi_local, light_pdf, 
                            nee, dummy, -1.0f, -1.0f, false, false, &intersect.tri, &intersect); // wi_local is used from the previous bounce
                        
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
                    nextEventEstimation(localState, materials, BVH, BVHindices, vertices, vertNum, scene, 
                    triNum, lights, lightNum, materialID, intersect, wi_local, light_pdf, nee, wo_local, -1.0f, -1.0f, false, false);
                    
                    
                    if (light_pdf > EPSILON)
                    {
                        // to calculate the bsdf pdf
                        pdf_eval(materials, materialID, wi_local, wo_local, -1.0f, -1.0f, pdf, false, false); // stores the bsdf pdf val in pdf
                        float neeWeight = light_pdf * light_pdf / (pdf * pdf + light_pdf * light_pdf);

                        Li += beta * nee * neeWeight;
                    }
                    
                }
                float4 f_val = f4();
                sample_f_eval(localState, materials, materialID, wi_local, etaI, etaT, intersect.backface, wo_local, f_val, pdf);

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
                

                beta *= (f_val * fabs(wo_local.z) / pdf);
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
        if (lengthSquared(Li) >= EPSILON)
            colorSum += Li;
    }
    colors[pixelIdx] = colorSum/numSample;
    rngStates[pixelIdx] = localState;
}

__host__ void launch(int maxDepth, Material* materials, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors)
{
    dim3 blockSize(16, 16);  
    dim3 gridSize((w+15)/16, (h+15)/16);
    curandState* d_rngStates;
    cudaMalloc(&d_rngStates, w * h * sizeof(curandState));

    unsigned long seed = 103033UL;
    initRNG<<<gridSize, blockSize>>>(d_rngStates, w, h, seed);
    cudaDeviceSynchronize();

    Li<<<gridSize, blockSize>>>(d_rngStates, materials, BVH, BVHindices, maxDepth, vertices, vertNum, scene, triNum, 
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