
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

    if (fabsf(a) < EPSILON)
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

                    intersect.uv = verts->uvs[tri->uvaInd] * barycentric.z + 
                        verts->uvs[tri->uvbInd] * barycentric.x + 
                        verts->uvs[tri->uvcInd] * barycentric.y;
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

__device__ void BVHShadowRay(const Ray& r, BVHnode* BVH, int* BVHindices, Vertices* verts, Triangle* scene, Material* materials, float4& throughputScale, float max_t, int skip_tri)
{
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
                Triangle* tri = &scene[idx];
                float4 barycentric;
                float t;
                triangleIntersect(verts, tri, r, barycentric, t);

                if (idx == skip_tri)
                    continue;

                if ((t > EPSILON) && (t < max_t))
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

__device__ void neePDF(Vertices* vertices, Triangle* scene, int lightNum, Triangle* light, const Intersection& intersect, 
    float& light_pdf, float etaI, float etaT, const Intersection* newIntersect)
{
    Triangle l = *light;
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
        f_eval(localState, materials, intersect.materialID, textures, wo, wi_local, etaI, etaT, f_val, intersect.uv);

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

__global__ void Li_unidirectional (curandState* rngStates, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, int maxDepth, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
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

        float xRot = 15.0f * (3.14159265f / 180.0f);
        float yRot = 25.0f * (3.14159265f / 180.0f);

        float aperture = 0.010f;      // lens radius, controls blur
        float focalDist = 2.15f;      // distance at which things are perfectly in focus

        float jitterX = 2.0f * curand_uniform(&localState) - 1.0f;
        float jitterY = 2.0f * curand_uniform(&localState) - 1.0f;
        float nx = (x + jitterX - w * 0.5f) / w;
        float ny = (y + jitterY - h * 0.5f) / h;

        //float4 cameraOrigin = f4(-0.8f, -0.05f, 2.4f);
        float4 cameraOrigin = f4(-0.5f, -0.55f, 2.4f);
        //float4 cameraOrigin = f4(-0.8f, 3.5f, 3.0f);
        float aspect = (float)w / (float)h;

        // Compute point on image plane (as before)
        float4 pixelPos = f4(cameraOrigin.x + nx * aspect,
                            cameraOrigin.y + ny,
                            cameraOrigin.z - 1.5f);

        // Compute initial ray from pinhole (for focal point calculation)
        float4 dir = pixelPos - cameraOrigin;
        dir = rotateX(dir, xRot);
        dir = rotateY(dir, yRot);
        float4 rayDir = normalize(dir);

        // Focal point along ray
        float4 focalPoint = cameraOrigin + rayDir * focalDist;

        // --- Sample point on lens ---
        float rad = sqrt(curand_uniform(&localState)) * aperture;
        float theta = 2.0f * 3.14159265f * curand_uniform(&localState);
        float lensX = rad * cosf(theta);
        float lensY = rad * sinf(theta);

        // Build camera basis vectors (assuming Z-forward, Y-up)
        float4 right = f4(1.0f, 0.0f, 0.0f);  // camera right
        float4 up = f4(0.0f, 1.0f, 0.0f);     // camera up

        // New ray origin on lens
        r.origin = cameraOrigin + lensX * right + lensY * up;

        // New ray direction through focal point
        r.direction = normalize(focalPoint - r.origin);



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
                float4 bg = f4(0.95f, 0.95f, 1.0f);
                Li += beta * bg;
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
                        
                        neePDF(vertices, scene, lightNum, &intersect.tri, previousintersectREAL, light_pdf, etaI, etaT, &intersect);
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
        Li = f4(fmaxf(Li.x, 0.0f), fmaxf(Li.y, 0.0f), fmaxf(Li.z, 0.0f));
        colorSum += Li;
    }
    colors[pixelIdx] = colorSum/numSample;
    rngStates[pixelIdx] = localState;
}

__host__ void launch_unidirectional(int maxDepth, Material* materials, float4* textures, BVHnode* BVH, int* BVHindices, Vertices* vertices, int vertNum, Triangle* scene, int triNum, 
    Triangle* lights, int lightNum, int numSample, bool useMIS, int w, int h, float4* colors)
{
    dim3 blockSize(16, 16);  
    dim3 gridSize((w+15)/16, (h+15)/16);
    curandState* d_rngStates;
    cudaMalloc(&d_rngStates, w * h * sizeof(curandState));

    unsigned long seed = 103033UL;
    initRNG<<<gridSize, blockSize>>>(d_rngStates, w, h, seed);
    cudaDeviceSynchronize();

    Li_unidirectional<<<gridSize, blockSize>>>(d_rngStates, materials, textures, BVH, BVHindices, maxDepth, vertices, vertNum, scene, triNum, 
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