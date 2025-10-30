#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>


class Image 
{
public:
    Image(int w, int h);
    ~Image();

    void setColor(int x, int y, float4 c);
    float4 getColor(int x, int y);
    void saveImageBMP(std::string fileName);

    int toIndex(int x, int y);
    std::pair<int,int> fromIndex(int i);


private:
    const int width, height;
    std::vector<float4> pixels;
};
