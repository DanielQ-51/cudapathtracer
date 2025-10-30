/*
Handles the image writing.

Uses a 1d vector of pixels instead of 2d for minor optimization.

*/
#pragma once
#include <cuda_runtime.h>
#include "imageUtil.cuh"
#include "util.cuh"
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>

#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)


void createBMPHeaders(int width, int height, BMPFileHeader &fileHeader, BMPInfoHeader &infoHeader);

Image::Image(int w, int h) : width(w), height(h), pixels(std::vector<float4>(w * h)) {}

Image::~Image() {}

int Image::toIndex(int x, int y) {
    return y * width + x;
}

std::pair<int,int> Image::fromIndex(int i) {
    int y = i / width; // integer division
    int x = i % width; // remainder
    return {x, y};
}

void Image::setColor(int x, int y, float4 c) {
    pixels[toIndex(x, y)] = c;
}

float4 Image::getColor(int x, int y) {
    return pixels[toIndex(x, y)];
}

void Image::saveImageBMP(std::string fileName) {
    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;

    createBMPHeaders(width, height, fileHeader, infoHeader);

    std::ofstream out(fileName, std::ios::binary);
    out.write((char*)&fileHeader, sizeof(fileHeader));
    out.write((char*)&infoHeader, sizeof(infoHeader));

    int rowSize = (3 * width + 3) & (~3); // each row padded to multiple of 4 bytes
    //int diff = width - rowSize;

    float4 c;
    std::vector<unsigned char> row(rowSize);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            c = getColor(x, y);
            //c /= std::max(c[2], std::max(c[0], c[1]));

            //std::cout << c.b << " " << c.g << " " << c.r << std::endl; c[2]/max(c[2].x(),c[2].y(),c[2].z())

            row[x*3 + 0] = static_cast<unsigned char>(clamp(c.z, 0.0f, 1.0f) * 255.0f + 0.5f);
            row[x*3 + 1] = static_cast<unsigned char>(clamp(c.y, 0.0f, 1.0f) * 255.0f + 0.5f);
            row[x*3 + 2] = static_cast<unsigned char>(clamp(c.x, 0.0f, 1.0f) * 255.0f + 0.5f);

            //std::cout << static_cast<unsigned char>(c.b * 255.0f + 0.5f) << " " << static_cast<unsigned char>(c.g * 255.0f + 0.5f) << " " << static_cast<unsigned char>(c.r * 255.0f + 0.5f) << std::endl;
        }

        //for (int i = 0; i < diff*3; i++) {
        //    row[width*3 + i] = 0;
        //}
        out.write(reinterpret_cast<char*>(row.data()), rowSize);
    }

    out.close();
}

void createBMPHeaders(int width, int height, BMPFileHeader &fileHeader, BMPInfoHeader &infoHeader) {
    int rowSize = (3 * width + 3) & (~3);
    int imageSize = rowSize * height;

    // File header
    fileHeader.bfType = 0x4D42;
    fileHeader.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + imageSize;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    // Info header
    infoHeader.biSize = sizeof(BMPInfoHeader);
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = imageSize;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;
}