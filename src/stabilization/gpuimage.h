#ifndef GPUIMAGE_H
#define GPUIMAGE_H


#include <vector>
#include <QImage>

class GPUImage
{
public:
    GPUImage(int width, int height, int channels);
    GPUImage(const GPUImage& other);
    ~GPUImage();

    inline float& value(int x, int y, int c) {
        return data[(y*width + x) * channels + c];
    }
    inline const float& value(int x, int y, int c) const {
        return data[(y*width + x) * channels + c];
    }

    void copyFrom(const GPUImage& other);
    void copyFrom(const std::vector<float>& vec);
    void copyFrom(const std::vector<std::byte>& vec);
    void copyFromCudaBuffer(const void* resourcePointer, size_t byteSize);
    void copyFromQImage(const QImage& image);
    void copyToQImage(QImage& image) const;

    float* data;
    int width, height, channels;
};

#endif // GPUIMAGE_H
