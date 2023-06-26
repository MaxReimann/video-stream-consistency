// flow_io.cpp. Adapted from https://vision.middlebury.edu/flow/code/flow-code/flowIO.cpp
//
// read and write our simple .flo flow file format

// ".flo" file format used for optical flow evaluation
//
// Stores 2-band float image for horizontal (u) and vertical (v) flow components.
// Floats are stored in little-endian order.
// A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
//
//  bytes  contents
//
//  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
//          (just a sanity check that floats are represented correctly)
//  4-7     width as an integer
//  8-11    height as an integer
//  12-end  data (width*height*2*4 bytes total)
//          the float values for u and v, interleaved, in row order, i.e.,
//          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
//
#include "flowIO.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file

// read a flow file into 2-band image
void ReadFlowFile(std::vector<float>& flow, int& width, int& height, std::string filename)
{
    FILE* stream = fopen(filename.c_str(), "rb");
    if (stream == 0) {
        throw std::runtime_error((std::string("ReadFlowFile: could not open ") + filename).c_str());
    }

    //int width, height;
    float tag;

    if ((int) fread(&tag, sizeof(float), 1, stream) != 1 ||
        (int) fread(&width, sizeof(int), 1, stream) != 1 ||
        (int) fread(&height, sizeof(int), 1, stream) != 1) {
        throw std::runtime_error((std::string("ReadFlowFile: problem reading file ") + filename).c_str());
    }

    if (tag != TAG_FLOAT) { // simple test for correct endian-ness
        throw std::runtime_error((std::string("ReadFlowFile: wrong tag (possibly due to big-endian machine?) ") + filename).c_str());
    }

    // another sanity check to see that integers were read correctly (99999 should do the trick...)
    if (width < 1 || width > 99999) {
        throw std::runtime_error((std::string("ReadFlowFile: illegal width ") + filename).c_str());
    }

    if (height < 1 || height > 99999) {
        throw std::runtime_error((std::string("ReadFlowFile: illegal height ") + filename).c_str());
    }

    int nBands = 2;
    flow.resize((size_t) height * width * nBands);

    //printf("reading %d x %d x 2 = %d floats\n", width, height, width*height*2);
    int n = nBands * width;
    for (int y = 0; y < height; y++) {
        //float* ptr = &img.Pixel(0, y, 0);
        float* ptr = &flow[(size_t) y * width * 2];
        if ((int) fread(ptr, sizeof(float), n, stream) != n) {
            throw std::runtime_error((std::string("ReadFlowFile: file is too short ") + filename).c_str());
        }
    }

    if (fgetc(stream) != EOF) {
        throw std::runtime_error((std::string("ReadFlowFile: file is too long ") + filename).c_str());
    }

    fclose(stream);
}

// write a 2-band image into flow file 
void WriteFlowFile(const std::vector<float>& flow, int W, int H, std::string filename)
{
    int width = W, height = H, nBands = 2;

    FILE* stream = fopen(filename.c_str(), "wb");
    if (stream == 0) {
        throw std::runtime_error((std::string("ReadFlowFile: could not open ") + filename).c_str());
    }

    // write the header
    fprintf(stream, TAG_STRING);
    if ((int) fwrite(&width, sizeof(int), 1, stream) != 1 ||
        (int) fwrite(&height, sizeof(int), 1, stream) != 1) {
        throw std::runtime_error((std::string("ReadFlowFile: can't write header to ") + filename).c_str());
    }

    // write the rows
    int n = nBands * width;
    for (int y = 0; y < height; y++) {
        const float* ptr = &flow[(size_t) y * width * 2];
        if ((int) fwrite(ptr, sizeof(float), n, stream) != n) {
            throw std::runtime_error((std::string("ReadFlowFile: can't write data to ") + filename).c_str());
        }
    }

    fclose(stream);
}
