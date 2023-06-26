#include "gpuimage.h"

void perform_consistency(
        GPUImage& originalCurr,
        GPUImage& processedPrev,
        GPUImage& processedCurr,
        GPUImage& processedNext,
        GPUImage& flowFwd,
        GPUImage& flowBwd,
        GPUImage& stabilizedPrev,
        GPUImage& stabilizedOut);

void get_bilinear(
        GPUImage& input,
        GPUImage& output);

void get_warp_result(
        GPUImage& input,
        GPUImage& flow,
        GPUImage& inputWarp);


void get_adap_comb(
        GPUImage& crntIn,
        GPUImage& crntPr,
        GPUImage& prevWarpIn,
        GPUImage& prevWarpPr,
        GPUImage& nextWarpIn,
        GPUImage& nextWarpPr,
        GPUImage& adapCmbIn,
        GPUImage& adapCmbPr,
        GPUImage& lastStabWarp,
        float alpha
        );

void get_consist_wt(
        GPUImage& adapCmbIn,
        GPUImage& crntIn,
        GPUImage& consistWt,
        float  beta,
        float gamma);

void get_consist_out(
        GPUImage& crntPr,
        GPUImage& prevStabWarp,
        GPUImage& consWt,
        int numIter,
        float stepSize,
        float momFac,
        GPUImage& consisOut);
