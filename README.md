# Interactive Temporal Video Consistency #

### [Project Page](https://maxreimann.github.io/stream-consistency/) | [Paper](https://arxiv.org/abs/2301.00750)

Blind video consistency produces temporally consistent videos from per-frame processed/stylized inputs without knowledge of the applied processing methods. 

This repo implements a low-latency method for improving temporal consistency in **high-resolution videos** and **video streams** and offers interactive control over the flickering amount.

Official Implementation of:<br/> 
**"Interactive Control over Temporal Consistency while Stylizing Video Streams"** <br/> 
Sumit Shekhar*, Max Reimann*, Moritz Hilscher, Amir Semmo, Jürgen Döllner, Matthias Trapp<br/> 
*equal contribution<br/> 
in CGF Journal (EGSR Special Edition), 2023

## Installation
Requires CUDA (11.4+) and QT 5.11+ 

To install, simply run the cmake (preferrably in a directory "build"), CMake will automatically download a CUDA-enabled build of ONNXRuntime. Then build (e.g with make).

After building, two binaries are found in the build directory:
`FlowVideoConsistency` is the CLI-based application for processing videos to file, `FlowVideoConsistencyPlayer` the interactive GUI application for live-viewing.

## Usage options

Process Frame directories:
```
Usage: ./FlowVideoConsistency -c pwcnet-light originalFrames processedFrames stabilizedFrames

Options:
  -?, -h, --help                    Displays this help.
  -c, --compute <model>             Compute optical flow using <model>. One of
                                    ['pwcnet', 'pwcnet-light']
  -f, --flow-directory <directory>  Read flow from <directory>. Alternative to compute

Arguments:
  originalFrames                    Original frame directory
  processedFrames                   Processed frames directory
  stabilizedFrames                  Output directory for stabilized frames
```

Process input videos, write frames: 
```
Usage: ./FlowVideoConsistency -c pwcnet-light test/input.mp4  test/processed.mp4 test/output
```

Process input videos, directly encode output video: 
```
Usage: ./FlowVideoConsistency -c pwcnet-light ../videos/input.mp4 ../videos/processed.mp4 output.mp4
```

Start FlowVideoConsistencyPlayer GUI:
```
./FlowVideoConsistencyPlayer
```

To use the fast setting (i.e. downscale flow computation), an environment variable FLOWDOWNSCALE can be set.
I.e., to downscale by 2x (a recommended factor when processing full-HD), set `FLOWDOWNSCALE=2 ./FlowVideoConsistencyPlayer`.

Original pytorchs models and conversion code are found in model-conversion/.
Code tested under linux 20.04, should work under windows as well.

## Citation
```
@article{shekhar2023consistency,
  author    = {Shekhar, Sumit and Reimann, Max and Hilscher, Moritz and Semmo, Amir and Döllner, Jürgen and Trapp, Matthias},
  title     = {Interactive Control over Temporal Consistency while Stylizing Video Streams},
  journal   = {Computer Graphics Forum},
  year      = {2023},
}
```
