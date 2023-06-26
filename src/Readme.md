# Flow Video Consistency #

## Build Instructions ##

### Windows ###

* Using Qt maintenance tool, install: Qt 5.12.10 with MSVC 2017 64-bit, CMake >= 3.19
* Install CUDA toolkit, 10.x preferrably
* Open CMakeLists.txt as project with Qt creator and select Qt kit with MSVC as compiler

### Linux ###

* Install CUDA/Qt for your distro, use CMake/make

## Run Instructions ##

* `stabilize.exe` takes as arguments directories of original frames, processed frames, optical flow and stabilized output frames
* Frames must be of format `123456.png` and start with index 1 (as ffmpeg outputs frames)
