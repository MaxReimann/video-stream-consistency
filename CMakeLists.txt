cmake_minimum_required(VERSION 3.17)

cmake_policy(SET CMP0104 NEW)
project(FlowVideoConsistency LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-D__STDC_CONSTANT_MACROS")
set(INSTALL_ASSETS ON)


set(SYSTEM_DIR_INSTALL FALSE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install")
set(BinaryInstallDirectory "bin")
set(CMakeInstallDirectory "cmake")
set(LibraryInstallDirectory "lib")
set(IncludeInstallRootDirectory "include")
set(InstallRootDirectory ".")

find_package(CUDAToolkit REQUIRED)
set(BUILD_WITH_CUDA 1)


set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LibraryInstallDirectory}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# if (UNIX)
#     set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${NeuralModelProcessor_DIR}/${LibraryInstallDirectory}")
# endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(Qt5 COMPONENTS Core Widgets Multimedia MultimediaWidgets Concurrent)

function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(NOT "${old_flags}" STREQUAL "")
        string(REPLACE ";" "," CUDA_flags "${old_flags}")
        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
            )
    endif()
endfunction()


if (WIN32)
    set(ZIPNAME ffmpeg-n5.1-latest-win64-gpl-shared-5.1) 
    set(ZIP_LINK "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${ZIPNAME}.zip")
    set(ZIP_FILE "ffmpeg.zip")
    set(DEP_EXTRACTION_DIRECTORY "${CMAKE_BINARY_DIR}/dependencies")
    if(NOT EXISTS "${DEP_EXTRACTION_DIRECTORY}/ffmpeg")
        file(MAKE_DIRECTORY "${DEP_EXTRACTION_DIRECTORY}")
        file(DOWNLOAD "${ZIP_LINK}" "${DEP_EXTRACTION_DIRECTORY}/${ZIP_FILE}" SHOW_PROGRESS STATUS DOWNLOAD_STATUS )
        # Separate the returned status code, and error message.
        list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
        list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)
        if(${STATUS_CODE} EQUAL 0)
        message(STATUS "Download ffmpeg completed successfully!")
        else()
        message(FATAL_ERROR "Error occurred during download ffmpeg: ${ERROR_MESSAGE}")
        endif()
        # Extract the zip file
        execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xzf "${DEP_EXTRACTION_DIRECTORY}/${ZIP_FILE}"
                        WORKING_DIRECTORY "${DEP_EXTRACTION_DIRECTORY}")
        file(RENAME "${DEP_EXTRACTION_DIRECTORY}/${ZIPNAME}" "${DEP_EXTRACTION_DIRECTORY}/ffmpeg")
    endif()

    set(LIBAV_ROOT_DIR "${DEP_EXTRACTION_DIRECTORY}/ffmpeg")
    FIND_PACKAGE(LibAV REQUIRED)
else() 
    # can be obtained with sudo apt-get install libavcodec-dev libavcodec-extra libavutil-dev libavfilter-dev libavformat-dev libswscale-dev
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
        libavfilter
        libavformat
        libavcodec
        libswresample
        libswscale
        libavutil
    )
endif()


#set(ORT_VERSION "1.13.1")
set(ORT_VERSION "1.20.1")
if (WIN32)
set(ZIPNAME "onnxruntime-win-x64-gpu-${ORT_VERSION}") 
set(ZIPENDING "zip")
else()
set(ZIPNAME "onnxruntime-linux-x64-gpu-${ORT_VERSION}") 
set(ZIPENDING "tgz")
endif()
set(ZIP_LINK "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ZIPNAME}.${ZIPENDING}")
message("${ZIP_LINK}")
set(ZIP_FILE "onnxruntime.tgz")
set(DEP_EXTRACTION_DIRECTORY "${CMAKE_BINARY_DIR}/dependencies")
if(NOT EXISTS "${DEP_EXTRACTION_DIRECTORY}/onnxruntime")
    file(MAKE_DIRECTORY "${DEP_EXTRACTION_DIRECTORY}")
    file(DOWNLOAD "${ZIP_LINK}" "${DEP_EXTRACTION_DIRECTORY}/${ZIP_FILE}" SHOW_PROGRESS STATUS DOWNLOAD_STATUS )
    # Separate the returned status code, and error message.
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)
    if(${STATUS_CODE} EQUAL 0)
    message(STATUS "Download onnxruntime completed successfully!")
    else()
    message(FATAL_ERROR "Error occurred during download onnxruntime: ${ERROR_MESSAGE}")
    endif()
    # Extract the zip file
    execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xzf "${DEP_EXTRACTION_DIRECTORY}/${ZIP_FILE}"
                    WORKING_DIRECTORY "${DEP_EXTRACTION_DIRECTORY}")
    file(RENAME "${DEP_EXTRACTION_DIRECTORY}/${ZIPNAME}" "${DEP_EXTRACTION_DIRECTORY}/onnxruntime")
endif()

FIND_PACKAGE(onnxruntime REQUIRED)

add_subdirectory(src/ort_custom_ops)

set (sources_stabilization
    src/stabilization/gpuimage.h
    src/stabilization/gpuimage.cpp 
    src/stabilization/gpuimage.cu
    src/stabilization/gpuimagewrapper.h
    src/stabilization/flowIO.h
    src/stabilization/flowIO.cpp
    src/stabilization/flowmodel.h
    src/stabilization/flowmodel.cpp
    src/stabilization/imagehelpers.h
    src/stabilization/imagehelpers.cpp
    src/stabilization/stabilizefiles.h
    src/stabilization/stabilizefiles.cpp 
    src/stabilization/stabilizestream.h
    src/stabilization/stabilizestream.cpp
    src/stabilization/videostabilizer.h
    src/stabilization/videostabilizer.cpp
    src/stabilization/flowconsistency.cu
    src/stabilization/flowconsistency.cuh
)

set(sources_gui
    src/gui/main.cpp
    src/gui/videoplayer.cpp
    src/gui/videoplayer.h
    src/gui/videowidget.cpp
    src/gui/videowidget.h
    src/gui/videoopenwidget.cpp
    src/gui/videoopenwidget.h
    src/gui/videowidgetsurface.cpp
    src/gui/videowidgetsurface.h
    src/gui/hyperparameterwidget.h
    src/gui/hyperparameterwidget.cpp
    src/gui/waiting_spinner.h
    src/gui/waiting_spinner.cpp
)


set(sources_decoding
    src/decoding/demuxer.cpp
    src/decoding/ffmpeg.cpp
    src/decoding/format_converter.cpp
    src/decoding/queue.h
    src/decoding/string_utils.cpp
    src/decoding/timer.cpp
    src/decoding/twostream_decoder.cpp
    src/decoding/video_decoder.cpp
    src/decoding/video_filterer.cpp
    src/decoding/video_control.h
    src/decoding/twostream_decoder.h
    src/decoding/video_decoder.h
    src/decoding/sorted_flat_deque.h
    src/decoding/video_filterer.h
    src/decoding/format_converter.h
    src/decoding/string_utils.h
    src/decoding/timer.h
    src/decoding/circular_buffer.h
    src/decoding/demuxer.h
    src/decoding/ffmpeg.h
    src/decoding/encoder.h
    src/decoding/encoder.cpp    
)


set(sources_inference
    src/inference/CpuIO.cpp
    src/inference/CpuIO.h
    src/inference/CudaIO.cpp
    src/inference/CudaIO.h
    src/inference/InferenceModelVariant.cpp
    src/inference/InferenceModelVariant.h
    src/inference/IOInterface.h
    src/inference/MemoryType.h
    src/inference/OrtContext.cpp
    src/inference/OrtContext.h
    src/inference/ImageArrayIOHelper.cpp
    src/inference/ImageArrayIOHelper.h
)


set_source_files_properties(src/stabilization/gpuimage.h PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/stabilization/gpuimage.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/stabilization/gpuimagewrapper.h PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/inference/CudaIO.cpp PROPERTIES LANGUAGE CUDA)
set(SOURCE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)


add_executable(FlowVideoConsistency ${sources_stabilization} ${sources_decoding} ${sources_inference} src/stabilization/main.cpp)
set_property(TARGET FlowVideoConsistency PROPERTY CUDA_ARCHITECTURES OFF)
target_link_libraries(FlowVideoConsistency PRIVATE Qt5::Core Qt5::Widgets  ${CUDA_LIBRARIES})
if (WIN32)
    target_include_directories(FlowVideoConsistency PRIVATE ${LIBAV_INCLUDE_DIRS})
    target_link_libraries(FlowVideoConsistency PRIVATE ${LIBAV_LIBRARIES})
else()
    target_include_directories(FlowVideoConsistency PRIVATE ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
    target_link_libraries(FlowVideoConsistency PRIVATE PkgConfig::LIBAV)
endif()
target_include_directories(FlowVideoConsistency PRIVATE ${onnxruntime_INCLUDE_DIRS} ${SOURCE_ROOT_DIR})
# target_link_libraries(FlowVideoConsistency PRIVATE nmp::Core)
target_link_libraries(FlowVideoConsistency PRIVATE ${onnxruntime_LIBRARIES} ${ProjectNamespace}::CustomOps)
target_compile_definitions(FlowVideoConsistency PRIVATE "D_NMP_ASSET_DIR=R\"(models)\"")


add_executable(FlowVideoConsistencyPlayer ${sources_stabilization} ${sources_decoding} ${sources_gui} ${sources_inference})
set_property(TARGET FlowVideoConsistencyPlayer PROPERTY CUDA_ARCHITECTURES OFF)
target_link_libraries(FlowVideoConsistencyPlayer PRIVATE Qt5::Core Qt5::Widgets Qt5::Concurrent Qt5::Multimedia ${CUDA_LIBRARIES})
if (WIN32)
target_include_directories(FlowVideoConsistencyPlayer PRIVATE ${LIBAV_INCLUDE_DIRS})
target_link_libraries(FlowVideoConsistencyPlayer PRIVATE ${LIBAV_LIBRARIES})
else()
target_include_directories(FlowVideoConsistencyPlayer PRIVATE ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
    target_link_libraries(FlowVideoConsistencyPlayer PRIVATE PkgConfig::LIBAV)
endif()
    # target_link_libraries(FlowVideoConsistencyPlayer PRIVATE nmp::Core)
target_include_directories(FlowVideoConsistencyPlayer PRIVATE ${onnxruntime_INCLUDE_DIRS}  ${SOURCE_ROOT_DIR})
target_link_libraries(FlowVideoConsistencyPlayer PRIVATE ${onnxruntime_LIBRARIES} ${ProjectNamespace}::CustomOps)
target_compile_definitions(FlowVideoConsistencyPlayer PRIVATE "D_NMP_ASSET_DIR=R\"(models)\"")

if (MSVC)
    # Disable warnings about nmp using the stl in its interface
    # This is not a problem because we always use the same implementation (compiler) for nmp and this project
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4251")
endif()

message(${CMAKE_SOURCE_DIR}/models)
ADD_CUSTOM_TARGET(symlink_assets ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/models "models")

if (WIN32)
    file(GLOB_RECURSE LIBAV_DLLS ${DEP_EXTRACTION_DIRECTORY}/ffmpeg/bin/*${CMAKE_SHARED_LIBRARY_SUFFIX})
    message("${CMAKE_COMMAND} -E copy_if_different ${LIBAV_DLLS} $<TARGET_FILE_DIR:FlowVideoConsistency>")
    add_custom_command(TARGET FlowVideoConsistency POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBAV_DLLS} $<TARGET_FILE_DIR:FlowVideoConsistency>)
endif()

install(TARGETS FlowVideoConsistency RUNTIME DESTINATION ${BinaryInstallDirectory} COMPONENT Runtime)
install(TARGETS FlowVideoConsistencyPlayer RUNTIME DESTINATION ${BinaryInstallDirectory} COMPONENT Runtime)
# install_target_common(FlowVideoConsistency)
if (MSVC)
    install(FILES $<TARGET_PROPERTY:onnxruntime,SHARED_LIBRARY_FILES> DESTINATION ${BinaryInstallDirectory})
    install(FILES ${LIBAV_DLLS} DESTINATION ${BinaryInstallDirectory})
    
    # Include Qt deployment helper function
    include(windeployqt)
    # Special package to find runtime libraries directory of qt
    windeployqtinstall(FlowVideoConsistencyPlayer)
else()
    install(FILES $<TARGET_PROPERTY:onnxruntime,SHARED_LIBRARY_FILES> DESTINATION ${LibraryInstallDirectory})
endif()


# get all files from model dir
file(GLOB_RECURSE INSTALL_MODEL_NAMES RELATIVE ${CMAKE_SOURCE_DIR}/models ${CMAKE_SOURCE_DIR}/models/*)
foreach(model IN LISTS INSTALL_MODEL_NAMES)
    install(FILES ${CMAKE_SOURCE_DIR}/models/${model} DESTINATION ${BinaryInstallDirectory}/models COMPONENT dev)
endforeach()