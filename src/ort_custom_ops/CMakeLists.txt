
set(CustomOpLibraryTarget "CustomOps")
set(CustomOpLibraryTargetExport "${CustomOpLibraryTarget}-export")

set(CustomOpsSourceDir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CustomOpsIncludeDir ${CMAKE_CURRENT_SOURCE_DIR}/include/ort_custom_ops)

set(CustomOpticalFlowOpsSourceDir ${CustomOpsSourceDir}/opticalflow)
set(CustomOpticalFlowOpsIncludeDir ${CustomOpsIncludeDir}/opticalflow)

set(CustomOpsSources
    ${CustomOpsSourceDir}/custom_ops.cpp)

set(CustomOpsIncludes
    ${CustomOpsIncludeDir}/basekernel.h
    ${CustomOpsIncludeDir}/custom_ops.h)

set(CustomOpsOpticalFlowSources
    ${CustomOpticalFlowOpsSourceDir}/correlation.cc
    ${CustomOpticalFlowOpsSourceDir}/warp.cc)

set(CustomOpsOpticalFlowIncludes
    ${CustomOpticalFlowOpsIncludeDir}/correlation.h
    ${CustomOpticalFlowOpsIncludeDir}/warp.h)

if(BUILD_WITH_CUDA)
    list(APPEND CustomOpsOpticalFlowSources
        ${CustomOpticalFlowOpsSourceDir}/correlation_cuda.cc
        ${CustomOpticalFlowOpsSourceDir}/correlation_cuda.cu
        ${CustomOpticalFlowOpsSourceDir}/warp_cuda.cc
        ${CustomOpticalFlowOpsSourceDir}/warp_cuda.cu)

        set_source_files_properties(${CustomOpticalFlowOpsSourceDir}/correlation_cuda.cc PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(${CustomOpticalFlowOpsSourceDir}/correlation_cuda.cu PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(${CustomOpticalFlowOpsSourceDir}/warp_cuda.cc PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(${CustomOpticalFlowOpsSourceDir}/warp_cuda.cu PROPERTIES LANGUAGE CUDA)
endif()

source_group("opticalflow" FILES ${CustomOpsOpticalFlowSources} ${CustomOpsOpticalFlowIncludes})
source_group("" FILES ${CustomOpsSources} ${CustomOpsIncludes})
source_group("util" FILES ${UtilSources} ${UtilIncludes})

add_library(${CustomOpLibraryTarget} SHARED 
    ${CustomOpsIncludes} ${CustomOpsSources}
    ${CustomOpsOpticalFlowIncludes} ${CustomOpsOpticalFlowSources}
    ${UtilIncludes} ${UtilSources})
target_link_libraries(${CustomOpLibraryTarget} PRIVATE ${onnxruntime_LIBRARIES})
set_property(TARGET ${CustomOpLibraryTarget} PROPERTY CUDA_ARCHITECTURES OFF)

message("BUILDWITHCUDA  ${BUILD_WITH_CUDA}")
if(BUILD_WITH_CUDA)
    target_link_libraries(${CustomOpLibraryTarget} PRIVATE CUDA::cudart_static)
    set_target_properties(${CustomOpLibraryTarget} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(${CustomOpLibraryTarget} PRIVATE D_BUILD_WITH_CUDA)
endif()

set_target_properties(${CustomOpLibraryTarget} PROPERTIES OUTPUT_NAME ${ProjectName}-${CustomOpLibraryTarget})
target_include_directories(${CustomOpLibraryTarget}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${UtilDir}

    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

add_library(${ProjectNamespace}::${CustomOpLibraryTarget} ALIAS ${CustomOpLibraryTarget})

install(TARGETS ${CustomOpLibraryTarget}
    EXPORT  "${CustomOpLibraryTarget}-export"       COMPONENT dev
    RUNTIME DESTINATION ${BinaryInstallDirectory}   COMPONENT runtime
)

# CMake config
# install(EXPORT ${CustomOpLibraryTarget}-export
#     NAMESPACE   ${InstallNamespace}::
#     DESTINATION ${CMakeInstallDirectory}/${CustomOpLibraryTarget}
#     COMPONENT   dev
# )