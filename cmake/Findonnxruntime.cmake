# This will define the following variables:
#   onnxruntime_FOUND        -- True if the system has the onnxruntime library
#   onnxruntime_INCLUDE_DIRS -- The include directories for onnxruntime
#   onnxruntime_LIBRARIES    -- Libraries to link against
#   onnxruntime_CXX_FLAGS    -- Additional (required) compiler flags

include(FindPackageHandleStandardArgs)

get_filename_component(onnxruntime_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dependencies/onnxruntime" ABSOLUTE)

set(onnxruntime_INCLUDE_DIRS ${onnxruntime_INSTALL_PREFIX}/include)
set(onnxruntime_LIBRARIES onnxruntime)
set(onnxruntime_CXX_FLAGS "") # no flags needed


find_library(onnxruntime_LIBRARY onnxruntime
    PATHS "${onnxruntime_INSTALL_PREFIX}/lib"
)

add_library(onnxruntime SHARED IMPORTED)
set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION "${onnxruntime_LIBRARY}")
set_property(TARGET onnxruntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIRS}")
set_property(TARGET onnxruntime PROPERTY INTERFACE_COMPILE_OPTIONS "${onnxruntime_CXX_FLAGS}")
set_property(TARGET onnxruntime PROPERTY IMPORTED_IMPLIB "${onnxruntime_LIBRARY}")

find_package_handle_standard_args(onnxruntime DEFAULT_MSG onnxruntime_LIBRARY onnxruntime_INCLUDE_DIRS)

get_filename_component(onnxruntime_LIBRARY_DIR ${onnxruntime_LIBRARY} DIRECTORY)
file(GLOB_RECURSE onnxruntime_SHARED_LIBRARY_OBJECTS
    ${onnxruntime_LIBRARY_DIR}/*${CMAKE_SHARED_LIBRARY_SUFFIX}*)


message("Onnx libraries dlls found: " ${onnxruntime_SHARED_LIBRARY_OBJECTS})

set_property(TARGET onnxruntime PROPERTY SHARED_LIBRARY_FILES "${onnxruntime_SHARED_LIBRARY_OBJECTS}")