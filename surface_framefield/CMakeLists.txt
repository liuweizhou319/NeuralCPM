cmake_minimum_required(VERSION 3.18)
project(surfaceframefield_igl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(PARAMETERIZATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/parameterization_libigl")
set(LIBIGL_ROOT "${PARAMETERIZATION_DIR}/deps/libigl" CACHE PATH
    "Path to libigl with external/CoMISo and external/eigen")
set(POLYSCOPE_SOURCE_DIR "" CACHE PATH
    "Optional local Polyscope source tree (with submodules)")

if(NOT EXISTS "${LIBIGL_ROOT}/include/igl/copyleft/comiso/miq.h")
    message(FATAL_ERROR "libigl MIQ headers missing; set LIBIGL_ROOT to the bundled libigl tree")
endif()

# Use the vendored solver overrides, matching parameterization_libigl/xmake.lua.
add_library(comiso STATIC
    "${PARAMETERIZATION_DIR}/vendor/CoMISo/Solver/EigenLDLTSolver.cc"
    "${PARAMETERIZATION_DIR}/vendor/CoMISo/Solver/MISolver.cc"
    "${PARAMETERIZATION_DIR}/vendor/CoMISo/Solver/ConstrainedSolver.cc"
    "${PARAMETERIZATION_DIR}/vendor/CoMISo/Solver/IterativeSolverT.cc"
    "${PARAMETERIZATION_DIR}/vendor/CoMISo/Solver/GMM_Tools.cc"
)
target_include_directories(comiso PUBLIC
    "${PARAMETERIZATION_DIR}/vendor"
    "${LIBIGL_ROOT}/external/eigen"
    "${LIBIGL_ROOT}/external"
    "${LIBIGL_ROOT}/external/CoMISo/ext/gmm-4.2/include"
)
target_compile_definitions(comiso PUBLIC INCLUDE_TEMPLATES)
if(MSVC)
    target_compile_definitions(comiso PUBLIC
        NOMINMAX _USE_MATH_DEFINES _SCL_SECURE_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS)
endif()

# Polyscope's CMake target provides its viewer dependencies transitively.
if(POLYSCOPE_SOURCE_DIR)
    add_subdirectory("${POLYSCOPE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/polyscope")
else()
    include(FetchContent)
    FetchContent_Declare(polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG v2.5.0
        GIT_SHALLOW TRUE
        GIT_SUBMODULES_RECURSE TRUE
    )
    FetchContent_MakeAvailable(polyscope)
endif()

add_executable(surfaceframefield_igl "${PARAMETERIZATION_DIR}/src/main.cpp")
target_include_directories(surfaceframefield_igl PRIVATE "${LIBIGL_ROOT}/include")
target_compile_definitions(surfaceframefield_igl PRIVATE
    GLM_ENABLE_EXPERIMENTAL GLFW_INCLUDE_NONE)
target_link_libraries(surfaceframefield_igl PRIVATE comiso polyscope)

option(BUILD_QEX "Build libQEx and its quad extraction tools" ON)
if(BUILD_QEX)
    include(cmake/libQEx.cmake)
endif()
