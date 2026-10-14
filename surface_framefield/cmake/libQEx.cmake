# Upstream's legacy CMake uses CMAKE_SOURCE_DIR and forces C++98 on its CLI.
# Define its targets here so it can coexist with this C++17 project unchanged.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
set(QEX_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../libQEx")
if(NOT EXISTS "${QEX_SOURCE_DIR}/interfaces/c/qex.cc")
    message(FATAL_ERROR
        "libQEx is missing. Run: git submodule update --init --recursive")
endif()

find_package(OpenMesh QUIET CONFIG)
if(NOT TARGET OpenMeshCore OR NOT TARGET OpenMeshTools)
    include(FetchContent)
    set(BUILD_APPS OFF CACHE BOOL "Build OpenMesh applications")
    set(OPENMESH_BUILD_UNIT_TESTS OFF CACHE BOOL "Build OpenMesh tests")
    FetchContent_Declare(openmesh
        URL https://www.graphics.rwth-aachen.de/media/openmesh_static/Releases/9.0/OpenMesh-9.0.tar.gz
        URL_HASH SHA256=b9574c921482798ce75a8bf578345a84b928ca26ee759219d21b310e2db9d006
    )
    # OpenMesh static libraries are also linked into the shared QEx library.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    FetchContent_MakeAvailable(openmesh)
endif()

set(qex_sources
    "${QEX_SOURCE_DIR}/interfaces/c/qex.cc"
    "${QEX_SOURCE_DIR}/src/predicates.c"
    "${QEX_SOURCE_DIR}/src/MeshExtractor.cc"
)
add_library(QEx SHARED ${qex_sources})
add_library(QExStatic STATIC ${qex_sources})
foreach(qex_target IN ITEMS QEx QExStatic)
    target_include_directories(${qex_target} PUBLIC "${QEX_SOURCE_DIR}/interfaces/c")
    target_link_libraries(${qex_target} PUBLIC OpenMeshCore OpenMeshTools)
    # Exact geometric predicates require strict floating-point evaluation.
    if(MSVC)
        target_compile_options(${qex_target} PRIVATE /fp:strict)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${qex_target} PRIVATE -fno-fast-math -ffp-contract=off)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$")
            target_compile_options(${qex_target} PRIVATE -msse2 -mfpmath=sse)
        endif()
    endif()
endforeach()
target_compile_definitions(QEx PRIVATE QEX_EXPORT_SYMBOLS)
if(WIN32)
    # qex.h declares dllimport for consumers unless QEX_EXPORT_SYMBOLS is set.
    set_target_properties(QEx PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    target_compile_definitions(QExStatic PUBLIC QEX_EXPORT_SYMBOLS)
endif()

if(NOT WIN32)
    # Upstream's command-line example uses POSIX APIs.
    add_executable(cmdline_tool "${QEX_SOURCE_DIR}/demo/cmdline_tool/main.cpp")
    target_link_libraries(cmdline_tool PRIVATE QEx)
    add_executable(demo_minimal_c "${QEX_SOURCE_DIR}/demo/minimal_c/main.c")
    target_link_libraries(demo_minimal_c PRIVATE QEx)
endif()
