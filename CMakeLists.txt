cmake_minimum_required(VERSION 3.21)

project("rwkv.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#
# Option list
#

# General
option(RWKV_BUILD_SHARED_LIBRARY   "rwkv: build as a shared library"                      ON)
option(RWKV_STATIC                 "rwkv: static link libraries"                          OFF)
option(RWKV_NATIVE                 "rwkv: enable -march=native flag"                      OFF)
option(RWKV_LTO                    "rwkv: enable link time optimization"                  OFF)

# Debug
option(RWKV_ALL_WARNINGS           "rwkv: enable all compiler warnings"                   ON)
option(RWKV_GPROF                  "rwkv: enable gprof"                                   OFF)

# Sanitizers
option(RWKV_SANITIZE_THREAD        "rwkv: enable thread sanitizer"                        OFF)
option(RWKV_SANITIZE_ADDRESS       "rwkv: enable address sanitizer"                       OFF)
option(RWKV_SANITIZE_UNDEFINED     "rwkv: enable undefined sanitizer"                     OFF)

# Instruction set specific
option(RWKV_AVX                    "rwkv: enable AVX"                                     ON)
option(RWKV_AVX2                   "rwkv: enable AVX2"                                    ON)
option(RWKV_AVX512                 "rwkv: enable AVX512"                                  OFF)
option(RWKV_FMA                    "rwkv: enable FMA"                                     ON)

# 3rd party libs
option(RWKV_ACCELERATE             "rwkv: enable Accelerate framework"                    ON)
option(RWKV_OPENBLAS               "rwkv: use OpenBLAS"                                   OFF)
option(RWKV_CUBLAS                 "rwkv: use cuBLAS"                                     OFF)
option(RWKV_CLBLAST                "rwkv: use CLBlast"                                    OFF)
option(RWKV_HIPBLAS                "rwkv: use hipBLAS"                                    OFF)
option(RWKV_METAL                  "rwkv: use Metal"                                      OFF)

# Build only shared library without building tests and extras
option(RWKV_STANDALONE             "rwkv: build only RWKV library"                        OFF)


# transition helpers (from llama.cpp)
function (rwkv_option_depr TYPE OLD NEW)
    if (${OLD})
        message(${TYPE} "${OLD} is deprecated and will be removed in the future.\nUse ${NEW} instead\n")
        set(${NEW} ON PARENT_SCOPE)
    endif()
endfunction()

set(GGML_ACCELERATE             ${RWKV_ACCELERATE})
set(GGML_CUDA                   ${RWKV_CUBLAS})
set(GGML_HIP                    ${RWKV_HIPBLAS})
set(GGML_METAL                  ${RWKV_METAL})
if (RWKV_OPENBLAS)
    set(GGML_BLAS_VENDOR "OpenBLAS")
    set(GGML_BLAS ON)
endif()

set(GGML_AVX                    ${RWKV_AVX})
set(GGML_AVX2                   ${RWKV_AVX2})
set(GGML_AVX512                 ${RWKV_AVX512})
set(GGML_FMA                    ${RWKV_FMA})

#
# Compile flags
#

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT MSVC)
    set(GGML_SANITIZE_THREAD ${RWKV_SANITIZE_THREAD})
    set(GGML_SANITIZE_ADDRESS ${RWKV_SANITIZE_ADDRESS})
    set(GGML_SANITIZE_UNDEFINED ${RWKV_SANITIZE_UNDEFINED})
endif()

if (RWKV_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wdouble-promotion
            -Wshadow
            -Wstrict-prototypes
            -Wpointer-arith
            -Wno-unused-function
            -Wno-ignored-attributes
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
            -Wno-multichar
            -Wno-nonnull
            -Wno-ignored-attributes
        )
    else()
        set(c_flags
            -W4
        )
        set(cxx_flags
            -W4
        )
    endif()

    add_compile_options(
            "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (RWKV_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
    set(GGML_LTO ON)
endif()

# Architecture specific
# TODO [llama.cpp]: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (RWKV_STATIC)
        set(GGML_STATIC ON)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (RWKV_GPROF)
        add_compile_options(-pg)
    endif()
    if (RWKV_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

#
# Build libraries
#

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (NOT RWKV_STANDALONE)
    set(GGML_STANDALONE OFF)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(extras)
elseif()
    set(GGML_STANDALONE ON)
endif()

if (NOT TARGET ggml)
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

if (RWKV_BUILD_SHARED_LIBRARY)
    add_library(rwkv SHARED rwkv.cpp rwkv.h)
else()
    add_library(rwkv rwkv.cpp rwkv.h)
endif()

if (GGML_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    endif()
endif()

if (GGML_CUDA)
    find_package(CUDAToolkit)

    if (CUDAToolkit_FOUND)
        add_compile_definitions(GGML_USE_CUDA)
        if (GGML_STATIC)
            if (WIN32)
                # As of 12.3.1 CUDA Toolkit for Windows does not offer a static cublas library
                set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas CUDA::cublasLt)
            else ()
                set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            endif()
        else()
            set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

        if (GGML_CUDA_NO_VMM)
            # No VMM requested, no need to link directly with the cuda driver lib (libcuda.so)
        else()
            set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} CUDA::cuda_driver) # required by cuDeviceGetAttribute(), cuMemGetAllocationGranularity(...), ...
        endif()
    endif()
endif()

if (APPLE AND GGML_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    endif()
endif()

if (GGML_METAL)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)
    add_compile_definitions(GGML_USE_METAL)

    set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS}
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        )
endif()

if (GGML_HIP)
    # CMake on Windows doesn't support the HIP language yet
    if (WIN32)
        set(CXX_IS_HIPCC TRUE)
    else()
        string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
    endif()

    find_package(hip     REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(rocblas REQUIRED)

    list(APPEND GGML_CDEF_PUBLIC GGML_USE_CUDA)

    add_compile_definitions(GGML_USE_HIPBLAS)

    if (CXX_IS_HIPCC)
        set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} hip::device)
    endif()

    if (GGML_STATIC)
        message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
    endif()

    set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} PUBLIC hip::host roc::rocblas roc::hipblas)
endif()

target_include_directories(rwkv PUBLIC .)
target_include_directories(rwkv PRIVATE ggml/include ggml/src)
target_compile_features(rwkv PUBLIC cxx_std_11)

if (GGML_METAL)
    set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} $<TARGET_OBJECTS:ggml-metal> $<TARGET_OBJECTS:ggml-blas>)
endif()
if (GGML_CUDA)
    set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} $<TARGET_OBJECTS:ggml-cuda>)
endif()
if (GGML_HIP)
    set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} $<TARGET_OBJECTS:ggml-hip>)
endif()
if (GGML_RPC)
    set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} $<TARGET_OBJECTS:ggml-rpc>)
endif()

target_link_libraries(rwkv PRIVATE $<TARGET_OBJECTS:ggml> $<TARGET_OBJECTS:ggml-base> $<TARGET_OBJECTS:ggml-cpu> ${RWKV_EXTRA_LIBS})

if (RWKV_BUILD_SHARED_LIBRARY)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(ggml-base PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(ggml-cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (GGML_METAL)
        set_target_properties(ggml-metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(ggml-blas PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    if (GGML_CUDA)
        set_target_properties(ggml-cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    if (GGML_HIP)
        set_target_properties(ggml-hip PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    target_compile_definitions(ggml PRIVATE GGML_SHARED GGML_BUILD)
    set_target_properties(rwkv PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(rwkv PRIVATE RWKV_SHARED RWKV_BUILD)
endif()
