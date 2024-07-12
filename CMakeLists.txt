cmake_minimum_required(VERSION 3.12)

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

# Build only shared library without building tests and extras
option(RWKV_STANDALONE             "rwkv: build only RWKV library"                        OFF)

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
    if (RWKV_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (RWKV_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (RWKV_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (APPLE AND RWKV_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (RWKV_OPENBLAS)
    if (RWKV_STATIC)
        set(BLA_STATIC ON)
    endif()

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "OpenBLAS found")

        add_compile_definitions(GGML_USE_OPENBLAS)
        add_link_options(${BLAS_LIBRARIES})
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

if (RWKV_CUBLAS)
    cmake_minimum_required(VERSION 3.17)
    set(CMAKE_CUDA_COMPILER_FORCED TRUE)

    find_package(CUDAToolkit)

    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        set(GGML_CUDA_SOURCES ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda.cu ${CMAKE_SOURCE_DIR}/ggml/include/ggml-cuda.h)

        add_compile_definitions(GGML_USE_CUDA)

        # By default, GGML_CUDA_MMV_Y is set to 1. This value leads to CUDA error on my machine:
        # CUDA error 9 at ...\rwkv.cpp\ggml\src\ggml-cuda.cu:6107: invalid configuration argument
        # The error appears when the head matrix of v5 3B and v5 7B models is offloaded. I guess the matrix is so large that block_num_y becomes too big.
        # Changing it to 2 makes it work. I did not see any performance impact when measuring v5 3B & v5 7B. Hopefully, this will not break other use-cases.
        # TODO Re-check after updating ggml whether this is needed
        add_compile_definitions(GGML_CUDA_MMV_Y=2)

        if (RWKV_STATIC)
            set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        else()
            set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

        # Architecture set-up copy-pasted from https://github.com/ggerganov/llama.cpp/blob/111163e2463171891680feed94371eb9becd9817/CMakeLists.txt#L317
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # 52: lowest CUDA 12 standard
            # 60: f16 CUDA intrinsics
            # 61: integer CUDA intrinsics
            # 70: compute capability at which unrolling a loop in mul_mat_q kernels is faster

            # Lowest CUDA 12 standard + lowest for integer intrinsics.
            set(CMAKE_CUDA_ARCHITECTURES "52;61;70")
        endif()
        message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (RWKV_CLBLAST)
    cmake_minimum_required(VERSION 3.17)

    file(GLOB OPENCL_INCLUDE_SEARCH_PATHS ${CMAKE_SOURCE_DIR}/OpenCL-*/)
    file(GLOB CLBLAST_INCLUDE_SEARCH_PATHS ${CMAKE_SOURCE_DIR}/CLBlast-*/)

    set(OPENCL_INCLUDE_SEARCH_PATHS
        /usr/include
        /usr/local/include
        $ENV{OPENCL_HOME}
        $ENV{OPENCL_HOME}/include
        ${OPENCL_INCLUDE_SEARCH_PATHS}
    )

    set(CLBLAST_INCLUDE_SEARCH_PATHS
        /usr/include
        /usr/local/include
        $ENV{CLBLAST_HOME}
        $ENV{CLBLAST_HOME}/include
        ${CLBLAST_INCLUDE_SEARCH_PATHS}
    )

    find_path(OPENCL_INC NAMES opencl.h PATHS ${OPENCL_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES include/CL)
    find_library(OPENCL_LIB NAMES OpenCL PATHS ${OPENCL_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES lib)
    find_path(CLBLAST_INC NAMES clblast.h PATHS ${CLBLAST_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES include)
    find_library(CLBLAST_LIB NAMES clblast PATHS ${CLBLAST_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES lib)

    if (OPENCL_LIB)
        set(OPENCL_INC ${OPENCL_INC}/..) # disgusting
        message(STATUS "OpenCL SDK found: ${OPENCL_INC}")

        if (CLBLAST_LIB)
            message(STATUS "CLBlast found: ${CLBLAST_INC}")
            add_compile_definitions(GGML_USE_CLBLAST)
            set(GGML_OPENCL_SOURCES ${CMAKE_SOURCE_DIR}/ggml/src/ggml-opencl.cpp ${CMAKE_SOURCE_DIR}/ggml/src/ggml-opencl.h)
            set(GGML_OPENCL_DIRS ${GGML_OPENCL_DIRS} ${OPENCL_INC} ${CLBLAST_INC})
            set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} ${OPENCL_LIB} ${CLBLAST_LIB})
            link_libraries("-Wl,--copy-dt-needed-entries")
        else()
            message(WARNING "CLBlast not found")
        endif()
    else()
       message(WARNING "OpenCL SDK not found, CLBlast cannot be enabled")
    endif()
endif()

if (RWKV_HIPBLAS)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

    if (NOT ${CMAKE_C_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CC=/opt/rocm/llvm/bin/clang")
    endif()

    if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
    endif()

    find_package(hip)
    find_package(hipblas)
    find_package(rocblas)

    if (${hipblas_FOUND} AND ${hip_FOUND})
        message(STATUS "HIP and hipBLAS found")
        add_compile_definitions(GGML_USE_HIPBLAS GGML_USE_CUDA)
        add_library(ggml-rocm OBJECT
                    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda.cu
                    ${CMAKE_SOURCE_DIR}/ggml/include/ggml-cuda.h)

        if (RWKV_BUILD_SHARED_LIBRARY)
            set_target_properties(ggml-rocm PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()

        target_include_directories(ggml-rocm PUBLIC ${CMAKE_SOURCE_DIR}/ggml/include)
        set_source_files_properties(${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda.cu PROPERTIES LANGUAGE CXX)
        target_link_libraries(ggml-rocm PRIVATE hip::device PUBLIC hip::host roc::rocblas roc::hipblas)

        if (RWKV_STATIC)
            message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
        endif()

        set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} ggml-rocm)
    else()
        message(WARNING "hipBLAS or HIP not found. Try setting CMAKE_PREFIX_PATH=/opt/rocm")
    endif()
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
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
            -Wno-multichar
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
endif()

# Architecture specific
# TODO [llama.cpp]: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (RWKV_STATIC)
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

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO [llama.cpp]: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            add_compile_options(-mcpu=native)
        endif()
        # TODO [llama.cpp]: armv6,7,8 version specific flags
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (RWKV_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
        elseif (RWKV_AVX2)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (RWKV_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        add_compile_options(-mf16c)
        if (RWKV_FMA)
            add_compile_options(-mfma)
        endif()
        if (RWKV_AVX)
            add_compile_options(-mavx)
        endif()
        if (RWKV_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (RWKV_AVX512)
            add_compile_options(-mavx512f)
            add_compile_options(-mavx512bw)
        endif()
    endif()
else()
    # TODO [llama.cpp]: support PowerPC
    message(STATUS "Unknown architecture")
endif()

#
# POSIX conformance
# Section copy-pasted from https://github.com/ggerganov/llama.cpp/blob/8781013ef654270cbead3e0011e33a6d690fb168/CMakeLists.txt#L580C20-L580C20
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)
add_compile_definitions(_XOPEN_SOURCE=600)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher.
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    remove_definitions(-D_XOPEN_SOURCE=600)
    add_compile_definitions(_XOPEN_SOURCE=700)
endif()

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc.
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions.
# Similarly on DragonFly, enabling BSD extensions is necessary.
if (
    CMAKE_SYSTEM_NAME MATCHES "Darwin" OR
    CMAKE_SYSTEM_NAME MATCHES "iOS" OR
    CMAKE_SYSTEM_NAME MATCHES "tvOS" OR
    CMAKE_SYSTEM_NAME MATCHES "DragonFly"
)
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases.
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

#
# Build libraries
#

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

add_library(ggml OBJECT
            ${CMAKE_SOURCE_DIR}/ggml/src/ggml.c
            ${CMAKE_SOURCE_DIR}/ggml/src/ggml-alloc.c
            ${CMAKE_SOURCE_DIR}/ggml/src/ggml-quants.c
            ${CMAKE_SOURCE_DIR}/ggml/src/ggml-backend.c
            ${CMAKE_SOURCE_DIR}/ggml/include/ggml.h
            ${CMAKE_SOURCE_DIR}/ggml/include/ggml-alloc.h
            ${CMAKE_SOURCE_DIR}/ggml/include/ggml-backend.h
            ${GGML_CUDA_SOURCES}
            ${GGML_OPENCL_SOURCES})

target_include_directories(ggml PUBLIC ${CMAKE_SOURCE_DIR}/ggml/include ${GGML_OPENCL_DIRS})
target_compile_features(ggml PUBLIC c_std_11) # Don't bump

if (MSVC)
    target_link_libraries(ggml PUBLIC ${RWKV_EXTRA_LIBS} Threads::Threads)
    if (RWKV_CUBLAS)
        target_compile_options(ggml PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -allow-unsupported-compiler
        >)
    endif()
else()
    if (WIN32 AND RWKV_HIPBLAS)
        target_link_libraries(ggml PUBLIC ${RWKV_EXTRA_LIBS} Threads::Threads)
    else()
        target_link_libraries(ggml PUBLIC m ${RWKV_EXTRA_LIBS} Threads::Threads)
    endif()
endif()

if (RWKV_BUILD_SHARED_LIBRARY)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if (RWKV_BUILD_SHARED_LIBRARY)
    add_library(rwkv SHARED rwkv.cpp rwkv.h)
else()
    add_library(rwkv rwkv.cpp rwkv.h)
endif()

target_include_directories(rwkv PUBLIC .)
target_compile_features(rwkv PUBLIC cxx_std_11)
target_link_libraries(rwkv PRIVATE ggml ${RWKV_EXTRA_LIBS})

if (RWKV_BUILD_SHARED_LIBRARY)
    set_target_properties(rwkv PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(rwkv PRIVATE RWKV_SHARED RWKV_BUILD)
endif()

if (NOT RWKV_STANDALONE)
    set_property(TARGET ggml PROPERTY GGML_STANDALONE OFF)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(extras)
elseif()
    set_property(TARGET ggml PROPERTY GGML_STANDALONE ON)
endif()
