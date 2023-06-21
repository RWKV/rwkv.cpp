# Using cuBLAS on Windows

To get cuBLAS in `rwkv.cpp` working on Windows, go through this guide section by section.

## Build Tools for Visual Studio 2019

Skip this step if you already have Build Tools installed.

To install Build Tools, go to [Visual Studio Older Downloads](https://visualstudio.microsoft.com/vs/older-downloads/), download `Visual Studio 2019 and other Products` and run the installer.

## CMake

Skip this step if you already have CMake installed: running `cmake --version` should output `cmake version x.y.z`.

Download latest `Windows x64 Installer` from [Download | CMake](https://cmake.org/download/) and run it.

## CUDA Toolkit

Skip this step if you already have CUDA Toolkit installed: running `nvcc --version` should output `nvcc: NVIDIA (R) Cuda compiler driver`.

CUDA Toolkit must be installed **after** CMake, or else CMake would not be able to see it and you will get error [No CUDA toolset found](https://stackoverflow.com/questions/56636714/cuda-compile-problems-on-windows-cmake-error-no-cuda-toolset-found).

Download an installer from [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and run it.

When installing:

- check `Visual Studio Integration`, or else CMake would not be able to see the toolkit
- optionally, uncheck driver installation — depending on the downloaded version of the toolkit, you may get an unwanted driver downgrade

## Building rwkv.cpp

The only thing different from the regular CPU build is `-DRWKV_CUBLAS=ON` option:

```commandline
cmake . -DRWKV_CUBLAS=ON
cmake --build . --config Release
```

If everything went OK, `bin\Release\rwkv.dll` file should appear.

## Using the GPU

You need to choose layer count that will be offloaded onto the GPU. In general, the more layers offloaded, the better will be the performance; but you may be constrained by VRAM size of your GPU. Increase offloaded layer count until you get "CUDA out of memory" errors.

If most of the computation is performed on GPU, you will not need high thread count. Optimal value may be as low as 1, since any additional threads would just eat CPU cycles while waiting for GPU operation to complete.

To offload layers to GPU:

- if using Python model: pass non-zero number in `gpu_layer_count` to constructor of `rwkv.rwkv_cpp_model.RWKVModel`
- if using Python wrapper for C library: call `rwkv.rwkv_cpp_shared_library.RWKVSharedLibrary.rwkv_gpu_offload_layers`
- if using C library directly: call `bool rwkv_gpu_offload_layers(struct rwkv_context * ctx, const uint32_t n_layers)`

## Fixing issues

You may get `FileNotFoundError: Could not find module '...\rwkv.dll' (or one of its dependencies). Try using the full path with constructor syntax.` error.

This means that the application couldn't find CUDA libraries that `rwkv.dll` depends on.

To fix this:

- navigate to the folder where CUDA Toolkit is installed
    - usually, it looks like `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin`
- find three DLLs in the `bin` folder:
  - `cudart64_110.dll`
  - `cublas64_11.dll`
  - `cublasLt64_11.dll`
- copy these DDLs to the folder containing `rwkv.dll`
  - usually, the folder is `rwkv.cpp/bin/Release`
