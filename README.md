# rwkv.cpp

This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) to [ggerganov/ggml](https://github.com/ggerganov/ggml).

Besides the usual **FP32**, it supports **FP16**, **quantized INT4, INT5 and INT8** inference. This project is **focused on CPU**, but cuBLAS is also supported.

This project provides [a C library rwkv.h](rwkv.h) and [a convinient Python wrapper](python%2Frwkv_cpp%2Frwkv_cpp_model.py) for it.

[RWKV](https://arxiv.org/abs/2305.13048) is a novel large language model architecture, [with the largest model in the family having 14B parameters](https://huggingface.co/BlinkDL/rwkv-4-pile-14b). In contrast to Transformer with `O(n^2)` attention, RWKV requires only state from previous step to calculate logits. This makes RWKV very CPU-friendly on large context lenghts.

Loading LoRA checkpoints in [Blealtan's format](https://github.com/Blealtan/RWKV-LM-LoRA) is supported through [merge_lora_into_ggml.py script](rwkv%2Fmerge_lora_into_ggml.py).

⚠️ **Python API was restructured on 2023-09-20**, you may need to change paths/package names in your code when updating `rwkv.cpp`.

## Quality and performance

If you use `rwkv.cpp` for anything serious, please [test all available formats for perplexity and latency](rwkv%2Fmeasure_pexplexity.py) on a representative dataset, and decide which trade-off is best for you.

Below table is for reference only. Measurements were made on 4C/8T x86 CPU with AVX2, 4 threads.

| Format    | Perplexity (169M) | Latency, ms (1.5B) | File size, GB (1.5B) |
|-----------|-------------------|--------------------|----------------------|
| `Q4_0`    | 17.507            | *76*               | **1.53**             |
| `Q4_1`    | 17.187            | **72**             | 1.68                 |
| `Q5_0`    | 16.194            | 78                 | *1.60*               |
| `Q5_1`    | 15.851            | 81                 | 1.68                 |
| `Q8_0`    | *15.652*          | 89                 | 2.13                 |
| `FP16`    | **15.623**        | 117                | 2.82                 |
| `FP32`    | **15.623**        | 198                | 5.64                 |

### With cuBLAS

Measurements were made on Intel i7 13700K & NVIDIA 3060 Ti 8 GB. Latency per token in ms shown.

| Model                 | Layers on GPU | Format | 1 thread | 2 threads | 4 threads | 8 threads | 24 threads |
|-----------------------|---------------|--------|----------|-----------|-----------|-----------|------------|
| `RWKV-4-Pile-169M`    | 12            | `Q4_0` | 7.9      | 6.2       | 6.9       | 8.6       | 20         |
| `RWKV-4-Pile-169M`    | 12            | `Q4_1` | 7.8      | 6.7       | 6.9       | 8.6       | 21         |
| `RWKV-4-Pile-169M`    | 12            | `Q5_1` | 8.1      | 6.7       | 6.9       | 9.0       | 22         |

| Model                 | Layers on GPU | Format | 1 thread | 2 threads | 4 threads | 8 threads | 24 threads |
|-----------------------|---------------|--------|----------|-----------|-----------|-----------|------------|
| `RWKV-4-Raven-7B-v11` | 32            | `Q4_0` | 59       | 51        | 50        | 54        | 94         |
| `RWKV-4-Raven-7B-v11` | 32            | `Q4_1` | 59       | 51        | 49        | 54        | 94         |
| `RWKV-4-Raven-7B-v11` | 32            | `Q5_1` | 77       | 69        | 67        | 72        | 101        |

Note: since cuBLAS is supported only for `ggml_mul_mat()`, we still need to use few CPU resources to execute remaining operations.

### With hipBLAS
Measurements were made on CPU AMD Ryzen 9 5900X & GPU AMD Radeon RX 7900 XTX. Latency per token in ms shown.

| Model                                    | Layers on GPU | Format | 1 thread | 2 threads | 4 threads | 8 threads | 24 threads |
|------------------------------------------|---------------|--------|----------|-----------|-----------|-----------|------------|
| `RWKV-novel-4-World-7B-20230810-ctx128k` | 32            | `f16`  | 94       | 91        | 94        | 106       | 944        |
| `RWKV-novel-4-World-7B-20230810-ctx128k` | 32            | `Q4_0` | 83       | 77        | 75        | 110       | 1692       |
| `RWKV-novel-4-World-7B-20230810-ctx128k` | 32            | `Q4_1` | 85       | 80        | 85        | 93        | 1691       |
| `RWKV-novel-4-World-7B-20230810-ctx128k` | 32            | `Q5_1` | 83       | 78        | 83        | 90        | 1115       |

Note: hipBLAS is same as cuBLAS.They only support `ggml_mul_mat()`, we still need to use few CPU resources to execute remaining operations.

## How to use

### 1. Clone the repo

**Requirements**: [git](https://gitforwindows.org/).

```commandline
git clone --recursive https://github.com/saharNooby/rwkv.cpp.git
cd rwkv.cpp
```

### 2. Get the rwkv.cpp library

#### Option 2.1. Download a pre-compiled library

##### Windows / Linux / MacOS

Check out [Releases](https://github.com/saharNooby/rwkv.cpp/releases), download appropriate ZIP for your OS and CPU, extract `rwkv` library file into the repository directory.

On Windows: to check whether your CPU supports AVX2 or AVX-512, [use CPU-Z](https://www.cpuid.com/softwares/cpu-z.html).

#### Option 2.2. Build the library yourself

This option is recommended for maximum performance, because the library would be built specifically for your CPU and OS.

##### Windows

**Requirements**: [CMake](https://cmake.org/download/) or [CMake from anaconda](https://anaconda.org/conda-forge/cmake), [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/vs/older-downloads/).

```commandline
cmake .
cmake --build . --config Release
```

If everything went OK, `bin\Release\rwkv.dll` file should appear.

##### Windows + cuBLAS

Refer to [docs/cuBLAS_on_Windows.md](docs%2FcuBLAS_on_Windows.md) for a comprehensive guide.

##### Windows + hipBLAS

Refer to [docs/hipBLAS_on_Windows.md](docs%2FhipBLAS_on_Windows.md) for a comprehensive guide.

##### Linux / MacOS

**Requirements**: CMake (Linux: `sudo apt install cmake`, MacOS: `brew install cmake`, anaconoda: [cmake package](https://anaconda.org/conda-forge/cmake)).

```commandline
cmake .
cmake --build . --config Release
```

**Anaconda & M1 users**: please verify that `CMAKE_SYSTEM_PROCESSOR: arm64` after running `cmake .` — if it detects `x86_64`, edit the `CMakeLists.txt` file under the `# Compile flags` to add `set(CMAKE_SYSTEM_PROCESSOR "arm64")`.

If everything went OK, `librwkv.so` (Linux) or `librwkv.dylib` (MacOS) file should appear in the base repo folder.

##### Linux / MacOS + cuBLAS

```commandline
cmake . -DRWKV_CUBLAS=ON
cmake --build . --config Release
```

If everything went OK, `librwkv.so` (Linux) or `librwkv.dylib` (MacOS) file should appear in the base repo folder.

### 3. Get an RWKV model

#### Option 3.1. Download pre-quantized Raven model

There are pre-quantized Raven models available on [Hugging Face](https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main). Check that you are downloading `.bin` file, **not** `.pth`.

#### Option 3.2. Convert and quantize PyTorch model

**Requirements**: Python 3.x with [PyTorch](https://pytorch.org/get-started/locally/).

This option would require a little more manual work, but you can use it with any RWKV model and any target format.

**First**, download a model from [Hugging Face](https://huggingface.co/BlinkDL) like [this one](https://huggingface.co/BlinkDL/rwkv-4-pile-169m/blob/main/RWKV-4-Pile-169M-20220807-8023.pth).

**Second**, convert it into `rwkv.cpp` format using following commands:

```commandline
# Windows
python python\convert_pytorch_to_ggml.py C:\RWKV-4-Pile-169M-20220807-8023.pth C:\rwkv.cpp-169M.bin FP16

# Linux / MacOS
python python/convert_pytorch_to_ggml.py ~/Downloads/RWKV-4-Pile-169M-20220807-8023.pth ~/Downloads/rwkv.cpp-169M.bin FP16
```

**Optionally**, quantize the model into one of quantized formats from the table above:

```commandline
# Windows
python python\quantize.py C:\rwkv.cpp-169M.bin C:\rwkv.cpp-169M-Q5_1.bin Q5_1

# Linux / MacOS
python python/quantize.py ~/Downloads/rwkv.cpp-169M.bin ~/Downloads/rwkv.cpp-169M-Q5_1.bin Q5_1
```

### 4. Run the model

#### Using the command line

**Requirements**: Python 3.x with [numpy](https://numpy.org/). If using `Pile` or `Raven` models, [tokenizers](https://pypi.org/project/tokenizers/) is also required.

To generate some text, run:

```commandline
# Windows
python python\generate_completions.py C:\rwkv.cpp-169M-Q5_1.bin

# Linux / MacOS
python python/generate_completions.py ~/Downloads/rwkv.cpp-169M-Q5_1.bin
```

To chat with a bot, run:

```commandline
# Windows
python python\chat_with_bot.py C:\rwkv.cpp-169M-Q5_1.bin

# Linux / MacOS
python python/chat_with_bot.py ~/Downloads/rwkv.cpp-169M-Q5_1.bin
```

Edit [generate_completions.py](rwkv%2Fgenerate_completions.py) or [chat_with_bot.py](rwkv%2Fchat_with_bot.py) to change prompts and sampling settings.

#### Using in your own code

The short and simple script [inference_example.py](python%2Finference_example.py) demostrates the use of `rwkv.cpp` in Python.

To use `rwkv.cpp` in C/C++, include the header [rwkv.h](rwkv.h).

To use `rwkv.cpp` in any other language, see [Bindings](#Bindings) section below. If your language is missing, you can try to bind to the C API using the tooling provided by your language.

## Bindings

These projects wrap `rwkv.cpp` for easier use in other languages/frameworks.

* Golang: [seasonjs/rwkv](https://github.com/seasonjs/rwkv)
* Node.js: [Atome-FE/llama-node](https://github.com/Atome-FE/llama-node)

## Compatibility

`ggml` moves fast, and can occasionally break compatibility with older file formats.

`rwkv.cpp` will attempt it's best to explain why a model file can't be loaded and what next steps are available to the user.

For reference only, here is a list of latest versions of `rwkv.cpp` that have supported older formats. **No support will be provided for these versions**.

- `Q4_2`, old layout of quantized formats
  - [commit 3ca9c7f](https://github.com/saharNooby/rwkv.cpp/commit/3ca9c7f7857a4b9f3de616ec938e71249cfb3f3f), [release with prebuilt binaries](https://github.com/saharNooby/rwkv.cpp/releases/tag/master-3ca9c7f)
- `Q4_3`, `Q4_1_O`
  - [commit c736ef5](https://github.com/saharNooby/rwkv.cpp/commit/c736ef5411606b529d3a74c139ee111ef1a28bb9), [release with prebuilt binaries](https://github.com/saharNooby/rwkv.cpp/releases/tag/master-1c363e6)

See also [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md) for version numbers of `rwkv.cpp` model files and their changelog.

## Contributing

Please follow the code style described in [docs/CODE_STYLE.md](docs/CODE_STYLE.md).
