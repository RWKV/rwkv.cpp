# rwkv.cpp

This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) to [ggerganov/ggml](https://github.com/ggerganov/ggml).

Besides the usual **FP32**, it supports **FP16**, **quantized INT4** and **quantized INT8** inference. This project is **CPU only**.

This project provides [a C library rwkv.h](rwkv.h) and [a convinient Python wrapper](rwkv%2Frwkv_cpp_model.py) for it.

RWKV is a novel large language model architecture, [with the largest model in the family having 14B parameters](https://huggingface.co/BlinkDL/rwkv-4-pile-14b). In contrast to Transformer with `O(n^2)` attention, RWKV requires only state from previous step to calculate logits. This makes RWKV very CPU-friendly on large context lenghts.

Loading LoRA checkpoints in [Blealtan's format](https://github.com/Blealtan/RWKV-LM-LoRA) is supported through [merge_lora_into_ggml.py script](rwkv%2Fmerge_lora_into_ggml.py).

### Quality and performance

If you use `rwkv.cpp` for anything serious, please [test all available formats for perplexity and latency](rwkv%2Fmeasure_pexplexity.py) on a representative dataset, and decide which trade-off is best for you.

Below table is for reference only. Measurements were made on 4C/8T x86 CPU with AVX2, 4 threads.

| Format    | Perplexity (169M) | Latency, ms (1.5B) | File size, GB (1.5B) |
|-----------|-------------------|--------------------|----------------------|
| `Q4_0`    | 17.507            | *76*               | **1.53**             |
| `Q4_1`    | 17.187            | **72**             | 1.68                 |
| `Q4_2`    | 17.060            | 85                 | **1.53**             |
| `Q5_0`    | 16.194            | 78                 | *1.60*               |
| `Q5_1`    | 15.851            | 81                 | 1.68                 |
| `Q8_0`    | *15.652*          | 89                 | 2.13                 |
| `FP16`    | **15.623**        | 117                | 2.82                 |
| `FP32`    | **15.623**        | 198                | 5.64                 |

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

##### Windows

**Requirements**: [CMake](https://cmake.org/download/) or [CMake from anaconda](https://anaconda.org/conda-forge/cmake), MSVC compiler.

```commandline
cmake .
cmake --build . --config Release
```

If everything went OK, `bin\Release\rwkv.dll` file should appear.

##### Linux / MacOS

**Requirements**: CMake (Linux: `sudo apt install cmake`, MacOS: `brew install cmake`, anaconoda: [cmake package](https://anaconda.org/conda-forge/cmake)).

```commandline
cmake .
cmake --build . --config Release
```

**Anaconda & M1 users**: please verify that `CMAKE_SYSTEM_PROCESSOR: arm64` after running `cmake .` — if it detects `x86_64`, edit the `CMakeLists.txt` file under the `# Compile flags` to add `set(CMAKE_SYSTEM_PROCESSOR "arm64")`.

If everything went OK, `librwkv.so` (Linux) or `librwkv.dylib` (MacOS) file should appear in the base repo folder.


### 3. Download an RWKV model from [Hugging Face](https://huggingface.co/BlinkDL) like [this one](https://huggingface.co/BlinkDL/rwkv-4-pile-169m/blob/main/RWKV-4-Pile-169M-20220807-8023.pth) and convert it into `ggml` format

**Requirements**: Python 3.x with [PyTorch](https://pytorch.org/get-started/locally/).

```commandline
# Windows
python rwkv\convert_pytorch_to_ggml.py C:\RWKV-4-Pile-169M-20220807-8023.pth C:\rwkv.cpp-169M.bin float16

# Linux / MacOS
python rwkv/convert_pytorch_to_ggml.py ~/Downloads/RWKV-4-Pile-169M-20220807-8023.pth ~/Downloads/rwkv.cpp-169M.bin float16
```

#### 3.1. Optionally, quantize the model

To convert the model into one of quantized formats from the table above, run:

```commandline
# Windows
python rwkv\quantize.py C:\rwkv.cpp-169M.bin C:\rwkv.cpp-169M-Q4_2.bin Q4_2

# Linux / MacOS
python rwkv/quantize.py ~/Downloads/rwkv.cpp-169M.bin ~/Downloads/rwkv.cpp-169M-Q4_2.bin Q4_2
```

### 4. Run the model

**Requirements**: Python 3.x with [PyTorch](https://pytorch.org/get-started/locally/) and [tokenizers](https://pypi.org/project/tokenizers/).

**Note**: change the model path with the non-quantized model for the full weights model.

To generate some text, run:

```commandline
# Windows
python rwkv\generate_completions.py C:\rwkv.cpp-169M-Q4_2.bin

# Linux / MacOS
python rwkv/generate_completions.py ~/Downloads/rwkv.cpp-169M-Q4_2.bin
```

To chat with a bot, run:

```commandline
# Windows
python rwkv\chat_with_bot.py C:\rwkv.cpp-169M-Q4_2.bin

# Linux / MacOS
python rwkv/chat_with_bot.py ~/Downloads/rwkv.cpp-169M-Q4_2.bin
```

Edit [generate_completions.py](rwkv%2Fgenerate_completions.py) or [chat_with_bot.py](rwkv%2Fchat_with_bot.py) to change prompts and sampling settings.

---

Example of using `rwkv.cpp` in your custom Python script:

```python
import rwkv_cpp_model
import rwkv_cpp_shared_library

# Change to model paths used above (quantized or full weights) 
model_path = r'C:\rwkv.cpp-169M.bin'


model = rwkv_cpp_model.RWKVModel(
    rwkv_cpp_shared_library.load_rwkv_shared_library(),
    model_path
)

logits, state = None, None

for token in [1, 2, 3]:
    logits, state = model.eval(token, state)
    
    print(f'Output logits: {logits}')

# Don't forget to free the memory after you've done working with the model
model.free()

```
