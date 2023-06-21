import os
import sys
import ctypes
import pathlib
from typing import Optional, List

QUANTIZED_FORMAT_NAMES = (
    'Q4_0',
    'Q4_1',
    'Q5_0',
    'Q5_1',
    'Q8_0'
)

P_FLOAT = ctypes.POINTER(ctypes.c_float)
P_INT = ctypes.POINTER(ctypes.c_int32)

class RWKVContext:

    def __init__(self, ptr: ctypes.pointer):
        self.ptr = ptr

class RWKVSharedLibrary:
    """
    Python wrapper around rwkv.cpp shared library.
    """

    def __init__(self, shared_library_path: str):
        """
        Loads the shared library from specified file.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        shared_library_path : str
            Path to rwkv.cpp shared library. On Windows, it would look like 'rwkv.dll'. On UNIX, 'rwkv.so'.
        """

        self.library = ctypes.cdll.LoadLibrary(shared_library_path)

        self.library.rwkv_init_from_file.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
        self.library.rwkv_init_from_file.restype = ctypes.c_void_p

        self.library.rwkv_gpu_offload_layers.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.library.rwkv_gpu_offload_layers.restype = ctypes.c_bool

        self.library.rwkv_eval.argtypes = [
            ctypes.c_void_p, # ctx
            ctypes.c_int32, # token
            P_FLOAT, # state_in
            P_FLOAT, # state_out
            P_FLOAT  # logits_out
        ]
        self.library.rwkv_eval.restype = ctypes.c_bool

        self.library.rwkv_eval_sequence.argtypes = [
            ctypes.c_void_p, # ctx
            P_INT, # tokens
            ctypes.c_size_t, # token count
            P_FLOAT, # state_in
            P_FLOAT, # state_out
            P_FLOAT  # logits_out
        ]
        self.library.rwkv_eval_sequence.restype = ctypes.c_bool

        self.library.rwkv_get_state_buffer_element_count.argtypes = [ctypes.c_void_p]
        self.library.rwkv_get_state_buffer_element_count.restype = ctypes.c_uint32

        self.library.rwkv_get_logits_buffer_element_count.argtypes = [ctypes.c_void_p]
        self.library.rwkv_get_logits_buffer_element_count.restype = ctypes.c_uint32

        self.library.rwkv_free.argtypes = [ctypes.c_void_p]
        self.library.rwkv_free.restype = None

        self.library.rwkv_free.argtypes = [ctypes.c_void_p]
        self.library.rwkv_free.restype = None

        self.library.rwkv_quantize_model_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.library.rwkv_quantize_model_file.restype = ctypes.c_bool

        self.library.rwkv_get_system_info_string.argtypes = []
        self.library.rwkv_get_system_info_string.restype = ctypes.c_char_p

    def rwkv_init_from_file(self, model_file_path: str, thread_count: int) -> RWKVContext:
        """
        Loads the model from a file and prepares it for inference.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        model_file_path : str
            Path to model file in ggml format.
        thread_count : int
            Count of threads to use, must be positive.
        """

        ptr = self.library.rwkv_init_from_file(model_file_path.encode('utf-8'), ctypes.c_uint32(thread_count))

        assert ptr is not None, 'rwkv_init_from_file failed, check stderr'

        return RWKVContext(ptr)

    def rwkv_gpu_offload_layers(self, ctx: RWKVContext, layer_count: int) -> bool:
        """
        Offloads specified count of model layers onto the GPU. Offloaded layers are evaluated using cuBLAS.
        Returns true if at least one layer was offloaded.
        If rwkv.cpp was compiled without cuBLAS support, this function is a no-op and always returns false.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        layer_count : int
            Count of layers to offload onto the GPU, must be >= 0.
        """

        assert layer_count >= 0, 'Layer count must be >= 0'

        return self.library.rwkv_gpu_offload_layers(ctx.ptr, ctypes.c_uint32(layer_count))

    def rwkv_eval(
            self,
            ctx: RWKVContext,
            token: int,
            state_in_address: Optional[int],
            state_out_address: int,
            logits_out_address: int
    ) -> None:
        """
        Evaluates the model for a single token.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        token : int
            Next token index, in range 0 <= token < n_vocab.
        state_in_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count; or None, if this is a first pass.
        state_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count. This buffer will be written to.
        logits_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_logits_buffer_element_count. This buffer will be written to.
        """

        assert self.library.rwkv_eval(
            ctx.ptr,
            ctypes.c_int32(token),
            ctypes.cast(0 if state_in_address is None else state_in_address, P_FLOAT),
            ctypes.cast(state_out_address, P_FLOAT),
            ctypes.cast(logits_out_address, P_FLOAT)
        ), 'rwkv_eval failed, check stderr'

    def rwkv_eval_sequence(
            self,
            ctx: RWKVContext,
            tokens: List[int],
            state_in_address: Optional[int],
            state_out_address: int,
            logits_out_address: int
    ) -> None:
        """
        Evaluates the model for a sequence of tokens.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        tokens : List[int]
            Next token indices, in range 0 <= token < n_vocab.
        state_in_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count; or None, if this is a first pass.
        state_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count. This buffer will be written to.
        logits_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_logits_buffer_element_count. This buffer will be written to.
        """

        assert self.library.rwkv_eval_sequence(
            ctx.ptr,
            ctypes.cast((ctypes.c_int32 * len(tokens))(*tokens), P_INT),
            ctypes.c_size_t(len(tokens)),
            ctypes.cast(0 if state_in_address is None else state_in_address, P_FLOAT),
            ctypes.cast(state_out_address, P_FLOAT),
            ctypes.cast(logits_out_address, P_FLOAT)
        ), 'rwkv_eval failed, check stderr'

    def rwkv_get_state_buffer_element_count(self, ctx: RWKVContext) -> int:
        """
        Returns count of FP32 elements in state buffer.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """

        return self.library.rwkv_get_state_buffer_element_count(ctx.ptr)

    def rwkv_get_logits_buffer_element_count(self, ctx: RWKVContext) -> int:
        """
        Returns count of FP32 elements in logits buffer.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """

        return self.library.rwkv_get_logits_buffer_element_count(ctx.ptr)

    def rwkv_free(self, ctx: RWKVContext) -> None:
        """
        Frees all allocated memory and the context.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """

        self.library.rwkv_free(ctx.ptr)

        ctx.ptr = ctypes.cast(0, ctypes.c_void_p)

    def rwkv_quantize_model_file(self, model_file_path_in: str, model_file_path_out: str, format_name: str) -> None:
        """
        Quantizes FP32 or FP16 model to one of INT4 formats.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        model_file_path_in : str
            Path to model file in ggml format, must be either FP32 or FP16.
        model_file_path_out : str
            Quantized model will be written here.
        format_name : str
            One of QUANTIZED_FORMAT_NAMES.
        """

        assert format_name in QUANTIZED_FORMAT_NAMES, f'Unknown format name {format_name}, use one of {QUANTIZED_FORMAT_NAMES}'

        assert self.library.rwkv_quantize_model_file(
            model_file_path_in.encode('utf-8'),
            model_file_path_out.encode('utf-8'),
            format_name.encode('utf-8')
        ), 'rwkv_quantize_model_file failed, check stderr'

    def rwkv_get_system_info_string(self) -> str:
        """
        Returns system information string.
        """

        return self.library.rwkv_get_system_info_string().decode('utf-8')


def load_rwkv_shared_library() -> RWKVSharedLibrary:
    """
    Attempts to find rwkv.cpp shared library and load it.
    To specify exact path to the library, create an instance of RWKVSharedLibrary explicitly.
    """

    file_name: str

    if 'win32' in sys.platform or 'cygwin' in sys.platform:
        file_name = 'rwkv.dll'
    elif 'darwin' in sys.platform:
        file_name = 'librwkv.dylib'
    else:
        file_name = 'librwkv.so'

    repo_root_dir: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent.parent

    paths = [
        # If we are in "rwkv" directory
        f'../bin/Release/{file_name}',
        # If we are in repo root directory
        f'bin/Release/{file_name}',
        # If we compiled in build directory
        f'build/bin/Release/{file_name}',
        # If we compiled in build directory
        f'build/{file_name}',
        # Search relative to this file
        str(repo_root_dir / 'bin' / 'Release' / file_name),
        # Fallback
        str(repo_root_dir / file_name)
    ]

    for path in paths:
        if os.path.isfile(path):
            return RWKVSharedLibrary(path)

    return RWKVSharedLibrary(paths[-1])
