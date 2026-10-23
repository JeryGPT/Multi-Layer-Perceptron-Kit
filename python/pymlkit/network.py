import ctypes
import os
import sys
from enum import Enum
from pathlib import Path


def _load_library():
    name = {"win32": "MLKit.dll", "darwin": "libMLKit.dylib"}.get(sys.platform,  "libMLKit.so")
    path = Path(__file__).resolve().parent / name
    if not path.exists():
        raise ImportError(f"Native library not found: {path}")
    return ctypes.CDLL(str(path))


class CNetwork(ctypes.Structure):
    pass

class Activations(Enum):
    ReLU = 0
    Linear = 1
    Softmax = 2
    Sigmoid = 3
    NONE = 4

class Losses(Enum):
    HalfMSE = 0
    CrossEntropy = 1

NetworkPtr = ctypes.POINTER(CNetwork)

pymlkit = _load_library()

nn_network_create_fxn = pymlkit.nn_network_create
nn_network_create_fxn.argtypes = [ctypes.c_int]
nn_network_create_fxn.restype = NetworkPtr




pymlkit.nn_network_destroy.argtypes = [NetworkPtr]
pymlkit.nn_network_destroy.restype = None

nn_add_layer_dense_fxn = pymlkit.nn_add_layer_dense
nn_add_layer_dense_fxn.argtypes = [NetworkPtr, ctypes.c_int, ctypes.c_int]


nn_network_forward_fxn = pymlkit.nn_network_forward
nn_network_forward_fxn.argtypes = [NetworkPtr, ctypes.POINTER(ctypes.c_float)]
nn_network_forward_fxn.restype = ctypes.POINTER(ctypes.c_float)

nn_network_set_loss_fxn = pymlkit.nn_network_set_loss
nn_network_set_loss_fxn.argtypes = [NetworkPtr,  ctypes.c_int]

nn_network_fit_fxn = pymlkit.nn_network_fit

nn_network_fit_fxn.restype = None
nn_network_fit_fxn.argtypes = [
    NetworkPtr,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

nn_network_export_fxn = pymlkit.nn_network_export
nn_network_export_fxn.argtypes = [NetworkPtr, ctypes.c_char_p]

nn_network_import_file_fxn = pymlkit.nn_network_import_file
nn_network_import_file_fxn.argtypes = [NetworkPtr, ctypes.c_char_p]
nn_network_import_file_fxn.restype = ctypes.c_int

nn_network_backward_fxn = pymlkit.nn_network_backward
nn_network_backward_fxn.argtypes = [NetworkPtr, ctypes.POINTER(ctypes.c_float)]
nn_network_backward_fxn.restype = None

class Network:
    def __init__(self, input_size):
        self.input_size = input_size


        self.network_ptr = pymlkit.nn_network_create(input_size)
        self.output_size = 1

    def __del__(self):
        if self.network_ptr:
            pymlkit.nn_network_destroy(self.network_ptr)
            self.network_ptr = None

    def add_dense(self, neuron_count, activation=Activations.Linear):
        pymlkit.nn_add_layer_dense(self.network_ptr, neuron_count, activation.value)
        self.output_size = neuron_count

    def backward(self, label = []):
        if len(label) != self.output_size:
            raise ValueError(f"[ERROR] Wrong tensor size ({len(label)} expected {self.output_size})")
        arr = (ctypes.c_float * len(label))(*label)
        pymlkit.nn_network_backward(self.network_ptr, arr)
        

    def forward(self, tensor = []):
        if len(tensor) != self.input_size:
            raise ValueError(f"[ERROR] Wrong tensor size ({len(tensor)} expected {self.input_size})")
        arr = (ctypes.c_float * len(tensor))(*tensor)
        out = pymlkit.nn_network_forward(self.network_ptr, arr)
        return [out[i] for i in range(self.output_size)]
        


    def fit(self, train_x, train_y, lr = 0.01, epochs = 10, batch = 1, log_step = 10):
        sample_count = len(train_x) // self.input_size
        if len(train_x) % self.input_size != 0:
            raise ValueError(
                f"Input data length {len(train_x)} is not divisible by input size {self.input_size}"
            )

        samples = len(train_x) // self.input_size

        if len(train_y) != samples * self.output_size:
            raise ValueError(
                f"Expected {samples * self.output_size} target values, got {len(train_y)}"
            )
        x_arr = (ctypes.c_float * len(train_x))(*train_x)
        y_arr = (ctypes.c_float * len(train_y))(*train_y)
        pymlkit.nn_network_fit(
            self.network_ptr,
            x_arr,
            sample_count,
            y_arr,
            lr,
            epochs,
            batch,
            log_step
        )
        
    def export(self, path = "model.mlk"):
        pymlkit.nn_network_export(self.network_ptr, path.encode("utf-8"))

    def load(self, path = "model.mlk"):
        if not os.path.exists(path):
            print(f"[WARN] File '{path}' does not exist.")
            return False
        pymlkit.nn_network_import_file(self.network_ptr, path.encode("utf-8"))
        return True
    
    def set_loss(self, loss = Losses.HalfMSE):
        pymlkit.nn_network_set_loss(self.network_ptr, loss.value)





