# pymlkit

A small, dependency-free neural network library with a C core and a Python
wrapper. Dense feed-forward networks, mini-batch SGD, binary weight
import/export, and REINFORCE reinforcement learning (C API).

## Python

```bash
pip install pymlkit        # or, from a clone: pip install .
```

```python
from pymlkit import Network, Activations, Losses

model = Network(2)                          # 2 inputs
model.add_dense(16, Activations.ReLU)
model.add_dense(1, Activations.Sigmoid)
model.set_loss(Losses.HalfMSE)

# data is a flat list: [x1a, x1b, x2a, x2b, ...] / [y1, y2, ...]
model.fit(train_x, train_y, lr=0.1, epochs=100, batch=16)

print(model.forward([0.1, 0.2]))            # -> [0.93]
model.export("model.mlk")                   # model.load("model.mlk") to restore
```

| Method | Description |
|---|---|
| `Network(input_size)` | Create a network |
| `add_dense(n, activation)` | Append a dense layer (output layer last) |
| `set_loss(loss)` | `Losses.HalfMSE` or `Losses.CrossEntropy` |
| `fit(x, y, lr, epochs, batch, log_step)` | Mini-batch SGD |
| `forward(x)` | One sample in, list of outputs out |
| `backward(y)` | Accumulate gradients for one sample (custom loops) |
| `export(path)` / `load(path)` | Save / load weights; `load` returns `False` if the file is missing |

Activations: `ReLU`, `Linear`, `Softmax`, `Sigmoid`. Example:
[`python/examples/circle_classifier.py`](python/examples/circle_classifier.py).

## C

The same library, usable directly from C (`MLKit.c` / `MLKit.h`):

```c
#include "MLKit.h"

Network* net = nn_network_create(2);
nn_add_layer_dense(net, 8, activation_reLU);
nn_add_layer_dense(net, 1, activation_sigmoid);
nn_network_set_loss(net, CrossEntropy);

nn_network_fit(net, x_train, 4, y_train, 0.1f, 2000, 4, 500);
float* out = nn_network_forward(net, sample);

nn_network_export(net, "model.mlk");
nn_network_destroy(net);
```

```sh
gcc -Ofast -o example example.c MLKit.c -lm   # or: cmake -S . -B build && cmake --build build
```

Compile with **`-Ofast`** (MSVC: `/O2 /fp:fast`) for much faster training. The
CMake build and the PyPI wheels already do this.

Main functions: `nn_network_create`, `nn_add_layer_dense`, `nn_network_set_loss`,
`nn_network_fit`, `nn_network_forward`, `nn_network_backward`,
`nn_apply_gradients`, `nn_network_test`, `nn_network_export`,
`nn_network_import_file`, `nn_network_destroy`. Signatures are in `MLKit.h`.

**Reinforcement learning (REINFORCE):** `rl_agent_create`, `rl_draw_action`,
`rl_remember`, `rl_policy_fit`, `rl_agent_destroy`. See
[`examples/snake/`](examples/snake/) for a full Snake agent.

## Notes

- Weights are stored as flat `float32` arrays; the `.mlk` file is a raw
  `float32` stream (bias, then weights, per neuron).
- Loading a model does not check that the architecture matches.
- No shuffling between epochs. CPU only. A learning/prototyping library, not
  production-hardened.

## License

MIT
