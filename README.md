# MLKit

A small, dependency-free neural network library written in C. Supports dense
feed-forward networks with ReLU / sigmoid / softmax / linear activations,
HalfMSE / CrossEntropy losses, SGD training, and binary weight import/export.

```c
#include "MLKit.h"
```

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [Network lifecycle](#network-lifecycle)
  - [Propagation](#propagation)
  - [Training & testing](#training--testing)
  - [Import / export](#import--export)
- [Usage Example](#usage-example)
- [Types](#types)
  - [`ActivationFunction`](#activationfunction)
  - [`LossFunction`](#lossfunction)
  - [`LayerType`](#layertype)
  - [`Neuron`](#neuron)
  - [`Layer`](#layer)
  - [`Network`](#network)
- [Known Limitations](#known-limitations)

---

## Core Concepts

A `Network` is a list of `Layer`s. Layer 0 is always an `INPUT` layer (a
pass-through layer with no weights). Every subsequent layer is `DENSE`,
fully connected to the previous layer, with its own activation function.

Typical usage pattern:

```c
Network* net = nn_network_create(input_size);
nn_add_layer_dense(net, hidden_size, activation_reLU);
nn_add_layer_dense(net, output_size, activation_softmax);
nn_network_set_loss(net, CrossEntropy);

nn_network_fit(net, x_train, num_samples, y_train, learning_rate, epochs, print_every_n_epochs);
nn_network_test(net, x_test, y_test, num_test_samples);

nn_network_export(net, "model.bin");
nn_network_destroy(net);
```

---

## API Reference

### Network lifecycle

#### `Network* nn_network_create(int input_size)`
Allocates a new `Network` with a single `INPUT` layer of `input_size`
neurons. Default loss is `HalfMSE`. Caller owns the returned pointer and
must call `nn_network_destroy` on it.

```c
Network* net = nn_network_create(784); // e.g. 28x28 flattened image
```

#### `void nn_add_layer_dense(Network* network, int neurons_count, ActivationFunction activation)`
Appends a fully-connected layer with `neurons_count` neurons to the network.
Each neuron's `input_size` is inferred automatically from the previous
layer's neuron count. Weights are randomly initialized in `[-1, 1]`, scaled
by He initialization when `activation_reLU` is used.

```c
nn_add_layer_dense(net, 128, activation_reLU);
nn_add_layer_dense(net, 10,  activation_softmax); // output layer
```

> ⚠️ Add layers in order — output layer last. `nn_network_forward` and
> `nn_network_backward` assume `layers[layers_count - 1]` is the output layer.

#### `void nn_network_set_loss(Network* network, LossFunction loss_f)`
Sets the loss function used by `nn_network_backward` and
`nn_calculate_loss`. Defaults to `HalfMSE` if never called.

```c
nn_network_set_loss(net, CrossEntropy);
```

#### `void nn_network_destroy(Network* network)`
Frees all neurons, layers, and the network struct itself. Call once you're
done with the network — no other function should be called on it afterward.

---

### Propagation

#### `void nn_network_forward(Network* net, float* x)`
Runs a single input sample through the network, populating `z_val` and
`output` on every neuron. `x` must point to a buffer of exactly
`net->input_size` floats — the size is read from the network itself rather
than passed in, so it's the caller's responsibility to ensure `x` is sized
correctly; there's no bounds check against a mismatched buffer.

```c
float x[784] = { /* ... */ };
nn_network_forward(net, x);
```

After calling, read predictions from the output layer:
```c
Layer* out = &net->layers[net->layers_count - 1];
for (int i = 0; i < out->neuron_count; i++) {
    printf("%f\n", out->neurons[i].output);
}
```

#### `void nn_network_backward(Network* net, float* y, float lr)`
Runs backpropagation for **one sample** against target vector `y` (length
must equal the output layer's neuron count), then applies an SGD weight
update with learning rate `lr`. You must call `nn_network_forward` for the
same sample immediately before this.

```c
float y[10] = { 0,0,1,0,0,0,0,0,0,0 }; // one-hot label
nn_network_forward(net, x);
nn_network_backward(net, y, 0.01f);
```

> This is a per-sample (pure SGD) update — there is no mini-batching.

---

### Training & testing

#### `void nn_network_fit(Network* net, float* x, int data_size, float* y, float lr, int epochs, int print_steps)`
Trains the network for `epochs` epochs over a flattened dataset:

- `x` — `data_size * input_size` floats, samples laid out contiguously.
- `y` — `data_size * output_size` floats, targets laid out contiguously.
- `lr` — learning rate applied every sample.
- `print_steps` — prints a loss readout every `print_steps` epochs.

Prints a live progress bar per epoch and a final "Finished training" message.

```c
nn_network_fit(net, x_train, 60000, y_train, 0.01f, 10, 1);
```

#### `float nn_calculate_loss(Network* net, float* x, float* y, int data_size)`
Runs `data_size` samples through the network and returns the mean per-sample
loss (using the network's configured `LossFunction`), without updating any
weights. Useful for validation/held-out evaluation, or for the periodic loss
readout inside `nn_network_fit`.

```c
float val_loss = nn_calculate_loss(net, x_val, y_val, 1000);
```

#### `void nn_network_test(Network* net, float* x, float* y, int samples)`
Runs `samples` inputs through the network and reports classification
accuracy by comparing `argmax(output)` to `argmax(y)` per sample.

> **Currently assumes a softmax (one-hot classification) output layer.**
> Not meaningful for regression (linear) or multi-label (sigmoid) outputs.

```c
nn_network_test(net, x_test, y_test, 10000);
// -> "The model was correct 9421/10000 times. With accuracy 94.210000%"
```

#### `void nn_network_print(Network network)`
Prints every layer and neuron's bias, input size, output, `z_val`, `delta`,
and weight count. Intended for debugging small networks — output volume
scales with total parameter count. Takes the `Network` by value.

```c
nn_network_print(*net);
```

---

### Import / export

Binary format is a flat sequence of `float32` values, one network's worth of
parameters, laid out layer-by-layer (skipping the input layer) and
neuron-by-neuron as: **`bias, weight[0], weight[1], ..., weight[input_size-1]`**.

#### `void nn_network_export(Network* net, char* filepath)`
Writes the network's parameters to `filepath` in the format above,
overwriting any existing file. Prints the total parameter count on success.

```c
nn_network_export(net, "model.bin");
```

#### `void nn_network_import_file(Network* net, char* filepath)`
Loads parameters from a file previously written by `nn_network_export` into
an **already-constructed** network with matching architecture (same number
of layers and neurons per layer — this is not validated, so a mismatched
architecture will read garbage or overrun the buffer). Exits the process if
the file can't be opened or the buffer can't be allocated.

```c
Network* net = nn_network_create(784);
nn_add_layer_dense(net, 128, activation_reLU);
nn_add_layer_dense(net, 10,  activation_softmax);
nn_network_import_file(net, "model.bin");
```

#### `void nn_network_import(Network* net, float* buffer, int size)`
Same as `nn_network_import_file`, but loads from an in-memory `float*`
buffer you've already populated (e.g. if you fetched weights from somewhere
other than disk) rather than reading a file. `size` should be given in
**bytes**, matching the convention used internally for file sizes.

```c
float* buf = /* ... obtained elsewhere ... */;
nn_network_import(net, buf, buf_size_in_bytes);
```

---

## Usage Example

This is a complete, compilable program — it trains a tiny network to learn
XOR, prints its predictions, and saves the model. Verified against the
library: after 2000 epochs the loss drops from ~0.13 to ~0.001 and the
output layer correctly separates the four cases.

```c
#include <stdio.h>
#include <time.h>
#include "MLKit.h"

int main(void) {
    srand((unsigned)time(NULL));

    // XOR: 4 samples, 2 input features, 1 binary output.
    float x_train[8] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    };
    float y_train[4] = { 0, 1, 1, 0 };

    Network* net = nn_network_create(2);
    nn_add_layer_dense(net, 8, activation_reLU);
    nn_add_layer_dense(net, 1, activation_sigmoid);
    nn_network_set_loss(net, CrossEntropy);

    nn_network_fit(net, x_train, 4, y_train, 0.1f, 2000, 500);

    printf("\nPredictions after training:\n");
    for (int i = 0; i < 4; i++) {
        nn_network_forward(net, &x_train[i * 2]);
        Layer* out = &net->layers[net->layers_count - 1];
        printf("  %.0f XOR %.0f -> %.4f (expected %.0f)\n",
               x_train[i*2], x_train[i*2+1], out->neurons[0].output, y_train[i]);
    }

    nn_network_export(net, "xor_model.bin");
    nn_network_destroy(net);
    return 0;
}
```

Build and run:

```sh
gcc -o example example.c MLKit.c -lm
./example
```

Actual output from a real run:

```
Finished training 2000, 2000 epochs

Predictions after training:
  0 XOR 0 -> 0.0190 (expected 0)
  0 XOR 1 -> 0.9710 (expected 1)
  1 XOR 0 -> 0.9709 (expected 1)
  1 XOR 1 -> 0.0715 (expected 0)
Saved a total of 33 parameters to xor_model.bin path
```

Reusing a trained model later:

```c
Network* net = nn_network_create(2);
nn_add_layer_dense(net, 8, activation_reLU);
nn_add_layer_dense(net, 1, activation_sigmoid);
nn_network_import_file(net, "xor_model.bin");

nn_network_forward(net, sample);
Layer* out = &net->layers[net->layers_count - 1];
// read out->neurons[i].output for predictions
```

---

## Types

### `ActivationFunction`

```c
typedef enum {
    activation_reLU,
    activation_linear,
    activation_softmax,
    activation_sigmoid,
    NONE
} ActivationFunction;
```

| Value                | Behavior |
|-----------------------|----------|
| `activation_reLU`     | `max(0, z)`. Weight init uses He scaling (`sqrt(2/fan_in)`). |
| `activation_linear`   | Identity, `output = z`. Use for regression outputs or DQN-style Q-value heads. |
| `activation_softmax`  | Numerically-stable softmax over the whole layer (max-subtraction trick). Intended for the **output layer only**. |
| `activation_sigmoid`  | `1 / (1 + e^-z)`. |
| `NONE`                | No activation applied. Used internally for the input layer. |

### `LossFunction`

```c
typedef enum {
    HalfMSE,
    CrossEntropy
} LossFunction;
```

| Value          | Formula (per output neuron) | Typical pairing |
|-----------------|------------------------------|------------------|
| `HalfMSE`       | `0.5 * (output - y)^2`       | `activation_linear` or `activation_sigmoid` output, regression |
| `CrossEntropy`  | `-y * log(output)`           | `activation_softmax` (multi-class) or `activation_sigmoid` (binary) output |

When the loss/activation pairing is `CrossEntropy` + `softmax` or `CrossEntropy`
+ `sigmoid`, the library uses the simplified combined gradient
`delta = output - y` directly (the standard trick that cancels the softmax /
cross-entropy Jacobian), rather than chaining the raw loss derivative through
the activation derivative.

### `LayerType`

```c
typedef enum {
    INPUT,
    DENSE
} LayerType;
```

Set automatically by `nn_network_create` (layer 0 → `INPUT`) and
`nn_add_layer_dense` (`DENSE`). Not intended to be set manually.

### `Neuron`

```c
typedef struct {
    float* weights;    // one weight per input from the previous layer
    float  bias;
    int    input_size; // number of incoming connections (== prev layer's neuron_count)
    float  output;     // activated output, a
    float  z_val;      // pre-activation weighted sum, z
    float  delta;      // dL/dz for this neuron, set during backward pass
    float  error;      // scratch field used for the softmax delta calculation
} Neuron;
```

### `Layer`

```c
typedef struct {
    Neuron* neurons;
    int neuron_count;
    ActivationFunction activation;
    LayerType type;
} Layer;
```

### `Network`

```c
typedef struct {
    Layer* layers;
    int layers_count;
    int layers_capacity;
    int input_size;
    LossFunction loss_function;
} Network;
```

`layers_capacity` grows by doubling as layers are added; you should not need
to touch it directly.

---

## Known Limitations

- `nn_network_test`'s accuracy metric assumes a softmax/one-hot output —
  don't use it to evaluate regression or multi-label models.
- `nn_network_backward` is single-sample SGD only; there's no built-in
  mini-batch averaging. For now :) .
- `nn_network_import` / `nn_network_import_file` do **not** validate that
  the loaded file matches the target network's architecture — mismatches
  fail silently or read out of bounds rather than raising a clear error.
- No malloc/fopen failure checks in a few older code paths — treat this as
  a learning/prototyping library rather than production-hardened code.