# MLKit

A small, dependency-free neural network library written in C. Supports dense
feed-forward networks with ReLU / sigmoid / softmax / linear activations,
HalfMSE / CrossEntropy losses, mini-batch SGD training, binary weight
import/export, and a **REINFORCE policy-gradient** reinforcement learning API.

```c
#include "MLKit.h"
```

## Python

A ctypes wrapper lives in `python/pymlkit` and is built from the same C sources.

```bash
pip install pymlkit        # or, from a clone: pip install .
```

```python
from pymlkit import Network, Activations, Losses

model = Network(2)
model.add_dense(16, Activations.ReLU)
model.add_dense(1, Activations.Sigmoid)
model.set_loss(Losses.HalfMSE)
model.fit(train_x, train_y, lr=0.1, epochs=100, batch=16)
print(model.forward([0.1, 0.2]))
```

Full example: `python/examples/circle_classifier.py`. To build only the C
library: `cmake -S . -B build && cmake --build build`.

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [Network lifecycle](#network-lifecycle)
  - [Propagation](#propagation)
  - [Training & testing](#training--testing)
  - [Import / export](#import--export)
  - [Helpers](#helpers)
  - [Reinforcement learning](#reinforcement-learning)
- [Usage Examples](#usage-examples)
  - [Supervised learning (XOR)](#supervised-learning-xor)
  - [Reinforcement learning (REINFORCE)](#reinforcement-learning-reinforce)
- [Types](#types)
  - [`ActivationFunction`](#activationfunction)
  - [`LossFunction`](#lossfunction)
  - [`LayerType`](#layertype)
  - [`Layer`](#layer)
  - [`Network`](#network)
  - [`ReplayBuffer`](#replaybuffer)
  - [`RLAgent`](#rlagent)
- [Known Limitations](#known-limitations)

---

## Core Concepts

A `Network` is a list of `Layer`s. Layer 0 is always a `LAYER_INPUT` layer (a
pass-through with no weights). Every subsequent layer is `LAYER_DENSE`,
fully connected to the previous layer, with its own activation function.

Each layer stores its parameters as **flat `float*` arrays** (weights, biases,
outputs, z-values, deltas, errors, gradients) rather than an array of
per-neuron structs. The weight for neuron `n`, incoming connection `w` in a
layer lives at index `n * layer->tensor + w` inside `layer->weights`.

Typical supervised usage:

```c
Network* net = nn_network_create(input_size);
nn_add_layer_dense(net, hidden_size, activation_reLU);
nn_add_layer_dense(net, output_size, activation_softmax);
nn_network_set_loss(net, CrossEntropy);

nn_network_fit(net, x_train, num_samples, y_train, lr, epochs, batch_size, print_every);
nn_network_test(net, x_test, y_test, num_test_samples);

nn_network_export(net, "model.mlk");
nn_network_destroy(net);
```

---

## API Reference

### Network lifecycle

#### `Network* nn_network_create(int input_size)`
Allocates a new `Network` with a single `LAYER_INPUT` layer of `input_size`
neurons. Default loss is `HalfMSE`. Caller owns the returned pointer and must
call `nn_network_destroy` on it.

```c
Network* net = nn_network_create(784); // 28x28 flattened image
```

#### `void nn_add_layer_dense(Network* network, int neuron_count, ActivationFunction activation)`
Appends a fully-connected layer with `neuron_count` neurons. The input size is
inferred automatically from the previous layer. Weights are randomly
initialized and scaled by the activation:

| Activation | Init scale |
|---|---|
| `activation_reLU` | He — `sqrt(2 / fan_in)` |
| `activation_softmax` | Xavier — `sqrt(1 / fan_in)` |
| everything else | `1.0` |

```c
nn_add_layer_dense(net, 128, activation_reLU);
nn_add_layer_dense(net, 10,  activation_softmax); // output layer
```

> ⚠️ Add layers in order — output layer last.

#### `void nn_network_set_loss(Network* network, LossFunction loss_f)`
Sets the loss function used during training and evaluation. Defaults to
`HalfMSE` if never called.

```c
nn_network_set_loss(net, CrossEntropy);
```

#### `void nn_network_destroy(Network* network)`
Frees all layer arrays and the network struct. No other function should be
called on the network after this.

---

### Propagation

#### `float* nn_network_forward(Network* net, float* x)`
Runs one input sample through the network. Returns a pointer directly into
`net->output_layer->outputs` — a `float*` of length `output_layer->neuron_count`.
The pointer is valid until the next forward call or `nn_network_destroy`.

`x` must be exactly `net->input_size` floats — there is no bounds check.

```c
float* out = nn_network_forward(net, x);
printf("class 3 score: %f\n", out[3]);
```

#### `void nn_network_backward(Network* net, float* y)`
Backpropagates for **one sample** against target vector `y` (length ==
output layer's `neuron_count`). Accumulates gradients into each layer's
`weights_gradient` / `bias_gradient` buffers — no weight update yet.
Call `nn_network_forward` for the same sample immediately before this.

```c
float y[10] = { 0,0,1,0,0,0,0,0,0,0 }; // one-hot label
nn_network_forward(net, x);
nn_network_backward(net, y);
```

> Gradients are averaged and applied only when `nn_apply_gradients` is called
> (which `nn_network_fit` does automatically every `batch` steps).

#### `void nn_apply_gradients(Network* net, float lr)`
Averages accumulated gradients over the current batch count, applies the SGD
update to all weights and biases, then zeroes the gradient buffers. Called
automatically by `nn_network_fit`; call directly only in a custom training loop.

```c
nn_network_forward(net, x);
nn_network_backward(net, y);
// ... repeat for batch_size samples ...
nn_apply_gradients(net, 0.01f);
```

---

### Training & testing

#### `void nn_network_fit(Network* net, float* x, int data_size, float* y, float lr, int epochs, int batch, int print_steps)`
Trains for `epochs` epochs over a flattened dataset using mini-batch gradient descent.

| Parameter | Description |
|---|---|
| `x` | `data_size x input_size` floats, samples contiguous |
| `y` | `data_size x output_size` floats, targets contiguous |
| `lr` | Learning rate |
| `batch` | Samples per mini-batch. Pass `1` for per-sample SGD. |
| `print_steps` | Print a loss readout every `print_steps` epochs |

Prints a live progress bar per epoch and a final "Finished training" message.

```c
nn_network_fit(net, x_train, 60000, y_train, 0.01f, 50, 32, 1);
```

#### `float nn_calculate_loss(Network* net, float* x, float* y, int data_size)`
Returns mean per-sample loss over `data_size` samples without updating weights.
Useful for validation sets or periodic diagnostics.

```c
float val_loss = nn_calculate_loss(net, x_val, y_val, 1000);
```

#### `void nn_network_test(Network* net, float* x, float* y, int samples)`
Reports classification accuracy by comparing `argmax(output)` to `argmax(y)` per sample.

> **Assumes a softmax / one-hot output.** Not meaningful for regression or multi-label outputs.

```c
nn_network_test(net, x_test, y_test, 10000);
// -> "The model was correct 9421/10000 times. With accuracy 94.210000%"
```

#### `void nn_network_print(Network network)`
Prints every layer's neurons with bias, input size, output, z-value, delta,
and weight count. Intended for debugging small networks.

```c
nn_network_print(*net);
```

---

### Import / export

Binary format: flat `float32` stream, layer-by-layer (input layer skipped),
neuron-by-neuron: **`bias, weight[0], weight[1], ..., weight[tensor-1]`**.

#### `void nn_network_export(Network* net, char* filepath)`
Writes all parameters to `filepath`, overwriting any existing file. Prints
the total parameter count on success.

```c
nn_network_export(net, "model.mlk");
```

#### `int nn_network_import_file(Network* net, char* filepath)`
Loads parameters from a file into an **already-constructed** network with
matching architecture. Returns `0` on success, `-1` if the file can't be opened.

> Architecture mismatch is **not** validated — mismatched sizes will silently
> read garbage or overrun the buffer.

```c
Network* net = nn_network_create(784);
nn_add_layer_dense(net, 128, activation_reLU);
nn_add_layer_dense(net, 10,  activation_softmax);
if (nn_network_import_file(net, "model.mlk") != 0) {
    fprintf(stderr, "Failed to load model\n");
}
```

#### `void nn_network_import(Network* net, float* buffer, int size)`
Same as `nn_network_import_file` but from an in-memory buffer. `size` is in **bytes**.

```c
nn_network_import(net, buf, buf_size_in_bytes);
```

---

### Helpers

#### `int nn_biggest_output_idx(Layer* output_l)`
Returns the index of the neuron with the highest output — equivalent to
`argmax` over the output layer.

```c
float* out = nn_network_forward(net, x);
int predicted_class = nn_biggest_output_idx(net->output_layer);
```

---

### Reinforcement learning

MLKit includes a REINFORCE policy-gradient trainer built on top of the core
network API. The workflow is:

1. Create a `Network` with a `softmax` output (action probabilities).
2. Wrap it in an `RLAgent` (one or more parallel replay buffers).
3. Each step: run forward, sample an action, remember the transition.
4. At the end of a rollout: call `rl_policy_fit` to update weights.

#### `RLAgent* rl_agent_create(Network* model, int agent_count)`
Creates an `RLAgent` owning `agent_count` independent `ReplayBuffer`s. The
agent borrows `model` — destroy the network separately.

```c
RLAgent* agent = rl_agent_create(net, 1);
```

#### `void rl_agent_destroy(RLAgent* agent)`
Frees all replay buffers and the agent struct. Does **not** free the network.

#### `int rl_draw_action(Layer* output_l)`
Samples an action index from the output layer's probability distribution
(expects softmax). Call after `nn_network_forward`.

```c
nn_network_forward(net, state);
int action = rl_draw_action(net->output_layer);
```

#### `void rl_remember(ReplayBuffer* buff, float* state, float reward, int action_idx, int should_discount, int done)`
Appends one transition to a replay buffer. Buffer grows dynamically (doubles
capacity when full).

| Parameter | Description |
|---|---|
| `state` | Input features for this step (copied) |
| `reward` | Immediate reward received |
| `action_idx` | The action that was taken |
| `should_discount` | `1` to include in return computation, `0` to skip |
| `done` | `1` if this step ends the episode |

```c
rl_remember(&agent->buffers[0], state, reward, action, 1, done);
```

#### `void rl_replaybuff_clear(ReplayBuffer* buff)`
Resets the buffer size to 0 (retains allocated capacity). Called automatically
by `rl_policy_fit` after consuming each buffer.

#### `void rl_policy_fit(RLAgent* agent, int* buffer_ids, int buffers_count, float lr, float gamma)`
Runs REINFORCE over the selected buffers:
1. Discounts rewards backward within each episode using `gamma`.
2. Computes policy gradient and accumulates gradients for each step.
3. Calls `nn_apply_gradients` once to update weights.
4. Clears all consumed buffers.

```c
int bufs[] = { 0 };
rl_policy_fit(agent, bufs, 1, 0.001f, 0.99f);
```

---

## Usage Examples

### Supervised learning (XOR)

```c
#include <stdio.h>
#include <time.h>
#include "MLKit.h"

int main(void) {
    srand((unsigned)time(NULL));

    float x_train[8] = { 0,0,  0,1,  1,0,  1,1 };
    float y_train[4] = { 0, 1, 1, 0 };

    Network* net = nn_network_create(2);
    nn_add_layer_dense(net, 8, activation_reLU);
    nn_add_layer_dense(net, 1, activation_sigmoid);
    nn_network_set_loss(net, CrossEntropy);

    nn_network_fit(net, x_train, 4, y_train, 0.1f, 2000, 4, 500);

    printf("\nPredictions after training:\n");
    for (int i = 0; i < 4; i++) {
        float* out = nn_network_forward(net, &x_train[i * 2]);
        printf("  %.0f XOR %.0f -> %.4f (expected %.0f)\n",
               x_train[i*2], x_train[i*2+1], out[0], y_train[i]);
    }

    nn_network_export(net, "xor_model.mlk");
    nn_network_destroy(net);
    return 0;
}
```

Build and run:

```sh
gcc -o example example.c MLKit.c -lm
./example
```

Expected output:

```
Finished training 2000, 2000 epochs

Predictions after training:
  0 XOR 0 -> 0.0046 (expected 0)
  0 XOR 1 -> 0.9965 (expected 1)
  1 XOR 0 -> 0.9964 (expected 1)
  1 XOR 1 -> 0.0049 (expected 0)
Saved a total of 33 parameters to xor_model.mlk path
```

Reusing a saved model:

```c
Network* net = nn_network_create(2);
nn_add_layer_dense(net, 8, activation_reLU);
nn_add_layer_dense(net, 1, activation_sigmoid);
nn_network_import_file(net, "xor_model.mlk");

float* out = nn_network_forward(net, sample);
// out[0] is the prediction
```

---

### Reinforcement learning (REINFORCE)

Skeleton for a single-agent policy gradient loop:

```c
#include <string.h>
#include <time.h>
#include "MLKit.h"

#define STATE_SIZE  8
#define NUM_ACTIONS 4

int main(void) {
    srand((unsigned)time(NULL));

    Network* net = nn_network_create(STATE_SIZE);
    nn_add_layer_dense(net, 64, activation_reLU);
    nn_add_layer_dense(net, NUM_ACTIONS, activation_softmax);
    nn_network_set_loss(net, CrossEntropy);

    RLAgent* agent = rl_agent_create(net, 1);

    for (int episode = 0; episode < 1000; episode++) {
        float state[STATE_SIZE] = { /* initial observation */ };
        int done = 0;

        while (!done) {
            nn_network_forward(net, state);
            int action = rl_draw_action(net->output_layer);

            float reward;
            float next_state[STATE_SIZE];
            // env_step(action, next_state, &reward, &done);

            rl_remember(&agent->buffers[0], state, reward, action, 1, done);
            memcpy(state, next_state, sizeof(state));
        }

        int buf_ids[] = { 0 };
        rl_policy_fit(agent, buf_ids, 1, 0.001f, 0.99f);
    }

    rl_agent_destroy(agent);
    nn_network_destroy(net);
    return 0;
}
```

See [`examples/snake/`](examples/snake/) for a complete Snake game agent using
this API.

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

| Value | Behavior |
|---|---|
| `activation_reLU` | `max(0, z)`. He weight init (`sqrt(2/fan_in)`). |
| `activation_linear` | Identity. Use for regression or DQN Q-value heads. |
| `activation_softmax` | Numerically-stable softmax. Output layer only. Xavier init (`sqrt(1/fan_in)`). |
| `activation_sigmoid` | `1 / (1 + e^-z)`. |
| `NONE` | No activation. Used internally for the input layer. |

### `LossFunction`

```c
typedef enum {
    HalfMSE,
    CrossEntropy
} LossFunction;
```

| Value | Formula (per output neuron) | Typical pairing |
|---|---|---|
| `HalfMSE` | `0.5 * (output - y)^2` | `activation_linear` / `activation_sigmoid`, regression |
| `CrossEntropy` | `-y * log(output)` | `activation_softmax` (multi-class) or `activation_sigmoid` (binary) |

When using `CrossEntropy` + `softmax` or `CrossEntropy` + `sigmoid`, the
library uses the simplified combined gradient `delta = output - y` directly,
cancelling the softmax / cross-entropy Jacobian.

### `LayerType`

```c
typedef enum {
    LAYER_INPUT,
    LAYER_DENSE
} LayerType;
```

Set automatically by `nn_network_create` (`LAYER_INPUT`) and
`nn_add_layer_dense` (`LAYER_DENSE`). Not intended to be set manually.

### `Layer`

```c
typedef struct {
    int neuron_count;
    float* weights;           // flat: weights[n * tensor + w]
    float* biases;            // one per neuron
    float* outputs;           // activated outputs (a)
    float* z_values;          // pre-activation weighted sums (z)
    float* deltas;            // dL/dz per neuron
    float* errors;            // scratch for softmax delta
    float* weights_gradient;  // accumulated per weight across a batch
    float* bias_gradient;     // accumulated per neuron across a batch
    int tensor;               // == previous layer neuron_count
    ActivationFunction activation;
    LayerType type;
} Layer;
```

To read the weight connecting input `w` to neuron `n`:

```c
float w = layer->weights[n * layer->tensor + w_idx];
```

### `Network`

```c
typedef struct {
    Layer* layers;
    Layer* output_layer;   // pointer into layers[layers_count - 1]
    int layers_count;
    int layers_capacity;   // grows by doubling; don't touch directly
    int input_size;
    int batched;           // samples accumulated in current batch
    LossFunction loss_function;
} Network;
```

### `ReplayBuffer`

```c
typedef struct {
    float* states;          // flat: states[step * state_size + i]
    int*   actions_idx;
    float* rewards;
    int*   should_discount;
    int*   done;
    int    state_size;
    int    size;            // current stored transitions
    int    capacity;        // allocated slots (doubles on overflow)
} ReplayBuffer;
```

### `RLAgent`

```c
typedef struct {
    Network*      network;
    ReplayBuffer* buffers;
    int           buffers_count;
} RLAgent;
```

---

## Known Limitations

- `nn_network_test` accuracy assumes a softmax/one-hot output — don't use it
  for regression or multi-label models.
- Mini-batching slices `x`/`y` in the order given — no shuffling between
  epochs, so batch composition is fixed across the whole training run.
- `nn_network_import` / `nn_network_import_file` do **not** validate that the
  loaded file matches the target architecture — mismatches fail silently or
  overread the buffer.
- No malloc/fopen failure checks in some code paths — treat this as a
  learning/prototyping library, not production-hardened code.
