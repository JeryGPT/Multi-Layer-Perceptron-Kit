#include "MLKit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

Network *nn_network_create(int input_size);

void nn_network_destroy(Network *network);

void nn_add_layer_dense(Network *network, int input_size,
                        ActivationFunction activation);

void nn_network_print(Network network);
float *nn_network_forward(Network *net, float *x);
void nn_network_backward(Network *net, float *y);
void nn_network_fit(Network *net, float *x, int data_size, float *y, float lr,
                    int epochs, int batch, int print_steps);

static void _layer_activate(Layer *layer);
static int _max_arr(float *arr, int arr_len);

static void _layer_init(Layer *layer, int neuron_count, int input_size,
                        ActivationFunction activation, LayerType type);
static void _replaybuff_init(ReplayBuffer *buff, int input_size);

void _replaybuff_destroy(ReplayBuffer *buff);
int _get_weight_idx(Layer *layer, int neuron_idx, int weight_idx);
float _get_neuron_weight(Layer *layer, int n_idx, int w_idx);
float _get_random_weight(ActivationFunction activation, int input_size);

// ACTIVATORS
float _reLU(float x) { return 0 > x ? 0 : x; }

float _sigmoid(float x) { return 1 / (1 + expf(-x)); }

float _get_derivative(Layer *output_layer, int neuron_idx) {
  ActivationFunction activation = output_layer->activation;
  float z_val = output_layer->z_values[neuron_idx];
  float a = output_layer->outputs[neuron_idx];
  switch (activation) {
  case activation_linear:
    return 1.0;
    break;
  case activation_reLU:
    return z_val < 0 ? 0.0 : 1.0;
    break;

  case activation_sigmoid: {
    float s = _sigmoid(z_val);
    return s * (1.0 - s);
  }
  case activation_softmax:
    break;
  default:
    return 1.0;
  }
  return 1.0f;
}

float _loss_value(LossFunction loss_f, float output,
                  float y) { // per/neuron calculation
  switch (loss_f) {
  case HalfMSE: {
    float error = output - y;
    return 0.5 * error * error;
  }
  case CrossEntropy: {
    float epsilon = 1e-7f;
    if (output < epsilon)
      output = epsilon;
    return -y * logf(output);
    break;
  }
  }
  return 0.0f;
}

float _loss_deriv_value(LossFunction loss_f, float output,
                        float y) { // per/neuron derivative
  switch (loss_f) {
  case HalfMSE: {
    return output - y;
  }
  case CrossEntropy: {
    float epsilon = 1e-7f;
    if (output < epsilon)
      output = epsilon;
    if (output > 1.0f - epsilon)
      output = 1.0f - epsilon;
    return -y / output + (1 - y) / (1 - output);
  }
  }
  return 0.0f;
}

static void _softmax_output_delta(Layer *output_l, LossFunction loss_f) {
  int n_count = output_l->neuron_count;

  float dot = 0.0f;

  for (int j = 0; j < n_count; j++) {
    dot += output_l->errors[j] * output_l->outputs[j];
  }

  for (int i = 0; i < n_count; i++) {
    float o_i = output_l->outputs[i];
    output_l->deltas[i] = o_i * (output_l->errors[i] - dot);
  }
}

void _calculate_output_delta(Network *net, float *y) {
  LossFunction loss_f = net->loss_function;
  Layer *output_l = net->output_layer;
  ActivationFunction activation = output_l->activation;
  switch (loss_f) {
  case (HalfMSE): {
    if (activation == activation_softmax) {
      for (int n = 0; n < output_l->neuron_count; n++) {
        float error = _loss_deriv_value(loss_f, output_l->outputs[n], y[n]);
        output_l->errors[n] = error;
      }
      _softmax_output_delta(output_l, loss_f);
    } else {

      for (int n = 0; n < output_l->neuron_count; n++) {
        float error = _loss_deriv_value(loss_f, output_l->outputs[n], y[n]);
        float z_deriv = _get_derivative(output_l, n);
        output_l->deltas[n] = error * z_deriv;
      }
    }
    break;
  }
  case (CrossEntropy): {
    if (activation == activation_softmax || activation == activation_sigmoid) {
      for (int n = 0; n < output_l->neuron_count; n++) {

        output_l->deltas[n] = output_l->outputs[n] - y[n];
      }
    } else {
      for (int n = 0; n < output_l->neuron_count; n++) {
        float error = _loss_deriv_value(loss_f, output_l->outputs[n], y[n]);
        output_l->deltas[n] = error * _get_derivative(output_l, n);
      }
    }
  }
  }
}

float nn_calculate_loss(Network *net, float *x, float *y, int data_size) {
  float total_loss = 0.0f;
  int input_size = net->input_size;
  int output_size = net->output_layer->neuron_count;

  Layer *output_layer = net->output_layer;
  for (int i = 0; i < data_size; i++) {
    float sample_loss = 0;
    nn_network_forward(net, &x[i * input_size]);
    for (int n = 0; n < output_size; n++) {
      sample_loss += _loss_value(net->loss_function, output_layer->outputs[n],
                                 y[i * output_size + n]);
    }
    total_loss += sample_loss / output_size;
  }
  total_loss /= (float)data_size;
  return total_loss;
}

void _clear_gradients(Network *net) {
  for (int i = 1; i < net->layers_count; i++) {
    Layer *output_l = &net->layers[i];
    for (int n = 0; n < output_l->neuron_count; n++) {
      output_l->bias_gradient[n] = 0;
      for (int w = 0; w < output_l->tensor; w++) {
        output_l->weights_gradient[_get_weight_idx(output_l, n, w)] = 0;
        output_l->weights_gradient[_get_weight_idx(output_l, n, w)] = 0;
      }
      output_l->deltas[n] = 0;
    }
  }
}

void nn_apply_gradients(Network *net, float lr) {
  int layers_count = net->layers_count;

  for (int l = 1; l < layers_count; l++) {
    Layer *layer = &net->layers[l];
    for (int n = 0; n < layer->neuron_count; n++) {
      for (int w = 0; w < layer->tensor; w++) {
        int w_i = _get_weight_idx(layer, n, w);
        layer->weights[w_i] -= lr * layer->weights_gradient[w_i] / net->batched;
      }
      layer->biases[n] -= lr * layer->bias_gradient[n] / net->batched;
    }
  }
  net->batched = 0;
  _clear_gradients(net);
}

void nn_network_fit(Network *net, float *x, int data_size, float *y, float lr,
                    int epochs, int batch, int print_steps) {
  int input_size = net->input_size;
  int output_size = net->output_layer->neuron_count;
  for (int epoch = 0; epoch < epochs; epoch++) {

    for (int i = 0; i < data_size; i++) {

      if (i % ((data_size + 10) / 10) == 0 || i == data_size - 1) {
        int progress = (i * 10) / data_size;
        if (i == data_size - 1)
          progress = 10;

        printf("\r[");
        for (int p = 0; p < progress; p++) {
          printf("=");
        }
        for (int p = progress; p < 10; p++) {
          printf(" ");
        }
        printf("] %d%% Epoch: %d/%d", progress * 10, epoch, epochs);

        fflush(stdout);
      }
      nn_network_forward(net, &x[i * input_size]);
      nn_network_backward(net, &y[i * output_size]);
      if ((i + 1) % batch == 0 || i == data_size - 1) {
        nn_apply_gradients(net, lr);
      }
    }

    if (epoch % print_steps == 0) {
      printf(" Loss: %f \n",
             nn_calculate_loss(
                 net, x, y,
                 (int)(data_size > 100000 ? data_size / 4 : data_size)));
    }
  }

  printf("Finished training %d, %d epochs \n", epochs, epochs);
}

void _calculate_hidden_deltas(Network *net) {
  int layers_count = net->layers_count;
  for (int cr = layers_count - 2; cr > 0; cr--) {
    Layer *current_l = &net->layers[cr];
    Layer *next_l = &net->layers[cr + 1];
    for (int n = 0; n < current_l->neuron_count; n++) {
      float error_sum = 0.0f;

      for (int next_n = 0; next_n < next_l->neuron_count; next_n++) {
        int weight_idx = _get_weight_idx(next_l, next_n, n);
        float next_neuron_delta = next_l->deltas[next_n];
        error_sum += next_neuron_delta * next_l->weights[weight_idx];
      }
      float deriv = _get_derivative(current_l, n);
      current_l->deltas[n] = error_sum * deriv;
    }
  }

  for (int l = 1; l < layers_count; l++) {
    Layer *layer = &net->layers[l];
    Layer *prev_layer = &net->layers[l - 1];
    for (int n = 0; n < layer->neuron_count; n++) {
      for (int w = 0; w < layer->tensor; w++) {
        layer->weights_gradient[_get_weight_idx(layer, n, w)] +=
            layer->deltas[n] * prev_layer->outputs[w];
      }
      layer->bias_gradient[n] += layer->deltas[n];
    }
  }
}

void nn_network_backward(Network *net, float *y) {
  int layers_count = net->layers_count;

  _calculate_output_delta(net, y);

  _calculate_hidden_deltas(net);
  net->batched += 1;

  return;
}

void nn_network_export(Network *net, char *filepath) {
  FILE *fptr = fopen(filepath, "wb");

  fclose(fptr);

  fptr = fopen(filepath, "ab");

  if (!fptr) {
    printf("Failed to export the model.");
  }
  int total_weights = 0;
  for (int l = 1; l < net->layers_count; l++) {
    Layer *layer = &net->layers[l];
    total_weights += layer->neuron_count * layer->tensor;
    for (int n = 0; n < layer->neuron_count; n++) {
      float bias = layer->biases[n];
      fwrite(&bias, sizeof(float), 1, fptr);
      total_weights++;
      for (int w = 0; w < layer->tensor; w++) {
        float weight = layer->weights[_get_weight_idx(layer, n, w)];
        fwrite(&weight, sizeof(float), 1, fptr);
      }
    }
  }
  printf("Saved a total of %d parameters to %s path\n", total_weights,
         filepath);
  fclose(fptr);
}

void nn_network_import(Network *net, float *buffer, int size) {
  int current_w = 0;
  for (int l = 1; l < net->layers_count; l++) {
    Layer *layer = &net->layers[l];
    for (int n = 0; n < layer->neuron_count; n++) {
      layer->biases[n] = buffer[current_w];
      current_w++;
      for (int w = 0; w < layer->tensor; w++) {
        layer->weights[_get_weight_idx(layer, n, w)] = buffer[current_w];
        current_w++;
      }
    }
  }
  if (current_w != (int)(size - sizeof(float))) {
    printf("Import buffer and network size mismatch");
  } else {
    printf("Loaded %d weights", (int)(size / sizeof(float) - 1));
  }
}

void _discount_rewards(ReplayBuffer *buffer, float gamma) {
  int size = buffer->size;
  float *rewards = buffer->rewards;
  float avg = 0;

  int *should_discount = buffer->should_discount;
  int jumps = 0;
  int episode_end = size;

  for (int i = size - 1; i >= 0; i--) {
    int done = buffer->done[i];
    if (done) {
      episode_end = i + 1;
      jumps = 0;
    }

    if (!should_discount[i]) {
      jumps += 1;

      continue;
    } else {
      int next_idx = i + 1 + jumps;
      if (!done && next_idx < episode_end) {
        rewards[i] += rewards[next_idx] * pow(gamma, jumps + 1);
      }
      jumps = 0; // Oh shit I feel so genius after this implementation wow
    }
    avg += rewards[i];
  }
  avg /= size;
}

void _policy_step(Network *net, float *state, int action_idx, float G) {
  nn_network_forward(net, state);
  if (net->output_layer->activation == activation_softmax) {
    Layer *output_l = net->output_layer;
    for (int n = 0; n < output_l->neuron_count; n++) {
      float target = n == action_idx;
      output_l->deltas[n] = (output_l->outputs[n] - target) *
                            G; // Policy gradient loss (-ln(output) * G) deriv
    }
  }
  net->batched += 1;

  _calculate_hidden_deltas(net);
}

void rl_policy_fit(RLAgent *agent, int *buffer_ids, int buffers_count, float lr,
                   float gamma) {

  Network *net = agent->network;
  int input_size = net->input_size;
  int output_size = net->output_layer->neuron_count;
  for (int i = 0; i < buffers_count; i++) {
    ReplayBuffer *buffer = &agent->buffers[buffer_ids[i]];
    int samples = buffer->size;
    _discount_rewards(buffer, gamma);
    for (int step = 0; step < samples; step++) {
      _policy_step(net, &buffer->states[step * input_size],
                   buffer->actions_idx[step], buffer->rewards[step]);
    }
    rl_replaybuff_clear(buffer);
  }
  nn_apply_gradients(net, lr);
}

int nn_network_import_file(Network *net, char *filepath) {

  FILE *fptr = fopen(filepath, "rb");
  if (!fptr) {
    printf("Failed to open the import file.");
    
    return -1;
  }

  fseek(fptr, 0L, SEEK_END);
  size_t f_size = ftell(fptr);
  rewind(fptr);

  float *buffer = malloc(f_size);
  if (!buffer) {
    printf("Failed to allocate memory for the import buffer.");
    fclose(fptr);
    exit(1);
  }

  fread(buffer, sizeof(float), f_size / sizeof(float), fptr);

  fclose(fptr);

  int current_w = 0;
  for (int l = 1; l < net->layers_count; l++) {
    Layer *layer = &net->layers[l];
    for (int n = 0; n < layer->neuron_count; n++) {
      layer->biases[n] = buffer[current_w];
      current_w++;
      for (int w = 0; w < layer->tensor; w++) {
        layer->weights[_get_weight_idx(layer, n, w)] = buffer[current_w];
        current_w++;
      }
    }
  }

  printf("Loaded %d parameters", (int)(f_size / sizeof(float)));
  free(buffer);
  return 0;
}

int rl_draw_action(Network* net, float* state) {
  nn_network_forward(net, state);
  Layer *output_l = net->output_layer;
  float r = (float)rand() / (float)RAND_MAX;
  float sum = 0.0f;

  for (int i = 0; i < output_l->neuron_count; i++) {
    sum += output_l->outputs[i];
    if (r <= sum) {
      return i;
    }
  }
  return output_l->neuron_count - 1; // Fallback
}
void rl_replaybuff_clear(ReplayBuffer *buff) {
  memset(buff->states, 0, buff->capacity * buff->state_size * sizeof(float));
  memset(buff->rewards, 0, buff->capacity * sizeof(float));
  memset(buff->actions_idx, 0, buff->capacity * sizeof(int));
  memset(buff->should_discount, 0, buff->capacity * sizeof(int));
  memset(buff->done, 0, buff->capacity * sizeof(int));

  buff->size = 0;
}

RLAgent *rl_agent_create(Network *model, int agent_count) {
  RLAgent *agent = malloc(sizeof(RLAgent));
  agent->network = model;
  agent->buffers = NULL;
  agent->buffers_count = 0;
  _agent_add_replaybuffers(agent, agent_count);
  agent->buffers_count = agent_count;

  return agent;
}

void _agent_add_replaybuffers(RLAgent *agent, int agent_count) {
  if (agent->buffers_count > 0) {
    printf("Replay buffers already exist. Cannot add more. \n");
    return;
  }
  agent->buffers = calloc(sizeof(ReplayBuffer), agent_count);
  for (int i = 0; i < agent_count; i++) {

    _replaybuff_init(&agent->buffers[i], agent->network->input_size);
  }
}

void rl_remember(ReplayBuffer *buff, float *state, float reward, int action_idx,
                 int should_discount, int done) {

  if (buff->size == buff->capacity) {

    buff->capacity *= 2;

    buff->states = realloc(buff->states,
                           buff->capacity * buff->state_size * sizeof(float));

    buff->rewards = realloc(buff->rewards, buff->capacity * sizeof(float));

    buff->actions_idx =
        realloc(buff->actions_idx, buff->capacity * sizeof(int));

    buff->should_discount =
        realloc(buff->should_discount, buff->capacity * sizeof(int));
    buff->done = realloc(buff->done, buff->capacity * sizeof(int));
  }

  for (int i = 0; i < buff->state_size; i++) {
    buff->states[buff->size * buff->state_size + i] = state[i];
  }

  buff->rewards[buff->size] = reward;
  buff->should_discount[buff->size] = should_discount;

  buff->actions_idx[buff->size] = action_idx;
  buff->done[buff->size] = done;

  buff->size += 1;
}

int nn_biggest_output_idx(Layer *output_layer) {
  int idx = 0;
  float max = output_layer->outputs[0];
  for (int n = 0; n < output_layer->neuron_count; n++) {
    if (output_layer->outputs[n] > max) {
      max = output_layer->outputs[n];
      idx = n;
    }
  }
  return idx;
}

void nn_network_test(Network *net, float *x, float *y,
                     int samples) { // works on softmax output only for now
  Layer *output_layer = net->output_layer;

  int correct_predictions = 0;

  for (int i = 0; i < samples; i++) {

    nn_network_forward(net, &x[i * net->input_size]);

    int correct_o = _max_arr(&y[i * output_layer->neuron_count],
                             output_layer->neuron_count);
    int predicted_o =
        _max_arr(output_layer->outputs, output_layer->neuron_count);

    if (correct_o == predicted_o) {
      correct_predictions++;
    }
  }
  printf("\nThe model was correct %d/%d times. With accuracy %f%%\n",
         correct_predictions, samples,
         ((float)correct_predictions / (float)samples) * 100.0f);
}

float *nn_network_forward(Network *net, float *x) {
  int x_size = net->input_size;
  if (x_size != net->layers[0].neuron_count) {
    printf("Couldn't load the data of size %d into %d input model", x_size,
           net->layers[0].neuron_count);
    exit(1);
  }
  for (int n = 0; n < x_size; n++) {
    net->layers[0].z_values[n] = x[n];
    net->layers[0].outputs[n] = x[n];
  }
  for (int l = 1; l < net->layers_count; l++) {
    Layer *current_l = &net->layers[l];
    Layer *prev_l = &net->layers[l - 1];

    for (int n = 0; n < current_l->neuron_count; n++) {
      float sum = current_l->biases[n];
      for (int prev_n = 0; prev_n < prev_l->neuron_count; prev_n++) {
        float w = current_l->weights[_get_weight_idx(current_l, n, prev_n)];
        sum += prev_l->outputs[prev_n] * w;
      }
      current_l->z_values[n] = sum;
    }
    _layer_activate(current_l);
  }
  return net->output_layer->outputs;
}

void nn_print_network(Network net) {

  Layer *layers = net.layers;

  int layer_count = net.layers_count;
  printf("Network: \n");
  for (int i = 0; i < layer_count; i++) {
    if (layers[i].type == LAYER_INPUT) {
      printf("\t INPUT");
    } else {
      printf("\t DENSE");
    }
    printf(" Layer %d - %d NEURONS: \n", i, layers[i].neuron_count);
    for (int n = 0; n < layers[i].neuron_count; n++) {
      printf(" Neuron %d: bias=%f, input_size=%d, output=%f, z_val=%f, "
             "delta=%f \nweights (%d): ",
             n, layers[i].biases[n], layers[i].tensor, layers[i].outputs[n],
             layers[i].z_values[n], layers[i].deltas[n], layers[i].tensor);
      printf("\t");

      printf("\n");
    }
  }
}

void _layer_activate(Layer *layer) {
  switch (layer->activation) {

  case activation_reLU:
    for (int n = 0; n < layer->neuron_count; n++) {
      layer->outputs[n] = _reLU(layer->z_values[n]);
    }
    break;

  case activation_softmax: {
    float max = layer->z_values[0];

    for (int n = 0; n < layer->neuron_count; n++) {
      layer->outputs[n] = layer->z_values[n];
      if (layer->outputs[n] > max) {
        max = layer->outputs[n];
      }
    }
    float sum_exp = 0;
    for (int n = 0; n < layer->neuron_count; n++) {
      layer->outputs[n] = expf(layer->z_values[n] - max);
      sum_exp += layer->outputs[n];
    }

    for (int n = 0; n < layer->neuron_count; n++) {
      layer->outputs[n] /= sum_exp;
    }
    break;
  }
  case activation_linear:
    for (int n = 0; n < layer->neuron_count; n++) {
      layer->outputs[n] = layer->z_values[n];
    }
    break;
  case activation_sigmoid:
    for (int n = 0; n < layer->neuron_count; n++) {
      layer->outputs[n] = _sigmoid(layer->z_values[n]);
    }
    break;
  case NONE:

    break;
  }
}

int _max_arr(float *arr, int arr_len) {
  float max = arr[0];
  int idx = 0;
  for (int i = 0; i < arr_len; i++) {
    if (arr[i] > max) {
      max = arr[i];
      idx = i;
    };
  }
  return idx;
}

void _layer_counter(Network *net) {
  if (net->layers_capacity == net->layers_count) {
    net->layers_capacity = net->layers_capacity * 2;
    net->layers = realloc(net->layers, net->layers_capacity * sizeof(Layer));
  }
}

void nn_add_layer_dense(Network *net, int neuron_count,
                        ActivationFunction activation) {

  int last_input_size = net->output_layer->neuron_count;
  _layer_counter(net);

  _layer_init(&net->layers[net->layers_count], neuron_count, last_input_size,
              activation, LAYER_DENSE);

  net->layers_count += 1;

  net->output_layer = &net->layers[net->layers_count - 1];
}

void nn_network_set_loss(Network *network, LossFunction loss_f) {
  network->loss_function = loss_f;
}

Network *nn_network_create(int input_size) {
  Network *net = (Network *)malloc(sizeof(Network));
  net->input_size = input_size;
  net->layers = calloc(2, sizeof(Layer));
  net->layers_count = 0;
  net->layers_capacity = 2;
  net->loss_function = HalfMSE;

  _layer_init(&net->layers[net->layers_count], input_size, 1, NONE,
              LAYER_INPUT);
  Layer input_layer = net->layers[net->layers_count];
  net->layers[0] = input_layer;
  net->layers_count += 1;
  net->output_layer = &net->layers[0];

  net->batched = 0;

  return net;
}

void rl_agent_destroy(RLAgent *agent) {
  if (agent->buffers != NULL) {
    for (int i = 0; i < agent->buffers_count; i++) {
      _replaybuff_destroy(&agent->buffers[i]);
    }
    free(agent->buffers);
  }
  free(agent);
}

void _replaybuff_init(ReplayBuffer *buff, int input_size) {
  int capacity = 8;
  buff->size = 0;
  buff->capacity = capacity;
  buff->states = (float *)malloc(sizeof(float) * capacity * input_size);
  buff->rewards = (float *)malloc(sizeof(float) * capacity);
  buff->actions_idx = (int *)malloc(sizeof(int) * capacity);
  buff->should_discount = (int *)malloc(sizeof(int) * capacity);
  buff->state_size = input_size;
  buff->done = (int *)malloc(sizeof(int) * capacity);
}

void _replaybuff_destroy(ReplayBuffer *buff) {
  free(buff->states);
  free(buff->rewards);
  free(buff->actions_idx);
  free(buff->should_discount);
}

void _layer_init(Layer *layer, int neuron_count, int input_size,
                 ActivationFunction activation, LayerType type) {
  layer->activation = activation;
  layer->neuron_count = neuron_count;
  layer->type = type;
  layer->weights_gradient = calloc(neuron_count * input_size, sizeof(float));
  layer->bias_gradient = calloc(neuron_count, sizeof(float));
  layer->z_values = calloc(neuron_count, sizeof(float));
  layer->weights = calloc(neuron_count * input_size, sizeof(float));
  layer->biases = calloc(neuron_count, sizeof(float));
  layer->outputs = calloc(neuron_count, sizeof(float));
  layer->deltas = calloc(neuron_count, sizeof(float));
  layer->errors = calloc(neuron_count, sizeof(float));
  layer->tensor = input_size;
  for (int n = 0; n < neuron_count; n++) {
    for (int i = 0; i < input_size; i++) {
      layer->weights[n * input_size + i] =
          type != LAYER_INPUT
              ? _get_random_weight(layer->activation, input_size)
              : 1;
    }
    layer->biases[n] = 0;
    layer->outputs[n] = 0;
    layer->errors[n] = 0;
  }
  return;
}

float _get_random_weight(ActivationFunction activation, int input_size) {
  float scale;
  switch (activation) {
  case activation_reLU:
    scale = sqrtf(2.0f / input_size);
    break;
  case activation_softmax:
    scale = sqrtf(1.0f / input_size);
    break;
  default:
    scale = 1.0f;
    break;
  }
  return scale * (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f);
}

float _get_neuron_weight(Layer *layer, int n_idx, int w_idx) {
  if (w_idx < 0 || w_idx >= layer->neuron_count) {
    printf("Index out of bounds for neuron weights.\n");
    exit(1);
  }
  return layer->weights[n_idx * layer->tensor + w_idx];
}

int _get_weight_idx(Layer *layer, int neuron_idx, int weight_idx) {
  return layer->tensor * neuron_idx + weight_idx;
}

void nn_network_destroy(Network *net) {

  for (int l = 0; l < net->layers_count; l++) {

    if (net->layers[l].type == LAYER_INPUT)
      continue;

        
    Layer* layer = &net->layers[l];

    free(layer->weights_gradient);
    free(layer->bias_gradient);

    free(layer->weights);
    free(layer->biases);

    free(layer->z_values);
    free(layer->outputs);
    free(layer->deltas);
    free(layer->errors);

  }

  free(net->layers);
  free(net);
}
