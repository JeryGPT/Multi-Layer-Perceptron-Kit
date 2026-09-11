#ifndef MLKIT_H
#define MLKIT_H
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

// ENUMS

typedef enum {
  LAYER_INPUT,
  LAYER_DENSE
} LayerType;

typedef enum {
    activation_reLU,
    activation_linear,
    activation_softmax,
    activation_sigmoid,
    NONE
} ActivationFunction;

typedef enum {
    HalfMSE,
    CrossEntropy
} LossFunction;

// STRUCTS



typedef struct {
    int neuron_count;
    float* weights_gradient;
    float* bias_gradient;
    float* z_values;
    float* biases;
    float* deltas;
    float* weights;
    float* outputs;
    float* errors;
    int tensor;
    ActivationFunction activation;
    LayerType type;
} Layer;

typedef struct {
    float* states;
    int* actions_idx;
    float* rewards; 
    int* should_discount;
    int state_size;
    int* done;
    
    int size;
    int capacity;
} ReplayBuffer;

typedef struct {
    Layer* layers;
    Layer* output_layer;
    int layers_count;
    int layers_capacity;
    int input_size;
    int batched;
    LossFunction loss_function;
    int episodes_batched;
} Network;

typedef struct {
    Network* network;
    ReplayBuffer* buffers;
    int buffers_count;
} RLAgent;

// Q-LEARNING


// API

// Managing the network struct

Network*    nn_network_create(int input_size);
void        nn_network_destroy(Network* network);
void        nn_add_layer_dense(Network* network, int input_size, ActivationFunction activation);
void        nn_network_set_loss(Network* network, LossFunction loss_f);

// Propagation 

float*        nn_network_forward(Network* net, float* x);
void        nn_network_backward(Network* net, float* y);


// Training and Testing

void        nn_network_fit(Network* net, float* x, int data_size, float* y, float lr, int epochs,int batch,  int print_steps);
void        nn_network_print(Network network);
void        nn_network_test(Network* net, float* x, float* y, int samples); // works on softmax output only for now
void        nn_apply_gradients(Network* net, float lr);


// Reinforcement Learning 

void        rl_policy_fit(RLAgent* agent, int* buffer_ids, int buffers_count, float lr, float gamma);

void        _agent_add_replaybuffers(RLAgent* agent, int agent_count);
RLAgent*    rl_agent_create(Network* model, int agent_count);


void        rl_remember(ReplayBuffer* buff, float* state, float reward, int action_idx, int should_discount, int done);
void        rl_replaybuff_clear(ReplayBuffer* buff);
int         rl_draw_action(Layer* output_l);
void        rl_agent_destroy(RLAgent* agent);



//Network import and export 

void        nn_network_export(Network* net, char* filepath);
void        nn_network_import(Network* net, float* buffer, int size);
int        nn_network_import_file(Network* net, char* filepath);

// Helpers 

int       nn_biggest_output_idx(Layer* output_l);

#endif