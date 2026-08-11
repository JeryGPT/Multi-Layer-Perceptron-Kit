#ifndef MLKIT_H
#define MLKIT_H
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

// ENUMS

typedef enum {
  INPUT,
  DENSE
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
    float* weights;
    float bias;
    int input_size;
    float output;
    float z_val;
    float delta;
    float error;
    float* weights_gradient;
    float bias_gradient;
} Neuron;


typedef struct {
    Neuron* neurons;
    int neuron_count;
    ActivationFunction activation;
    LayerType type;

} Layer;

typedef struct {
    Layer* layers;
    Layer* output_layer;
    int layers_count;
    int layers_capacity;
    int input_size;
    LossFunction loss_function;
} Network;


// API

// Managing the network struct

Network*    nn_network_create(int input_size);
void        nn_network_destroy(Network* network);
void        nn_add_layer_dense(Network* network, int input_size, ActivationFunction activation);
void        nn_network_set_loss(Network* network, LossFunction loss_f);

// Propagation 

void        nn_network_forward(Network* net, float* x);
void        nn_network_backward(Network* net, float* y);

// Training and Testing

void        nn_network_fit(Network* net, float* x, int data_size, float* y, float lr, int epochs,int batch,  int print_steps);
void        nn_network_print(Network network);
void        nn_network_test(Network* net, float* x, float* y, int samples); // works on softmax output only for now


//Network import and export 

void        nn_network_export(Network* net, char* filepath);
void        nn_network_import(Network* net, float* buffer, int size);
void        nn_network_import_file(Network* net, char* filepath);

#endif