#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<math.h>
#include "MLKit.h"




Network* nn_network_create(int input_size);

void nn_network_destroy(Network* network);

void nn_add_layer_dense(Network* network, int input_size, ActivationFunction activation);

void nn_network_print(Network network);
void nn_network_forward(Network* net, float* x);
void nn_network_backward(Network* net, float* y, float lr);
void nn_network_fit(Network* net, float* x, int data_size, float* y, float lr, int epochs, int print_steps);

static void _layer_activate(Layer* layer);
static int _max_arr(float* arr, int arr_len);

static void _layer_init(Layer* layer, int neuron_count, int input_size, ActivationFunction activation, LayerType type);
static void _neuron_init(Neuron* neuron, int input_size, ActivationFunction activation);

// ACTIVATORS
float _reLU(float x) {
    return 0 > x ? 0 : x;
}

float _sigmoid(float x) {
    return 1/ (1 + expf(-x));
}

float _get_derivative(Layer* output_layer, int neuron_idx) {
    ActivationFunction activation = output_layer->activation;
    float z_val = output_layer->neurons[neuron_idx].z_val;
    float a = output_layer->neurons[neuron_idx].output;
    switch (activation)
    {
    case activation_linear:
        return 1.0;
        break;
    case activation_reLU:
        return z_val < 0 ? 0.0 : 1.0;
        break;

    case activation_sigmoid:
        float s = _sigmoid(z_val);
        return s * (1.0 -s);
    case activation_softmax:
        
    default:
        return 1.0;
    }
}

float _loss_value(LossFunction loss_f, float output, float y)  { // per/neuron calculation
    switch (loss_f) {
        case HalfMSE:
            float error = output - y;
            return 0.5 * error * error;
        case CrossEntropy: {
            float epsilon = 1e-7f;
            if (output < epsilon) output = epsilon;
            return - y * logf(output);
            break;
        }
    }
}

float _loss_deriv_value(LossFunction loss_f, float output, float y) { // per/neuron derivative
    switch (loss_f) {
        case HalfMSE: {
            return output - y;
        }
        case CrossEntropy: {
            float epsilon = 1e-7f;
            if (output < epsilon) output = epsilon;
            if (output > 1.0f - epsilon) output = 1.0f - epsilon;
            return -y/output + (1-y)/(1-output);
        }
    }

}

static void _softmax_output_delta(Layer* output_l, LossFunction loss_f){
    int n_count = output_l->neuron_count;

    float dot = 0.0f;

    for (int j = 0; j < n_count; j++) {
        dot += output_l->neurons[j].error * output_l->neurons[j].output;
    }

    for (int i = 0; i < n_count; i++) {
        float o_i = output_l->neurons[i].output;
        output_l->neurons[i].delta = o_i * (output_l->neurons[i].error - dot);
    }
}

void _calculate_output_delta(Network* net, float* y){
    LossFunction loss_f = net->loss_function;
    Layer* output_l = &net->layers[net->layers_count-1];
    ActivationFunction activation = output_l->activation;
    switch (loss_f) {
        case (HalfMSE): {
            if (activation == activation_softmax) {
                for (int n = 0; n < output_l->neuron_count; n++) {
                    Neuron* neuron = &output_l->neurons[n];
                    float error = _loss_deriv_value(loss_f, neuron->output, y[n]);
                    neuron->error = error;
                }
                _softmax_output_delta(output_l, loss_f);
            }else{
              
                for (int n = 0; n < output_l->neuron_count; n++) {
                    Neuron* neuron = &output_l->neurons[n];
                    float error = _loss_deriv_value(loss_f, neuron->output, y[n]);
                    float z_deriv = _get_derivative(output_l, n);
                    neuron->delta = error * z_deriv;
                }
            }
            break;
        }
        case (CrossEntropy) : {
            if (activation == activation_softmax || activation == activation_sigmoid){
                for (int n = 0; n < output_l->neuron_count; n++) {
                    Neuron* neuron = &output_l->neurons[n];
                    neuron->delta = output_l->neurons[n].output - y[n];
                }
            }else {
                for (int n = 0; n < output_l->neuron_count; n++) {
                    Neuron* neuron = &output_l->neurons[n];
                    float error = _loss_deriv_value(loss_f, neuron->output, y[n]);
                    neuron->delta = error * _get_derivative(output_l, n);
                }
            }
        }
    }
}


float nn_calculate_loss(Network* net, float* x, float* y, int data_size) {
    float total_loss = 0.0f;
    int input_size = net->input_size;
    int output_size = net->layers[net->layers_count-1].neuron_count;

    Layer* output_layer = &net->layers[net->layers_count-1];
    for (int i = 0; i < data_size; i ++) {
        float sample_loss = 0;
        nn_network_forward(net, &x[i * input_size]);
        for (int n = 0; n < output_size; n++) {
            sample_loss += _loss_value(net->loss_function, output_layer->neurons[n].output, y[i * output_size + n]);
        } 
        total_loss += sample_loss/output_size;
    }   
    total_loss /= (float)data_size;
    return total_loss;
}

void nn_network_fit(Network* net, float* x, int data_size, float* y, float lr, int epochs, int print_steps) {
    int input_size = net->input_size;
    int output_size = net->layers[net->layers_count-1].neuron_count;
    for (int epoch = 0; epoch < epochs; epoch ++) {

        for (int i = 0; i < data_size; i++) {
            if (i % ((data_size + 10 )/ 10) == 0 || i == data_size - 1) {
                int progress = (i * 10) / data_size;
                if (i == data_size - 1) progress = 10;
                
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
            nn_network_forward(net, &x[i*input_size]);
            nn_network_backward(net, &y[i*output_size], lr);


        }
        
        if (epoch % print_steps == 0) {
            printf(" Loss: %f \n", 
            nn_calculate_loss(
                net, 
                x,
                y, 
                (int)(data_size/10 > 0 ? data_size/10 : data_size)
            )
            );  
        }

    }

    printf("Finished training %d, %d epochs \n", epochs, epochs);
}

void nn_network_backward(Network* net, float* y, float lr) {
    int layers_count = net->layers_count;

    _calculate_output_delta(net, y);

    for (int cr = layers_count -2; cr > 0; cr--){
        Layer* current_l = &net->layers[cr];
        Layer* next_l = &net->layers[cr+1];
        for (int n = 0; n < current_l->neuron_count; n++) {
            float error_sum = 0.0f;
            Neuron* current_neuron = &current_l->neurons[n];

            for (int next_n = 0; next_n < next_l->neuron_count; next_n++) {
                Neuron* next_neuron = &next_l->neurons[next_n];
                float next_neuron_delta = next_neuron->delta;
                error_sum += next_neuron_delta * next_neuron->weights[n];
            }
            float deriv = _get_derivative(current_l, n);
            current_neuron->delta = error_sum * deriv;
                
        }
        
    } 

    for (int l = 1; l < layers_count; l++) {
        Layer* layer = &net->layers[l];
        Layer* prev_layer = &net->layers[l-1];
        for (int n = 0; n < layer->neuron_count; n++) {
            Neuron* neuron = &layer->neurons[n];
            for (int w = 0; w < neuron->input_size; w++) {
                neuron->weights[w] -= lr * neuron->delta * prev_layer->neurons[w].output;
            }
            neuron->bias -= lr * neuron->delta;
        }
    }

    return;
}

void nn_network_export(Network* net, char* filepath){
    FILE* fptr = fopen(filepath, "wb");

    fclose(fptr);

    
    fptr = fopen(filepath, "ab");
    
    
    if (!fptr) {
        printf("Failed to export the model.");
        
    }
    int total_weights = 0;
    for (int l = 1; l < net->layers_count; l++){
        Layer* layer = &net->layers[l];
        total_weights += layer->neuron_count * layer->neurons[0].input_size;
        for (int n = 0; n < layer->neuron_count; n++){
            Neuron* neuron = &layer->neurons[n];
            float bias = neuron->bias;
            fwrite(&bias, sizeof(float), 1, fptr);
            total_weights++;
            for (int w = 0; w < neuron->input_size; w++){
                float weight = neuron->weights[w];
                fwrite(&weight, sizeof(float),1,fptr);
            }
        }
    }
    printf("Saved a total of %d parameters to %s path\n", total_weights, filepath);
    fclose(fptr);
}

void nn_network_import(Network* net, float* buffer, int size) {
    int current_w = 0;
    for (int l = 1; l < net->layers_count; l++){
        Layer* layer = &net->layers[l];
        for (int n = 0; n < layer->neuron_count; n++){
            Neuron* neuron = &layer->neurons[n];
            neuron->bias=buffer[current_w];
            current_w++;
            for (int w = 0; w < neuron->input_size; w++){
                neuron->weights[w]=buffer[current_w];
                current_w++;
            }

        }
    }
    if (current_w != (int)(size  - sizeof(float))) {
        printf("Import buffer and network size mismatch");
    }else{
        printf("Loaded %d weights", (int)(size/sizeof(float) - 1));
    }
}

void nn_network_import_file(Network* net, char* filepath) {

    FILE* fptr = fopen(filepath, "rb");
    if (!fptr) {
        printf("Failed to open the import file.");
        exit(1);
    }

    fseek(fptr, 0L, SEEK_END);
    size_t f_size = ftell(fptr);
    rewind(fptr);

    float* buffer = malloc(f_size);
    if (!buffer) {
        printf("Failed to allocate memory for the import buffer.");
        fclose(fptr);
        exit(1);
    }

    fread(buffer, sizeof(float), f_size/sizeof(float), fptr);

    fclose(fptr);




    int current_w = 0;
    for (int l = 1; l < net->layers_count; l++){
        Layer* layer = &net->layers[l];
        for (int n = 0; n < layer->neuron_count; n++){
            Neuron* neuron = &layer->neurons[n];
            neuron->bias=buffer[current_w];
            current_w++;
            for (int w = 0; w < neuron->input_size; w++){
                neuron->weights[w]=buffer[current_w];
                current_w++;
            }
        }
    }

    printf("Loaded %d parameters", (int)(f_size/sizeof(float)));
    free(buffer);

}


int _biggest_output(Layer* output_layer) {
    int idx = 0;
    float biggest = output_layer->neurons[0].output;
    for (int n = 0; n < output_layer->neuron_count; n++) {
        Neuron* neuron = &output_layer->neurons[n];
        if (neuron->output > biggest) {
            biggest = neuron->output;
            idx = n;
        }
    }
    return idx;
}

void nn_network_test(Network* net, float* x, float* y, int samples) { // works on softmax output only for now
    Layer* output_layer = &net->layers[net->layers_count-1];

    Neuron* output_neurons = net->layers[net->layers_count-1].neurons;

    int correct_predictions = 0;

    for (int i = 0; i < samples; i++) {

        nn_network_forward(net, &x[i*net->input_size]);

        int correct_o = _max_arr(&y[i*output_layer->neuron_count],output_layer->neuron_count);
        int predicted_o = _biggest_output(output_layer);

        if (correct_o == predicted_o ) {
            correct_predictions ++ ;
        }

    }
    printf("\nThe model was correct %d/%d times. With accuracy %f%%\n",correct_predictions, samples, ((float)correct_predictions / (float)samples) * 100.0f);


}

void nn_network_forward(Network* net, float* x) {
    int x_size = net->input_size;
    if (x_size != net->layers[0].neuron_count) {
        printf("Couldn't load the data of size %d into %d input model", x_size, net->layers[0].neuron_count);
        exit(1);
    }
    for (int n = 0; n <x_size; n++){
        net->layers[0].neurons[n].z_val = x[n];
        net->layers[0].neurons[n].output = x[n];
    }
    for (int l = 1; l < net->layers_count; l++) {
        Layer* current_l    = &net->layers[l];
        Layer* prev_l       = &net->layers[l-1];


        for (int n = 0; n < current_l->neuron_count; n++) {
            float sum = current_l->neurons[n].bias;
            for (int prev_n = 0; prev_n < prev_l->neuron_count; prev_n ++ ){
                float w  = current_l->neurons[n].weights[prev_n];
                sum     += prev_l->neurons[prev_n].output * w;
            }
            current_l->neurons[n].z_val = sum;

        }
        _layer_activate(current_l);

    }



}

void nn_print_network(Network net) {

    Layer* layers = net.layers;

    int layer_count = net.layers_count;
    printf("Network: \n");
    for (int i = 0; i < layer_count ; i++) {
        if (layers[i].type == INPUT) {
            printf("\t INPUT");
        }else{
            printf("\t DENSE");
            
        }
        printf(" Layer %d - %d NEURONS: \n", i, layers[i].neuron_count);
        for (int n = 0; n < layers[i].neuron_count; n++) {
            printf(" Neuron %d: bias=%f, input_size=%d, output=%f, z_val=%f, delta=%f \nweights (%d): ", 
                n, 
                layers[i].neurons[n].bias, 
                layers[i].neurons[n].input_size,
                layers[i].neurons[n].output, 
                layers[i].neurons[n].z_val,layers[i].neurons[n].delta, 
                layers[i].neurons[n].input_size);
            printf("\t");


            printf("\n");
        }
    }
}



void _layer_activate(Layer* layer) {
    switch (layer->activation) {

        case activation_reLU:
            for (int n =0; n < layer->neuron_count; n++) {
                layer->neurons[n].output = _reLU(layer->neurons[n].z_val);
            }
            break;
        
        case activation_softmax:
            float max = layer->neurons[0].z_val;
            for (int n = 0; n < layer->neuron_count; n++) {
                layer->neurons[n].output = layer->neurons[n].z_val;
                if (layer->neurons[n].z_val > max) {
                    max = layer->neurons[n].output;
                }
            }
            float sum_exp = 0;
            for (int n = 0; n < layer->neuron_count; n++) {
                layer->neurons[n].output = expf(layer->neurons[n].z_val - max);
                sum_exp                 += layer->neurons[n].output;
            }


            for (int n = 0; n < layer->neuron_count; n++) {
                layer->neurons[n].output /= sum_exp;
            }
            break;
        case activation_linear:
            for (int n = 0; n < layer->neuron_count; n++) {
                layer->neurons[n].output = layer->neurons[n].z_val;
            }
            break;
        case activation_sigmoid:
            for (int n =0; n < layer->neuron_count; n++) {
                layer->neurons[n].output = _sigmoid(layer->neurons[n].z_val);
            }
            break;
        case NONE:
            break;


            
    }
}


int _max_arr(float* arr, int arr_len) {
    float max = arr[0];
    int idx = 0;
    for (int i =0; i < arr_len; i++) {
        if (arr[i] > max) {
            max = arr[i];
            idx = i;
        };
    }
    return idx;
} 


void _layer_counter(Network* net) {
    if ( net->layers_capacity == net->layers_count) {
        net->layers_capacity    = net->layers_capacity * 2;
        net->layers             = realloc(net->layers, net->layers_capacity * sizeof(Layer));
    }
}

void nn_add_layer_dense(Network* net, int neurons_count, ActivationFunction activation) {

    int last_input_size = net->layers[net->layers_count-1].neuron_count;
    _layer_counter(net);
    

    _layer_init(&net->layers[net->layers_count], neurons_count, last_input_size, activation, DENSE);
    net->layers_count += 1;
}

void nn_network_set_loss(Network* network, LossFunction loss_f) {
    network->loss_function = loss_f;
}


Network* nn_network_create(int input_size) {
    Network* net            = (Network*)malloc(sizeof(Network));
    net->input_size         = input_size;
    net->layers             = calloc(2, sizeof(Layer));
    net->layers_count       = 0;
    net->layers_capacity    = 2;
    net->loss_function      = HalfMSE;

    Layer input_layer;
    input_layer.activation    = NONE;
    input_layer.neuron_count  = input_size;
    input_layer.type          = INPUT;
    input_layer.neurons       = malloc(input_size * sizeof(Neuron));
    for (int n = 0; n < input_size; n++) {
        _neuron_init(&input_layer.neurons[n], 1, NONE);
    }
    net->layers[0]      = input_layer;
    net->layers_count   += 1;
    

    return net;
}

void _layer_init(Layer* layer, int neuron_count, int input_size, ActivationFunction activation, LayerType type) {
    layer->activation   = activation;
    layer->neuron_count = neuron_count;
    layer->neurons      = calloc(layer->neuron_count, sizeof(Neuron));
    layer->type         = type;

    for (int n = 0; n < neuron_count; n++) {
        _neuron_init(&layer->neurons[n], input_size, layer->activation);
    }
    return;
}

void _neuron_init(Neuron* neuron, int input_size, ActivationFunction activation) {
    neuron->weights     = calloc(input_size, sizeof(float));
    neuron->bias        = 0;
    neuron->input_size  = input_size;
    neuron->delta       = 0;
    neuron->z_val       = 0;
    neuron->output      = 0;
    neuron->error       = 0;

    float scale = 1.0f;
    switch (activation){
        case activation_reLU:
            scale = sqrtf(2.0f / input_size); 
            break;
            
    }

    for (int w = 0; w < input_size; w++) {
        neuron->weights[w] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale; // random weights between -1 and 1 scaled by the activation function
    }
    
    return;
}

void    nn_network_destroy(Network* net) {
    
    for (int l = 0; l < net->layers_count; l ++) {

        for (int n = 0; n < net->layers[l].neuron_count; n++) {
            free(net->layers[l].neurons[n].weights);
        }

        free(net->layers[l].neurons);

    }
    free(net->layers);
    free(net);
}

