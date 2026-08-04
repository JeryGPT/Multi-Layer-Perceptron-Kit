#include<stdio.h>
#include<stdlib.h>
#include "../../MLKit.h"


int main() {
    srand(time(NULL));
    int TRAIN_SAMPLES = 1000;
    float* x_data = calloc(TRAIN_SAMPLES * 2, sizeof(float));
    float* y_data = calloc(TRAIN_SAMPLES, sizeof(float));

    for (int i = 0 ;i < TRAIN_SAMPLES; i++) {
        x_data[i*2] = (i) ;
        x_data[i*2 + 1] = 2 ;
        y_data[i] = (i) * x_data[i*2 + 1] ;
    }

    Network* model = nn_network_create(2);
    nn_add_layer_dense(model, 32, activation_reLU);
    nn_add_layer_dense(model, 1, activation_linear);

    nn_network_fit(model, x_data, TRAIN_SAMPLES, y_data, 0.0000003, 300, 5);

    while(1) {
        printf("Input the number to test model on: ");
        float input = 2;
        scanf("%f", &input);
        float x_data[2] = {input, 2.0};
        Layer* output_l = &model->layers[model->layers_count-1];
        nn_network_forward(model, &input, 2);
        printf("Model prediction: %f * 2 = %f",input , output_l->neurons[0].output );
    }
    



    return 0;
}