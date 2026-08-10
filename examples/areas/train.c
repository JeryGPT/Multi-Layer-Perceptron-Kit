#include "../../MLKit.h"

#define TRAIN_SAMPLE 30000

float arr_max(float* arr, int arr_size) {
    float max_val = arr[0];
    for (int i = 0; i < arr_size; i++) {
        if (max_val < arr[i]){
            max_val = arr[i];
        }
    }
    return max_val;
}

int main () {

    srand(time(NULL));

    Network* model = nn_network_create(3);
    nn_add_layer_dense(model , 128, activation_reLU);
    nn_add_layer_dense(model , 64, activation_reLU);
    nn_add_layer_dense(model , 32, activation_reLU);
    nn_add_layer_dense(model , 10, activation_reLU);
    nn_add_layer_dense(model , 1, activation_linear);

    float* x_train = calloc(TRAIN_SAMPLE * model->input_size, sizeof(float));
    float* y_train = calloc(TRAIN_SAMPLE, sizeof(float));

    for (int i = 0; i < TRAIN_SAMPLE; i++) {
        float a = (float)rand() / RAND_MAX;
        float b = (float)rand() / RAND_MAX;
        float c = (float)rand() / RAND_MAX;
        int idx = i * model->input_size;
        x_train[idx] = a;
        x_train[idx + 1] = b;
        x_train[idx + 2] = c;
        y_train[i] = c > 0.5 ? a * b / 2 : a * b; // triangle
        
    }
    nn_network_fit(model, x_train, TRAIN_SAMPLE, y_train, 0.1, 70, 32, 1);
    nn_network_export(model, "./model_snapshots/model.mlk");

    Layer* last_l = &model->layers[model->layers_count-1];;



    while (1) {
        float a = 75;
        float b = 22;
        float figure = 1;
        printf("\nInput a: ");
        scanf("%f", &a);
        printf("\nInput h: ");
        scanf("%f", &b);
        printf("\n Figuer (>0.5 triangle 0.5< square): ");
        scanf("%f", &figure);
        float max_val = arr_max((float[]){a,b}, 2);
        float tens_a = pow(10, (int)log10(a) +1);
        float tens_b = pow(10, (int)log10(b) +1);
        a/=tens_a;
        b/=tens_b;
        nn_network_forward(model, (float[]){a,b, figure}, model->input_size);
        float prediction = last_l->neurons[0].output;
        float expected = figure > 0.5 ? (a*tens_a*b*tens_b/2) : (a*tens_a*b*tens_b);
        printf("\n The area of %s : a %f, b %f= %f, %f", figure  > 0.5 ? "triangle" : "square", a*tens_a, b*tens_b, prediction*tens_a*tens_b,expected);
    }





    return 0;
}