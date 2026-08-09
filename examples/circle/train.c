#include "../../MLKit.h"

#define TRAINING_SAMPLES 3000
#define RADIUS 1.0f

int main () {
    srand(1);

    Network* model = nn_network_create(2);
    nn_add_layer_dense(model, 32, activation_reLU);
    nn_add_layer_dense(model, 16, activation_reLU);
    nn_add_layer_dense(model, 1, activation_sigmoid);
    nn_network_set_loss(model, HalfMSE);
    

    float* x_data = calloc(TRAINING_SAMPLES * 2, sizeof(float));
    float* y_data = calloc(TRAINING_SAMPLES, sizeof(float));

    // [-1, 1] max circle's edge range
    for (int i = 0; i < TRAINING_SAMPLES; i++) {
        float x = (float)rand()/RAND_MAX * 2 -1;
        float y = (float)rand()/RAND_MAX * 2 -1;

        float is_in_circle = x*x + y*y <= RADIUS * RADIUS;

        x_data[i*2]     = x;
        x_data[i*2 + 1] = y;
        y_data[i]        = is_in_circle;
    }

    nn_network_fit(model, x_data, TRAINING_SAMPLES, y_data, 0.001, 300, 25);
    char export_path[] = "./model_snapshots/model.mlk";
    nn_network_export(model, export_path);

    free(x_data);
    free(y_data);
    nn_network_destroy(model);

    // honestly this project and code is a mess, but it works and I learned a lot from it. I hope you enjoy it too.
}