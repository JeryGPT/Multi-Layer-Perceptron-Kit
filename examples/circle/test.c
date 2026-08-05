#include "../../MLKit.h"

int main () {
    srand(time(NULL));

    Network* model = nn_network_create(2);
    nn_add_layer_dense(model, 32, activation_reLU);
    nn_add_layer_dense(model, 16, activation_reLU);
    nn_add_layer_dense(model, 1, activation_sigmoid);

    char model_path[] = "./model_snapshots/model.mlk";
    nn_network_import_file(model, model_path);

    Layer* last_l = &model->layers[model->layers_count-1];
    int grid_size = 0;

    while (1) {

        printf("\nInput the size of grid: ");
        scanf("%d", &grid_size);
        printf("\n");

        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x ++) {
                float x_p = (float)x / grid_size * 2 - 1;
                float y_p = (float)y / grid_size * 2 - 1;

                nn_network_forward(model, (float[]){x_p, y_p}, 2);

                float prediction = last_l->neurons[0].output;

                printf("%c ", prediction > 0.5 ? '#' : ' ');
            }
            printf("\n");
        }
    }

    nn_network_destroy(model);

}