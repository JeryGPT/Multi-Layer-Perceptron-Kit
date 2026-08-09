#include "../../MLKit.h"
#include "mnist_manage.h"

void    network_test_mnist(Network* network);
void    use_network(Network* network, float* x, float* y, int dataset_size);

int main () {
    srand(time(NULL));
    //  Create the model
    Network* network = nn_network_create(28*28); 

    //  Define the network
    nn_add_layer_dense(network, 64, activation_reLU); 
    nn_add_layer_dense(network, 32, activation_reLU); 
    nn_add_layer_dense(network, 10, activation_softmax);
    nn_network_set_loss(network, CrossEntropy);

    //  Import weights and biases from saved model snapshhot
    char model_snapshot_path[] = "./model_snapshots/mnist_model.mlk";
    nn_network_import_file(network, model_snapshot_path); // import weights and biases

    network_test_mnist(network); // run the test

    nn_network_destroy(network); // free the malloc
}


int _arr_max(float* arr, int size) {
    int idx = 0;
    float max = 0;
    for (int i = 0; i < size; i++){
        if (arr[i] > max) {
            max = arr[i];
            idx = i;
        }
    }
    return idx;
}

void use_network(Network* network, float* x, float* y, int dataset_size) {

    Layer* output_layer     = &network->layers[network->layers_count-1];
    Neuron* output_neurons  = output_layer->neurons;

    while (1) {
        printf("Select sample from test dataset to check on: ");
        int sample = 0;
        scanf("%d", &sample);

        nn_network_forward(network, &x[sample*network->input_size], network->input_size);

        int sample_data = _arr_max(&y[sample*output_layer->neuron_count], output_layer->neuron_count);
        int prediction = 0;
        float max_output = 0;
        for (int n = 0; n < output_layer->neuron_count; n++){
            Neuron current_n = output_neurons[n];
            if (current_n.output > max_output) {
                max_output = current_n.output;
                prediction = n;
            }
        }
        printf("Sample %d: correct label: %d, prediction: %d (%f%%)\n", sample, sample_data, prediction, (float)max_output*100.0f);
    }
}

void network_test_mnist(Network* network) {

    //  Generate test data
    float* x_test = malloc(10000 * 28 * 28 * sizeof(float));
    float* y_test = calloc(10000 * 10, sizeof(float));

    char path[]             = "./dataset/";
    char filename[]         = "t10k-images-idx3-ubyte";
    char filename_labels[]  = "t10k-labels-idx1-ubyte";

    unsigned char* x_test_c = get_mnist_dataset(path, filename);
    unsigned char* y_test_c = get_mnist_labels(path, filename_labels);

    for (int i = 0; i < 10000 * 28 * 28; i++) {
        x_test[i] = (float)x_test_c[i + 16] / 255.0f;
    }


    for (int i = 0; i < 10000; i++) {
        int correct_label = y_test_c[i + 8]; // skip the header
        y_test[i * 10 + correct_label] = 1.0f; 
    }

    //  Run the test
    nn_network_test(network, x_test, y_test, 10000);
    use_network(network, x_test, y_test, 10000); // Use the network on samples


    //  Free the dataset
    free(x_test);
    free(y_test);

    free(x_test_c);
    free(y_test_c);
}