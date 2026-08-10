#include "../../MLKit.h"
#include "mnist_manage.h"

void network_train_mnist(Network* network);

#define SAMPLES 60000
int main() {

    //  Create the model
    Network* network = nn_network_create(28*28);

    //  Define the network
    nn_add_layer_dense(network, 64, activation_reLU); 
    nn_add_layer_dense(network, 32, activation_reLU); 
    nn_add_layer_dense(network, 10, activation_softmax);
    nn_network_set_loss(network, CrossEntropy);

    //  Start training
    network_train_mnist(network);

    nn_network_destroy(network);


    return 0;
}



void network_train_mnist(Network* network) {

    //  Generate training data 
    float* x_train = malloc(60000 * 28 * 28 * sizeof(float));
    float* y_train = calloc(60000 * 10, sizeof(float));

    char path[]             = "./dataset/";
    char filename[]         = "train-images-idx3-ubyte";
    char filename_labels[]  = "train-labels-idx1-ubyte";

    unsigned char* x_train_c = get_mnist_dataset(path, filename);
    unsigned char* y_train_c = get_mnist_labels(path, filename_labels);

    
    for (int i = 0; i < 60000 * 28 * 28; i++) {
        x_train[i] = (float)x_train_c[i + 16] / 255.0f;
    }
    

    for (int i = 0; i < 60000; i++) {
        int correct_label = y_train_c[i + 8]; // 8 bytes header
        y_train[i * 10 + correct_label] = 1.0f; 
    }

    //  Start the training
    nn_network_fit(network, x_train, SAMPLES, y_train, 0.08, 10, 32, 1);

    //  Save the final model 
    char filepath[] = "./model_snapshots/mnist_model.mlk";
    
    nn_network_export(network, filepath); // export weights and biases

    //  Free the training data memory
    free(x_train_c); 
    free(y_train_c);
    free(x_train);
    free(y_train);
}