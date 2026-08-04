#include<stdio.h>
#include<stdlib.h>
#include<string.h>

char* get_mnist_dataset(char* path, char* file_name) {
    size_t path_len = strlen(path);
    size_t name_len = strlen(file_name);
    char* full_path = malloc(path_len + name_len + 1);
    if (!full_path) {
        perror("Malloc failed");
        exit(1);
    }

    strcpy(full_path, path);
    strcat(full_path, file_name);

    FILE* ptr = fopen(full_path, "rb");


    free(full_path);
    if (!ptr) {
        printf("\nThe %s file not found!\n", file_name);
        exit(1);
    }


    fseek(ptr,0L, SEEK_END);
    size_t data_size = ftell(ptr);
    char* dataset = malloc(data_size);
    rewind(ptr);
    fread(dataset, 1, data_size, ptr);

    fclose(ptr);

    return dataset;
}


unsigned char* get_mnist_labels(char* path, char* file_name) {
    size_t path_len = strlen(path);
    size_t name_len = strlen(file_name);
    char* full_path = malloc(path_len + name_len + 1);
    if (!full_path) {
        perror("Malloc failed");
        exit(1);
    }

    strcpy(full_path, path);
    strcat(full_path, file_name);
    FILE* file = fopen(full_path, "rb");
    if (!file) {
        printf("Failed to open the %s file\n", file_name);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);


    unsigned char* labels = malloc(file_size);
    fread(labels, 1, file_size, file);
    free(full_path);
    fclose(file);

    return labels;
}



