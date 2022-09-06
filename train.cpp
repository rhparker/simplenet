#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "classifier.h"

int main(int argc, char* argv[]) {

  //
  // load MNIST data
  //
  mnist_data *train_data;
  unsigned int cnt;
  int result;

  // load MNIST data using external routine
  // data is loaded as an array of structs, each struct has image and label
  if (result = mnist_load("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", &train_data, &cnt)) {
    printf("An error occured: %d\n\n", result);
  } 
  else {
    printf("training image count: %d\n\n", cnt);
  }

  // put loaded data into a more standard format for training
  // one array for labels, one array for images
  // for data, just creates a new array of pointers, which point to the loaded data
  double **data = new double*[cnt];
  unsigned int *labels = new unsigned int[cnt];
  int input_size = 784;
  int output_size = 10;
  for (int i = 0; i < cnt; i++) {
    labels[i] = train_data[i].label;
    data[i] = train_data[i].data;
  }

  //
  // initialize network
  // 

  // // no hidden layers
  // int layer_sizes[2] = {input_size,output_size};
  // Classifier C(2,layer_sizes,0.1);

  // one hidden layer
  int layer_sizes[3] = {input_size,64,output_size};
  Classifier C(3,layer_sizes,0.1);

  // print network properties
  C.properties();


  //
  // run training epochs
  // 

  // initial accuracy and cross entropy (pre-training)
  std::cout << "TRAINING" << std::endl;
  std::cout << "epoch  accuracy  cross-entropy loss" << std::endl;
  std::cout << 0 << "    " << C.compute_loss(cnt, data, labels) << "    " << C.train_loss << std::endl;

  int epochs = 5;
  int batch_size = 10;
  double learning_rate = 0.1;
  for (int i = 1; i <= epochs; i++) {
    C.train_epoch(cnt, data, labels, learning_rate, batch_size);
    std::cout << i << "    " << C.compute_loss(cnt, data, labels) << "   " << C.train_loss << std::endl;
  }

  // unallocate training data
  free(train_data);
  delete[] data;
  delete[] labels;

  //
  // run everything on test data
  //

  mnist_data *t10k_data;
  unsigned int t10k_cnt;

  // load MNIST data using external routine
  // data is loaded as an array of structs, each struct has image and label
  std::cout << std::endl << "TEST" << std::endl;
  if (result = mnist_load("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte", &t10k_data, &t10k_cnt)) {
    printf("An error occured: %d\n\n", result);
  } 
  else {
    printf("test image count: %d\n", t10k_cnt);
  }

  double **test_data = new double*[t10k_cnt];
  unsigned int *test_labels = new unsigned int[t10k_cnt];
  for (int i = 0; i < t10k_cnt; i++) {
    test_labels[i] = t10k_data[i].label;
    test_data[i] = t10k_data[i].data;
  }

  std::cout << "accuracy: " << C.compute_loss(t10k_cnt, test_data, test_labels) << std::endl << std::endl;

  // unallocate test data
  free(t10k_data);
  delete[] test_data;
  delete[] test_labels;

  return 0;
}

