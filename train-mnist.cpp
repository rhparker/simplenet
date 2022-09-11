// trains classifier neural network on MNIST data

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "classifier.h"
#include "loadmnist.h"

int main(int argc, char* argv[]) {

  //
  // load MNIST data
  //

  int input_size = 784;
  int output_size = 10;
  unsigned int cnt;

  // arrays to be allocated for data and labels
  double** data;
  unsigned int* labels;

  // load MNIST training data
  cnt = mnist_load("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", data, labels);
  if (cnt <= 0) {
    printf("An error occured: %d\n\n", cnt);
  } 
  else {
    printf("training image count: %d\n\n", cnt);
  }

  //
  // initialize network
  // 

  // no hidden layers
  int layer_sizes[2] = {input_size,output_size};
  Classifier C(2,layer_sizes,0.1);

  // // one hidden layer
  // int layer_sizes[3] = {input_size,64,output_size};
  // Classifier C(3,layer_sizes,0.1);

  // print network properties
  C.properties();

  //
  // run training epochs
  // 

  // initial accuracy and cross entropy (pre-training)
  std::cout << "TRAINING" << std::endl;
  std::cout << "epoch  accuracy  cross-entropy loss" << std::endl;
  std::cout << 0 << "    " << C.compute_loss(cnt, data, labels) << "    " << C.train_loss << std::endl;

  int epochs = 2;
  int batch_size = 10;
  double learning_rate = 0.1;
  for (int i = 1; i <= epochs; i++) {
    C.train_epoch(cnt, data, labels, learning_rate, batch_size);
    std::cout << i << "    " << C.compute_loss(cnt, data, labels) << "   " << C.train_loss << std::endl;
  }

  // unallocate training data
  for (int i = 0; i < cnt; i++) {
    delete[] data[i];
  }
  delete[] data;
  delete[] labels;

  //
  // run network on test data
  //

  // load test data
  cnt = mnist_load("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte", data, labels);
  std::cout << std::endl;
  if (cnt <= 0) {
    printf("An error occured: %d\n\n", cnt);
  } 
  else {
    printf("test image count: %d\n\n", cnt);
  }

  // compute and print loss/accuracy
  std::cout << "accuracy: " << C.compute_loss(cnt, data, labels) << std::endl << std::endl;

  // unallocate test data
  for (int i = 0; i < cnt; i++) {
    delete[] data[i];
  }
  delete[] data;
  delete[] labels;

  return 0;
}

