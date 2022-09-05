#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "classifier.h"

int main(int argc, char* argv[]) {

  // load and process training data
  mnist_data *train_data;
  unsigned int cnt;
  int result;

  // load MNIST data
  if (result = mnist_load("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", &train_data, &cnt)) {
    printf("An error occured: %d\n", result);
  } 
  else {
    printf("image count: %d\n", cnt);
  }

  double **data = new double*[cnt];
  unsigned int *labels = new unsigned int[cnt];
  int input_size = 784;
  int output_size = 10;
  for (int i = 0; i < cnt; i++) {
    labels[i] = train_data[i].label;
    data[i] = train_data[i].data;
  }

  // network
  Classifier C(input_size,output_size,0.1);

  std::cout << C.compute_loss(cnt, data, labels) << " " << C.train_loss << std::endl;

  for (int i = 0; i < 3; i++) {
    C.train_epoch(cnt, data, labels, 0.1, 10);
    std::cout << C.compute_loss(cnt, data, labels) << " " << C.train_loss << std::endl;
  }

  free(train_data);

  return 1;
}

