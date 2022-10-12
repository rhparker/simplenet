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

  // std dev for weight initialization
  double sigma = 0.1;

  // timers
  double batch_time;
  double loss_time;

  // no hidden layers
  // int layer_sizes[2] = {input_size,output_size};
  // double drop_probs[1] = {0};
  // Classifier C(2,layer_sizes,sigma,drop_probs);

  // one hidden layer
  int layer_sizes[3] = {input_size,128,output_size};
  double drop_probs[2] = {0, 0};
  Classifier C(3,layer_sizes,sigma,drop_probs);

  // print network properties
  C.properties();

  //
  // run training epochs
  // 

  // initial accuracy and cross entropy (pre-training)
  std::cout << "TRAINING" << std::endl;
  std::cout 
    << "epoch  accuracy  cross-entropy loss       squared L2 norm of weights      batch time      loss_time" 
    << std::endl;
  loss_time = C.compute_loss(cnt, data, labels);
  std::cout << 0 << "    " << C.train_accuracy
    << "    " << C.train_loss << "   " << C.weight_L2sq << std::endl;

  int epochs = 4;
  int batch_size = 10;
  double learning_rate = 0.1;
  double weight_decay = 0;
  for (int i = 1; i <= epochs; i++) {
    batch_time = C.train_epoch(cnt, data, labels, learning_rate, weight_decay, batch_size);
    loss_time = C.compute_loss(cnt, data, labels);
    std::cout << i << "      " << C.train_accuracy
      << "   " << C.train_loss << "   " << C.weight_L2sq 
      << "   " << batch_time << "   " << loss_time << std::endl;
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
  loss_time = C.compute_loss(cnt, data, labels);
  std::cout << "accuracy: " << C.train_accuracy << std::endl << std::endl;

  // unallocate test data
  for (int i = 0; i < cnt; i++) {
    delete[] data[i];
  }
  delete[] data;
  delete[] labels;

  return 0;
}

