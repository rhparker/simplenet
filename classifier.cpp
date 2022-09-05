#include "classifier.h"

#include <iostream>
#include <cmath>
#include <algorithm>    
#include <vector>
#include <ctime> 
#include <cstdlib>

// random number generator
int myrandom(int i) { 
  return std::rand() % i;
}

// argmax
unsigned int argmax(int len, double* values) {
  double current_max;
  unsigned int current_arg = 0;
  current_max = values[0];
  for (int i = 0; i < len; i++) {
    if (values[i] > current_max) {
      current_max = values[i];
      current_arg = i;
    }
  }
  return current_arg;
}

// sigmoid activation function
inline double sig(double x) {
  return 1.0/(1.0 + exp(-x));
}

// derivative of sigmoid
inline double d_sig(double x) {
  return sig(x)*(1.0 - sig(x));
}

// constructor
Classifier::Classifier(int in_size, int out_size, double sigma) {
  input_size  = in_size;
  output_size = out_size;

  L = new Layer(input_size, output_size, sigma);
  output = new double[output_size];
  // initialize outputs to 0
  for (int i = 0; i < output_size; i++) {
    output[i] = 0;
  }
}

// destructor
Classifier::~Classifier() {
  // delete L;
}

// generate output from input
void Classifier::run(double *input) {
  double normalizer = 0.0;
  L->run_layer(input);

  // softmax operation
  // exponentiate each entry and compute normalizer
  for (int i = 0; i < output_size; i++) {
    output[i] = exp(L->output[i]);
    normalizer += output[i];
  }
  // divide each entry by normalizer
  for (int i = 0; i < output_size; i++) {
    output[i] /= normalizer;
  }
}

// print layer output
void Classifier::print_output() {
  for (int i = 0; i < output_size; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
}

// 
double Classifier::compute_loss(int cnt, double **data, unsigned int *labels) {
  double correct = 0;
  train_loss = 0;
  for (int i = 0; i < cnt; i++) {

    run(data[i]);
    if ( argmax(output_size, output) == labels[i] ) {
      correct += 1;
    }

    train_loss -= log( output[ labels[i] ] );
  }
  train_accuracy = correct/cnt;
  return train_accuracy;
}

// train for one epoch
double Classifier::train_epoch(int cnt, double **data, unsigned int *labels, double lr) {
  // randomly shuffle training samples
  int index;
  std::srand ( unsigned ( std::time(0) ) );
  int* order = new int[cnt];
  for (int i = 0; i < cnt; i++) {
    order[i] = i;
  }
  std::random_shuffle(order, order+cnt, myrandom);

  for (int i = 0; i < cnt; i++) {
    double* delta = new double[output_size];
    unsigned int yj;
    index = order[i];

    run(data[index]);

    // compute delta
    for (int j = 0; j < output_size; j++) {
      yj = (j == labels[index]);
      delta[j] = output[j] - yj;

      // update biases
      L->bias[j] -= lr*delta[j];
      // update weights
      for (int k = 0; k < input_size; k++) {
        L->weight[j*input_size + k] -= lr*delta[j] * data[index][k];
      }
    } 

    delete[] delta;
  }
  delete[] order;
  return 0;
}
