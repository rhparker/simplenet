#include "layer.h"

#include <random>
#include <iostream>
#include <cmath>

// macro for getting index of 2D array stored as 1D flattened array
#define ind(i,j,rows) i*rows + j 

// constructor
Layer::Layer(int in_size, int out_size, double sigma, double drop) {
  // number of inputs and outputs
  input_size  = in_size;
  output_size = out_size;

  // dropout probability
  drop_prob = drop;

  // allocate memory for weights, biases, and output
  weight = new double[input_size*output_size];
  bias = new double[output_size];

  // random device initialization
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  // instance of class std::normal_distribution with mean 0, std dev sigma
  std::normal_distribution<double> d(0, sigma); 

  // initialize weights to normal(0, sigma)
  for (int i = 0; i < input_size*output_size; i++) {
    weight[i] = d(gen);
  }

  // initialize biases to 0
  for (int i = 0; i < output_size; i++) {
    bias[i] = 0; 
  }
}

// destructor
Layer::~Layer(void) {
  delete[] weight;
  delete[] bias; 
}

// print out weight matrix
void Layer::print_weights() {
  // iterate over rows
  for (int i = 0; i < output_size; i++) {
    // go across each row, i.e. iterate over columns
    for (int j = 0; j < input_size; j++) {
      std::cout << weight[ ind(i,j,input_size) ] << " ";
    }
    std::cout << std::endl << std::endl;
  }
}

// generate output from input: z = W x + b
void Layer::run_layer(double* input, double* output) {
  // iterate over rows
  for (int i = 0; i < output_size; i++) {
    output[i] = bias[i];
    // iterate over columns
    for (int j = 0; j < input_size; j++) {
      output[i] += weight[i*input_size + j] * input[j];
    }
  }
}

