// multi-layer classifier

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

// argmax function
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

// sigmoid activation function (single element)
inline double sig(double x) {
  return 1.0/(1.0 + exp(-x));
}

// derivative of sigmoid (single element)
inline double d_sig(double x) {
  return sig(x)*(1.0 - sig(x));
}

// sigmoid activation
// data : input vector
// size : size of input
// out : output
void sig_act(int size, double* data, double* out) {
  for (int i = 0; i < size; i++) {
    out[i] = sig(data[i]);
  }
}

// softmax operation
// data : input vector
// size : size of input
// out : output
void softmax(int size, double* data, double* out) {
  double normalizer = 0.0;
  for (int i = 0; i < size; i++) {
    out[i] = exp(data[i]);
    normalizer += out[i];
  }
  // divide each entry by normalizer
  for (int i = 0; i < size; i++) {
    out[i] /= normalizer;
  }
}

// constructor
// num_l : number of layers
// l_sizes: array of sizes of layers (including input and output)
// sigma: weights initialized to Normal(0, sigma) (biases intialized to 0)
Classifier::Classifier(int num_l, int* l_sizes, double sigma) {
  num_layers = num_l;
  input_size = l_sizes[0];
  output_size = l_sizes[num_layers-1];
  // store layer sizes
  layer_sizes = new int[num_layers];
  for (int i=0; i < num_layers; i++) {
    layer_sizes[i] = l_sizes[i];
  }  
  // create layers using Layer objects
  // one fewer needed than num_layers, so L[0] is not used
  L = new Layer*[num_layers];
  for (int i=1; i < num_layers; i++) {
    L[i] = new Layer(layer_sizes[i-1],layer_sizes[i], sigma);
  }
  // initialize output to all 0s
  output = new double[output_size];
  for (int i = 0; i < output_size; i++) {
    output[i] = 0;
  }
}

// destructor
Classifier::~Classifier() {
  delete L;
  delete[] output;
}

// print properties of network
void Classifier::properties() {
  std::cout << "Network properties" << std::endl;
  std::cout << "Input layer:  " << input_size << std::endl;
  if (num_layers > 2) {
    for (int i = 0; i < num_layers-2; i++) {
      std::cout << "Hidden layer: " << layer_sizes[i+1] << std::endl;
    }
  }
  std::cout << "Output layer: " << output_size << std::endl << std::endl;
}

// runs classifier on input
// outputs are: raw layer output (z) and layer output after activation (a)
// activation is sigmoid for all but last layer, softmax for last layer
void Classifier::forward(double* input, double** , double** a) {
  // process all but last layer
  for (int i = 1; i < num_layers-1; i++) {
    L[i]->run_layer(a[i-1], z[i]);
    // intermediate layers get sigmoid activation function
    sig_act(layer_sizes[i], z[i], a[i]);
  }
  // output layer
  L[num_layers-1]->run_layer(a[num_layers-2], z[num_layers-1]);
  // output layer gets softmax activation
  softmax(output_size, z[num_layers-1], a[num_layers-1]);
}

// generate and store output of network from input
void Classifier::run(double* input) {
  // allocate pointers for layer data
  // raw layer output (z) and layer output after activation (a)
  double** z = new double*[num_layers];
  double** a = new double*[num_layers];
  // first layer is input layer
  z[0] = input;
  a[0] = input;
  // allocate memory for intermediate layers
  for (int i = 1; i < num_layers; i++) {
    a[i] = new double[ L[i]->output_size ];
    z[i] = new double[ L[i]->output_size ];
  }

  // run forward propagation
  forward(input, z, a);
  // copy output from forward propagation to output variable
  for (int i = 0; i < output_size; i++) {
    output[i] = a[num_layers-1][i];
  }

  // free memory taken by intermediate layers
  for (int i = 1; i < num_layers; i++) {
    delete[] z[i];
    delete[] a[i];
  }
  delete[] z;
  delete[] a;
}

// print stored network output
void Classifier::print_output() {
  for (int i = 0; i < output_size; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
}

// compute loss (cross-entropy) and training accuracy from labeled data
// cnt : number of samples
// data : data array
// labels : corresponding labels
double Classifier::compute_loss(int cnt, double** data, unsigned int* labels) {
  // running total of number correct
  double correct = 0;
  train_loss = 0;
  // iterate over all samples
  for (int i = 0; i < cnt; i++) {
    // evaluate network at current sample
    run(data[i]);
    // increment number correct if classification output from network (argmax of probability vector)
    // matches label
    if ( argmax(output_size, output) == labels[i] ) {
      correct += 1;
    }
    // update cross-entropy with sample
    train_loss -= log( output[ labels[i] ] );
  }
  // compute training accuracy
  train_accuracy = correct/cnt;
  return train_accuracy;
}

// run one epoch of training using mini-batch stochastic gradient descent
// cnt:  number of data samples
// data: array containing data
// labels: labels corresponding to data
// lr: learning rate
// batch_size: size of each mini-batch
double Classifier::train_epoch(int cnt, double** data, unsigned int* labels, 
                                  double lr, unsigned int batch_size) {
  int index;
  int num_batches;
  num_batches = cnt / batch_size;

  // randomly shuffle training samples
  std::srand ( unsigned ( std::time(0) ) );
  int* order = new int[cnt];
  for (int i = 0; i < cnt; i++) {
    order[i] = i;
  }
  std::random_shuffle(order, order+cnt, myrandom);

  // backpropagation: iterate over batches
  for (int b = 0; b < num_batches; b++) {
    // partials of bias and weights for each layer (index 0 is unused)
    double** d_bias   = new double*[num_layers];
    double** d_weight = new double*[num_layers];
    // for each layer, allocate memory for partials and initialize to 0
    for (int j = 1; j < num_layers; j++) {
      d_bias[j] = new double[ L[j]->output_size ];
      d_weight[j] = new double[ L[j]->input_size*L[j]->output_size ];
      for (int k = 0; k < L[j]->output_size; k++) d_bias[j][k] = 0;
      for (int k = 0; k < L[j]->input_size*L[j]->output_size; k++) d_weight[j][k] = 0;
    }

    // compute contributions to paritals from each training sample in batch
    for (int i = 0; i < batch_size; i++) {
      // variable to store label for current sample
      unsigned int yj;
      // index of training sample in data array
      index = order[ b*batch_size + i ];

      // step 1: forward propagation on training sample
      // allocate pointers for layer data
      // need raw layer output (z) and layer output after activation (a)
      double** z = new double*[num_layers];
      double** a = new double*[num_layers];
      // first layer is input layer, so set pointer to current sample
      a[0] = data[index];
      z[0] = data[index];
      // allocate memory for intermediate layers
      for (int j = 1; j < num_layers; j++) {
        a[j] = new double[ L[j]->output_size ];
        z[j] = new double[ L[j]->output_size ];
      }
      // run forward propagation, fills z and a
      forward(data[index], z, a);

      // step 2: compute delta. delta is partial derivative of loss with respect to
      // raw layer output z (delta[0] is unused)
      double** delta = new double*[num_layers];
      for (int j = 1; j < num_layers; j++) {
        delta[j] = new double[ L[j]->output_size ];
      }
      
      // do last delta first, since that one is different
      // due to softmax activation function in last layer
      for (int j = 0; j < output_size; j++) {
        yj = (j == labels[index]);
        delta[num_layers-1][j] = a[num_layers-1][j] - yj;
      }

      // iteratively compute remaining deltas, working backwards
      for (int l = num_layers - 2; l > 0; l--) {
        for (int j = 0; j < L[l]->output_size; j++) {
          delta[l][j] = 0;
          for (int k = 0; k < L[l+1]->output_size; k++) {
            delta[l][j] += delta[l+1][k] * L[l+1]->weight[k*L[l+1]->input_size + j];
          }
          delta[l][j] *= d_sig( z[l][j] );
        }
      }

      // step 3: using the deltas, update bias and weight partials, one layer at a time
      for (int l = 1; l < num_layers; l++) {
        for (int j = 0; j < L[l]->output_size; j++) {
          d_bias[l][j] += delta[l][j];
          for (int k = 0; k < L[l]->input_size; k++) {
            d_weight[l][j*L[l]->input_size + k] += delta[l][j] * a[l-1][k];
          }
        }
      }

      // delete z, a, and delta; no longer needed
      for (int j = 1; j < num_layers; j++) {
        delete[] z[j];
        delete[] a[j];
        delete[] delta[j];
      }
      delete[] z;
      delete[] a;
      delete[] delta;
    }

    // step 4: now that we have finished with our batch, update the weights
    // using partial derivatives for entire mini-batch, one layer at a time
    for (int l = 1; l < num_layers; l++) {
      for (int j = 0; j < L[l]->output_size; j++) {
        // update biases
        L[l]->bias[j] -= (lr/batch_size)*d_bias[l][j];
        // update weights
        for (int k = 0; k < L[l]->input_size; k++) {
          L[l]->weight[j*L[l]->input_size + k] -= 
            (lr/batch_size)*d_weight[l][j*L[l]->input_size + k];
        }
      }
    }

    // delete partial derivative d_bias and d_weight
    for (int j = 1; j < num_layers; j++) {
      delete[] d_bias[j];
      delete[] d_weight[j];
    }
    delete[] d_bias;
    delete[] d_weight;
  }

  delete[] order;
  return 0;
}
