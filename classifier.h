// Classifier with input layer, output layer, and (optional) hidden layers
// layers are fully connected
// all but last layer use sigmoid activation function
// last layer uses softmax to get probability vector
// loss function is cross-entropy

#include "layer.h"

class Classifier {
  public:
    // layer parameters: num_layers includes input, hidden, and output layers
    int num_layers;
    int* layer_sizes;
    int input_size;
    int output_size;

    // layers
    Layer** L;

    // stores output of net
    double* output;

    // stores current training accuracy and cross-entropy loss
    double train_accuracy;
    double train_loss;

    // constructor and destructor
    Classifier(int num_l, int* l_sizes, double sigma);
    ~Classifier(); 

    // print properties
    void properties();

    // forward propagation on input, generates output from all layers
    void forward(double *input, double** z, double** a);
    // runs net on input
    void run(double *input);

    // print output of network
    void print_output();

    // compute loss function (cross entropy) and accuracy
    double compute_loss(int cnt, double **data, unsigned int *labels);

    // run one epoch of training with supplied data and labels 
    // using mini-batch stochastic gradient descent
    double train_epoch(int cnt, double **data, unsigned int *labels, double lr, unsigned int batch_size);
};