// fully connected linear layer
// with weight matrix and biases
// linear portion only

class Layer {
  public:
    // number of inputs and outputs
    int input_size;
    int output_size;
    // weights and biases (weight matrix stored as flattened array)
    double* weight;
    double* bias;
    // dropout probability : dropout is done on the output of the layer
    double drop_prob;

    // constructor and destructor
    Layer(int in_size, int out_size, double sigma, double drop);
    ~Layer(); 

    // print weight matrix
    void print_weights();

    // run layer (store output in the input variable)
    // used for testing purposes
    void run_layer(double* input, double* output);
};
