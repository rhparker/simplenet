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

    // constructor and destructor
    Layer(int in_size, int out_size, double sigma);
    ~Layer(); 

    // get weight as specific position in weight matrix
    // (weight matrix is flattened into 1D array)
    double get_weight(int i, int j);
    // print weight matrix
    void print_weights();

    // run layer (store output in the input variable)
    void run_layer(double* input, double* output);
};
