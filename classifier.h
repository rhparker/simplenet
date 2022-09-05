#include "layer.h"

class Classifier {
  public:
    int input_size;
    int output_size;

    Layer* L;
    double* output;

    double train_accuracy;
    double train_loss;

    // constructor and destructor
    Classifier(int in_size, int out_size, double sigma);
    ~Classifier(); 

    void run(double *input);
    void print_output();

    double compute_loss(int cnt, double **data, unsigned int *labels);

    double train_epoch(int cnt, double **data, unsigned int *labels, double lr, unsigned int batch_size);
};