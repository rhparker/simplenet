###################################################################
#  Makefile for neural net
###################################################################

# compilers
CXX = g++
CC  = gcc
F90 = gfortran

# flags
CXXFLAGS = -O2
CFLAGS   = -O2
FFLAGS   = -O2

# makefile targets
all : train-mnist train-fashion

train-mnist : train-mnist.cpp
	$(CXX) $(CXXFLAGS) train-mnist.cpp loadmnist.cpp layer.cpp classifier.cpp -o $@

train-fashion : train-fashion.cpp
	$(CXX) $(CXXFLAGS) train-fashion.cpp loadmnist.cpp layer.cpp classifier.cpp -o $@

clean :
	\rm -f *.o *.out train-mnist train-fashion train-mnist-omp

####### End of Makefile #######