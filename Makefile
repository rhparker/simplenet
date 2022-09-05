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
all : train

train : train.cpp
	$(CXX) $(CXXFLAGS) train.cpp layer.cpp classifier.cpp -o $@

clean :
	\rm -f *.o *.out train

####### End of Makefile #######