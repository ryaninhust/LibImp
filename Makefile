CXX = g++
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXX = g++-6
endif
CXXFLAGS = -Wall -O3 -std=c++0x -march=native

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: train predict

predict: predict.cpp mf.o
	$(CXX) $(CXXFLAGS) -o $@ $^

train: train.cpp mf.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm.o: mf.cpp mf.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f train predict mf.o *.bin.*
