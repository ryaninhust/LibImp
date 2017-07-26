CXX = g++
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXX = g++-6
endif
CXXFLAGS = -Wall -g -std=c++0x -march=native

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: train

train: train.cpp mf.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm.o: mf.cpp mf.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f train predict mf.o *.bin.*
