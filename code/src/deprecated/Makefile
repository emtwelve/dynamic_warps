EXECUTABLE := graph

CU_FILES   := warp_mapcu.cu

CU_DEPS    :=

CC_FILES   := warp_map.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -std=c++0x -m64
CXXFLAGS=-O3 -Wall
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib/ -lcudart
else
# Building on Linux
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
endif
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35


OBJS=$(OBJDIR)/warp_map.o  $(OBJDIR)/warp_mapcu.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@ -lpthread

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
