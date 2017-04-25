FOLDER     := tests/$(TEST)

EXECUTABLE := $(FOLDER)/$(TEST) $(FOLDER)/log_$(TEST) $(FOLDER)/opt_$(TEST)

CU_FILES   := $(FOLDER)/$(TEST)cu.cu

CU_DEPS    :=

CC_FILES   := $(FOLDER)/$(TEST).cpp

EXTRA      := $(FOLDER)/*.log $(FOLDER)/log_* $(FOLDER)/opt_*

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=$(FOLDER)/objs
CXX=g++ -m64
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


OBJS=$(OBJDIR)/simple.o  $(OBJDIR)/simplecu.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(EXTRA)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: $(FOLDER)/%.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(FOLDER)/%.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
