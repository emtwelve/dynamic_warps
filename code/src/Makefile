all: PrintfLogs.so

CXXFLAGS = -rdynamic $(shell ./llvm-config --cxxflags) -g -O2

%.so: %.o
	$(CXX) -dylib -shared $^ -o $@
clean:
	rm -f *.o *~ *.so