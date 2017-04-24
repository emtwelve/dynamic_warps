

#mkdir -p objs/


#../../src/codegen.py simplecu.cu

#g++ -m64 simple.cpp -O3 -Wall -c -o objs/simple.o

#nvcc log_simplecu.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/log_simplecu.o

#g++ -m64 -O3 -Wall -o simple objs/simple.o  objs/log_simplecu.o -L/usr/local/cuda/lib64/ -lcudart

#./simple > log_simple.log

#../../src/analyze.py log_simple.log
../../src/codegen_opt.py simplecu.cu
