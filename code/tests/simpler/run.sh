

mkdir -p objs/

echo -e "\e[0;49;91m" # Red
echo "Annotating to make logger"
../../src/codegen_log.py simplecu.cu

echo -e "\e[0;49;32m" # Green
echo "Compiling logger"
g++ -m64 simple.cpp -O3 -Wall -c -o objs/simple.o
nvcc log_simplecu.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/log_simplecu.o
g++ -m64 -O3 -Wall -o log_simple objs/simple.o  objs/log_simplecu.o -L/usr/local/cuda/lib64/ -lcudart


echo -e "\e[0;49;95m" # Pink
echo "Running logger (output in log_<filename>.log)"
./log_simple > log_simple.log

echo -e "\e[0;49;93m" # Yellow
echo "Running analyzer and optimizer to generate opt_<filename>cu.cu"
../../src/codegen_opt.py simplecu.cu

echo -e "\e[0;49;96m" # Cyan
echo "Compiling optimized code"
g++ -m64 simple.cpp -O3 -Wall -c -o objs/simple.o
nvcc opt_simplecu.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/opt_simplecu.o
g++ -m64 -O3 -Wall -o opt_simple objs/simple.o  objs/opt_simplecu.o -L/usr/local/cuda/lib64/ -lcudart

echo -e "\e[0;49;94m" # Teal
echo "Running optimized code"
./opt_simple

echo -e "\e[0;49;32m" # Green
echo "Running unoptimized code"
g++ -m64 simple.cpp -O3 -Wall -c -o objs/simple.o
nvcc simplecu.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/simplecu.o
g++ -m64 -O3 -Wall -o simple objs/simple.o  objs/simplecu.o -L/usr/local/cuda/lib64/ -lcudart
./simple

