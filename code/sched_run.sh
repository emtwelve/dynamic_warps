make clean TEST=$1
make TEST=$1

cd tests/$1
echo -e "\e[0;49;91m" # Red
echo "Annotating to make logger"
../../src/codegen_log.py $1cu.cu

echo -e "\e[0;49;32m" # Green
echo "Compiling logger"
g++ -m64 $1.cpp -O3 -Wall -c -o objs/$1.o
nvcc log_$1cu.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/log_$1cu.o
g++ -m64 -O3 -Wall -o log_$1 objs/$1.o objs/log_$1cu.o -L/usr/local/cuda/lib64/ -lcudart

echo -e "\e[0;49;95m" # Pink
echo "Running logger (output in log_<filename>.log)"
./log_$1 > log_$1.log

echo -e "\e[0;49;93m" # Yellow
echo "Running analyzer and optimizer to generate opt_<filename>cu.cu"
../../src/codegen_opt.py $1cu.cu

echo -e "\e[0;49;96m" # Cyan
echo "Compiling optimized code"
g++ -m64 $1.cpp -O3 -Wall -c -o objs/$1.o
nvcc opt_$1cu.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/opt_$1cu.o
g++ -m64 -O3 -Wall -o opt_$1 objs/$1.o objs/opt_$1cu.o -L/usr/local/cuda/lib64/ -lcudart

echo -e "\e[0;49;94m" # Teal
echo "Running optimized code"
./opt_$1

echo -e "\e[0;49;32m" # Green
echo "Running unoptimized code"
g++ -m64 $1.cpp -O3 -Wall -c -o objs/$1.o
nvcc $1cu.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/$1cu.o
g++ -m64 -O3 -Wall -o $1 objs/$1.o objs/$1cu.o -L/usr/local/cuda/lib64/ -lcudart
./$1


# TO RUN SCRIPT
# ./run.sh (test_name)

# TO RUN MAKEFILE
# make clean TEST=(test_name)
# make TEST=(test_name)
