cd tests/$1
rm -rf objs/
mkdir objs

echo -e "\e[0;49;93m" # Yellow
echo "Creating the different versions of the cuda code"

echo -e "\e[0;49;32m" # Green
echo "Running program with annotations"
nvcc anno_$1.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/anno_$1.o
g++ -m64 -O3 -Wall -o anno_$1 objs/anno_$1.o -L/usr/local/cuda/lib64/ -lcudart
./anno_$1 > log.csv

echo -e "\e[0;49;95m" # Pink
echo "Creating warp remapping"
g++ -std=c++0x -m64 ../../src/warp_map.cpp -O3 -Wall -c -o objs/warp_map.o -lpthread
nvcc ../../src/warp_mapcu.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/warp_mapcu.o
g++ -std=c++0x -m64 -O3 -Wall -o remap_warp objs/warp_map.o  objs/warp_mapcu.o -L/usr/local/cuda/lib64/ -lcudart
./remap_warp log.csv $2 $3 > warp.log

echo -e "\e[0;49;96m" # Cyan
echo "Run unoptimized code"
nvcc $1.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/$1.o
g++ -m64 -O3 -Wall -o $1 objs/$1.o -L/usr/local/cuda/lib64/ -lcudart
./$1

echo -e "\e[0;49;91m" # Red
echo "Generating optimized code"
../../src/./codegen_remap.py $1.cu warp.log > opt_simple.cu


echo -e "\e[0;49;94m" # Teal
echo "Run optimized code"
nvcc opt_$1.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/opt_$1.o
g++ -m64 -O3 -Wall -o opt_$1 objs/opt_$1.o -L/usr/local/cuda/lib64/ -lcudart
./opt_$1

tput sgr0

# TO RUN SCRIPT
# ./run.sh (test_name)

# TO RUN MAKEFILE
# make clean TEST=(test_name)
# make TEST=(test_name)
