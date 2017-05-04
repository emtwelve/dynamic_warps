cd tests/$1
mkdir objs
# Make the test

echo -e "\e[0;49;93m" # Yellow
echo "Running Annotation Engine"
../../src/codegen_log.py $1.cu > anno.log

echo -e "\e[0;49;32m" # Green
echo "Running program with annotations"
make clean > make.log
mv $1.cu temp$1.cu
mv anno_$1.cu $1.cu
num_bbs=`cat num_bbs`
make > make.log
./$1 > log.csv
make clean > make.log
mv $1.cu anno_$1.cu
mv temp$1.cu $1.cu

echo -e "\e[0;49;95m" # Pink
echo "Creating warp remapping"
g++ -std=c++0x -m64 ../../src/warp_map.cpp -O3 -Wall -c -o objs/warp_map.o -lpthread
nvcc ../../src/warp_mapcu.cu -O3 -m64 --gpu-architecture compute_35 -c -o objs/warp_mapcu.o
g++ -std=c++0x -m64 -O3 -Wall -o remap_warp objs/warp_map.o  objs/warp_mapcu.o -L/usr/local/cuda/lib64/ -lcudart
./remap_warp log.csv $num_bbs $2 > warp.log

echo -e "\e[0;49;96m" # Cyan
echo "Compile + Run + Profile Unoptimized Code ($1.cu)"
# Profile the unoptimized code run
make clean > make.log
make > make.log
nvprof --analysis-metrics -o profile_unopt$1.nvvp -f ./$1

echo -e "\e[0;49;91m" # Red
echo "Generating Optimized Code (opt_$1.cu)"
../../src/codegen_remap.py $1.cu warp.log > opt_$1.cu


echo -e "\e[0;49;94m" # Teal
echo "Compile + Run + Profile Optimized Code (opt_$1.cu)"
mv $1.cu temp$1.cu
mv opt_$1.cu $1.cu
make clean > make.log
make > make.log
nvprof --analysis-metrics -o profile_opt$1.nvvp -f ./$1
make clean > make.log
mv $1.cu opt_$1.cu
mv temp$1.cu $1.cu

tput sgr0

# TO RUN SCRIPT
# ./run.sh (test_name)

# TO RUN MAKEFILE
# make clean TEST=(test_name)
# make TEST=(test_name)

# To check NVIDIA GPU specs
# nvidia-smi
