clang -O -emit-llvm -c ../tests/$1.c

opt -load ./FunctionInfo.so -function-info ./$1.bc -o out

