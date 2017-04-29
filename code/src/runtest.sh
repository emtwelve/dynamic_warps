RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;34m'
BLUE='\033[0;35m'

export PATH=$PATH:/afs/cs.cmu.edu/user/fp/courses/15411-f08/llvm/bin

echo -e "${RED}Running tests"

echo -e "${BLUE}"
clang -O0 -emit-llvm -c ../tests/$1.cu
llvm-dis $1.bc
cat $1.ll

echo -e "${GREEN}"
opt -mem2reg $1.bc -o $1-m2r.bc
llvm-dis $1-m2r.bc
cat $1-m2r.ll

echo -e "${ORANGE}"
opt -load ./PrintfLogs.so -printf-bb-logger ./$1-m2r.bc -o out.bc
llvm-dis out.bc
cat out.ll

rm $1.bc $1-m2r.bc $1.ll $1-m2r.ll

