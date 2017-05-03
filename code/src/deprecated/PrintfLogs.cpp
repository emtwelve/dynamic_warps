// Printf Basic Block Logger
// Author: Joey, Norman
////////////////////////////////////////////////////////////////////////////////

//#include "llvm/Module.h"
// /afs/cs.cmu.edu/user/fp/courses/15411-f08/llvm
// /afs/cs.cmu.edu/user/fp/courses/15411-f08/llvm/include
#include "llvm/lib/Pass.h"
#include "llvm/lib/IR/Function.h"
#include "llvm/lib/Support/raw_ostream.h"



#include <vector>
#include <utility>
#include <iostream>
#include <assert.h>

using namespace llvm;

namespace {
  class PrintfLogs : public BasicBlockPass {

  public:
    static char ID;
    PrintfLogs() : BasicBlockPass(ID) { }
    ~PrintfLogs() { }

    // Do some initialization
    bool doInitialization(Function &F) override {

      return false;
    }


    // Print output for each function
    bool runOnBasicBlock(BasicBlock &BB) override {
      outs() << BB << "\n";

      //for (BasicBlock::iterator BBI = FI->begin(); BBI != FI->end(); /*++BBI*/) 
      //          Instruction *I = BBI;

      return false;
    }

    bool doFinalization(Function &F) override {

      return false;
    }

  private:


  };
}

char PrintfLogs::ID = 0;
static RegisterPass<PrintfLogs> X("printf-bb-logger", "Printf Call Logger", false, false);