// 15-745 PrintOpts.cpp
////////////////////////////////////////////////////////////////////////////////

#include "llvm/Pass.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <vector>
#include <utility>
#include <iostream>
#include <assert.h>

using namespace llvm;

namespace {

  class PrintOpts : public FunctionPass {

  public:
    static char ID;
    PrintOpts() : FunctionPass(ID) { }
    ~PrintOpts() { }

    bool runOnFunction(llvm::Function &F) override {
      for (Function::iterator bb = F.begin(), bbe = F.end(); bb != bbe; ++bb) {
        BasicBlock &b = *bb;
        for (BasicBlock::iterator i = b.begin(), ie = b.end(); i != ie; ++i) {
          Instruction *I = i;
          Instruction *newinst;
          b.getInstList().insertAfter(I, newinst);
        }
      }
      return true;
    }

  private:

  };
}

// LLVM uses the address of this static member to identify the pass, so the
// initialization value is unimportant.
char PrintOpts::ID = 0;
static RegisterPass<PrintOpts> X("print-opts", "15745: Printing", false, false);
