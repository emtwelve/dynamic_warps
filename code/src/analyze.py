#!/usr/bin/python

from sys import argv
from collections import Counter

class Param(object):
    def __init__(self): pass
    def add_type(self, mytype): self.mytype = mytype
    def add_value(self, value): self.value = value
    def add_name(self, name):   self.name = name
    def get_type(self): return self.mytype
    def get_value(self):return self.value
    def get_name(self): return self.name
    def tostr(self):    return str(self.mytype) + " " + str(self.value) + " " + str(self.name)
    def __repr__(self):
        return self.tostr()
    def __str__(self):
        return self.tostr()

class FnCallLog(object):
    def __init__(self, thr_idx, fn_name, argv):
        # 1:1 type to its value
        assert len(argv) % 3 == 0
        self.thr_idx = thr_idx
        self.name = fn_name
        self.params = []
        for i in xrange(len(argv)):
            if i % 3 == 0: # type
                param = Param()
                param.add_type(argv[i])
            elif i % 3 == 1: # value
                param.add_value(argv[i])
            elif i % 3 == 2: # argument name
                param.add_name(argv[i])
                self.params.append(param)
            else: assert(False)


    def getFnName(self): return self.name
    def getThrIdx(self): return self.thr_idx
    def getParams(self): return self.params
    def tostr(self):
        ans = self.name + "[" + self.thr_idx + "]("
        for param in self.params:
            ans += param.tostr() + ","
        ans += ")"
        return ans
    def __repr__(self): return self.tostr()
    def __str__(self): return self.tostr()

class FixedArgument(object):
    def __init__(self, _fn_name, _arg_idx, _arg_name, _arg_typ):
        self.fn_name = _fn_name;    self.arg_idx = _arg_idx
        self.arg_values = []
        self.arg_name = _arg_name;  self.arg_type = _arg_typ
    def addValue(self, arg_value, arg_value_count):
        self.arg_values += [(arg_value, arg_value_count)]
    def __str__(self):
        return str(self.fn_name) + " " + str(self.arg_idx) + " " + \
               str(self.arg_name) + " " + str(self.arg_type) + " " + \
               str(self.arg_values)
    def __repr__(self):
        return self.__str__()

def getArgumentName(fn_name, index, call_logs):
    # indexing into the call_logs's function's list can be anywhere
    #   because a specific function has the same call log parameters
    #   across all calls
    return call_logs[fn_name][0].params[index].name

def getArgumentType(fn_name, index, call_logs):
    # indexing into the call_logs's function's list can be anywhere
    #   because a specific function has the same call log parameters
    #   across all calls
    return call_logs[fn_name][0].params[index].mytype


def getCounts(call_logs):

    # TODO: properly get number of params, specific
    #  for all different functions
    len_params = len(call_logs["test"][0].getParams())

    # counts[i] is param i's dictionary of value to count of value
    counts = [] # list of dictionaries
    for i in xrange(len_params):
        counts += [[Counter(), None]]

    for fn_name in call_logs:
        for fnCallLog in call_logs[fn_name]:
            params = fnCallLog.getParams()
            for i in xrange(len(params)):
                value = params[i].get_value()
                counts[i][0][value] += 1
                counts[i][1] = params[i]
    return counts

def initialize_gens(args_to_fix):
    arg_fixed_functions_to_gen = {}
    for fn_name in args_to_fix:
        arg_fixed_functions_to_gen[fn_name] = {}

    for fn_name, args in args_to_fix.items():
        for arg in args:
            arg_fixed_functions_to_gen[fn_name][arg] = {}

    for fn_name, args in args_to_fix.items():
        for arg in args:
            for value, count in args_to_fix[fn_name][arg].arg_values:
                arg_fixed_functions_to_gen[fn_name][arg][value] = []

    return arg_fixed_functions_to_gen

def generate_arg_fixed_functions(args_to_fix):
    arg_fixed_functions_to_gen = initialize_gens(args_to_fix)

    for fn_name, args in args_to_fix.items():
        for arg in args:
            for value, count in args_to_fix[fn_name][arg].arg_values:
                argFxdName = fn_name + "_" + arg + "_" + value
                rest_of_function = "###REST_" + argFxdName
                arg_fixed_functions_to_gen[fn_name][arg][value] = \
                     "__device__ int " + \
                     argFxdName + " ( " + \
                     " int y , int z " + \
                     " ) { \n\t" + \
                     args_to_fix[fn_name][arg].arg_type + " " + \
                     arg + " = " + \
                     value + ";\n" + \
                     rest_of_function


    #print arg_fixed_functions_to_gen
    """
    for fn_name, args in args_to_fix.items():
        for arg in args:
            for value, count in args_to_fix[fn_name][arg].arg_values:
                print "~~~~"
                print arg_fixed_functions_to_gen[fn_name][arg][value]
    """

    return arg_fixed_functions_to_gen

def generate_branching_function(fn_name, args_to_fix):
    branch_function = "__device__ int branch_" + fn_name + " ( bool x , int y , int z ) {\n"
    branch_function += "\tswitch (x) {\n" # TODO: generalize
    for arg in args_to_fix[fn_name]:
        fixedArg = args_to_fix[fn_name][arg]
        for val, cnt in fixedArg.arg_values:
            branch_function += "\t\tcase " + str(val) + ":\n" + \
                "\t\t\treturn " + fn_name + "_" + arg + "_" + str(val) + " ( y , z ) " + ";\n"
    branch_function += "\t}\n}"
    return branch_function

def analyze(fname):
    print "Beginning analyze.py"
    print "Reading ", fname
    assert fname.split('.')[1] == 'log'
    fd = open(fname, 'r')
    lines = fd.readlines()

    call_logs = {}
    for line in lines:
        toks = line.strip().split(" ")
        if "CALL_LOG" == toks[0]:
            thr_idx = toks[1]
            fn_name = toks[2]
            argv = toks[3:]
            fnCallLog = FnCallLog(thr_idx, fn_name, argv)
            if fn_name not in call_logs:
                call_logs[fn_name] = [fnCallLog]
            else:
                call_logs[fn_name] += [fnCallLog]


    # e.g.
    #   [Counter({'1': 32, '0': 32}),
    #    Counter({'11': 5, '10': 5, '1': 5, '0': 5, ...}),
    #    Counter({'0': 10, '1': 9, '3': 9, '2': 9, ...})]
    # meaning: argument 0 is called with 1 32 times and 0 32 times
    #          argument 1 is called with 11 5 times, 10 5 times, etc...
    #          argument 2 is called with 0 10 times, 1 9 times, etc...
    argCounts = getCounts(call_logs)

    # The following prints out the call log:
    """
    for fn_name, calllogs in call_logs.items():
        print fn_name
        for cl in calllogs:
            print cl
    """
    # The following prints out the per argument value count:
    print argCounts

    fn_name = "test"
    num_argfixed_fns = Counter()
    args_to_fix = {} # per function args_to_fix
    MIN_THRESHOLD = 0.45
    # Iterate over all Counter objects:
    for i in xrange(len(argCounts)):
        arg_counter = argCounts[i][0]
        total = float(sum(arg_counter.values())) # total of all counts
        # Iterate over all (argument, argument count) pairs:
        for arg, count in arg_counter.items():
            arg_name = getArgumentName(fn_name, i, call_logs)
            arg_type = getArgumentType(fn_name, i, call_logs)
            if count / total >= MIN_THRESHOLD:
                if fn_name not in args_to_fix:
                    args_to_fix[fn_name] = {}

                if arg_name not in args_to_fix[fn_name]:
                    fixedArg = FixedArgument(fn_name, i, arg_name, arg_type)
                    args_to_fix[fn_name][arg_name] = fixedArg

                args_to_fix[fn_name][arg_name].addValue(arg, count)
                num_argfixed_fns[fn_name] += 1
            else:
                # TODO: add to nonfixed arguments list
                pass

    #print args_to_fix
    arg_fixed_functions = generate_arg_fixed_functions(args_to_fix)
    branch_function = generate_branching_function(fn_name, args_to_fix)
    #print branch_function
    return branch_function, arg_fixed_functions, num_argfixed_fns

if __name__ == "__main__":
    if len(argv) != 2:
        print "Usage: ./analyze.py <log file>"
    fname = argv[1]
    analyze(fname)
