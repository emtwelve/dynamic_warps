#!/usr/bin/python

from sys import argv
from collections import Counter, defaultdict

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
        # thr_idx: the CUDA thread index that makes this function call
        # fn_name: the name of the device function
        # argv: the arguments into the device function

        # 1:1:1 type to its value to its name
        assert len(argv) % 3 == 0
        self.thr_idx = thr_idx
        self.name = fn_name
        self.params = []
        # The following code is kind of confusing,
        #   i will be 0,1,2,3,4,5,6,...
        #   for each grouping of 3 (e.g. i = {0,1,2}),
        #       generate another parameter and add it to params list
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
    """ Class to represent a given argument.

        For example,
        there is one FixedArgument object for parameter hello in
        __device__ int example_dev(int bye, bool hello, char yo) { ... }
            hello.fn_name is 'example_dev'
            hello.arg_idx is 1
            hello.arg_name is 'hello'
            hello.arg_type is bool
            hello.arg_values is a list that will be populated with all
                ArgValues, ArgValueCounts, Lists of thread_indices
                this argument is called with.
                arg_values gets populated with calls to addValue
    """
    def __init__(self, _fn_name, _arg_idx, _arg_name, _arg_typ):
        # fn_name: the name of the device function
        # arg_idx: the parameter position into the function (0,1,2,...)
        # arg_name: the name of the parameter
        # arg_typ: the type of the parameter
        self.fn_name = _fn_name;    self.arg_idx = _arg_idx
        self.arg_values = [] # initially empty, to be later populated
        self.arg_name = _arg_name;  self.arg_type = _arg_typ
    def addValue(self, arg_value, arg_value_count, thread_indices):
        """ Populate arg_values """
        self.arg_values += [(arg_value, arg_value_count, thread_indices)]
    def __str__(self):
        return str(self.fn_name) + " " + str(self.arg_idx) + " " + \
               str(self.arg_name) + " " + str(self.arg_type) + " " + \
               str(self.arg_values)
    def __repr__(self):
        return self.__str__()

class NotFixedArgument(object):
    """ Class to represent a given argument that will not be fixed.  Simliar
        to a FixedArgument except no values list.
    """
    def __init__(self, _fn_name, _arg_idx, _arg_name, _arg_typ):
        # fn_name: the name of the device function
        # arg_idx: the parameter position into the function (0,1,2,...)
        # arg_name: the name of the parameter
        # arg_typ: the type of the parameter
        self.fn_name = _fn_name;    self.arg_idx = _arg_idx
        self.arg_name = _arg_name;  self.arg_type = _arg_typ
    def __str__(self):
        return str(self.fn_name) + " " + str(self.arg_idx) + " " + \
               str(self.arg_name) + " " + str(self.arg_type)
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

    # TODO: properly get length of params, specific
    #  for all different functions
    #  HACK: Right now just getting the length of parameters
    #    for a single function, but we are assuming we have only
    #    one device function called test at the moment:
    len_params = len(call_logs["test"][0].getParams())

    # counts[i] is param i's dictionary of value to count
    counts = []
    # list of tuples where
    #   tuple[0] has type Counter: (param value -> count) --AND--
    #   tuple[1] has type Param list                      --AND--
    #   tuple[2] has type dictionary: (param value -> thread_idx list) 
    for i in xrange(len_params):
        counts += [[Counter(), None, defaultdict(list)]]

    for fn_name in call_logs:
        for fnCallLog in call_logs[fn_name]:
            params = fnCallLog.getParams()
            for i in xrange(len(params)):
                value = params[i].get_value()
                caller_thread_idx = fnCallLog.getThrIdx()
                counts[i][0][value] += 1
                counts[i][1] = params[i]
                counts[i][2][value] += [caller_thread_idx]

    return counts

def initialize_gens(args_to_fix):
    """ Makes an empty dictionary of dictionaries, where the
        inner dictionary maps to a list (this list will
        eventually contain strings of the arg_fixed functions
        to write to the opt_<filename>cu.cu file) """
    arg_fixed_functions_to_gen = {}
    for fn_name in args_to_fix:
        arg_fixed_functions_to_gen[fn_name] = {}

    for fn_name, args in args_to_fix.items():
        for arg in args:
            arg_fixed_functions_to_gen[fn_name][arg] = {}

    for fn_name, args in args_to_fix.items():
        for arg in args:
            for value, count, thread_indices in args_to_fix[fn_name][arg].arg_values:
                arg_fixed_functions_to_gen[fn_name][arg][value] = []

    return arg_fixed_functions_to_gen

def not_fixedArgStringGen(fn_name, args_no_fix, include_type):
    res = ""
    var_to_notFixedArg = args_no_fix[fn_name]
    for var, not_fixedArg in var_to_notFixedArg.items():
        if include_type:
            res += " " + not_fixedArg.arg_type + " " + \
                         not_fixedArg.arg_name + " ,"
        else:
            res += " " + not_fixedArg.arg_name + " ,"
    return res[:-1] # remove last comma

def generate_arg_fixed_functions(args_to_fix, args_no_fix):
    # Initialize empty data structure (a dictionary-dictionary-list):
    #   It maps a function name to a dictionary D, where
    #   D maps an argument name to a list L, where
    #   L contains the string of the device function to generate
    arg_fixed_functions_to_gen = initialize_gens(args_to_fix)

    # Populate the dictionary-dictionary-list data structure:
    for fn_name, args in args_to_fix.items():
        for arg in args:
            for value, count, thread_indices in args_to_fix[fn_name][arg].arg_values:
                # name of device arg_fixed function to generate:
                argFxdName = fn_name + "_" + arg + "_" + value

                # placeholder to be filled in later:
                rest_of_function = "###REST_" + argFxdName

                # add the device function string to the dictionary:
                arg_fixed_functions_to_gen[fn_name][arg][value] = \
                     "__device__ int " + \
                     argFxdName + " ( " + \
                     not_fixedArgStringGen(fn_name, args_no_fix, 1) + \
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

def generate_branching_function(fn_name, args_to_fix, args_no_fix):
    # Function prolog: TODO: generalize the input parameters:
    branch_function = "__device__ int branch_" + fn_name + " ( bool x , int y , int z ) {\n"
    
    # Switch statement to call all the seperate arg_fixed device functions:
    branch_function += "\tswitch (x) {\n" # TODO: generalize (what do we do on multiple parameters?)
    for arg in args_to_fix[fn_name]:
        fixedArg = args_to_fix[fn_name][arg]
        for val, cnt, _ in fixedArg.arg_values:
            branch_function += "\t\tcase " + str(val) + ":\n" + \
                "\t\t\treturn " + fn_name + "_" + arg + "_" + str(val) + \
                " ( " + not_fixedArgStringGen(fn_name, args_no_fix, 0) + " ) " + ";\n"

    # Failure case and function epilog:
    branch_function += "\t}\n\tint *asdffdsa12344321 = NULL;\n\t" + \
                       "return " + "(int) " + "*asdffdsa12344321;" + "\n}"

    return branch_function

def generate_warp_rescheduler(args_to_fix):
    # Generate list of lists, where each inner list is a group
    #   of at most 32 CUDA thread indices.  All threads in an
    #   inner list should be remapped to the same warp:
    thread_groups = []
    for function_name in args_to_fix:
        argToFixedArg = args_to_fix[function_name]
        for arg in argToFixedArg:
            fixedArg = argToFixedArg[arg]

            assert(fixedArg.fn_name == function_name)
            for val, cnt, thr_indices in fixedArg.arg_values:
                thread_groups += [thr_indices]

    WARP_SIZE = 32
    def warpGroupRemap(warp_idx):
        return range(warp_idx*WARP_SIZE, (warp_idx+1)*WARP_SIZE)

    warp_rescheduler = [None]*64
    for i in xrange(len(thread_groups)):
        group = thread_groups[i]
        remap = warpGroupRemap(i)
        for j in xrange(len(group)):
            thr_idx = int(group[j])
            warp_rescheduler[int(thr_idx)] = remap[j]

    return warp_rescheduler



def parseCallLogFile(fname):
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

    return call_logs

def analyze(fname):
    print "Beginning analyze.py"
    print "Reading ", fname
    assert fname.split('.')[1] == 'log'

    # Parse the call log file into a List of FnCallLog objects:
    call_logs = parseCallLogFile(fname)

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
    #print argCounts

    fn_name = "test"
    num_argfixed_fns = Counter() # Not used at the moment
    args_to_fix = {} # per function args to fix
    args_no_fix = {} # per function args not to fix
    MIN_THRESHOLD = 0.45
    # Iterate over all Counter objects:
    for i in xrange(len(argCounts)):
        arg_counter = argCounts[i][0]

        thr_indices = argCounts[i][2] # dictionary from ArgValue to thr_idx list
        total = float(sum(arg_counter.values())) # total of all counts
        # Iterate over all (argument, argument count) pairs:
        for arg_value, count in arg_counter.items():
            arg_name = getArgumentName(fn_name, i, call_logs)
            arg_type = getArgumentType(fn_name, i, call_logs)
            if count / total >= MIN_THRESHOLD:
                if fn_name not in args_to_fix:
                    args_to_fix[fn_name] = {}

                if arg_name not in args_to_fix[fn_name]:
                    fixedArg = FixedArgument(fn_name, i, arg_name, arg_type)
                    args_to_fix[fn_name][arg_name] = fixedArg

                args_to_fix[fn_name][arg_name].addValue(arg_value, count, thr_indices[arg_value])
                num_argfixed_fns[fn_name] += 1
            else:
                # TODO: add to nonfixed arguments list
                if fn_name not in args_no_fix:
                    args_no_fix[fn_name] = {}
                if arg_name not in args_no_fix[fn_name]:
                    not_fixedArg = NotFixedArgument(fn_name, i, arg_name, arg_type)
                    args_no_fix[fn_name][arg_name] = not_fixedArg
                assert(args_no_fix[fn_name][arg_name].arg_name == arg_name)
                assert(args_no_fix[fn_name][arg_name].fn_name == fn_name)
                assert(args_no_fix[fn_name][arg_name].arg_type == arg_type)

    print "Arguments not to fix:"
    print args_no_fix

    warp_rescheduler = generate_warp_rescheduler(args_to_fix)
    arg_fixed_functions = generate_arg_fixed_functions(args_to_fix, args_no_fix)
    branch_function = generate_branching_function(fn_name, args_to_fix, args_no_fix)

    print "Arguments to fix:"
    print args_to_fix
    print "Warp rescheduler:"
    print warp_rescheduler
    print "Argfixed functions:"
    print arg_fixed_functions
    print "Branch function:"
    print repr(branch_function)

    return branch_function, arg_fixed_functions, warp_rescheduler

if __name__ == "__main__":
    if len(argv) != 2:
        print "Usage: ./analyze.py <log file>"
    fname = argv[1]
    analyze(fname)
