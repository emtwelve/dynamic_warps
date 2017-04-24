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

class FixedArgument(object):
    def __init__(self, _fn_name, _arg_idx, _arg_id, _arg_count, _arg_name):
        self.fn_name = _fn_name;    self.arg_idx = _arg_idx
        self.arg_id = _arg_id;      self.arg_count = _arg_count
        self.arg_name = _arg_name
    def __str__(self):
        return str(self.fn_name) + " " + str(self.arg_idx) + " " + \
               str(self.arg_id) + " " + str(self.arg_count) + " " + str(self.arg_name)
    def __repr__(self):
        return self.__str__()

def getArgumentName(fn_name, index, call_logs):
    # indexing into the call_logs's function's list can be anywhere
    #   because a specific function has the same call log parameters
    #   across all calls
    return call_logs[fn_name][0].params[index].name

def getCounts(call_logs):

    # TODO: properly get number of params, specific
    #  for all different functions
    len_params = len(call_logs["test"][0].getParams())

    # counts[i] is param i's dictionary of value to count of value
    counts = [] # list of dictionaries
    for i in xrange(len_params):
        counts += [Counter()]

    for fn_name in call_logs:
        for fnCallLog in call_logs[fn_name]:
            params = fnCallLog.getParams()
            for i in xrange(len(params)):
                value = params[i].get_value()
                counts[i][value] += 1

    return counts

if __name__ == "__main__":
    if len(argv) != 2:
        print "Usage: ./analyze.py <log file>"
    fname = argv[1]
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

    for fn_name, calllogs in call_logs.items():
        print fn_name
        for cl in calllogs:
            print cl.tostr()
    print argCounts

    fn_name = "test"
    args_to_fix = []
    MIN_THRESHOLD = 0.45
    # Iterate over all Counter objects:
    for i in xrange(len(argCounts)):
        arg_counter = argCounts[i]
        total = float(sum(arg_counter.values())) # total of all counts
        # Iterate over all (argument, argument count) pairs:
        for arg, count in arg_counter.items():
            arg_name = getArgumentName(fn_name, i, call_logs)
            if count / total >= MIN_THRESHOLD:
                args_to_fix += [FixedArgument(fn_name, i, arg, count, arg_name)]

    # 
    for arg_to_fix in args_to_fix:

