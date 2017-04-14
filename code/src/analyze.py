#!/usr/bin/python

from sys import argv
from collections import Counter

class Param(object):
    def __init__(self): pass
    def add_type(self, mytype): self.mytype = mytype
    def add_value(self, value): self.value = value
    def get_type(self): return self.mytype
    def get_value(self):return self.value
    def tostr(self):    return str(self.mytype) + " " + str(self.value)

class FnCallLog(object):
    def __init__(self, thr_idx, fn_name, argv):
        # 1:1 type to its value
        assert len(argv) % 2 == 0
        self.thr_idx = thr_idx
        self.name = fn_name
        self.params = []
        for i in xrange(len(argv)):
            if i % 2 == 0: # type
                param = Param()
                param.add_type(argv[i])
            else: # value
                param.add_value(argv[i])
                self.params.append(param)
    def getFnName(self): return self.name
    def getThrIdx(self): return self.thr_idx
    def getParams(self): return self.params
    def tostr(self):
        ans = self.name + "[" + self.thr_idx + "]("
        for param in self.params:
            ans += param.tostr() + ","
        ans += ")"
        return ans

def getCounts(fnCallLogLst):

    # TODO: properly get number of params, specific
    #  for all different functions
    len_params = len(fnCallLogLst[0].getParams())

    # counts[i] is param i's dictionary of value to count of value
    counts = [] # list of dictionaries
    for i in xrange(len_params):
        counts += [Counter()]

    for k in xrange(len(fnCallLogLst)):
        fnCallLog = fnCallLogLst[k]
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

    call_logs = []
    for line in lines:
        toks = line.strip().split(" ")
        if "CALL_LOG" == toks[0]:
            fnCallLog = FnCallLog(toks[1], toks[2], toks[3:])
            call_logs += [fnCallLog]

    # e.g.
    #   [Counter({'1': 32, '0': 32}),
    #    Counter({'11': 5, '10': 5, '1': 5, '0': 5, ...}),
    #    Counter({'0': 10, '1': 9, '3': 9, '2': 9, ...})]
    # meaning: argument 0 is called with 1 32 times and 0 32 times
    #          argument 1 is called with 11 5 times, 10 5 times, etc...
    #          argument 2 is called with 0 10 times, 1 9 times, etc...
    countsList = getCounts(call_logs)

    print call_logs
    print countsList
