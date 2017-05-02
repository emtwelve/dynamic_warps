#!/usr/bin/python

from sys import argv
from analyze import analyze
from collections import Counter

def splitKeep(s, delim):
    """ split method on s except keep delim at end rather than removing it """
    return map(lambda x: x + delim, s.split(delim))

def removeEnd(s, delim):
    """ Remove everything past the last instance of delim """
    length_to_end = len(s.split(delim)[-1])
    return s[:len(s)-length_to_end-1]

if __name__ == "__main__":
    if len(argv) != 2:
        print "Usage: ./codegen.py <cuda file>"; assert(False)
    fname = argv[1]
    print "Creating optimized ", fname
    fd = open(fname, 'r')

    # :-5 slicing to chop off the cu.cu part of the filename
    #   For example simplecu.cu gets converted to log_simple.log:
    log_file = "log_" + fname[:-5] + ".log"

    # Obtain the actual branch function to write to the opt_<filename>cu.cu file,
    #   and the arg_fixed functions.
    branch_functions, arg_fixed_functions, warp_rescheduler = analyze(log_file)
    
    print "Beginning codegen_opt.py"

    lines = fd.readlines()

    hasArgFixedGend = {}
    gen = []
    looking_for_end = False
    fn_ends = {}
    argfixed_count = Counter()
    for i in xrange(len(lines)):
        line = lines[i]

        if "__device__" in line:
            looking_for_end = True
            fn_name = line.split(" ")[2]

            print "Adding placeholders for", fn_name
            # Add argfixed function and placeholders:
            for arg in arg_fixed_functions[fn_name]:
                for val, argfixed_code in arg_fixed_functions[fn_name][arg].items():
                    gen += splitKeep(argfixed_code, "\n")

            print "Adding branching function for", fn_name
            # Add branching function after argfixed functions:
            gen += splitKeep(branch_functions[fn_name], '\n')

        else:

            # Not looking for end of a device function:
            if not looking_for_end:
                gen += [line]
                fn_end = []

            # Looking for end of a device function:
            else:
                fn_end += [line]
                if line == "}\n": # Marker to find end of device function
                    looking_for_end = False
                    # keep track of the end to fill in later at placeholder
                    fn_ends[fn_name] = fn_end
            

    print "Adding argfixed functions at ###REST_ points"
    # Now fill in rest of argfixed fn device code at placeholders:
    look_for = "###REST_"
    for i in xrange(len(gen)):
        line = gen[i]
        if look_for == line[:len(look_for)]:
            # Get arg fixed fn name by parsing the ####REST_ line:
            argfixed_fn_name = line.split(look_for)[1].strip() # gen[i][len(gen[i])-len(look_for)-1:-1]
            # Get original fn name by removing everything past the last two underscores: 
            fn_name = removeEnd(removeEnd(argfixed_fn_name, "_"), "_")
            # Replace placholder with the actual end-of-device-function code:
            gen[i] = ''.join(fn_ends[fn_name])

    print "Replacing original device function calls with branching function call"
    for i in xrange(len(gen)):
        line = gen[i]
        for fn_name in arg_fixed_functions:
            if " " + fn_name + "(" in line: # must match this in source:
                sep = line.split(fn_name)
                if len(sep) != 2:
                    print line, sep, len(sep); assert(False)
                gen[i] = sep[0] + "branch_" + fn_name + sep[1]


    gen = ''.join(gen)
    fd = open("opt_" + fname, 'w')
    fd.write(gen)


