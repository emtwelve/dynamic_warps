#!/usr/bin/python

from sys import argv
from analyze import analyze
from collections import Counter

if __name__ == "__main__":
    if len(argv) != 3:
        print "Usage: ./codegen_remap.py <cuda file> <warp log file>"; assert(False)

    test_fname = argv[1]
    warplog_fname = argv[2]

    fd_test = open(test_fname, 'r')
    fd_log  = open(warplog_fname, 'r')

    remap_array_str = fd_log.readlines()[0]
    length = str(len(remap_array_str.split(',')))

    prolog = \
"""
int host_remap[""" + length + """] = {""" + remap_array_str + """};
__device__ volatile int device_remap[""" + length + """];
int *remap() {
    //int *device_remap;
    cudaMemcpyToSymbol(device_remap, host_remap, """ + length + """ * sizeof(int));
    //return device_remap;
    return NULL;
}
"""

    lines = fd_test.readlines()
    for i in xrange(len(lines)):
        if (i < 8): continue # ignore header comments
        if "int index" in lines[i]:
            lines[i] += "\nvolatile int ridx = device_remap[index];\n"
        elif "index" in lines[i] and "int index" not in lines[i]:
            index_split = lines[i].split("index")
            ridx_line = ""
            for j in xrange(len(index_split)-1):
                ridx_line += index_split[j] + "ridx"
            ridx_line += index_split[j+1]

            lines[i] = ridx_line

        if "int main(" in lines[i] or "int main (" in lines[i]:
            lines[i] += "\nremap();\n"



    print prolog + ''.join(lines)
    # Get device function names
    """
    device_fn_names = []
    for line in lines:
        if "__device__" in line:
            # __device__ int fn_name ( ... ) {

            # [__device__, int, fn_name, ...]
            toks = line.split()

            # 2nd element in toks list is function name
            fn_name = toks[2]

            # chop off everything after the '('
            last_idx = fn_name.find("(")
            if last_idx != -1: fn_name = fn_name[:last_idx]

            device_fn_names += [fn_name]

    print device_fn_names
    """

    # Insert remap into device function calls and declarations
    """
    for i in xrange(len(lines)):
        for fn_name in device_fn_names:
            if fn_name in lines[i] and "(" in lines[i]:
                rparen_split = lines[i].split(")")
                print rparen_split
                assert(len(rparen_split) == 2)
                if "__device__" in lines[i]:
                    lines[i] = rparen_split[0] + ", int *remap)" + rparen_split[1]
                else:
                    lines[i] = rparen_split[0] + ", remap)" + rparen_split[1]

    print prolog + ''.join(lines)
    """


