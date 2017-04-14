#!/usr/bin/python

from sys import argv


if __name__ == "__main__":
    if len(argv) != 2:
        print "Usage: ./codegen.py <cuda file>"
    fname = argv[1]
    print "Annotating ", fname
    fd = open(fname, 'r')
    lines = fd.readlines()

    gen = []
    for line in lines:
        if
