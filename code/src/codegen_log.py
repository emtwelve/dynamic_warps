#!/usr/bin/python

from sys import argv

THREAD_IDX = "blockIdx.x * blockDim.x + threadIdx.x"

def add_call_logger(i, lines):
    assert("__device__" in lines[i])
    newline = []
    tokens = lines[i].split(' ')
    dev = tokens[0]
    typ = tokens[1]
    fn_name = tokens[2]

    assert(typ in ['int', 'void', 'char', 'float'])
    assert(dev == "__device__")

    # Create new code to log function name and params
    newline = '\n\t/* GENERATED */ printf("CALL_LOG %d ' + fn_name + " "
    params = ", " + THREAD_IDX
    for j in xrange(3, len(tokens)):
        if (tokens[j] in ['int', 'void', 'char', 'float',
                          'unsigned', 'bool', 'void*']):
            if tokens[j] == 'int':
                newline += 'int %d '
            elif tokens[j] == 'void':
                pass
            elif tokens[j] == 'char':
                newline += 'char %c '
            elif tokens[j] == 'float':
                newline += 'float %f '
            elif tokens[j] == 'unsigned':
                newline += 'unsigned %u '
            elif tokens[j] == 'void*':
                newline += 'void* %p '
            elif tokens[j] == 'bool':
                newline += 'bool %u '
            newline += tokens[j+1] + ' ' # add parameter name
            params += ", " + tokens[j+1]

    newline += '\\n"' + params + ");\n"

    return list(newline)


if __name__ == "__main__":
    print "Beginning codegen_log.py"
    if len(argv) != 2:
        print "Usage: ./codegen_log.py <cuda file>"; assert(False)
    fname = argv[1]
    print "Annotating ", fname
    fd = open(fname, 'r')

    lines = fd.readlines()

    gen = []
    for i in xrange(len(lines)):
        line = lines[i]

        if "__device__" in line:
            gen += [line] + add_call_logger(i, lines)

        else:
            gen += [line]

    gen = ''.join(gen)
    fd = open("log_" + fname, 'w')
    fd.write(gen)
