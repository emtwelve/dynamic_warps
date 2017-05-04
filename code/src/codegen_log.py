#!/usr/bin/python

from sys import argv

THREAD_IDX = "blockIdx.x * blockDim.x + threadIdx.x"

# TO LOG FUNCTION CALLS:
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

def add_all_call_loggers(lines):
    gen = []
    for i in xrange(len(lines)):
        line = lines[i]

        if "__device__" in line:
            gen += [line] + add_call_logger(i, lines)

        else:
            gen += [line]

    gen = ''.join(gen)
    return gen

def find_parens(s):
    """ Source: http://stackoverflow.com/questions/29991917/
                indices-of-matching-parentheses-in-python """
    toret = {}
    pstack = []
    for i, c in enumerate(s):
        if c == '{':
            pstack.append(i)
        elif c == '}':
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i
    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))
    return toret

def add_thread_basicblock_loggers(fd):
    source = fd.read()

    prolog = '#define BBLOG(bbid) printf("%d,%d\\n", blockIdx.x * blockDim.x + threadIdx.x, bbid)\n'

    # Firstly, find all global and device "{"'s
    #           and all for    and while  "{"'s:
    fnLbrackets = []
    for line in source.split('\n'):
        if "__device__" in line or "__global__" in line:
            # Get index of line in entire source:
            offset = source.find(line)
            # Get location of function's basic block left bracket
            fnLbracketIdx = offset + line.find("{")
            fnLbrackets.append(fnLbracketIdx)

    # Get matching parens
    all_matching_brackets = find_parens(source)

    # Find all other internal lparens:
    allLbrackets = []
    for fnLbracket in fnLbrackets:
        fnRbracket = all_matching_brackets[fnLbracket]

        for subLbracket in all_matching_brackets:
            if (fnLbracket < subLbracket and
                subLbracket < fnRbracket):
                allLbrackets.append(subLbracket)
    allLbrackets = sorted(allLbrackets)

    # Insert the logging statements:
    parts = [source[i:j] for i,j in zip(allLbrackets,
                                   allLbrackets[1:]+[None])]
    print parts
    for i in xrange(len(parts)):
        parts[i] = "{ BBLOG(" + str(i) + ");\n" + parts[i][1:]

    fd = open("num_bbs", 'w')
    fd.write(str(len(parts)))
    fd.close()

    source_anno = prolog + source[:allLbrackets[0]] + ''.join(parts)
    return source_anno

if __name__ == "__main__":
    print "Beginning codegen_log.py"
    if len(argv) != 2:
        print "Usage: ./codegen_log.py <cuda file>"; assert(False)
    fname = argv[1]
    print "Annotating ", fname
    fd = open(fname, 'r')

    #lines = fd.readlines()

    # Entire source code annotated with call loggers
    # gen = add_all_call_loggers(lines)

    # Entire source code annotated with (thread_idx, basic_block_idx) loggers
    gen = add_thread_basicblock_loggers(fd)

    fd = open("anno_" + fname, 'w')
    fd.write(gen)
    fd.close()

