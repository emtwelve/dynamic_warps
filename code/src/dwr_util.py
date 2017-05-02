
def pprint_argCounts(counts):
    for fn_name in counts:
        print fn_name + ":"
        for elem in counts[fn_name]:
            assert(type(elem) == list and len(elem) == 3)
            counter, param, d = elem[0], elem[1], elem[2]
            print "    " + "ValueCounter:\n        ", counter
            print "    " + "Param:\n        ", param
            print "    " + "ValueThreads:"
            for key in d:
                threadList = d[key]
                print "            ", str(key)+":", threadList
            print "    ~~~~~~~~"


