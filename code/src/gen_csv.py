from analyze import parseCallLogFile, getArgCounts
from dwr_util import pprint_argCounts
from collections import defaultdict
from sys import argv

def gen_csv(call_log_filename):
  call_logs = parseCallLogFile(call_log_filename)
  argCounts = getArgCounts(call_logs)

  NUM_THREADS = 4
  NUM_ARGS = 8

  threadCounts = defaultdict(list)
  # initialize the argCounts for all threads:
  for i in xrange(NUM_THREADS):
    threadCounts[i] = [None]*NUM_ARGS


  global_param_idx = 0
  for fn_name, tuples in argCounts.items():
    for i in xrange(len(tuples)):
      value_counter, param, thread_dict = tuples[i]


      global_param_idx += 1

    print fn_name, tuples

if __name__ == "__main__":
  if len(argv) != 2:
    print "Usage: ./gen_csv.py <call log file>.log"

  call_log_filename = argv[1]
  gen_csv(call_log_filename)