# Simple script to read the file comparing log-likelihood values and print out a summary
import sys
import os
import csv
import numpy as np

n_adapt = int(sys.argv[1])
n_trials = 50
index_col = 1

methods = ['no_adaptation', 'proposed', 'transfer', 'transfer_last_layer', 'retrained_target']
loglike = {m: [] for m in methods}

fname = os.path.join('AWGN_to_fading_{:d}'.format(n_adapt), 'loglike_comparison.csv')
fp = open(fname, mode='r')
cr = csv.reader(fp, delimiter=',')
for i, row in enumerate(cr):
    if i == 0:
        continue

    loglike[row[0]].append(float(row[index_col]))

fp.close()
for m in methods:
    vals = np.array(loglike[m])[:n_trials]
    p = np.percentile(vals, [2.5, 50, 97.5])
    if vals.shape[0] != n_trials:
        print("WARNING: method {} calculates using {} trials.".format(m, vals.shape[0]))

    print("\n{}: mean = {:.4f}, median = {:.4f}, CI = [{:.4f}, {:.4f}]".format(m, np.mean(vals), p[1], p[0], p[2]))

print('\n')
