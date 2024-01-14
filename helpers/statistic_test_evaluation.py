from distutils.log import error
import numpy as np
import warnings
import os
from tabulate import tabulate
from scipy.stats import ttest_ind
from scipy.stats import ranksums


""" 
Class generate tables with paired statistic tests and store it
"""


class StatisticTest():
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def process(self, table_name, alpha=0.05, m_fmt="%3f", std_fmt=None, nc="---", db_fmt="%s", tablefmt="plain"):
        """Process"""

        # Ignore warnings
        warnings.filterwarnings("ignore")

        try:
            # PARAM X FOLD X CLASSIFIER X METRIC 
            scores = self.evaluator.scores
            mean_scores = self.evaluator.mean
            std = self.evaluator.std
            metrics = list(self.evaluator.metrics.keys())
            clfs = list(self.evaluator.clfs.keys())
            parameters = ["highestSlot", "avgHighestSlot", "sumOfSlots", "avgActiveTransceivers"]
            n_clfs = len(clfs)
            t = []
            # Generate tables 
            for m_id, m_name in enumerate(metrics):
                t.append([db_fmt % m_name])
                for param_idx, param_name in enumerate(parameters):
                    # Mean value
                    t.append(['']+[db_fmt % param_name] + ["%.3f" % v for v in mean_scores[param_idx, :, m_id]])
                    # If std_fmt is not None, std will appear in tables
                    if std_fmt:
                        t.append([''] + [std_fmt % v for v in std[param_idx, :, m_id]])
                    # Calculate T and P for T-studenta test
                    T, p = np.array(
                        [[ttest_ind(scores[param_idx, :, i],
                            scores[param_idx, :, j])
                        for i in range(len(clfs))]
                        for j in range(len(clfs))]
                    ).swapaxes(0, 2)
                    _ = np.where((p < alpha) * (T > 0))
                    conclusions = [list(1 + _[1][_[0] == i])
                                for i in range(n_clfs)]
            
                    t.append([''] + [''] + [", ".join(["%i" % i for i in c])
                                    if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else nc)
                                    for c in conclusions])

            # Show outputs  
            headers = ['Metric','Parameter']
            for i in clfs:
                headers.append(i)
            print(tabulate(t, headers))

            # Save outputs as .tex extension
            os.chdir('../latexTable')
            with open('%s.txt' % (table_name), 'w') as f:
                    f.write(tabulate(t, headers, tablefmt='latex'))

        except ValueError:
            error('Incorrect value!')

    def rank_process(self, table_name, alpha=.05, nc="---", tablefmt="plain"):
        ranks = self.evaluator.ranks
        clfs = list(self.evaluator.clfs.keys())
        mean_ranks = np.mean(ranks, axis=1)
        t = []

        for m, metric in enumerate(self.evaluator.metrics):
            metric_ranks = ranks[m,:,:]
            length = len(clfs)

            s = np.zeros((length, length))
            p = np.zeros((length, length))

            for i in range(length):
                for j in range(length):
                    s[i, j], p[i, j] = ranksums(metric_ranks.T[i], metric_ranks.T[j])
            _ = np.where((p < alpha) * (s > 0))
            conclusions = [list(1 + _[1][_[0] == i])
                           for i in range(length)]

            t.append(["%s" % metric] + ["%.3f" %
                                           v for v in
                                           mean_ranks[m]])

            # t.append([''] + [", ".join(["%i" % i for i in c])
            #                  if len(c) > 0 else nc
            #                  for c in conclusions])
            t.append([''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else nc)
                             for c in conclusions])

        # Show outputs
        print(tabulate(t, headers=(clfs), tablefmt='plain'))

        # Save outputs
        #print(os.getcwd())
        os.chdir('../latexTable')
        with open('%s.txt' % (table_name), 'w') as f:
            f.write(tabulate(t, headers=(clfs), tablefmt='latex'))