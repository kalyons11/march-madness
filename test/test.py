"""Simple test script.
"""

import sys
sys.path.append('..')

from classes import *


def main():
    # Create features
    f1 = Feature('test', lambda d: d['score'].std())
    f2 = Feature('test_again', lambda d: d['fta'].mean())

    # Create model
    m = Model('base', [f1, f2], "LogisticRegression")

    # Create trainer and run
    t = Trainer(m)
    result = t.run()

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
