"""Used to create output file for course staff to examine.
"""

import sys
from functools import partial

sys.path.append('..')


from classes import Feature, Main, Model, Trainer

from test import mean_func


def main():
    print('Feature creation...')
    feat = Feature('score', partial(mean_func, column='score'))

    print('Model creation...')
    m = Model('base', [feat])

    print('Training model...')
    t = Trainer(m)
    print(t.run())

    print('Running model...')
    result = Main.main(m)

    print('Saving result...')
    result.to_csv(f'out_{m.name}.csv')


if __name__ == '__main__':
    main()
