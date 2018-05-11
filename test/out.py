"""Used to create output file for course staff to examine.
"""

import sys
from functools import partial

sys.path.append('..')


from classes import Feature, Main, Model, Trainer

from test import seed_func, rpi_func


def main():
    print('Feature creation...')
    feat = Feature('seed', seed_func)

    print('Model creation...')
    m = Model('seed_and_rpi', [
        Feature('seed', seed_func),
        Feature('rpi', rpi_func)
    ])

    print('Training model...')
    t = Trainer(m)
    print(t.run())

    print('Running model...')
    result = Main.main(m)

    print('Saving result...')
    result.to_csv(f'out_{m.name}.csv')


if __name__ == '__main__':
    main()
