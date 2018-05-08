"""Simple test script.
"""

import sys
sys.path.append('..')

from classes import *


def f1_func(dm, team, context, **kwargs):
    # Filter
    filt = dm.get_team_in_season(context.season, team)

    # Return std score
    return filt['score'].std()


def f2_func(dm, team, context, **kwargs):
    # Filter
    filt = dm.get_team_in_season(context.season, team)

    # Return mean fta
    return filt['fta'].mean()


def f3_func(dm, team, context, **kwargs):
    # Get the other team
    me = team.team_id
    other = context.other.team_id

    # Get all games where these 2 teams have played
    reg = dm.data.reg
    filt = reg[((reg.Wteam == me) & (reg.Lteam == other)) |
               ((reg.Lteam == me) & (reg.Wteam == other))]

    if filt.shape[0] == 0:
        return 0

    filt_win = filt[filt.Wteam == me]
    filt_win_mean = 0 if filt_win.shape[0] == 0 else filt_win['Wscore'].mean()

    filt_lose = filt[filt.Lteam == me]
    filt_lose_mean = 0 if filt_lose.shape[0] == 0 else filt_lose['Lscore'].mean()

    return filt_win_mean - filt_lose_mean


def main():
    # Create features
    f1 = Feature('test', f1_func, key='value')
    f2 = Feature('test_again', f2_func)
    f3 = Feature('omg', f3_func)

    # Create model
    m = Model('base', [f1, f2, f3], "LogisticRegression")

    # Create trainer and run
    t = Trainer(m)
    result = t.run()

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
