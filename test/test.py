"""Simple test script.
"""

import sys
sys.path.append('..')

from classes import *


def f1_func(dm, team, context, **kwargs):
    filt = dm.get_team_in_season(context.season, team)
    return dm.get_mean_stat(df=filt, team=team, column='score')


def f2_func(dm, team, context, **kwargs):
    filt = dm.get_team_in_season(context.season, team)
    return dm.get_mean_stat(df=filt, team=team, column='to')


def f3_func(dm, team, context, **kwargs):
    filt = dm.get_team_in_season(context.season, team)
    return dm.get_mean_stat(df=filt, team=team, column='fgm')


def win_func(dm, team, context, **kwargs):
    # Filter on the team
    filt = dm.get_team_in_season(context.season, team)

    # Count the number of wins
    wins = filt[filt.Wteam == team.team_id]

    # Win percentage
    return wins.shape[0] / filt.shape[0]


# average margin of victory

# include the seed


def previous_matchups_with_postseason(dm, team, context, **kwargs):
    # Get the other team
    me = team.team_id
    other = context.other.team_id

    # Filter back 5 years
    reg = dm.data.reg
    post = dm.data.tourney
    reg_this_yr = reg[reg.Season >= context.season.yr - 5]
    reg_this_yr = reg_this_yr.append(post)

    # Get win percentage against this team
    these = reg_this_yr[((reg.Wteam == me) & (reg.Lteam == other)) |
                        ((reg.Lteam == me) & (reg.Wteam == other))]

    if these.shape[0] > 0:
        # Get the win percentage
        wins = these[these.Wteam == me]
        return wins.shape[0] / these.shape[0]

    return 0


def main():
    # Create features
    f1 = Feature('score', f1_func, key='value')
    f2 = Feature('to', f2_func)
    f3 = Feature('fgm', f3_func)
    f4 = Feature('win_pct', win_func)
    f6 = Feature('previous_matchups_with_postseason_5',
                 previous_matchups_with_postseason)

    # Create model
    m = Model('base', [f1, f2, f3, f4, f6], "LogisticRegression",
              mode="concat")

    # Create trainer and run
    t = Trainer(m)
    result = t.run()

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
