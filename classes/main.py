import pandas as pd

from .model import Model
from .runner import Runner
from .season import Season


class Main:
    """Immutable class we will call main() on to run the project."""

    @classmethod
    def main(self):
        """Runs runners on each season that we need to. Returns dataframe
        with predictions."""
        result = pd.DataFrame()
        for season in [2011, 2012, 2013]:
            r = Runner(Season(season), Model.default())
            current = r.run()
            result = result.append(current)
            break
        return result
