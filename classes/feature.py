class Feature:
    """Represents a particular entry in a team vector. Computed on a team in a
    given season."""

    def __init__(self, name, compute_func, **kwargs):
        """
        name: string of name describing this feature
        compute_func: df (Data, team, context, **kwargs) -> numeric value
        """
        self.name = name
        self.func = compute_func
        self.kwargs = kwargs if kwargs is not None else {}

    def compute(self, dm, team, context):
        return self.func(dm, team, context, **self.kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
