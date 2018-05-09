from .feature import Feature


class FeatureVector:
    """Represents a vector of features. Immutable."""

    def __init__(self, feature_dict):
        """feature_dict: dict[Feature -> numeric]"""
        self.feature_dict = feature_dict

    def __getitem__(self, k):
        return self.feature_dict[k]

    def __str__(self):
        return str(self.feature_dict)

    def __sub__(self, other):
        """computes self - other"""
        final = {}
        for k in self.feature_dict:
            final[k] = self[k] - other[k]
        return FeatureVector(final)

    def concat(self, other):
        """combines self and other into new vect"""
        result = {**self.feature_dict}
        for k, v in other.feature_dict.items():
            result[Feature(str(k) + '_other', k.func, **k.kwargs)] = v
        return FeatureVector(result)

    def to_list(self):
        """uses sorted key order so it is deterministic"""
        result = []
        for k in sorted(self.feature_dict, key=lambda f: f.name):
            result.append(self[k])
        return result
