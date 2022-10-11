from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from alphashape import alphashape


class AlphaShapeFeatures(BaseEstimator, TransformerMixin):
    """
    mask based features
    """
    def __init__(self, alpha = 0.5):
        self.alpha = alpha

    def fit(self, X = None, y = None):
        return self

    def transform(self,X):
        localization = X[0].copy()

        proteins = localization["protein"].unique()

        features = dict()
        for pr in proteins:
            indx = localization["protein"] == pr
            indx = indx & (localization.loc[:,"cluster"] != -1)
            if indx.sum() > 0:
                points_3d = localization.loc[indx,["x","y","z"]].to_numpy()
                alpha_shape_pr = alphashape(points_3d, self.alpha)
                features["AS_volume_" + pr] = alpha_shape_pr.volume
                features["AS_density_" + pr] = alpha_shape_pr.volume / indx.sum()
                features["AS_area_" + pr] = alpha_shape_pr.area
                alpha_shape_pr = None
            else:
                features["AS_volume_" + pr] = 0.
                features["AS_area_" + pr] = 0.

        return features