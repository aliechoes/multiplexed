import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import cosine
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class HistogramGenerator(BaseEstimator, TransformerMixin):
    """
    mask based features
    """
    def __init__(self, diameter = 7, bins = 20):
        self.diameter = diameter
        self.bins = bins
    
    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        localization = X.copy() 

        xmin, xmax = localization["x"].min(),localization["x"].max()
        ymin, ymax = localization["y"].min(),localization["y"].max()
        zmin, zmax = localization["z"].min(),localization["z"].max()

        assert (xmax - xmin) < 2*self.diameter, "small diameter"
        assert (ymax - ymin) < 2*self.diameter, "small diameter"
        assert (zmax - zmin) < 2*self.diameter, "small diameter"

        center_x = (xmax + xmin)/2.
        center_y = (ymax + ymin)/2.
        center_z = (zmax + zmin)/2.

        cols = ["x","y","z"]
        localization[cols] = localization[cols].subtract([  center_x,
                                                            center_y,
                                                            center_z] )
        localization[cols] = localization[cols].divide([    self.diameter,
                                                            self.diameter,
                                                            self.diameter])

        histograms = dict()
        for i, ch in  enumerate(localization["channel"].unique()):
            indx = localization["channel"] == ch 
            hist, _ = np.histogramdd(  localization[indx,cols].to_numpy(), 
                                            bins = self.bins,
                                            range = ((-1,1),(-1,1),(-1,1)) )
            
            histograms[ch] = hist
            hist = None
            
        return [X, histograms]

class HistogramsStatistics(BaseEstimator, TransformerMixin):
    """
    mask based features
    """
    def __init__(self): 
        pass
    
    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        localization = X[0].copy() 
        histograms = X[1].copy() 

        channles = localization["channel"].unique()
        
        features = dict()
        for ch in channles:
            features["mean_" + ch] = histograms[ch].ravel().mean()
            features["std_" + ch] = histograms[ch].ravel().std()
            features["skewness_" + ch] = skew(histograms[ch].ravel())
            features["kurtosis_" + ch] = kurtosis(histograms[ch].ravel())
            features["entropy_" + ch] = entropy(histograms[ch].ravel())
            features["min_" + ch] = histograms[ch].ravel().min()
            features["max_" + ch] = histograms[ch].ravel().max()
            
        return features


class HistogramsDistances(BaseEstimator, TransformerMixin):
    """
    mask based features
    """
    def __init__(self, eps=1e-16): 
        self.eps = eps
    
    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        localization = X[0].copy() 
        histograms = X[1].copy() 

        channles = localization["channel"].unique()
        
        features = dict()
        for i, ch1 in enumerate(channles):
            for j, ch2 in enumerate(channles)[i+1:]:
                features["ws_" + ch1 + "_" +  ch2] = wasserstein_distance(  histograms[ch1], 
                                                                            histograms[ch1])
                features["js_" + ch1 + "_" +  ch2] = jensenshannon( histograms[ch1], 
                                                                    histograms[ch2])
                features["cosine_distance_" + ch1 + "_" +  ch2] = cosine(   histograms[ch1], 
                                                                            histograms[ch2])

                features["mean_" + ch1 + "/mean_" + ch2] =  histograms[ch1].ravel().mean() / \
                                                            (histograms[ch2].ravel().mean() +  eps)
                                                        
                                                        
                hist1_maxvalues = np.array(np.unravel_index(histograms[ch1].argmax(), 
                                                            histograms[ch1].shape))
                hist2_maxvalues = np.array(np.unravel_index(histograms[ch2].argmax(), 
                                                            histograms[ch2].shape))
                features["hist_max_" +  ch1 + "_" +  ch2] = np.linalg.norm( hist2_maxvalues-\
                                                                            hist1_maxvalues)
                
            
        return features