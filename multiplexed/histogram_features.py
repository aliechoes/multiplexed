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
    def __init__(self, diameter = 10, bins = 20):
        self.diameter = diameter
        self.bins = bins
    
    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        localization = X.copy() 

        xmin, xmax = localization["x"].min(),localization["x"].max()
        ymin, ymax = localization["y"].min(),localization["y"].max()
        zmin, zmax = localization["z"].min(),localization["z"].max()


        if (xmax - xmin) > self.diameter:
            raise ValueError("small diameter. The data has a range of", 
                            (xmax - xmin),
                            "in comparison to the input diameter:", 
                            self.diameter)
        
        if (ymax - ymin) > self.diameter:
            raise ValueError("small diameter. The data has a range of", 
                            (ymax - ymin),
                            "in comparison to the input diameter:", 
                            self.diameter)
        
        if (zmax - zmin) > self.diameter:
            raise ValueError("small diameter. The data has a range of", 
                            (zmax - zmin),
                            "in comparison to the input diameter:", 
                            self.diameter)

        center_x = (xmax + xmin)/2.
        center_y = (ymax + ymin)/2.
        center_z = (zmax + zmin)/2.

        cols_ = ["x","y","z"]
        localization[cols_] = localization[cols_].subtract([  center_x,
                                                            center_y,
                                                            center_z] )
        localization[cols_] = localization[cols_].divide([    self.diameter,
                                                            self.diameter,
                                                            self.diameter])

        histograms = dict()
        for _ , pr in  enumerate(localization["protein"].unique()):
            indx = localization["protein"] == pr 
            hist, _ = np.histogramdd(  localization.loc[indx,cols_].to_numpy(), 
                                            bins = self.bins,
                                            range = ((-1,1),(-1,1),(-1,1)) )
            
            histograms[pr] = hist
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

        proteins = localization["protein"].unique()
        
        features = dict()
        features["HF_diameter_x"] = localization.x.max() -  localization.x.min()
        features["HF_diameter_y"] = localization.y.max() -  localization.y.min()
        features["HF_diameter_z"] = localization.z.max() -  localization.z.min()
        for pr in proteins:
            features["HF_mean_" + pr] = histograms[pr].ravel().mean() # intensities
            features["HF_std_" + pr] = histograms[pr].ravel().std() # intensities
            features["HF_skewness_" + pr] = skew(histograms[pr].ravel()) # intensities
            features["HF_kurtosis_" + pr] = kurtosis(histograms[pr].ravel()) # intensities
            features["HF_entropy_" + pr] = entropy(histograms[pr].ravel()) # intensities
            features["HF_min_" + pr] = histograms[pr].ravel().min() # intensities
            features["HF_max_" + pr] = histograms[pr].ravel().max() # intensities
            
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

        proteins = list(histograms.keys())
        features = dict()
        for i, pr1 in enumerate(proteins):
            for _, pr2 in enumerate(proteins[i+1:]):
                features["HD_ws_" + pr1 + "_" +  pr2] = wasserstein_distance(  histograms[pr1].ravel(), 
                                                                            histograms[pr2].ravel())
                features["HD_js_" + pr1 + "_" +  pr2] = jensenshannon( histograms[pr1].ravel(), 
                                                                    histograms[pr2].ravel())
                features["HD_cosine_distance_" + pr1 + "_" +  pr2] = cosine(   histograms[pr1].ravel(), 
                                                                            histograms[pr2].ravel())

                features["HD_mean_" + pr1 + "/mean_" + pr2] =  histograms[pr1].ravel().mean() / \
                                                            (histograms[pr2].ravel().mean() +  self.eps)
                                                        
                                                        
                hist1_maxvalues = np.array(np.unravel_index(histograms[pr1].argmax(), 
                                                            histograms[pr1].shape))
                
                hist2_maxvalues = np.array(np.unravel_index(histograms[pr2].argmax(), 
                                                            histograms[pr2].shape))
                
                features["HD_hist_max_" +  pr1 + "_" +  pr2] = np.linalg.norm( hist2_maxvalues-\
                                                                            hist1_maxvalues)
                
            
        return features

