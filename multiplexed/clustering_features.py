from sklearn.cluster import DBSCAN 
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

def get_cluster_features(localization_group, 
                         cluster_info ):
    
    localization_group_ = localization_group.copy()
    ## finding biggest clusters and filter them
    value_counts = localization_group_.cluster.value_counts()
    
    main_clusters = value_counts.argmax()
    
    min_samples = cluster_info["min_samples"][0]

    if min_samples <= value_counts.max():
    
        indx = localization_group_.cluster.isin([main_clusters])
        localization_group_ = localization_group_.copy().loc[indx,:]

        ## calculating the Center of Mass
        com_x = localization_group_.x.mean()
        com_y = localization_group_.y.mean()
        com_z = localization_group_.z.mean()

        ## calculating std
        std_x = localization_group_.x.std()
        std_y = localization_group_.y.std()
        std_z = localization_group_.z.std()
        
        ## calculating convex hull
        convex_hull = ConvexHull(localization_group_.loc[:,["x","y","z"]].to_numpy()).volume
        
    else:
        ## calculating the Center of Mass
        com_x = -1.
        com_y = -1.
        com_z = -1.

        ## calculating std
        std_x = 0.
        std_y = 0.
        std_z = 0.
        convex_hull = 0

    ## calculating volume
    volume = np.power( (std_x + std_y + (std_z)) / 3 * 2, 3 ) * np.pi * 4 / 3
    
    ch = cluster_info["Proteins"][0]
    features = dict()
    features["cluster_size_" + ch] = len(localization_group_)
    features["com_x_" + ch] = com_x
    features["com_y_" + ch] = com_y
    features["com_z_" + ch] = com_z

    features["std_x_" + ch] = std_x
    features["std_y_" + ch] = std_y
    features["std_z_" + ch] = std_z
    
    features["volume_" + ch] = volume
    features["convex_hull_" + ch] = convex_hull
    return features


    

class ClusteringGenerator(BaseEstimator, TransformerMixin):
    """
    mask based features
    """
    def __init__(   self, 
                    path_to_clustering_info, 
                    default_radius = 0.6, 
                    default_min_density = 5, 
                    default_min_samples = 400):
        self.cluster_info = pd.read_csv(path_to_clustering_info)
        self.default_radius = default_radius
        self.default_min_density = default_min_density
        self.default_min_samples = default_min_samples
    
    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        localization = X[0].copy() 
        histograms = X[1].copy()  

        localization["cluster"] = -1
        cols = ["x","y","z"]
        channles = localization["channel"].unique()

        for ch in channles:
            if ch in self.cluster_info.Proteins.tolist():
                ch_indx = self.cluster_info.Proteins == ch
                ch_cluster_info = self.cluster_info.loc[ch_indx,:].to_dict('list')
            else:
                ch_cluster_info = { 'Proteins': [ch],
                                    'radius': [self.default_radius],
                                    'min_density': [self.default_min_density],
                                    'min_samples': [self.default_min_samples]}
            
            indx = localization["channel"] == ch
            db = DBSCAN(    eps=ch_cluster_info["radius"][0], 
                            min_samples=ch_cluster_info["min_density"][0])
            
            db.fit(localization.loc[indx,cols].astype(float).to_numpy())
            localization.loc[indx,"cluster"] = db.labels_  

        return [localization, histograms]


class ClusteringFeatures(BaseEstimator, TransformerMixin):
    """
    mask based features
    """
    def __init__(   self, 
                    path_to_clustering_info, 
                    default_radius = 0.6, 
                    default_min_density = 5, 
                    default_min_samples = 400,
                    eps = 1e-16):
        self.cluster_info = pd.read_csv(path_to_clustering_info)
        self.default_radius = default_radius
        self.default_min_density = default_min_density
        self.default_min_samples = default_min_samples
        self.eps = eps
    
    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        localization = X[0].copy() 
        
        channles = localization["channel"].unique()

        features = dict()
        for ch in channles:
            if ch in self.cluster_info.Proteins.tolist():
                ch_indx = self.cluster_info.Proteins == ch
                ch_cluster_info = self.cluster_info.loc[ch_indx,:].to_dict('list')
            else:
                ch_cluster_info = { 'Proteins': [ch],
                                    'radius': [self.default_radius],
                                    'min_density': [self.default_min_density],
                                    'min_samples': [self.default_min_samples]}
            
            indx = localization["channel"] == ch
            features_ch = get_cluster_features(localization.loc[indx,:], 
                                            ch_cluster_info )
            features.update(features_ch)
            
        for i, ch1 in enumerate(channles):
            for _, ch2 in enumerate(channles[i+1:]):
                com_ch1 = [ features["com_x_" + ch1],
                            features["com_y_" + ch1],
                            features["com_z_" + ch1]]  
                com_ch1 = np.array(com_ch1)
                
                com_ch2 = [ features["com_x_" + ch2],
                            features["com_y_" + ch2],
                            features["com_z_" + ch2]]   
                com_ch2 = np.array(com_ch2)
    
                features["com_distance_" + ch1 + "_" + ch2] = np.linalg.norm(com_ch2-com_ch1)
                
                cluster_size_ch1 = features["cluster_size_" + ch1]
                cluster_size_ch2 = features["cluster_size_" + ch2]
                features["cluster_size_" + ch1 + "/cluster_size_" + ch2] = \
                                                cluster_size_ch1 / (cluster_size_ch2 +  self.eps)

        return features