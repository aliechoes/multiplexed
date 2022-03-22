from sklearn.cluster import DBSCAN 
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


def get_pr_cluster_info(cluster_info, 
                        pr,
                        default_radius,
                        default_min_density,
                        default_min_samples):
    
    if pr in cluster_info.proteins.tolist():
        pr_indx = cluster_info.proteins == pr
        pr_cluster_info = cluster_info.loc[pr_indx,:].to_dict('list')
    else:
        pr_cluster_info = { 'proteins': [pr],
                            'radius': [default_radius],
                            'min_density': [default_min_density],
                            'min_samples': [default_min_samples]}
    return pr_cluster_info



def get_cluster_features(localization_group, 
                         cluster_info ):
    
    localization_group_ = localization_group.copy()
    ## finding biggest clusters and filter them
    value_counts = localization_group_.cluster.value_counts()
    try:
        biggest_cluster_size = value_counts.max()
    except ValueError:
        biggest_cluster_size = 0
    
    min_samples = cluster_info["min_samples"][0]

    if min_samples <= biggest_cluster_size:
        main_clusters = value_counts == biggest_cluster_size
        main_clusters = main_clusters[main_clusters].index.tolist()
        
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
    
    pr = cluster_info["proteins"][0]
    features = dict()
    features["CF_cluster_size_" + pr] = len(localization_group_)
    features["CF_com_x_" + pr] = com_x
    features["CF_com_y_" + pr] = com_y
    features["CF_com_z_" + pr] = com_z

    features["CF_std_x_" + pr] = std_x
    features["CF_std_y_" + pr] = std_y
    features["CF_std_z_" + pr] = std_z
    
    features["CF_volume_" + pr] = volume
    features["CF_convex_hull_" + pr] = convex_hull
    return features


def find_protein_center_of_mass(localization_group, 
                        pr_cluster_info):
    
    pr = pr_cluster_info["proteins"][0]
    indx = localization_group.protein == pr
    clustering_group1 = get_cluster_features(localization_group.loc[indx,:], 
                                             pr_cluster_info )
    pr_com = [  clustering_group1["CF_com_x_" + pr],
                clustering_group1["CF_com_y_" + pr],
                clustering_group1["CF_com_z_" + pr]]
    pr_com = np.array(pr_com)
        
    return pr_com


def find_distance_from_overal_com( localization_group, 
                            pr_cluster_info,
                            pr1_com, 
                            pr2_com):
    
    ## finding the com of the proteins
    pr = pr_cluster_info["proteins"][0]
    pr_com  = find_protein_center_of_mass(localization_group,  
                                          pr_cluster_info)
    
    overal_com = (pr1_com + pr2_com)/2.
       
    dist1 = np.linalg.norm(pr1_com-pr_com)
    dist2 = np.linalg.norm(pr2_com-pr_com)
    
    if dist1 < dist2:
        dist = -1*np.linalg.norm(overal_com-pr_com)
    else: 
        dist = np.linalg.norm(overal_com-pr_com)  
    
    return dist
    

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
        localization = X.copy() 

        localization["cluster"] = -1
        cols = ["x","y","z"]
        proteins = localization["protein"].unique()

        for pr in proteins:
            pr_cluster_info =get_pr_cluster_info(   self.cluster_info, 
                                                    pr,
                                                    self.default_radius,
                                                    self.default_min_density,
                                                    self.default_min_samples)
            
            indx = localization["protein"] == pr
            db = DBSCAN(    eps=pr_cluster_info["radius"][0], 
                            min_samples=pr_cluster_info["min_density"][0])
            
            db.fit(localization.loc[indx,cols].astype(float).to_numpy())
            localization.loc[indx,"cluster"] = db.labels_  

        return localization


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
        
        proteins = localization["protein"].unique()

        features = dict()
        for pr in proteins:
            pr_cluster_info =get_pr_cluster_info(   self.cluster_info, 
                                                    pr,
                                                    self.default_radius,
                                                    self.default_min_density,
                                                    self.default_min_samples)
            
            indx = localization["protein"] == pr
            features_pr = get_cluster_features(localization.loc[indx,:], 
                                            pr_cluster_info )
            features.update(features_pr)
            
        for i, pr1 in enumerate(proteins):
            for _, pr2 in enumerate(proteins[i+1:]):
                com_pr1 = [ features["CF_com_x_" + pr1],
                            features["CF_com_y_" + pr1],
                            features["CF_com_z_" + pr1]]  
                com_pr1 = np.array(com_pr1)
                
                com_pr2 = [ features["CF_com_x_" + pr2],
                            features["CF_com_y_" + pr2],
                            features["CF_com_z_" + pr2]]   
                com_pr2 = np.array(com_pr2)
    
                features["CF_com_distance_" + pr1 + "_" + pr2] = np.linalg.norm(com_pr2-com_pr1)
                
                cluster_size_pr1 = features["CF_cluster_size_" + pr1]
                cluster_size_pr2 = features["CF_cluster_size_" + pr2]
                features["CF_cluster_size_" + pr1 + "/cluster_size_" + pr2] = \
                                                cluster_size_pr1 / (cluster_size_pr2 +  self.eps)

        return features



class DistanceToCOMFeatures(BaseEstimator, TransformerMixin):
    """
    mask based features
    """
    def __init__(   self, 
                    path_to_clustering_info, 
                    pr1,
                    pr2,
                    default_radius = 0.6, 
                    default_min_density = 5, 
                    default_min_samples = 400,
                    eps = 1e-16):
        self.pr1 = pr1
        self.pr2 = pr2
        self.cluster_info = pd.read_csv(path_to_clustering_info)
        self.default_radius = default_radius
        self.default_min_density = default_min_density
        self.default_min_samples = default_min_samples
        self.eps = eps
    
    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        localization_group = X[0].copy()
        
        pr1_cluster_info = get_pr_cluster_info(self.cluster_info, 
                                               self.pr1,
                                               self.default_radius,
                                               self.default_min_density,
                                               self.default_min_samples)
        
        pr1_com  = find_protein_center_of_mass(localization_group,  
                                               pr1_cluster_info)
        
        pr2_cluster_info = get_pr_cluster_info(self.cluster_info, 
                                               self.pr2,
                                               self.default_radius,
                                               self.default_min_density,
                                               self.default_min_samples)
        
        pr2_com  = find_protein_center_of_mass(localization_group,  
                                               pr2_cluster_info)
        
        proteins = localization_group["protein"].unique()

        features = dict()
        for pr in proteins:
            pr_cluster_info = get_pr_cluster_info(self.cluster_info, 
                                               pr,
                                               self.default_radius,
                                               self.default_min_density,
                                               self.default_min_samples)
            dist = find_distance_from_overal_com( localization_group, 
                                                                pr_cluster_info,
                                                                pr1_com, 
                                                                pr2_com)
            features["CD_distance_of_" + pr + "_from_" + self.pr1 + "&" + self.pr2 ] = dist
         
        return features