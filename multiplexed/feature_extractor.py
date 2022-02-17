import pandas as pd
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
from multiplexed.utils import list_of_dict_to_dict

class FeatureExtractor(object):
    def __init__(self, feature_unions):
        self.feature_unions = feature_unions

    def extract_(self, localization, grp):
        try: 
            features = self.feature_unions.transform(localization).copy()
            features = list_of_dict_to_dict(features)
        except:
            print("corrupted group", grp)
            features = []
        return features

    def extract_features(self, localization_groups, n_jobs = -1):
        groups = localization_groups.group.unique()
        features = []
        for grp in tqdm(groups):
            indx = localization_groups.group == grp
            localization = localization_groups.loc[indx,:].copy()
            features.append(self.extract_(  localization, 
                                            grp )) 
        return features 