import pandas as pd
from tqdm import tqdm
from multiplexed.utils import list_of_dict_to_dict

class FeatureExtractor(object):
    def __init__(self, feature_unions):
        self.feature_unions = feature_unions

    def extract_(self, localization_group):
        features = self.feature_unions.transform(localization_group).copy()
        features = list_of_dict_to_dict(features)
        return features

    def extract_features(self, metadata, localizations):

        features = []
        for i in tqdm(range(metadata.shape[0])):
            ds = metadata.loc[i, "dataset"]
            grp = metadata.loc[i, "group"]

            indx = localizations.dataset == ds
            indx = indx & (localizations.group == grp)

            localization_group = localizations.loc[indx,:].copy()
            features.append(self.extract_(localization_group))
        features = pd.DataFrame(features)
        return features