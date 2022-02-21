
import os
import glob
import numpy as np
import pandas as pd
import h5py
import random
from joblib import Parallel, delayed
from tqdm import tqdm

def list_of_dict_to_dict(list_of_dicts):
    new_dict = dict()
    for one_dict in list_of_dicts:
        new_dict.update(one_dict)
    return new_dict


# def get_label(h5_file_path):
#     h5_file = h5py.File(h5_file_path, "r")  
#     ## label
#     results = dict()
#     try:
#         results["label"] = h5_file.get("label")[()]
#         results["set"] = "labeled"
#     except TypeError:
#         results["label"] = "-1"
#         results["set"] = "unlabeled"
#     try:
#         results["object_number"] = os.path.split(h5_file_path)[-1]
#         results["object_number"] = results["object_number"].replace(".h5","")
#         results["object_number"] = results["object_number"].split("_")[-1]
#     except TypeError:
#         results["object_number"] = None
#     h5_file.close()
#     return results

    
def metadata_generator(datasets):
    metadata = pd.DataFrame(columns=[   "dataset",
                                        "group",
                                        "label",
                                        "cluster"])
    for ds in tqdm(datasets):
        files = glob.glob(datasets[ds]+"*.hdf5")
        groups = set([])
        for f in files:
            temp = h5py.File(f, "r")
            temp = temp.get("locs")[()]
            temp_group = pd.DataFrame(temp)["group"]
            temp_group = set(temp_group.unique())
            groups = set.union(groups, temp_group )
        for gr in groups:
            metadata = metadata.append({
                "dataset" : ds,
                "group"   : gr,
                "label"   : -1,
                "cluster"   : -1,
            }, ignore_index=True)
    return metadata