import pandas as pd
from pathlib import Path
from src._preprocessing import *
from typing import List


class DataLoader(object):
    def __init__(self, data_dir):
        self._data_dir = Path(data_dir)
        self._name = self._data_dir.stem
        self._wide_dfs = self._load_data(self._data_dir)

    @property
    def name(self):
        return self._name

    @staticmethod
    def _load_data(data_dir):
        dfs = {}
        col_order = None
        for file in data_dir.iterdir():
            if file.suffix == '.csv':
                df = pd.read_csv(file, index_col=0)
            elif file.suffix == ".xlsx":
                df = pd.read_excel(file, index_col=0)
            elif file.suffix in [".tsv", ".txt"]:
                df = pd.read_csv(file, index_col=0, sep='\t')
            else:
                continue

            if "Patient No." in df.columns:
                p_id = df["Patient No."].unique()[0]
            else:
                p_id = file.stem.split(" ")[0]
            df = df.loc[df.index[df.index != "Patient No."], df.columns[df.columns != "Patient No."]]
            if col_order is None:
                col_order = df.columns
            dfs[p_id] = df.loc[:, col_order]

        return dfs

    def get_merged(self, name_or_id):
        assert isinstance(name_or_id, str) or isinstance(name_or_id, int)
        sel_cols = []
        for d_name, df in self._wide_dfs.items():
            sel_col = df.loc[:, name_or_id] if isinstance(name_or_id, str) else df.iloc[:, name_or_id]
            sel_col.name = d_name
            sel_cols.append(sel_col)
        return pd.concat(sel_cols, axis=1)


class DataProcessor(object):
    def __init__(self,
                 parent_data_dir,
                 sample_metadata: pd.DataFrame,
                 group_cols: List[str],
                 cluster_orders=None,
                 group_orders=None):
        self._parent_data_dir = parent_data_dir

        self._sample_metadata = sample_metadata
        self._group_cols = group_cols # and this is the groups of samples
        self._group_info = self._parse_metadata(sample_metadata, group_cols)
        self._group_orders = group_orders if cluster_orders is not None else list(
            self._group_info[group_cols[0]].keys())
        self._group_preprocessings = {}

        self._data_loaders = {f.stem: DataLoader(f)
                              for f in Path(self._parent_data_dir).iterdir() if f.is_dir()}

        self._cluster_orders = cluster_orders if cluster_orders is not None else list(self._data_loaders.keys())
        #  for the ease of clarifying, we use cluster to denote the groups of cell types (sep by panels)
        self._cluster_preprocessings = {}

    @staticmethod
    def _parse_metadata(sample_metadata, group_cols):
        group_dic = {}
        for g in group_cols:
            group_dic[g] = sample_metadata.groupby(g).apply(lambda x: list(x.index)).to_dict()
        return group_dic

    def add_preprocessing(self,
                          preprocessing_name,
                          axis=0,
                          **kwargs):
        if axis == 0:
            self._cluster_preprocessings[preprocessing_name] = PREPROCESSINGS[preprocessing_name](**kwargs)
        elif axis == 1:
            self._group_preprocessings[preprocessing_name] = PREPROCESSINGS[preprocessing_name](**kwargs)

    def get_processed_data(self,
                           property_name_or_id: str,
                           cluster_preprocessings="all",
                           group_preprocessings="all"):
        mg_data = pd.concat([self._data_loaders[c_order].get_merged(property_name_or_id)
                             for c_order in self._cluster_orders], axis=0)

        if cluster_preprocessings == "all":
            cluster_preprocessings = list(self._cluster_preprocessings.keys())
        if group_preprocessings == "all":
            group_preprocessings = list(self._group_preprocessings.keys())

        for c_p in cluster_preprocessings:
            mg_data = self._cluster_preprocessings[c_p](mg_data, axis=1)

        for g_p in group_preprocessings:
            mg_data = self._group_preprocessings[g_p](mg_data, axis=0)

        return mg_data