import pandas as pd
from pathlib import Path


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
            sel_cols.append(sel_col)
        return pd.concat(sel_cols, axis=1)


class DataProcessor(object):
    def __init__(self, parent_data_dir):
        self._parent_data_dir = parent_data_dir

        self._data_loaders = {f.stem: DataLoader(f)
                              for f in Path(self._parent_data_dir).iterdir() if f.is_dir()}