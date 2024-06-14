from src._utils import gen_dummy, gen_metadata, gen_cluster_metadata
from src.data import DataLoader, DataProcessor
from src.plotting import Plotter
import tomllib
from pathlib import Path

import pandas as pd
from argparse import ArgumentParser
import tomllib


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="default")
    return parser.parse_args()


def load_metadata(path_dic):
    row_metadata, col_metadata = None, None
    if "row_metadata_path" in path_dic:
        row_metadata = pd.read_csv(path_dic["row_metadata_path"], index_col=0)
    if "col_metadata_path" in path_dic:
        col_metadata = pd.read_csv(path_dic["col_metadata_path"], index_col=0)
    return row_metadata, col_metadata


def load_data_processor(path_dic, col_metadata):
    processor = DataProcessor(path_dic["parent_dir"],
                              sample_metadata=col_metadata,
                              group_cols=path_dic["col_cluster_names"] if "col_cluster_names" in path_dic else None,
                              cluster_orders=path_dic["row_cluster_orders"] if "row_cluster_orders" in path_dic else None,
                              group_orders=path_dic["col_cluster_orders"] if "col_cluster_orders" in path_dic else None, )
    # processor.add_preprocessing(preprocessing_name="standardize", axis=0)
    if "preprocessing" in path_dic:
        for (name, axis) in path_dic["preprocessing"]:
            processor.add_preprocessing(preprocessing_name=name,
                                        axis=axis)
    merged_data = processor.get_processed_data(path_dic["property_name_or_id"])

    return merged_data, processor


def setup_plotter(conf_dic, merged_data, processor, col_metadata, row_metadata):
    plotter = Plotter(data=merged_data,
                      row_split_dic=processor.row_split_dic,
                      col_split_dic=processor.col_split_dic[conf_dic["col_split_factor"]],
                      col_meta_data=col_metadata,
                      row_meta_data=row_metadata,
                      row_cluster_orders=processor.row_orders,
                      col_cluster_orders=processor.col_orders, )
    plotter.plot(figsize=conf_dic["figsize"] if conf_dic["figsize"] != "default" else None,
                 col_colors=conf_dic["col_colors"],
                 row_colors=conf_dic["row_colors"],
                 wspace=conf_dic["wspace"],
                 hspace=conf_dic["hspace"],
                 file_name=conf_dic["file_name"],
                 heatmap_cmap=conf_dic["heatmap_cmap"] if "heatmap_cmap" in conf_dic else "RdBu_r",
                 dpi=conf_dic["dpi"])


def main():
    args = get_args()
    config_path = args.config if Path(args.config).is_file() else (
            Path(__file__).parent.parent / "configs/{}.toml".format(args.config))

    with open(config_path, "rb") as f:
        configs = tomllib.load(f)

    print(configs)
    row_metadata, col_metadata = load_metadata(configs["metadata"])
    merged_data, processor = load_data_processor(configs["DataProcessing"], col_metadata)
    setup_plotter(configs["Plotter"], merged_data, processor, col_metadata, row_metadata)


if __name__ == '__main__':
    main()