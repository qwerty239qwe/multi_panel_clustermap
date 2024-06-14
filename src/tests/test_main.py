from src._utils import gen_dummy, gen_metadata, gen_cluster_metadata
from src.data import DataLoader, DataProcessor
from src.plotting import Plotter
import pandas as pd


def setup_cluster_data():
    for i in range(1, 10):
        gen_dummy("./test_data/panel_1/DM01 quantification.T.xlsx",
                  f"{i:02}", None)
        gen_dummy("./test_data/panel_1/DM01 quantification.T.xlsx",
                  f"{i:02}", "panel_2")
        gen_dummy("./test_data/panel_1/DM01 quantification.T.xlsx",
                  f"{i:02}", "panel_3")

    metadata = gen_metadata("./test_data/panel_1")
    metadata.to_csv("./test_data/metadata.csv")


def test_main():
    row_metadata = gen_cluster_metadata("./test_data")
    row_metadata.to_csv("./test_data/row_metadata.csv")

    # setup_cluster_data()
    metadata = pd.read_csv("./test_data/metadata.csv", index_col=0)
    processor = DataProcessor("./test_data",
                              sample_metadata=metadata,
                              group_cols=["sex"],
                              cluster_orders=["panel_1", "panel_2", "panel_3"],
                              group_orders=["M", "F"],)
    # processor.add_preprocessing(preprocessing_name="standardize", axis=0)
    processor.add_preprocessing(preprocessing_name="z_score", axis=0)
    merged_data = processor.get_processed_data("Total (cells)")
    plotter = Plotter(data=merged_data,
                      row_split_dic=processor.row_split_dic,
                      col_split_dic=processor.col_split_dic["sex"],
                      col_meta_data=metadata,
                      row_meta_data=row_metadata,
                      row_cluster_orders=processor.row_orders,
                      col_cluster_orders=processor.col_orders,)
    plotter.plot(col_colors=["sex", "treatment"],
                 row_colors=["panel", "property"],
                 wspace=0.01, hspace=0.01,
                 file_name="./test_pad.svg")
