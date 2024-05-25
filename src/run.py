from src._utils import gen_dummy, gen_metadata
from src.data import DataLoader, DataProcessor
from src.plotting import Plotter

from pathlib import Path

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


def main():
    # setup_cluster_data()
    #loader = DataLoader(Path("./test_data/panel_1"))
    #merged_data = loader.get_merged("Total (cells)")
    metadata = pd.read_csv("./test_data/metadata.csv", index_col=0)

    processor = DataProcessor("./test_data",
                              metadata,
                              ["sex"],
                              ["panel_1", "panel_2", "panel_3"],
                              ["M", "F"],)
    processor.add_preprocessing(preprocessing_name="standardize", axis=0)
    # processor.add_preprocessing(preprocessing_name="z_score", axis=0)
    merged_data = processor.get_processed_data("Total (cells)")
    print(processor.row_split_dic)
    print(processor.col_split_dic)
    print(processor.row_orders)
    print(processor.col_orders)
    #
    plotter = Plotter(data=merged_data,
                      row_split_dic=processor.row_split_dic,
                      col_split_dic=processor.col_split_dic["sex"],
                      meta_data=metadata,
                      row_cluster_orders=processor.row_orders,
                      col_cluster_orders=processor.col_orders,)
    plotter.plot()


if __name__ == '__main__':
    main()