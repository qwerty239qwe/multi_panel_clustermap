from src._utils import gen_dummy, gen_metadata
from src.data import DataLoader

from pathlib import Path


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
    loader = DataLoader(Path("./test_data/panel_1"))
    print(loader.get_merged("Total (cells)"))

if __name__ == '__main__':
    main()