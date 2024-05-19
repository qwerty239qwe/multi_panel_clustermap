from src._utils import gen_dummy


def setup_cluster_data():
    for i in range(2, 10):
        gen_dummy("./test_data/panel_1/DM01 quantification.T.xlsx",
                  i, None)

    for i in range(10, 20):
        gen_dummy("./test_data/panel_1/DM01 quantification.T.xlsx",
                  i, "panel_2")

    for i in range(20, 30):
        gen_dummy("./test_data/panel_1/DM01 quantification.T.xlsx",
                  i, "panel_3")


def main():
    pass


if __name__ == '__main__':
    main()