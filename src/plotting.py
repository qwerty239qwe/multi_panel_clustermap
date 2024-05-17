import matplotlib.pyplot as plt
from seaborn.matrix import dendrogram


class Plotter(object):
    def __init__(self,
                 data,
                 row_split_dic,
                 col_split_dic,
                 meta_data,
                 row_cluster_orders=None,
                 col_cluster_orders=None,
                 figsize="default"):
        self.data = data
        self.row_split_dic = row_split_dic
        self.col_split_dic = col_split_dic
        self.meta_data = meta_data
        self.figsize = self._determine_figsize(figsize, self.data)
        self.fig, self.axes = plt.subplots(len(self.row_split_dic)+1,
                                           len(self.col_split_dic)+1,
                                           figsize=self.figsize)

        self._row_dendrograms = {}
        self._col_dendrograms = {}

        self._split_cluster(self.data,
                            self.row_split_dic,
                            self._row_dendrograms,
                            row_cluster_orders, 0)

        self._split_cluster(self.data,
                            self.col_split_dic,
                            self._col_dendrograms,
                            col_cluster_orders, 1)



        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.,
                            hspace=0.)

    @staticmethod
    def _determine_figsize(figsize, data):
        if figsize == "default":
            return data.shape
        return figsize

    def _split_cluster(self, data, split_dic, dendrogram_dic, orders=None, axis=0):
        if orders is None:
            orders = list(split_dic.keys())

        for i, name in enumerate(orders):
            subset = data.loc[split_dic[name], :] if axis == 0 else data.loc[:, split_dic[name]]
            dendrogram_dic[name] = dendrogram(subset, ax=self.axes[1, 0], axis=0, rotate=90)

    def _fill_heatmaps(self, data, row_dendrograms, col_dendrograms):
        pass