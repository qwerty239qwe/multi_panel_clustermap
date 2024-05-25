import matplotlib.pyplot as plt
from seaborn.matrix import dendrogram
from seaborn.utils import despine
import seaborn as sns
import numpy as np

from typing import Dict, List


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

        self._row_dendrograms = {}
        self._col_dendrograms = {}
        self.col_cluster_orders = col_cluster_orders
        self.row_cluster_orders = row_cluster_orders

    @staticmethod
    def _determine_figsize(figsize, data):
        if figsize == "default":
            print(f"Using default figsize: ({(data.shape[1] // 2, data.shape[0] // 2)})")
            return data.shape[1], data.shape[0] // 2
        return figsize

    def _split_cluster(self, data, split_dic, dendrogram_dic, orders=None, axis=0):
        if orders is None:
            orders = list(split_dic.keys())

        for i, name in enumerate(orders):
            subset = data.loc[split_dic[name], :] if axis == 0 else data.loc[:, split_dic[name]]
            dendrogram_dic[name] = dendrogram(subset,
                                              ax=self.axes[i+1, 0] if axis == 0 else self.axes[0, i+1],
                                              axis=axis,
                                              rotate=90 if axis == 0 else 0)
            despine(ax=self.axes[i+1, 0], bottom=True, left=True)

    def _fill_heatmaps(self,
                       data,
                       row_dendrograms,
                       col_dendrograms,
                       row_cluster_orders: List[str],
                       col_cluster_orders: List[str],
                       row_split_dic: Dict[str, List[str]],
                       col_split_dic: Dict[str, List[str]],
                       vmin=None, vmax=None,
                       **kwargs):
        if row_cluster_orders is None:
            row_cluster_orders = list(row_split_dic.keys())

        if col_cluster_orders is None:
            col_cluster_orders = list(col_split_dic.keys())

        if vmin is None:
            vmin = data.min().min()
        if vmax is None:
            vmax = data.max().max()

        for i, row in enumerate(row_cluster_orders):
            sel_row_samples: List[str] = np.array(row_split_dic[row])[row_dendrograms[row].reordered_ind]
            for j, col in enumerate(col_cluster_orders):
                sel_col_samples: List[str] = np.array(col_split_dic[col])[col_dendrograms[col].reordered_ind]
                subset_data = data.loc[sel_row_samples, sel_col_samples]
                sns.heatmap(subset_data, ax=self.axes[i+1, j+1], cbar=False if not (i == 0 and j == 0) else True,
                            cbar_ax=self.axes[0, 0],
                            vmin=vmin, vmax=vmax,
                            cbar_kws={'label': 'score', 'pad': 0.15},
                            **kwargs)
                if j < len(col_cluster_orders)-1:
                    self.axes[i+1, j+1].set_yticklabels([])
                else:
                    self.axes[i+1, j+1].yaxis.tick_right()
                    self.axes[i+1, j+1].yaxis.set_label_position("right")
                    self.axes[i+1, j+1].tick_params(axis='y', labelrotation=0)

                if i < len(row_cluster_orders) - 1:
                    self.axes[i + 1, j + 1].tick_params(axis='x',
                                                        which='both',
                                                        bottom=False,
                                                        top=False,
                                                        left=False,
                                                        labelbottom=False)

    def pad_colorbar(self, pad=0.05, width_adj_coef=0.5, height_adj_coef=1):
        pos2 = self.axes[0, 0].get_position()
        new_pos2 = [pos2.x0 - pad, pos2.y0 + pad,
                    (pos2.width - pad) * width_adj_coef,
                    (pos2.height - pad) * height_adj_coef]
        self.axes[0, 0].set_position(new_pos2)

    def setup_figure(self, figsize=None):
        col_nums = [len(self.col_split_dic[c]) for c in self.col_cluster_orders]
        row_nums = [len(self.row_split_dic[r]) for r in self.row_cluster_orders]
        self.fig, self.axes = plt.subplots(len(self.row_split_dic) + 1,
                                           len(self.col_split_dic) + 1,
                                           figsize=self.figsize if figsize is None else figsize,
                                           gridspec_kw={'width_ratios': [1, ] +
                                                                        [2 / sum(col_nums) * cn for cn in col_nums],
                                                        'height_ratios': [0.5, ] +
                                                                         [2 / sum(row_nums) * rn for rn in row_nums]})

    def plot(self,
             figsize=None,
             wspace=0.05,
             hspace=0.05,
             cbar_pad=0.01):
        self.setup_figure(figsize=figsize)
        self._split_cluster(self.data,
                            self.row_split_dic,
                            self._row_dendrograms,
                            self.row_cluster_orders, 0)
        self._split_cluster(self.data,
                            self.col_split_dic,
                            self._col_dendrograms,
                            self.col_cluster_orders, 1)
        self._fill_heatmaps(self.data,
                            row_dendrograms=self._row_dendrograms,
                            col_dendrograms=self._col_dendrograms,
                            row_cluster_orders=self.row_cluster_orders,
                            col_cluster_orders=self.col_cluster_orders,
                            row_split_dic=self.row_split_dic,
                            col_split_dic=self.col_split_dic
                            )

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.8,
                            top=0.9,
                            wspace=wspace,
                            hspace=hspace)

        self.pad_colorbar(pad=cbar_pad)

        plt.savefig("./test.svg")
        plt.show()