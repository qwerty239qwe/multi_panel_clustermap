import matplotlib.pyplot as plt
from seaborn.matrix import dendrogram
from seaborn.utils import despine
import seaborn as sns
import numpy as np

from typing import Dict, List


class Plotter(object):
    DEFAULT_PATETTES = ["deep", "hls", "husl", "Set2", "icefire"]
    def __init__(self,
                 data,
                 row_split_dic,
                 col_split_dic,
                 col_meta_data=None,
                 row_meta_data=None,
                 row_cluster_orders=None,
                 col_cluster_orders=None,
                 figsize="default",
                 palette_orders="default"):
        self.data = data
        self.row_split_dic = row_split_dic
        self.col_split_dic = col_split_dic
        self.col_meta_data = col_meta_data
        self.row_meta_data = row_meta_data

        self.figsize = self._determine_figsize(figsize, self.data)
        self._row_dendrograms = {}
        self._col_dendrograms = {}
        self.col_cluster_orders = col_cluster_orders
        self.row_cluster_orders = row_cluster_orders

        self._n_used_palettes = 0
        if palette_orders == "default":
            self.palette_orders = self.DEFAULT_PATETTES

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
            if axis == 0:
                self.axes[i + 1, 0].set_yticklabels([])
                self.axes[i + 1, 0].set_yticks([])
            else:
                self.axes[0, i+1].set_xticklabels([])
                self.axes[0, i+1].set_xticks([])

    def _add_colors(self,
                    ax,
                    ordered_colors,
                    start_i=0,
                    append_on_yaxis=True,
                    width=0.05,
                    deviation_idx=0,
                    x_dev=0,
                    y_dev=0,
                    height=1,
                    color_name=None,
                    label_deviation=0.05,):
        for i, color in enumerate(ordered_colors):
            ax.add_patch(plt.Rectangle(xy=(-0.05 * deviation_idx + x_dev, i+start_i)
                                          if append_on_yaxis else (i+start_i, y_dev + 0.05 * deviation_idx),
                                       width=width if append_on_yaxis else height,
                                       height=height if append_on_yaxis else width,
                                       color=color,
                                       lw=0,
                                       transform=ax.get_yaxis_transform()
                                            if append_on_yaxis else ax.get_xaxis_transform(),
                                       clip_on=False))
        if color_name is not None:
            if append_on_yaxis:
                ax.text(-0.05 * deviation_idx + x_dev + width / 2, -label_deviation,
                        color_name, transform=ax.transAxes, rotation=90,
                        horizontalalignment="right", verticalalignment="center",
                        rotation_mode="anchor")
            else:
                ax.text(1+label_deviation, y_dev + 0.05 * deviation_idx,
                        color_name, transform=ax.transAxes)

    def _fill_heatmaps(self,
                       data,
                       row_dendrograms,
                       col_dendrograms,
                       row_cluster_orders: List[str],
                       col_cluster_orders: List[str],
                       row_split_dic: Dict[str, List[str]],
                       col_split_dic: Dict[str, List[str]],
                       row_colors: List[Dict[str, any]] = None,
                       col_colors: List[Dict[str, any]] = None,
                       row_color_names: List[str] = None,
                       col_color_names: List[str] = None,
                       vmin=None,
                       vmax=None,
                       heatmap_cmap="RdBu_r",
                       **kwargs):
        if row_cluster_orders is None:
            row_cluster_orders = list(row_split_dic.keys())

        if col_cluster_orders is None:
            col_cluster_orders = list(col_split_dic.keys())

        if vmin is None:
            vmin = data.min().min()
        if vmax is None:
            vmax = data.max().max()

        all_row_samples, all_col_samples = [], []
        for i, row in enumerate(row_cluster_orders):
            sel_row_samples: List[str] = np.array(row_split_dic[row])[row_dendrograms[row].reordered_ind]
            all_row_samples.append(sel_row_samples)
            for j, col in enumerate(col_cluster_orders):
                sel_col_samples: List[str] = np.array(col_split_dic[col])[col_dendrograms[col].reordered_ind]
                if len(all_col_samples) <= j:
                    all_col_samples.append(sel_col_samples)

                subset_data = data.loc[sel_row_samples, sel_col_samples]
                sns.heatmap(subset_data, ax=self.axes[i+1, j+1],
                            cbar=False if not (i == 0 and j == 0) else True,
                            cbar_ax=self.axes[0, 0],
                            vmin=vmin,
                            vmax=vmax,
                            cbar_kws={'label': 'score', 'pad': 0.15},
                            cmap=heatmap_cmap,
                            **kwargs)
                if j < len(col_cluster_orders)-1:
                    self.axes[i+1, j+1].set_yticklabels([])
                else:
                    self.axes[i+1, j+1].yaxis.tick_right()
                    self.axes[i+1, j+1].yaxis.set_label_position("right")
                    self.axes[i+1, j+1].tick_params(axis='y', labelrotation=0)

                if i < len(row_cluster_orders) - 1:
                    self.axes[i + 1, j + 1].set_xticklabels([])
                    self.axes[i + 1, j + 1].set_xticks([])
                    self.axes[i + 1, j + 1].tick_params(axis='x',
                                                        which='both',
                                                        bottom=False,
                                                        top=False,
                                                        left=False,
                                                        labelbottom=False)
        if row_colors is not None:
            for ri, row_color_dic in enumerate(row_colors):
                rendered_i = 0
                for i, row in enumerate(row_cluster_orders):
                    samples = np.array(row_split_dic[row])[row_dendrograms[row].reordered_ind]
                    self._add_colors(self.axes[i + 1, 1],
                                     ordered_colors=[row_color_dic[s] for s in samples],
                                     start_i=0,
                                     deviation_idx=ri+1,
                                     color_name=row_color_names[ri] if i == len(row_cluster_orders) - 1 else None)
                    rendered_i += len(samples)

        if col_colors is not None:
            for ci, col_color_dic in enumerate(col_colors):
                rendered_i = 0
                for i, col in enumerate(col_cluster_orders):
                    samples = np.array(col_split_dic[col])[col_dendrograms[col].reordered_ind]
                    self._add_colors(self.axes[1, i + 1],
                                     ordered_colors=[col_color_dic[s] for s in samples],
                                     start_i=0,
                                     deviation_idx=ci,
                                     append_on_yaxis=False,
                                     y_dev=1,
                                     color_name=col_color_names[ci] if i == len(col_cluster_orders) - 1 else None)
                    rendered_i += len(samples)

    def pad_colorbar(self, pad=0.05, width_adj_coef=0.5, height_adj_coef=1):
        pos2 = self.axes[0, 0].get_position()
        new_pos2 = [pos2.x0 - pad, pos2.y0 + pad,
                    (pos2.width - pad) * width_adj_coef,
                    (pos2.height - pad) * height_adj_coef]
        self.axes[0, 0].set_position(new_pos2)

    def setup_figure(self, figsize=None):
        col_nums = [len(self.col_split_dic[c]) for c in self.col_cluster_orders]
        row_nums = [len(self.row_split_dic[r]) for r in self.row_cluster_orders]
        print(self.row_split_dic)
        print(self.col_split_dic)

        print(col_nums)
        print(row_nums)
        print(self.figsize if figsize is None else figsize)

        self.fig, self.axes = plt.subplots(len(self.row_split_dic) + 1,
                                           len(self.col_split_dic) + 1,
                                           figsize=self.figsize if figsize is None else figsize,
                                           gridspec_kw={'width_ratios': [1, ] +
                                                                        [2 / sum(col_nums) * cn for cn in col_nums],
                                                        'height_ratios': [0.5, ] +
                                                                         [2 / sum(row_nums) * rn for rn in row_nums]})

    def _get_luts(self, group, by="row"):
        if by == "row":
            id_to_group = self.row_meta_data[group]
        elif by == "col":
            id_to_group = self.col_meta_data[group]
        else:
            raise ValueError("By must be either 'row' or 'col'")
        groups = id_to_group.unique()
        colors = sns.color_palette(self.palette_orders[self._n_used_palettes], len(groups))
        self._n_used_palettes += 1
        return id_to_group.map(dict(zip(groups, colors))).to_dict()

    def plot(self,
             figsize=None,
             wspace=0.0,
             hspace=0.0,
             cbar_pad=0.01,
             row_colors=None,
             col_colors=None,
             file_name=None,
             annotate_row_colors=True,
             annotate_col_colors=True,
             heatmap_cmap="RdBu_r",
             dpi=450):
        self.setup_figure(figsize=figsize)
        row_color_names = None
        if row_colors is not None:
            row_color_names = [gp if annotate_row_colors else None for gp in row_colors]
            row_colors = [self._get_luts(gp, by='row') for gp in row_colors]

        col_color_names = None
        if col_colors is not None:
            col_color_names = [gp if annotate_col_colors else None for gp in col_colors]
            col_colors = [self._get_luts(gp, by='col') for gp in col_colors]

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
                            col_split_dic=self.col_split_dic,
                            row_colors=row_colors,
                            col_colors=col_colors,
                            row_color_names=row_color_names,
                            col_color_names=col_color_names,
                            heatmap_cmap=heatmap_cmap,
                            )

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.8,
                            top=0.9,
                            wspace=wspace,
                            hspace=hspace)

        self.pad_colorbar(pad=cbar_pad)

        if file_name is not None:
            plt.savefig(file_name, dpi=dpi)
        plt.show()