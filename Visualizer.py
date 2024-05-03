from Helper import *
from Datasets import Datasets

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# from pySankey.sankey import sankey

import cebra
import pandas as pd


class Vizualizer:
    def __init__(self, root_dir) -> None:
        self.save_dir = Path(root_dir).joinpath("figures")
        dir_exist_create(self.save_dir)

    def plot_dataplot_summary(self, plot_dir, title="Data Summary"):
        # TODO: combine plots from top to bottom aligned by time
        # distance, velocity, acceleration, position, raster
        pass

    def plot_embedding(
        self,
        ax,
        embedding,
        embedding_labels: dict,
        title="Embedding",
        cmap="rainbow",
        plot_legend=True,
        colorbar_ticks=None,
        markersize=0.05,
        alpha=0.4,
        figsize=(10, 10),
        dpi=300,
    ):
        embedding, labels = Datasets.force_equal_dimensions(
            embedding, embedding_labels["labels"]
        )

        ax = cebra.plot_embedding(
            ax=ax,
            embedding=embedding,
            embedding_labels=labels,
            markersize=markersize,
            alpha=alpha,
            dpi=dpi,
            title=title,
            figsize=figsize,
            cmap=cmap,
        )
        if plot_legend:
            # Create a ScalarMappable object using the specified colormap
            sm = plt.cm.ScalarMappable(cmap=cmap)
            unique_labels = np.unique(labels)
            unique_labels.sort()
            sm.set_array(unique_labels)  # Set the range of values for the colorbar

            # Manually create colorbar
            cbar = plt.colorbar(sm, ax=ax)
            # Adjust colorbar ticks if specified
            cbar.set_label(embedding_labels["name"])  # Set the label for the colorbar

            if colorbar_ticks:
                cbar.ax.yaxis.set_major_locator(
                    MaxNLocator(integer=True)
                )  # Adjust ticks to integers
                cbar.set_ticks(colorbar_ticks)  # Set custom ticks
        return ax

    def plot_multiple_embeddings(
        self,
        embeddings: dict,
        labels: dict,
        title="Embeddings",
        cmap="rainbow",
        projection="3d",
        figsize=(20, 4),
        plot_legend=True,
        markersize=0.05,
        alpha=0.4,
        dpi=300,
    ):
        figsize_x, figsize_y = figsize
        fig = plt.figure(figsize=figsize)

        # Compute the number of subplots
        num_subplots = len(embeddings)
        rows = 1
        cols = num_subplots
        if num_subplots > 4:
            rows = int(num_subplots**0.5)
            cols = (num_subplots + rows - 1) // rows
            figsize = (figsize_x, figsize_y * rows)

        fig, axes = plt.subplots(
            rows, cols, figsize=figsize, subplot_kw={"projection": projection}
        )

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, (subplot_title, embedding) in enumerate(embeddings.items()):
            ax = axes[i]
            ax = self.plot_embedding(
                ax,
                embedding,
                labels,
                subplot_title,
                cmap,
                plot_legend=False,
                markersize=markersize,
                alpha=alpha,
                dpi=dpi,
            )

        for ax in axes[num_subplots:]:
            ax.remove()  # Remove any excess subplot axes

        if plot_legend:
            # Create a ScalarMappable object using the specified colormap
            sm = plt.cm.ScalarMappable(cmap=cmap)
            unique_labels = np.unique(labels["labels"])
            unique_labels.sort()
            sm.set_array(unique_labels)  # Set the range of values for the colorbar

            # Manually create colorbar
            cbar = plt.colorbar(sm, ax=ax)
            # Adjust colorbar ticks if specified
            cbar.set_label(labels["name"])  # Set the label for the colorbar

        self.plot_ending(title)

    def plot_losses(
        self,
        models,
        models_shuffled=[],
        title="Losses",
        coloring_type="rainbow",
        plot_original=True,
        plot_shuffled=True,
        plot_model_iterations=False,
        alpha=0.8,
        figsize=(10, 10),
    ):
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

        if coloring_type == "rainbow":
            num_colors = len(models) + len(models_shuffled)
            rainbow_colors = [
                mcolors.to_rgba(c, alpha=alpha)
                for c in plt.cm.rainbow(np.linspace(0, 1, num_colors))
            ]
            colors = (
                rainbow_colors[: len(models)],
                rainbow_colors[len(models) :],
            )

        elif coloring_type == "distinct":
            # Generate distinct colors for models and models_shuffled
            colors_original = [
                mcolors.to_rgba(c, alpha=alpha)
                for c in plt.cm.tab10(np.linspace(0, 1, len(models)))
            ]
            colors_shuffled = [
                mcolors.to_rgba(c, alpha=alpha)
                for c in plt.cm.Set3(np.linspace(0, 1, len(models_shuffled)))
            ]
            colors = colors_original, colors_shuffled

        elif coloring_type == "mono":  # Blues and Reds
            blue_colors = [
                mcolors.to_rgba(c, alpha=alpha)
                for c in plt.cm.Blues(np.linspace(0.3, 1, len(models)))
            ]
            reds_colors = [
                mcolors.to_rgba(c, alpha=alpha)
                for c in plt.cm.Reds(np.linspace(0.3, 1, len(models_shuffled)))
            ]
            colors = (
                blue_colors,  # colors_original
                reds_colors,  # colors_shuffled
            )

        else:
            raise ValueError("Invalid coloring type. Choose 'rainbow' or 'distinct'.")

        # Plotting
        if plot_original and plot_shuffled:
            models_to_plot = models + models_shuffled
            colors_to_use = colors[0] + colors[1]
        elif plot_original:
            models_to_plot = models
            colors_to_use = colors[0]
            title += f"{title} not shuffled"
        else:
            models_to_plot = models_shuffled
            title += f"{title} shuffled"
            colors_to_use = colors[1]

        for color, model in zip(colors_to_use, models_to_plot):
            label = model.name.split("behavior_")[-1]
            label += f" - {model.max_iterations} Iter" if plot_model_iterations else ""
            if model.fitted:
                ax = cebra.plot_loss(
                    model, color=color, alpha=alpha, label=label, ax=ax
                )
            else:
                global_logger.error(f"{label} Not fitted.")
                global_logger.warning(f"Skipping model {label}.")
                print(f"Skipping model {label}.")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("InfoNCE Loss")
        plt.legend(bbox_to_anchor=(0.5, 0.3), frameon=False)
        self.plot_ending(title)

    def plot_consistency_scores(self, ax1, title, embeddings, labels, dataset_ids):
        (
            time_scores,
            time_pairs,
            time_subjects,
        ) = cebra.sklearn.metrics.consistency_score(
            embeddings=embeddings,
            labels=labels,
            dataset_ids=dataset_ids,
            between="datasets",
        )
        ax1 = cebra.plot_consistency(
            time_scores,
            pairs=time_pairs,
            datasets=time_subjects,
            ax=ax1,
            title=title,
            colorbar_label="consistency score",
        )
        return ax1

    def plot_multiple_consistency_scores(
        self,
        animals,
        wanted_stimulus_types,
        wanted_embeddings,
        exclude_properties=None,
        figsize=(7, 7),
    ):
        # TODO: change this to a more modular funciton, integrate into classes
        # labels to align the subjects is the position of the mouse in the arena
        # labels = {}  # {wanted_embedding 1: {animal_session_task_id: embedding}, ...}
        # for wanted_embedding in wanted_embeddings:
        #    labels[wanted_embedding] = {"embeddings": {}, "labels": {}}
        #    for wanted_stimulus_type in wanted_stimulus_types:
        #        for animal, session, task in yield_animal_session_task(animals):
        #            if task.behavior_metadata["stimulus_type"] == wanted_stimulus_type:
        #                wanted_embeddings_dict = filter_dict_by_properties(
        #                    task.embeddings,
        #                    include_properties=wanted_embedding,
        #                    exclude_properties=exclude_properties,
        #                )
        #                for embedding_key, embedding in wanted_embeddings_dict.items():
        #                    labels_id = f"{session.date[-3:]}_{task.task} {wanted_stimulus_type}"
        #                    position_lables = task.behavior.position.data
        #                    position_lables, embedding = force_equal_dimensions(
        #                        position_lables, embedding
        #                    )
        #                    labels[wanted_embedding]["embeddings"][
        #                        labels_id
        #                    ] = embedding
        #                    labels[wanted_embedding]["labels"][
        #                        labels_id
        #                    ] = position_lables
        #
        #    dataset_ids = list(labels[wanted_embedding]["embeddings"].keys())
        #    embeddings = list(labels[wanted_embedding]["embeddings"].values())
        #    labeling = list(labels[wanted_embedding]["labels"].values())
        #
        #    title = f"CEBRA-{wanted_embedding} embedding consistency"
        #    fig = plt.figure(figsize=figsize)
        #    ax1 = plt.subplot(111)
        #    ax1 = self.plot_consistency_score(
        #        ax1, title, embeddings, labeling, dataset_ids
        #    )
        #    plt.show()
        pass
        # self.plot_ending(title)

    def plot_decoding_score(
        self,
        decoded_model_lists,
        labels,
        title="Behavioral Decoding of Position",
        colors=["deepskyblue", "gray"],
        figsize=(13, 5),
    ):
        # TODO: improve this function, modularity, flexibility
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        overall_num = 0
        for color, docoded_model_list in zip(colors, decoded_model_lists):
            for num, decoded_model in enumerate(docoded_model_list):
                alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
                x_pos = overall_num + num
                width = 0.4  # Width of the bars
                ax1.bar(
                    x_pos, decoded_model.decoded[1], width=0.4, color=color, alpha=alpha
                )
                label = "".join(decoded_model.name.split("_train")).split("behavior_")[
                    -1
                ]
                ax2.scatter(
                    decoded_model.state_dict_["loss"][-1],
                    decoded_model.decoded[1],
                    s=50,
                    c=color,
                    alpha=alpha,
                    label=label,
                )
            overall_num += x_pos + 1

        x_label = "InfoNCE Loss (contrastive learning)"
        ylabel = "Median position error in cm"

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_ylabel(ylabel)
        labels = labels
        label_pos = np.arange(len(labels))
        ax1.set_xticks(label_pos)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1, 1), frameon=False)
        plt.show()

    def plot_histogram(self, data, title, bins=100, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.hist(data, bins=bins)
        plt.show()
        plt.close()

    def histogam_subplot(
        self,
        data: np.ndarray,
        title: str,
        ax,
        bins=100,
        xlim=[0, 1],
        xlabel="",
        ylabel="Frequency",
        xticklabels=None,
        color=None,
    ):
        ax.set_title(title)
        ax.hist(data.flatten(), bins=bins, color=color)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticklabels == "empty":
            ax.set_xticklabels("")

    def heatmap_subplot(
        self,
        matrix,
        title,
        ax,
        sort=False,
        xlabel="Cell ID",
        ylabel="Cell ID",
        cmap="YlGnBu",
    ):
        if sort:
            # Assuming correlations is your correlation matrix as a NumPy array
            # Convert it to a Pandas DataFrame
            correlations_df = pd.DataFrame(matrix)
            # sort the correlation matrix
            matrix = correlations_df.sort_values(by=0, axis=1, ascending=False)

        # Creating a heatmap with sort correlations
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.heatmap(matrix, annot=False, cmap=cmap, ax=ax)

    def plot_corr_hist_heat_salience(
        self,
        correlation: np.ndarray,
        saliences,
        title: str,
        bins: int = 100,
        sort=False,
        figsize=(17, 5),
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        self.histogam_subplot(
            correlation,
            "Correlation",
            ax1,
            bins=bins,
            xlim=[-1, 1],
            xlabel="Correlation Value",
            ylabel="Frequency",
        )
        self.heatmap_subplot(correlation, "Correlation Heatmap", ax2, sort=sort)
        self.histogam_subplot(
            saliences,
            "Saliences",
            ax3,
            xlim=[0, 2],
            bins=bins,
            xlabel="n",
            ylabel="Frequency",
        )
        self.plot_ending(title, save=True)

    def plot_dist_sal_dims(
        self, distances, saliences, normalized_saliences, title, bins=100
    ):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(17, 10))
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        title = title + " Histograms"

        self.histogam_subplot(
            distances,
            "Distance from Origin",
            ax1,
            bins=bins,
            color=colors[0],
            xlim=[0, 2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            saliences,
            "Normalized Distances",
            ax2,
            bins=bins,
            color=colors[1],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 0],
            "normalized X",
            ax3,
            bins=bins,
            color=colors[2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 1],
            "normalized Y",
            ax4,
            bins=bins,
            color=colors[3],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 2], "normalized Z", ax5, bins=bins, color=colors[4]
        )
        self.plot_ending(title, save=True)

    def plot_dist_sal_dims_2(
        self,
        distances,
        saliences,
        normalized_saliences,
        distances2,
        saliences2,
        normalized_saliences2,
        title,
        bins=100,
    ):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 2, figsize=(17, 10))
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        title = title + " Histograms"

        self.histogam_subplot(
            distances,
            "Distance from Origin",
            ax1[0],
            bins=bins,
            color=colors[0],
            xlim=[0, 2],
        )
        self.histogam_subplot(
            saliences,
            "Normalized Distances",
            ax2[0],
            bins=bins,
            color=colors[1],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 0],
            "normalized X",
            ax3[0],
            bins=bins,
            color=colors[2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 1],
            "normalized Y",
            ax4[0],
            bins=bins,
            color=colors[3],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences[:, 2],
            "normalized Z",
            ax5[0],
            bins=bins,
            color=colors[4],
        )

        self.histogam_subplot(
            distances2,
            "Distance from Origin",
            ax1[1],
            bins=bins,
            color=colors[0],
            xlim=[0, 2],
        )
        self.histogam_subplot(
            saliences2,
            "Normalized Distances",
            ax2[1],
            bins=bins,
            color=colors[1],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences2[:, 0],
            "normalized X",
            ax3[1],
            bins=bins,
            color=colors[2],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences2[:, 1],
            "normalized Y",
            ax4[1],
            bins=bins,
            color=colors[3],
            xticklabels="empty",
        )
        self.histogam_subplot(
            normalized_saliences2[:, 2],
            "normalized Z",
            ax5[1],
            bins=bins,
            color=colors[4],
        )
        self.plot_ending(title, save=True)

    def plot_corr_heat_corr_heat(
        self, correlation1, correlation2, title1, title2, sort=False, figsize=(17, 5)
    ):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
        title = title1 + " vs " + title2
        self.histogam_subplot(
            correlation1,
            title1 + " Correlation",
            ax1,
            bins=100,
            xlim=[-1, 1],
            xlabel="Correlation Value",
            ylabel="Frequency",
            color="tab:blue",
        )
        self.heatmap_subplot(correlation1, title1, ax2, sort=sort)
        self.histogam_subplot(
            correlation2,
            title2 + " Correlation",
            ax3,
            bins=100,
            xlim=[-1, 1],
            xlabel="Correlation Value",
            ylabel="Frequency",
            color="tab:orange",
        )
        self.heatmap_subplot(correlation2, title2, ax4, sort=sort)
        self.plot_ending(title, save=True)

    def plot_ending(self, title, save=True):
        plt.suptitle(title)
        plt.tight_layout()  # Ensure subplots fit within figure area
        plot_save_path = (
            str(self.save_dir.joinpath(title + ".png"))
            .replace(">", "bigger")
            .replace("<", "smaller")
        )
        if save:
            plt.savefig(plot_save_path, dpi=300)
        plt.show()
        plt.close()

    @staticmethod
    def plot_traces_shifted(
        traces, figsize_x=20, labels=None, additional_title=None, savepath=None
    ):
        """
        Plots traces shifted up by 10 for each trace
        """
        title = "Activity"
        if additional_title:
            title += f" {additional_title}"

        lines_per_y = 1
        y_size = int(len(traces) / lines_per_y)
        figsize = (figsize_x, y_size / 2)
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)
        neg_counter = 0
        for i, trace in enumerate(traces):
            if np.isnan(trace).any():
                neg_counter += 1
                continue
            if np.sum(trace) == 0:
                continue
            trace = normalize_vector_01(trace)
            if labels is not None:
                label = f"{labels[i]:.3f}"
                ax.plot(trace + (i - neg_counter) / lines_per_y, label=label)
            ax.plot(trace + (i - neg_counter) / lines_per_y)
        plt.ylim(-1, y_size)
        plt.xlim(0, traces.shape[1])
        plt.title(title)
        if savepath:
            plt.savefig(savepath)
        if labels is not None:
            plt.legend(loc="upper right")
        plt.show()
        plt.close()

    @staticmethod
    def plot_cell_activites_heatmap(
        rate_map, additional_title=None, norm_rate=False, sorting_indices=None, xlabel="location (cm)", ylabel="Cell id"
    ):
        # ordered activities by peak
        title = "Cell Activities"
        if additional_title:
            title = title + f" {additional_title}"
        if not norm_rate:
            title += " not normalized"
        if sorting_indices is not None:
            title += " order provided"

        # normalize by cell firering rate
        if norm_rate:
            plot_rate_map = rate_map.copy()
            for i, cell_rate in enumerate(rate_map):
                plot_rate_map[i] = normalize_vector_01(cell_rate)
        else:
            plot_rate_map = rate_map

        sorted_rate_map, indices = sort_arr_by(
            plot_rate_map, axis=1, sorting_indices=sorting_indices
        )

        plt.figure()
        plt.imshow(sorted_rate_map, aspect="auto", interpolation="None")

        plt.ylabel(ylabel)
        # remove yticks
        plt.xlabel(xlabel)
        plt.title(title)
        plt.show(block=False)
        plt.close()
        return sorted_rate_map, indices

    @staticmethod
    def create_histogram(
        data,
        title,
        xlabel,
        ylabel,
        data_labels=None,
        bins=100,
        red_line_pos=None,
        color=None,
        interactive=False,
        stacked=True,
    ):
        if interactive:
            fig = px.histogram(
                data,
                title=title,
                labels={"value": xlabel, "count": ylabel},
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False)
            fig.show()
        else:
            plt.figure(figsize=(10, 3))
            plt.hist(data, bins=bins, label=data_labels, color=color, stacked=True)
            plt.legend()
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if red_line_pos is not None:
                plt.axvline(red_line_pos, color="r", linestyle="dashed", linewidth=1)
            plt.show(block=False)
            plt.close()

    @staticmethod
    def plot_zscore(
        zscore,
        additional_title=None,
        data_labels=None,
        color=None,
        interactive=False,
        zscore_threshold=2.5,
        stacked=True,
    ):
        zscore = np.array(zscore)
        title = "Zscore distribution of Spatial Information"
        xlabel = "Zscore"
        ylable = "# of cells"
        if additional_title is not None:
            title = title + f" ({additional_title})"
        zscores = split_array_by_zscore(zscore, zscore, threshold=2.5)
        if data_labels is None:
            data_labels = [
                f"{len(zscores[0])} values > {zscore_threshold}",
                f" {len(zscores[1])} values < {zscore_threshold}",
            ]
        Vizualizer.create_histogram(
            zscores,
            title,
            xlabel,
            ylable,
            bins=100,
            data_labels=data_labels,
            red_line_pos=zscore_threshold,
            color=color,
            interactive=interactive,
            stacked=stacked,
        )

    @staticmethod
    def plot_si_rates(
        si_rate,
        data_labels=None,
        additional_title=None,
        zscore=None,
        zscore_threshold=2.5,
        color=None,
        interactive=False,
        stacked=True,
    ):
        si_rate = np.array(si_rate)
        title = "Spatial Information Rate"
        xlabel = "Spatial Information Rate [bits/sec]"
        ylabel = "# of cells"
        if additional_title is not None:
            title = title + f" ({additional_title})"
        data = (
            split_array_by_zscore(si_rate, zscore, threshold=zscore_threshold)
            if zscore is not None
            else si_rate
        )
        if data_labels is None and len(data) == 2:
            data_labels = [
                f"{len(data[0])} <= 5% probability",
                f" {len(data[1])} > 5% probability",
            ]
        Vizualizer.create_histogram(
            data,
            title,
            xlabel,
            ylabel,
            bins=100,
            data_labels=data_labels,
            color=color,
            interactive=interactive,
            stacked=stacked,
        )

    @staticmethod
    def plot_sanky_example():
        #####################################################
        ################### SANKEY PLOTS ####################
        #####################################################
        #
        label_list = ["cat", "dog", "domesticated", "female", "male", "wild", "neither"]
        # cat: 0, dog: 1, domesticated: 2, female: 3, male: 4, wild: 5
        source = [0, 0, 1, 3, 4, 4, 1]
        target = [3, 4, 4, 2, 2, 5, 6]
        count = [5, 6, 22, 21, 6, 22, 5]

        fig = go.Figure(
            data=[
                go.Sankey(
                    node={"label": label_list},
                    link={"source": source, "target": target, "value": count},
                )
            ]
        )

        fig.show()
