# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import plotly
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import colorsys


from scipy.spatial import ConvexHull

import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

# from pySankey.sankey import sankey
import cebra
import pandas as pd

# OWN
from Helper import *


class Vizualizer:
    def __init__(self, root_dir) -> None:
        self.save_dir = Path(root_dir).joinpath("figures")
        dir_exist_create(self.save_dir)

    def plot_dataplot_summary(self, plot_dir, title="Data Summary"):
        # TODO: combine plots from top to bottom aligned by time
        # distance, velocity, acceleration, position, raster
        pass

    def calculate_alpha(value, min_value, max_value, min_alpha=0.1, max_alpha=1.0):
        """
        Calculates the alpha (transparency) value based on the given value and its range.

        Parameters
        ----------
        value : float
            The value for which to calculate the alpha.
        min_value : float
            The minimum value of the range.
        max_value : float
            The maximum value of the range.
        min_alpha : float, optional
            The minimum alpha value (default is 0.1).
        max_alpha : float, optional
            The maximum alpha value (default is 1.0).

        Returns
        -------
        float
            The calculated alpha value.
        """
        if max_value == min_value:
            return 0.1  # max_alpha
        normalized_value = (value - min_value) / (max_value - min_value)
        return min_alpha + (max_alpha - min_alpha) * normalized_value

    @staticmethod
    def generate_similar_colors(base_color, num_colors):
        """
        Generates a list of similar colors based on a base color.
        """
        base_hls = colorsys.rgb_to_hls(*base_color[:3])
        colors = []
        for i in range(num_colors):
            hue = (base_hls[0] + i * 0.05) % 1.0  # Small hue variations
            lightness = max(
                0, min(1, base_hls[1] + (i - num_colors / 2) * 0.1)
            )  # Slight lightness variations
            rgb = colorsys.hls_to_rgb(hue, lightness, base_hls[2])
            colors.append(rgb)
        return colors

    #############################  Data Plots #################################################
    @staticmethod
    def default_plot_attributes():
        return {
            "fps": None,
            "title": None,
            "ylable": None,
            "ylimits": None,
            "yticks": None,
            "xlable": None,
            "xlimits": None,
            "xticks": None,
            "num_ticks": None,
            "figsize": None,
            "save_path": None,
        }

    @staticmethod
    def define_plot_parameter(
        plot_attributes=None,
        fps=None,
        title=None,
        ylable=None,
        ylimits=None,
        yticks=None,
        xlable=None,
        xticks=None,
        num_ticks=None,
        xlimits=None,
        figsize=None,
        save_path=None,
    ):
        if plot_attributes is None:
            plot_attributes = Vizualizer.default_plot_attributes()
        plot_attributes["fps"] = fps or plot_attributes["fps"]
        plot_attributes["title"] = title or plot_attributes["title"]
        plot_attributes["title"] = (
            plot_attributes["title"]
            if plot_attributes["title"][-4:] == "data"
            else plot_attributes["title"] + " data"
        )

        plot_attributes["ylable"] = ylable or plot_attributes["ylable"] or None
        plot_attributes["ylimits"] = ylimits or plot_attributes["ylimits"] or None
        plot_attributes["yticks"] = yticks or plot_attributes["yticks"] or None
        plot_attributes["xlable"] = xlable or plot_attributes["xlable"] or "time"
        plot_attributes["xlimits"] = xlimits or plot_attributes["xlimits"] or None
        plot_attributes["xticks"] = xticks or plot_attributes["xticks"] or None
        plot_attributes["num_ticks"] = num_ticks or plot_attributes["num_ticks"] or 100
        plot_attributes["figsize"] = figsize or plot_attributes["figsize"] or (20, 3)
        plot_attributes["save_path"] = save_path or plot_attributes["save_path"]

        # create plot dir if missing
        plot_attributes["save_path"] = Path(plot_attributes["save_path"])
        if not plot_attributes["save_path"].parent.exists():
            plot_attributes["save_path"].parent.mkdir(parents=True, exist_ok=True)

        return plot_attributes

    @staticmethod
    def default_plot_start(
        plot_attributes: dict = None,
        figsize=None,
        title=None,
        xlable=None,
        xlimits=None,
        xticks=None,
        ylable=None,
        ylimits=None,
        yticks=None,
        fps=None,
        num_ticks=50,
        save_path=None,
    ):
        plot_attributes = Vizualizer.define_plot_parameter(
            plot_attributes=plot_attributes,
            figsize=figsize,
            title=title,
            ylable=ylable,
            ylimits=ylimits,
            yticks=yticks,
            xlable=xlable,
            xlimits=xlimits,
            xticks=xticks,
            num_ticks=num_ticks,
            fps=fps,
            save_path=save_path,
        )
        plt.figure(figsize=plot_attributes["figsize"])
        plt.title(plot_attributes["title"])
        plt.ylabel(plot_attributes["ylable"])
        if plot_attributes["ylimits"]:
            plt.ylim(plot_attributes["ylimits"])
        if plot_attributes["yticks"]:
            plt.yticks(plot_attributes["yticks"][0], plot_attributes["yticks"][1])
        plt.xlabel(plot_attributes["xlable"])
        plt.tight_layout()
        plt.xlim(plot_attributes["xlimits"])
        return plot_attributes

    @staticmethod
    def plot_image(
        plot_attributes=None,
        figsize=(10, 10),
        save_path=None,
        show=False,
    ):
        if plot_attributes is None:
            plot_attributes = Vizualizer.default_plot_attributes()

        plot_attributes["figsize"] = plot_attributes["figsize"] or figsize
        plot_attributes["save_path"] = plot_attributes["save_path"] or save_path

        plt.figure(figsize=plot_attributes["figsize"])
        # Load the image
        image = plt.imread(plot_attributes["save_path"])
        plt.imshow(image)
        plt.axis("off")

        if show:
            plt.show()

    @staticmethod
    def default_plot_ending(
        plot_attributes=None, regenerate_plot=False, save_path=None, show=False, dpi=300
    ):
        if plot_attributes is None:
            plot_attributes = Vizualizer.default_plot_attributes()
        plot_attributes["save_path"] = plot_attributes["save_path"] or save_path

        if regenerate_plot:
            plt.savefig(plot_attributes["save_path"], dpi=dpi)

        if show:
            plt.show()
            plt.close()

    def data_plot_1D(
        data,
        plot_attributes: dict = None,
        marker_pos=None,
        marker=None,
        seconds_interval=5,
    ):
        if data.ndim == 1:
            plt.plot(data, zorder=1)
        else:
            named_dimensions = {0: "x", 1: "y", 2: "z"}
            for dim in range(data.shape[1]):
                plt.plot(data[:, dim], zorder=1, label=f"{named_dimensions[dim]}")
            plt.legend()

        if marker_pos is not None:
            marker = "." if marker is None else marker
        else:
            marker_pos = range(len(data)) if marker is not None else None

        if marker:
            plt.scatter(marker_pos, data[marker_pos], marker=marker, s=10, color="red")

        num_frames = data.shape[0]

        if plot_attributes["xticks"]:
            if plot_attributes["xticks"] == "auto":
                plt.xticks()
            else:
                plt.xticks(plot_attributes["xticks"])
        else:
            xticks, xpos = Vizualizer.define_xticks(
                plot_attributes=plot_attributes,
                num_frames=num_frames,
                fps=plot_attributes["fps"],
                num_ticks=plot_attributes["num_ticks"],
                seconds_interval=seconds_interval,
            )
            plt.xticks(xpos, xticks, rotation=45)

    @staticmethod
    def data_plot_2D(
        data,
        plot_attributes,
        position_data,
        border_limits=None,
        marker_pos=None,
        marker=None,
        fps=None,
        seconds_interval=5,
        colormap_label: str = None,
        tick_label=None,
        tick_pos=None,
        color_by="value",
        cmap="plasma",  # "viridis", "winter", "plasma"
    ):
        # data = data[:15000, :]
        # data = may_butter_lowpass_filter(
        #    data,
        #    smooth=True,
        #    cutoff=1,
        #    fps=20,
        #    order=2,
        # )
        data, position_data = force_equal_dimensions(data, position_data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # Convert coordinates to a numpy array if it's not already
        coordinates = np.array(position_data) * 100  # Convert to cm

        # Extract x and y coordinates
        x_coords = coordinates[:, 0]
        y_coords = coordinates[:, 1]

        # Generate a time array based on the number of coordinates
        if color_by == "time":
            num_frames = data.shape[0]
            color_map_label = f"Time"
            color_value_reference = range(num_frames)
            tick_label, tick_pos = Vizualizer.define_xticks(
                plot_attributes=plot_attributes,
                num_frames=num_frames,
                fps=plot_attributes["fps"],
                num_ticks=int(plot_attributes["num_ticks"] / 2),
                seconds_interval=seconds_interval,
            )
            scatter_alpha = 0.8
            dot_size = 1
        elif color_by == "value":
            absolute_data = np.linalg.norm(np.abs(data), axis=1)
            color_value_reference = np.array(absolute_data)
            color_map_label = colormap_label or plot_attributes["ylable"]

            if tick_label is None:
                if plot_attributes["yticks"] is not None:
                    tick_pos, tick_label = plot_attributes["yticks"]
                if tick_label is not None and tick_pos is None:
                    tick_pos = np.range(0, 1, len(tick_label))
            scatter_alpha = 0.8
            dot_size = 3
        # Create the plot
        scatter = plt.scatter(
            x_coords,
            y_coords,
            c=color_value_reference,
            cmap=cmap,
            s=dot_size,
            alpha=scatter_alpha,
        )

        if border_limits is not None:
            # Add border lines
            plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)
            plt.axvline(x=border_limits[0], color="r", linestyle="--", alpha=0.5)
            plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            plt.axhline(y=border_limits[1], color="r", linestyle="--", alpha=0.5)

        x_data_range = max(coordinates[:, 0]) - min(coordinates[:, 0])
        y_data_range = max(coordinates[:, 1]) - min(coordinates[:, 1])
        xlimits = (
            min(coordinates[:, 0] - x_data_range * 0.03),
            max(coordinates[:, 0] + x_data_range * 0.03),
        )
        ylimits = (
            min(coordinates[:, 1] - y_data_range * 0.03),
            max(coordinates[:, 1] + y_data_range * 0.03),
        )

        plt.xlabel("X position (cm)")
        plt.ylabel("Y position (cm)")
        plt.xlim(xlimits)
        plt.ylim(ylimits)

        # Add a colorbar to show the time mapping
        cbar = plt.colorbar(scatter, label=color_map_label)
        if tick_label is not None and tick_pos is not None:
            cbar.set_ticks(tick_pos)
            cbar.set_ticklabels(tick_label)

        plt.grid(True, alpha=0.5)

    @staticmethod
    def define_xticks(
        plot_attributes=None,
        num_frames=None,
        fps=None,
        num_ticks=None,
        seconds_interval=5,
    ):
        if num_frames is not None:
            xticks, xpos = range_to_times_xlables_xpos(
                end=num_frames,
                fps=fps,
                seconds_per_label=seconds_interval,
            )

            # reduce number of xticks
            if len(xpos) > num_ticks:
                steps = round(len(xpos) / num_ticks)
                xticks = xticks[::steps]
                xpos = xpos[::steps]
        else:
            xticks, xpos = None, None
        return xticks, xpos

    @staticmethod
    def plot_neural_activity_raster(
        data,
        fps,
        num_ticks=None,
        seconds_interval=5,
    ):

        binarized_data = data
        num_time_steps, num_neurons = binarized_data.shape
        # Find spike indices for each neuron
        spike_indices = np.nonzero(binarized_data)
        # Creating an empty image grid
        image = np.zeros((num_neurons, num_time_steps))
        # Marking spikes as pixels in the image grid
        image[spike_indices[1], spike_indices[0]] = 1
        # Plotting the raster plot using pixels
        plt.imshow(image, cmap="gray", aspect="auto", interpolation="none")
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization of trials/neurons

        xticks, xpos = Vizualizer.define_xticks(
            num_frames=num_time_steps,
            fps=fps,
            num_ticks=num_ticks,
            seconds_interval=seconds_interval,
        )
        plt.xticks(xpos, xticks, rotation=45)

    ########################################################################################################################
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
        embedding, labels = force_equal_dimensions(
            embedding, embedding_labels["labels"]
        )
        if embedding.shape[1] == 2:
            ax = Vizualizer.plot_embedding_2d(
                axis=ax,
                embedding=embedding,
                embedding_labels=labels,
                markersize=markersize,
                alpha=alpha,
                cmap=cmap,
                title=title,
                figsize=figsize,
                dpi=dpi,
                plot_legend=plot_legend,
            )
        elif embedding.shape[1] == 3:
            ax = Vizualizer.plot_embedding_3d(
                axis=ax,
                embedding=embedding,
                embedding_labels=labels,
                markersize=markersize,
                alpha=alpha,
                cmap=cmap,
                title=title,
                figsize=figsize,
                dpi=dpi,
                plot_legend=plot_legend,
            )
        else:
            raise NotImplementedError(
                "Invalid labels dimension. Choose 2D or 3D labels."
            )
        return ax

    def plot_embedding_2d(
        embedding: Union[npt.NDArray, torch.Tensor],
        embedding_labels: Optional[Union[npt.NDArray, torch.Tensor, str]],
        idx_order: Optional[Tuple[int]] = None,
        markersize: float = 0.05,
        alpha: float = 0.4,
        cmap: str = "cool",
        title: str = "2D Embedding",
        axis: Optional[matplotlib.axes.Axes] = None,
        figsize: tuple = (5, 5),
        dpi: float = 100,
        grey_fig: bool = False,
        plot_legend: bool = True,
        **kwargs,
    ):
        """
        This function is based on the plot_embedding function from the cebra library.
        """
        # define the axis
        if axis is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot()
        else:
            ax = axis

        # define idx order
        if idx_order is None:
            idx_order = (0, 1)

        else:
            # If the idx_order was provided by the user
            ## Check size validity
            if len(idx_order) != 2:
                raise ValueError(
                    f"idx_order must contain 2 dimension values, got {len(idx_order)}."
                )

            # Check value validity
            for dim in idx_order:
                if dim < 0 or dim > embedding.shape[1] - 1:
                    raise ValueError(
                        f"List of dimensions to plot is invalid, got {idx_order}, with {dim} invalid."
                        f"Values should be between 0 and {embedding.shape[1]}."
                    )

        # plot the embedding
        (
            idx1,
            idx2,
        ) = idx_order
        ax.scatter(
            xs=embedding[:, idx1],
            ys=embedding[:, idx2],
            c=embedding_labels,
            cmap=cmap,
            alpha=alpha,
            s=markersize,
            **kwargs,
        )

        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.set_title(title, y=1.0, pad=-10)

        if grey_fig:
            ax.xaxis.pane.set_edgecolor("grey")
            ax.yaxis.pane.set_edgecolor("grey")

        if plot_legend:
            Vizualizer.add_2d_colormap_legend(fig, move_right=1)

        return ax

    def plot_embedding_3d_by_groups(
        groups: dict,
        plot_hulls=False,
        alpha=0.8,
        hull_alpha=0.2,
        legend=True,
        title="3D Plot",
        filter_outliers=True,
    ):
        """
        Create 3D plot with convex hull surfaces for each group of points.
        parameters:
        - groups: dict
            - keys: labels for each group (Only numeric labels are supported for now.)
            - values: list of 3D points for each group
        """
        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        np_labels = np.array(list(groups.keys()))
        rgba_colors = Vizualizer.create_rgba_labels(np_labels, alpha=alpha)

        # Plot the points
        for rgba, (label, points) in zip(rgba_colors, groups.items()):
            points = np.array(points)
            colors = np.array([rgba for _ in range(len(points))])
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=1,
                label=label,
            )

        # Plot the convex hull surfaces
        if plot_hulls:
            title += " with Convex Hulls"

            hulls = {}
            for label, points in groups.items():
                points = np.array(points)
                filtered_points = filter_outlier(points) if filter_outliers else points
                if (
                    len(filtered_points) >= 4
                ):  # Minimum 4 filtered_points needed to form a 3D convex hull
                    hull = ConvexHull(filtered_points)
                    hulls[label] = hull

            for rgba, (label, hull) in zip(rgba_colors, hulls.items()):
                vertices = hull.points[hull.vertices]
                poly3d = []
                for s in hull.simplices:
                    available_vertices = []
                    for v in s:
                        if len(vertices) <= v:
                            bad = True
                        else:
                            available_vertices.append(vertices[v])
                    if np.array(available_vertices).shape[0] == 3:
                        poly3d.append(np.array(available_vertices))

                poly3d = np.array(poly3d)
                ax.add_collection3d(
                    Poly3DCollection(
                        poly3d,
                        facecolors=rgba,
                        linewidths=0.01,
                        edgecolors="r",
                        alpha=hull_alpha,
                    )
                )

        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set plot title
        ax.set_title(title)

        # Add legend
        if legend:
            ax.legend()

        Vizualizer.add_2d_colormap_legend(fig, move_right=0.9)

        # Show the plot
        plt.show()

    def plot_embedding_3d(
        embedding: Union[npt.NDArray, torch.Tensor],
        embedding_labels: Optional[Union[npt.NDArray, torch.Tensor, str]],
        idx_order: Optional[Tuple[int]] = None,
        markersize: float = 0.05,
        alpha: float = 0.4,
        cmap: str = "cool",
        title: str = "3D Embedding",
        axis: Optional[matplotlib.axes.Axes] = None,
        figsize: tuple = (5, 5),
        dpi: float = 100,
        grey_fig: bool = False,
        plot_legend: bool = True,
        **kwargs,
    ):
        """
        This function is based on the plot_embedding function from the cebra library.
        """

        # define the axis
        if axis is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(projection="3d")
        else:
            ax = axis

        # define idx order
        if idx_order is None:
            idx_order = (0, 1, 2)

        # plot the embedding
        idx1, idx2, idx3 = idx_order
        ax.scatter(
            xs=embedding[:, idx1],
            ys=embedding[:, idx2],
            zs=embedding[:, idx3],
            c=embedding_labels,
            cmap=cmap,
            alpha=alpha,
            s=markersize,
            **kwargs,
        )

        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        ax.set_title(title, y=1.08, pad=-10)

        if grey_fig:
            ax.xaxis.pane.set_edgecolor("grey")
            ax.yaxis.pane.set_edgecolor("grey")
            ax.zaxis.pane.set_edgecolor("grey")

        if plot_legend:
            if embedding_labels.ndim == 1:
                Vizualizer.add_1d_colormap_legend(
                    ax, embedding_labels, label_name="labels", ticks=None, cmap=cmap
                )
            elif embedding_labels.ndim == 2:
                Vizualizer.add_2d_colormap_legend(fig)
            else:
                raise ValueError("Invalid labels dimension. Choose 2D or 3D labels.")

        return ax

    def add_1d_colormap_legend(
        ax, labels, label_name="labels", ticks=None, cmap="rainbow"
    ):
        # Create a ScalarMappable object using the specified colormap
        sm = plt.cm.ScalarMappable(cmap=cmap)
        unique_labels = np.unique(labels)
        unique_labels.sort()
        sm.set_array(unique_labels)  # Set the range of values for the colorbar

        # Manually create colorbar
        cbar = plt.colorbar(sm, ax=ax)
        # Adjust colorbar ticks if specified
        cbar.set_label(label_name)  # Set the label for the colorbar
        if ticks is not None:
            cbar.ax.yaxis.set_major_locator(
                MaxNLocator(integer=True)
            )  # Adjust ticks to integers
            cbar.set_ticks(
                np.linspace(cbar.vmin, cbar.vmax, len(ticks))
            )  # Set custom ticks
            cbar.set_ticklabels(ticks)

    def add_2d_colormap_legend(
        fig, move_right=1, xticks=None, yticks=None, additional_title=""
    ):
        # plt.subplots_adjust(left=0.1, right=0.65, top=0.85)
        cax = fig.add_axes([move_right, 0.55, 0.3, 0.3])
        cp1 = np.linspace(0, 1)
        cp2 = np.linspace(0, 1)
        Cp1, Cp2 = np.meshgrid(cp1, cp2)
        C0 = np.zeros_like(Cp1) + 0.5
        # make RGB image, p1 to red channel, p2 to blue channel
        Legend = np.dstack((Cp1, C0, Cp2))
        # parameters range between 0 and 1
        cax.imshow(Legend, origin="lower", extent=[0, 1, 0, 1])
        cax.set_xlabel("X")
        cax.set_ylabel("Y")
        if xticks is None:
            xticks = []
        if yticks is None:
            yticks = []
        title = f"2D colormap - {additional_title}"
        cax.set_xticks(np.linspace(0, 1, len(xticks)))
        cax.set_yticks(np.linspace(0, 1, len(yticks)))
        xlabels = [f"{x:.1f}" for x in xticks]
        ylabels = [f"{y:.1f}" for y in yticks]
        cax.set_xticklabels(xlabels, rotation=45, ha="right")
        cax.set_yticklabels(ylabels)
        cax.set_title(title, fontsize=10)

    def plot_multiple_embeddings(
        self,
        embeddings: dict,
        labels: dict,
        ticks=None,
        title="Embeddings",
        cmap="rainbow",
        projection="3d",
        figsize=(20, 4),
        plot_legend=True,
        markersize=0.05,
        alpha=0.4,
        max_plot_per_row=4,
        dpi=300,
    ):
        figsize_y = 4
        # Compute the number of subplots
        num_subplots = len(embeddings)
        rows = 1
        cols = num_subplots
        if num_subplots > max_plot_per_row:
            rows = int(num_subplots**0.5)
            cols = (num_subplots + rows - 1) // rows
            figsize_x = 5 * max_plot_per_row
        else:
            figsize_x = 5 * num_subplots
        figsize = (figsize_x, figsize_y * rows)

        fig = plt.figure(figsize=figsize)
        fig, axes = plt.subplots(
            rows, cols, figsize=figsize, subplot_kw={"projection": projection}
        )

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        # create 2D RGBA labels to overwrite 1D cmap coloring
        if labels["labels"].shape[1] == 2:
            min_vals = np.min(labels["labels"], axis=0)
            max_vals = np.max(labels["labels"], axis=0)
            # steps = 5
            xticks_2d_colormap = np.linspace(min_vals[0], max_vals[0], 5)
            yticks_2d_colormap = np.linspace(min_vals[1], max_vals[1], 5)

            rgba_colors = Vizualizer.create_rgba_labels(labels["labels"])
            labels["labels"] = rgba_colors

        for i, (subplot_title, embedding) in enumerate(embeddings.items()):
            ax = axes[i]
            ax = self.plot_embedding(
                ax=ax,
                embedding=embedding,
                embedding_labels=labels,
                title=subplot_title,
                cmap=cmap,
                plot_legend=False,
                markersize=markersize,
                alpha=alpha,
                dpi=dpi,
            )

        if plot_legend:
            if labels["labels"].shape[1] == 1:
                Vizualizer.add_1d_colormap_legend(
                    ax,
                    labels["labels"],
                    label_name=labels["name"],
                    ticks=ticks,
                    cmap=cmap,
                )
            else:
                Vizualizer.add_2d_colormap_legend(
                    fig,
                    move_right=0.95,
                    xticks=xticks_2d_colormap,
                    yticks=yticks_2d_colormap,
                    additional_title=labels["name"],
                )

        for ax in axes[num_subplots:]:
            ax.remove()  # Remove any excess subplot axes

        self.plot_ending(title)

    def create_rgba_labels(values, alpha=0.8):
        """c : array-like or list of colors or color, optional
        The marker colors. Possible values:

        - A scalar or sequence of n numbers to be mapped to colors using
            *cmap* and *norm*.
        - A 2D array in which the rows are RGB or RGBA.
        - A sequence of colors of length n.
        - A single color format string.

        Note that *c* should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values to be
        colormapped. If you want to specify the same RGB or RGBA value for
        all points, use a 2D array with a single row.  Otherwise,
        value-matching will have precedence in case of a size matching with
        *x* and *y*.

        If you wish to specify a single color for all points
        prefer the *color* keyword argument.

        Defaults to `None`. In that case the marker color is determined
        by the value of *color*, *facecolor* or *facecolors*. In case
        those are not specified or `None`, the marker color is determined
        by the next color of the ``Axes``' current "shape and fill" color
        cycle. This cycle defaults to :rc:`axes.prop_cycle`."""

        cmap = lambda x, y: (x, 0.5, y, alpha)
        normalized_values = normalize_01(values, axis=0)
        values = np.array(values)
        if values.ndim == 1:
            raise ValueError("1D values not supported yet.")
        elif values.ndim == 2:
            # Create a 2D array of RGBA values
            rgba_colors = np.array([cmap(x, y) for x, y in normalized_values])
        elif values.ndim == 3:
            raise ValueError("3D values not supported yet.")
        return rgba_colors

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
        Vizualizer.heatmap_subplot(correlation, "Correlation Heatmap", ax2, sort=sort)
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
        Vizualizer.heatmap_subplot(correlation1, title1, ax2, sort=sort)
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
        Vizualizer.heatmap_subplot(correlation2, title2, ax4, sort=sort)
        self.plot_ending(title, save=True)

    def plot_ending(self, title, title_size=20, save=True):
        plt.suptitle(title, fontsize=title_size)
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

    #################################################################
    ##### statistics of decoding (accuracy, precision, recall, f1-score)
    @staticmethod
    def plot_decoding_statistics(
        decoder_results: List[float],
        additional_title: str = "",
    ):
        decoded_test_datasets_reverse = None
        decoded_lists = [[decoder_results]]
        labels = [["All Cells"]]
        labels_flattened = ["All Cells"]
        for (
            scoring_type_reverse,
            decoded_test_sets,
        ) in decoded_test_datasets_reverse.items():
            decoded_lists.append([])
            labels.append([])
            for percentage, decoded_test_set in decoded_test_sets.items():
                decoded_lists[-1].append(decoded_test_set)
                label = f"{scoring_type_reverse} - {percentage}% cells"
                labels[-1].append(label)
                labels_flattened.append(label)
        print(labels)

        # viz = Vizualizer(root_dir=root_dir)
        # TODO: Is this working????????

        # viz.plot_decoding_score(decoded_model_lists=decoded_model_lists, labels=labels, figsize=(13, 5))
        title = f"{additional_title} lowest % cells fro Behavioral Decoding of stimulus"
        fig = plt.figure(figsize=(13, 5))
        fig.suptitle(title, fontsize=16)
        colors = ["green", "red", "deepskyblue"]
        ax1 = plt.subplot(111)
        # ax2 = plt.subplot(211)

        overall_num = 0
        for color, docoded_model_list, labels_list in zip(
            colors, decoded_lists, labels
        ):
            for num, (decoded, label) in enumerate(
                zip(docoded_model_list, labels_list)
            ):
                # color = "deepskyblue" if "A\'" == "".join(label[:2]) else "red" if "B" == label[0] else "green"
                alpha = 1 - ((1 / len(docoded_model_list)) * num / 1.3)
                x_pos = overall_num + num
                width = 0.4  # Width of the bars
                ax1.bar(
                    # x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
                    x_pos,
                    decoded[1],
                    width=0.4,
                    color=color,
                    alpha=alpha,
                    label=label,
                )
                # ax2.bar(
                # x_pos, decoded[1], width=0.4, color=color, alpha=alpha, label = label
                # x_pos, decoded[2], width=0.4, color=color, alpha=alpha, label = label
                # )
                ##ax2.scatter(
                #    middle_A_model.state_dict_["loss"][-1],
                #    decoded[1],
                #    s=50,
                #    c=color,
                #    alpha=alpha,
                #    label=label,
                # )
            overall_num = x_pos + 1

        ylabel = "Mean stimulus error"

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_ylabel(ylabel)
        ax1.grid(axis="y", alpha=0.2)
        print_labels = labels_flattened
        label_pos = np.arange(len(labels_flattened))
        ax1.set_xticks(label_pos)
        # ax1.set_ylim([0, 1])
        ax1.set_xticklabels(print_labels, rotation=45, ha="right")

        ylabel = "mean stimulus in cm"

        # ax2.spines["top"].set_visible(False)
        # ax2.spines["right"].set_visible(False)
        # ax2.set_ylabel(ylabel)
        # ax2.grid(axis="y", alpha=0.5)
        # print_labels = labels_flattened
        # label_pos = np.arange(len(labels_flattened))
        # ax2.set_xticks(label_pos)
        ##ax2.set_ylim([0, 130])
        # ax2.set_xticklabels(print_labels, rotation=45, ha="right")

        # plt.legend()
        plt.show()

    @staticmethod
    def plot_decoding_statistics_line(
        models,
        by="task",
        additional_title: Optional[str] = None,
        markersize: float = 0.05,
        alpha: float = 0.4,
        dpi: int = 300,
    ):
        """
        decoding results by trainings iterations (left to right), 1 line for every task
        Assumtions:
            - models is a dictionary of dictionaries of dictionaries
                - models[session_date][task_name][model_name] = model
                - model has decoding_statistics attribute
            - models are sorted by time
        """
        continuouse_stats = ["mse", "rmse", "r2"]
        discrete_stats = ["accuracy", "precision", "recall", "f1-score"]

        summary_decodings_by_iterations = {}
        for session_date, session_dict in models.items():
            for task_name, models_dict in session_dict.items():
                # Assuming task names are unique
                init_dict_in_dict(summary_decodings_by_iterations, task_name)
                for model_name, model in models_dict.items():
                    # Assume modles are sorted by time
                    # extract iterations from model name
                    iterations = None
                    for name_part in model_name.split("_"):
                        if "iter-" in name_part:
                            iterations = int(name_part.replace("iter-", ""))
                    if iterations is None:
                        print(
                            f"No iterations information was found in model name: {model_name}. Skipping..."
                        )
                        continue
                    else:
                        current_dict = init_dict_in_dict(
                            summary_decodings_by_iterations[task_name], iterations
                        )

                    # create a list of decodings stats
                    for stat_name, stat in model.decoding_statistics.items():
                        init_dict_in_dict(current_dict, stat_name)
                        current_dict[stat_name] = stat

        if stat_name in discrete_stats:
            performance_measure_type = "discrete"
        elif stat_name in continuouse_stats:
            performance_measure_type = "continuouse"
        else:
            raise ValueError("Invalid performance measure type.")

        if by == "task":
            summary_decodings_by_task_array = {}
            task_names = list(summary_decodings_by_iterations.keys())
            for task_num, (task_name, iteration_datas) in enumerate(
                summary_decodings_by_iterations.items()
            ):
                sorted_iteraion_datas = sort_dict(iteration_datas)
                for iteration, stat_datas in sorted_iteraion_datas.items():
                    current_dict = init_dict_in_dict(
                        summary_decodings_by_task_array, iteration
                    )
                    for stat_name, stat in stat_datas.items():
                        if stat_name in discrete_stats:
                            add_to_list_in_dict(current_dict, stat_name, stat)
                        elif stat_name in continuouse_stats:
                            init_dict_in_dict(current_dict, stat_name)
                            # create mean and variance list
                            for moment, value in stat.items():
                                add_to_list_in_dict(
                                    current_dict[stat_name], moment, value
                                )

            if performance_measure_type == "discrete":
                Vizualizer.plot_discrete_decoding_statistics_by_task(
                    summary_decodings_by_task_array, xticks=task_names
                )
            elif performance_measure_type == "continuouse":
                Vizualizer.plot_continuous_decoding_statistics_by_task(
                    summary_decodings_by_task_array,
                    xticks=task_names,
                    additional_title=additional_title,
                )

        elif by == "iterations":
            summary_decodings_by_iterations_array = {}
            task_names = list(summary_decodings_by_iterations.keys())
            iteration_values = []
            for task_num, (task_name, iteration_datas) in enumerate(
                summary_decodings_by_iterations.items()
            ):
                sorted_iteraion_datas = sort_dict(iteration_datas)
                iteration_values.append(list(sorted_iteraion_datas.keys()))
                current_dict = init_dict_in_dict(
                    summary_decodings_by_iterations_array, task_name
                )
                for iteration, stat_datas in sorted_iteraion_datas.items():
                    for stat_name, stat in stat_datas.items():
                        if stat_name in discrete_stats:
                            add_to_list_in_dict(current_dict, stat_name, stat)
                        elif stat_name in continuouse_stats:
                            init_dict_in_dict(current_dict, stat_name)
                            # create mean and variance list
                            for moment, value in stat.items():
                                add_to_list_in_dict(
                                    current_dict[stat_name], moment, value
                                )

            if performance_measure_type == "discrete":
                Vizualizer.plot_discrete_decoding_statistics_by_training_iterations(
                    summary_decodings_by_iterations_array, xticks=iteration_values
                )
            elif performance_measure_type == "continuouse":
                Vizualizer.plot_continuous_decoding_statistics_by_training_iterations(
                    summary_decodings_by_iterations_array,
                    xticks=iteration_values,
                    additional_title=additional_title,
                )

    @staticmethod
    def plot_discrete_decoding_statistics_by_task(
        decodings, xticks=None, min_max_roc_auc=(0, 1)
    ):
        raise NotImplementedError

    @staticmethod
    def plot_continuous_decoding_statistics_by_task(
        decodings,
        xticks=None,
        # min_max_r2=(-1, 1),
        # min_max_rmse=(0, 1),
        additional_title="",
    ):
        # Plot decodings
        c_dec = len(decodings)
        # discrete colormap (nipy_spectral) and discrete
        colormap = plt.get_cmap("tab10")
        fig, axes = plt.subplots(len(decodings), 2, figsize=(15, 3 * c_dec))
        fig.suptitle(
            f"Decoding statistics for different tasks {additional_title}", fontsize=20
        )
        # ........... improve this function
        for dec_num, (iter, task_data) in enumerate(decodings.items()):
            for i, (eval_name, eval_stat) in enumerate(task_data.items()):
                if "values" in eval_stat.keys():
                    values = eval_stat["values"]
                else:
                    values = eval_stat["mean"]

                var = None
                if "var" in eval_stat.keys():
                    var = eval_stat["var"]
                elif "variance" in eval_stat.keys():
                    var = eval_stat["variance"]

                axes[dec_num, i].plot(
                    xticks,
                    values,
                    # label=f"{animal_id}",
                    # color=colormap(animal_num),
                )
                axes[dec_num, i].set_title(
                    f"{eval_name} score for tasks with {iter} iterations"
                )
                axes[dec_num, i].set_ylabel(eval_name)
                min_y = -1 if eval_name == "r2" else 0
                max_y = 1 if eval_name == "r2" else None
                axes[dec_num, i].set_ylim(min_y, max_y)
                # set xticks
                xtick_pos = np.arange(len(xticks))
                axes[dec_num, i].legend()
                if dec_num == len(decodings) - 1:
                    axes[dec_num, i].set_xlabel("Task")
                    axes[dec_num, i].set_xticks(xtick_pos, xticks)
                else:
                    axes[dec_num, i].set_xticks(xtick_pos, [])

                if var is not None:
                    axes[dec_num, i].errorbar(
                        xticks,
                        values,
                        # yerr=animal_eval_stat_var,
                        # color=colormap(animal_num),
                        alpha=0.5,
                        fmt="o",
                        capsize=5,
                    )
                    # label=f"{task_name}")#, capsize=5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_discrete_decoding_statistics_by_training_iterations(decodings):
        # plot accuracy, precision, recall, f1-score
        classes = ["center", "corner", "boarder"]
        max_iter_count = len(list(decodings.keys()))

        for i, (animal_id, task_data) in enumerate(decodings.items()):
            task_num = len(task_data)
            fig, axes = plt.subplots(
                task_num, max_iter_count, figsize=(5 * max_iter_count, 5 * task_num)
            )
            fig.suptitle(
                f"ROC curves for spatial zones for different tasks with different iterations for {animal_id}",
                fontsize=40,
            )
            for task_num, (task_name, eval_stat) in enumerate(task_data.items()):
                eval_stat = decoding_data["roc_auc"]
                for iter_eval_num, roc_auc_dict in enumerate(eval_stat):
                    for class_num, values in roc_auc_dict.items():
                        max_iter_value = list(decodings.keys())[iter_eval_num]
                        max_iter_pos = max_iter_count - iter_eval_num - 1
                        ax = axes[task_num, max_iter_pos]
                        fpr, tpr, auc = values.values()
                        ax.plot(
                            fpr,
                            tpr,
                            label=f"{classes[class_num].capitalize()} AUC {auc:.2f}",
                        )

                        ax.set_title(
                            f"{task_name}: iter. {max_iter_value}", fontsize=20
                        )
                        if iter_eval_num == 0:
                            ax.set_ylabel("TPR")
                        if task_num == len(task_data) - 1:
                            ax.set_xlabel("FPR")
                        ax.legend(loc="lower right")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_continuous_decoding_statistics_by_training_iterations(
        decodings,
        xticks=None,
        additional_title="",
        cmap="tab10",
        figsize=(15, 8),
    ):
        # Create a color map for tasks
        colormap = plt.get_cmap(cmap)
        xticks = np.array(xticks)
        if xticks.ndim < 2:
            xticks = np.array([xticks] * len(decodings))
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        for task_num, (task_name, decoding_data) in enumerate(decodings.items()):
            for i, (eval_name, eval_stat) in enumerate(decoding_data.items()):
                if "values" in eval_stat.keys():
                    values = eval_stat["values"]
                else:
                    values = eval_stat["mean"]

                var = None
                if "var" in eval_stat.keys():
                    var = eval_stat["var"]
                elif "variance" in eval_stat.keys():
                    var = eval_stat["variance"]

                axes[i].plot(
                    xticks[i], values, label=f"{task_name}", color=colormap(task_num)
                )
                if var:
                    axes[i].errorbar(
                        xticks[i],
                        values,
                        yerr=var,
                        capsize=5,
                        fmt=".",
                        color=colormap(task_num),
                        alpha=0.4,
                    )

                axes[i].set_title(
                    f"{eval_name.upper()} score for different tasks with different iterations {additional_title}"
                )
                axes[i].set_ylabel(eval_name)
                axes[i].legend()
                axes[i].set_xlabel("Iterations")
                axes[i].set_xticks(xticks[i], xticks[i], rotation=45, fontsize=8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cell_activites_heatmap(
        rate_map,
        additional_title=None,
        norm_rate=False,
        sorting_indices=None,
        xlabel="location (cm)",
        ylabel="Cell id",
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
        plot_rate_map = rate_map if not norm_rate else normalize_01(rate_map)

        sorted_rate_map, indices = sort_arr_by(
            plot_rate_map, axis=1, sorting_indices=sorting_indices
        )

        plt.figure()
        plt.imshow(sorted_rate_map, aspect="auto")  # , interpolation="None")

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
            fig = plotly.express.histogram(
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
            percentage = (len(zscores[0]) / (len(zscores[0]) + len(zscores[1]))) * 100
            data_labels = [
                f"{percentage:.0f}% ({len(zscores[0])}) cells > {zscore_threshold}",
                f" {len(zscores[1])} cells < {zscore_threshold}",
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
        zscores=None,
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
            split_array_by_zscore(si_rate, zscores, threshold=zscore_threshold)
            if zscores is not None
            else si_rate
        )
        if data_labels is None and len(data) == 2:
            percentage = (len(data[0]) / (len(data[0]) + len(data[1]))) * 100
            data_labels = [
                f" {percentage:.0f}% ({len(data[0])}) <= 5% p",
                f"{len(data[1])} > 5% p",
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

        fig = plotly.graph_object.Figure(
            data=[
                plotly.graph_object.Sankey(
                    node={"label": label_list},
                    link={"source": source, "target": target, "value": count},
                )
            ]
        )

        fig.show()

    @staticmethod
    def plot_multi_task_cell_activity_pos_by_time(
        task_cell_activity,
        figsize_x=20,
        norm=False,
        smooth=False,
        window_size=5,
        additional_title=None,
        savepath=None,
        lines_per_y=1,
        use_discrete_colors=False,
        cmap="inferno",
        show=True,
    ):
        """
        Plots the activity of cells across multiple tasks, with each task's activity plotted in separate subplots.
        Cell plots are normalized and smoothed. Top subplot shows the average activity across all laps, and the bottom subplot shows the activity of each lap.

        Parameters
        ----------
        task_cell_activity : dict
            Dictionary where keys are task identifiers and values are dictionaries with keys "lap_activity" and "additional_title".
            "lap_activity" is a numpy array of cell activities, and "additional_title" is a string to be added to the subplot title.
        figsize_x : int, optional
            Width of the figure in inches (default is 20).
        norm : bool, optional
            Whether to normalize the traces (default is False).
        smooth : bool, optional
            Whether to smooth the traces (default is False).
        window_size : int, optional
            Window size for smoothing (default is 5).
        additional_title : str, optional
            Additional title to be added to the main title (default is None).
        savepath : str, optional
            Path to save the plot (default is None).
        lines_per_y : int, optional
            Number of lines per y-axis unit (default is 1).
        use_discrete_colors : bool, optional
            Whether to use discrete colors for the traces (default is False).
        cmap : str, optional
            Colormap to use for the traces (default is "inferno").
        show : bool, optional
            Whether to show the plot (default is True).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        # create 2 subplots
        fig, axes = plt.subplots(
            2, len(task_cell_activity), gridspec_kw={"height_ratios": [1, 10]}
        )  # Set relative heights of subplots

        for task_num, (task, activity_and_title) in enumerate(
            task_cell_activity.items()
        ):
            traces = activity_and_title["lap_activity"]
            additional_cell_title = activity_and_title["additional_title"]

            sum_traces = np.array([np.sum(traces, axis=0)])
            if "label" in activity_and_title.keys():
                label = make_list_ifnot(activity_and_title["label"])
            else:
                label = None

            if axes.ndim == 1:
                axes = axes.reshape(1, -1)

            axes[0, task_num] = Vizualizer.traces_subplot(
                axes[0, task_num],
                sum_traces,
                labels=label,
                norm=norm,
                smooth=smooth,
                window_size=window_size,
                lines_per_y=1.1,
                xlabel="",
                yticks=None,
                additional_title=f"avg. {additional_cell_title}",
                ylabel="",
                figsize_x=figsize_x,
                use_discrete_colors=use_discrete_colors,
                cmap=cmap,
            )

            norm_traces = normalize_01(traces) if norm else traces
            norm_traces = np.nan_to_num(norm_traces)

            axes[1, task_num] = Vizualizer.traces_subplot(
                axes[1, task_num],
                norm_traces,
                labels=None,
                norm=False,
                smooth=smooth,
                window_size=window_size,
                lines_per_y=lines_per_y,
                additional_title=additional_cell_title,
                ylabel="lap",
                figsize_x=figsize_x,
                use_discrete_colors=use_discrete_colors,
                cmap=cmap,
            )

        title = "Cell Activity"
        if additional_title:
            title += f" {additional_title}"

        fig.subplots_adjust(hspace=0.08, top=0.93)  # Decrease gap between subplots
        fig.suptitle(title, fontsize=17)

        if savepath:
            plt.savefig(savepath)
        if show:
            plt.show()
            plt.close()

        return fig

    @staticmethod
    def plot_single_cell_activity(
        traces,
        figsize_x=20,
        labels=None,
        norm=False,
        smooth=False,
        window_size=5,
        additional_title=None,
        savepath=None,
        lines_per_y=1,
        use_discrete_colors=False,
        cmap="inferno",
        show=True,
    ):
        """
        Plots the activity of a single cell. Top subplot shows the average activity across all laps, and the bottom subplot shows the activity of each lap.

        Parameters
        ----------
        traces : np.ndarray
            Array of cell activity traces.
        figsize_x : int, optional
            Width of the figure in inches (default is 20).
        labels : list, optional
            Labels for the traces (default is None).
        norm : bool, optional
            Whether to normalize the traces (default is False).
        smooth : bool, optional
            Whether to smooth the traces (default is False).
        window_size : int, optional
            Window size for smoothing (default is 5).
        additional_title : str, optional
            Additional title to be added to the main title (default is None).
        savepath : str, optional
            Path to save the plot (default is None).
        lines_per_y : int, optional
            Number of lines per y-axis unit (default is 1).
        use_discrete_colors : bool, optional
            Whether to use discrete colors for the traces (default is False).
        cmap : str, optional
            Colormap to use for the traces (default is "inferno").
        show : bool, optional
            Whether to show the plot (default is True).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        # create 2 subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 10]}
        )  # Set relative heights of subplots
        fig.subplots_adjust(hspace=0.08)  # Decrease gap between subplots

        sum_traces = np.array([np.sum(traces, axis=0)])

        if labels:
            labels = make_list_ifnot(labels)

        ax1 = Vizualizer.traces_subplot(
            ax1,
            sum_traces,
            labels=labels,
            norm=norm,
            smooth=smooth,
            window_size=window_size,
            lines_per_y=1.1,
            xlabel="",
            yticks=None,
            additional_title=f"avg. {additional_title}",
            ylabel="",
            figsize_x=figsize_x,
            use_discrete_colors=use_discrete_colors,
            cmap=cmap,
        )

        norm_traces = normalize_01(traces) if norm else traces
        norm_traces = np.nan_to_num(norm_traces)

        ax2 = Vizualizer.traces_subplot(
            ax2,
            norm_traces,
            labels=None,
            norm=False,
            smooth=smooth,
            window_size=window_size,
            lines_per_y=lines_per_y,
            additional_title=additional_title,
            ylabel="lap",
            figsize_x=figsize_x,
            use_discrete_colors=use_discrete_colors,
            cmap=cmap,
        )
        if savepath:
            plt.savefig(savepath)
        if show:
            plt.show()
            plt.close()

        return fig

    @staticmethod
    def plot_traces_shifted(
        traces,
        figsize_x=20,
        labels=None,
        norm=False,
        smooth=False,
        window_size=5,
        additional_title=None,
        savepath=None,
        lines_per_y=1,
        use_discrete_colors=True,
        cmap="inferno",  # gray, magma, plasma, viridis
    ):
        """
        Plots traces shifted up by a fixed amount for each trace.

        Parameters
        ----------
        traces : np.ndarray
            Array of traces to plot.
        figsize_x : int, optional
            Width of the figure in inches (default is 20).
        labels : list, optional
            Labels for the traces (default is None).
        norm : bool, optional
            Whether to normalize the traces (default is False).
        smooth : bool, optional
            Whether to smooth the traces (default is False).
        window_size : int, optional
            Window size for smoothing (default is 5).
        additional_title : str, optional
            Additional title to be added to the main title (default is None).
        savepath : str, optional
            Path to save the plot (default is None).
        lines_per_y : int, optional
            Number of lines per y-axis unit (default is 1).
        use_discrete_colors : bool, optional
            Whether to use discrete colors for the traces (default is True).
        cmap : str, optional
            Colormap to use for the traces (default is "inferno").
            good colormaps for black background: gray, inferno, magma, plasma, viridis
            white colormaps for black background: binary, blues

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
        fig, ax = plt.subplots()
        ax = Vizualizer.traces_subplot(
            ax,
            traces,
            labels=labels,
            norm=norm,
            smooth=smooth,
            window_size=window_size,
            lines_per_y=lines_per_y,
            additional_title=additional_title,
            figsize_x=figsize_x,
            use_discrete_colors=use_discrete_colors,
            cmap=cmap,
        )
        if savepath:
            plt.savefig(savepath)
        plt.show()
        plt.close()

        return ax

    # Subplots
    def traces_subplot(
        ax,
        traces,
        additional_title=None,
        color=None,
        labels=None,
        norm=False,
        smooth=False,
        window_size=5,
        lines_per_y=1,
        plot_legend=True,
        yticks="default",
        xlabel="Bins",
        ylabel="Cell",
        figsize_x=20,
        figsize_y=None,
        use_discrete_colors=True,
        cmap="inferno",
    ):
        """
        Plots traces on a given axis with options for normalization, smoothing, and color mapping.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to plot on.
        traces : np.ndarray
            Array of traces to plot.
        additional_title : str, optional
            Additional title to be added to the subplot title (default is None).
        color : str or None, optional
            Color for the traces (default is None).
        labels : list, optional
            Labels for the traces (default is None).
        norm : bool, optional
            Whether to normalize the traces (default is False).
        smooth : bool, optional
            Whether to smooth the traces (default is False).
        window_size : int, optional
            Window size for smoothing (default is 5).
        lines_per_y : int, optional
            Number of lines per y-axis unit (default is 1).
        plot_legend : bool, optional
            Whether to plot the legend (default is True).
        yticks : str or list, optional
            Y-axis tick labels (default is "default").
        xlabel : str, optional
            Label for the x-axis (default is "Bins").
        ylabel : str, optional
            Label for the y-axis (default is "Cell").
        figsize_x : int, optional
            Width of the figure in inches (default is 20).
        figsize_y : int or None, optional
            Height of the figure in inches (default is None).
        use_discrete_colors : bool, optional
            Whether to use discrete colors for the traces (default is True).
        cmap : str, optional
            Colormap to use for the traces (default is "inferno").
            good colormaps for black background: gray, inferno, magma, plasma, viridis
            white colormaps for black background: binary, blues

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the plotted traces.
        """
        if smooth:
            traces = smooth_array(traces, window_size=window_size, axis=1)
        if norm:
            traces = normalize_01(traces)
            traces = np.nan_to_num(traces)

        if labels is None:
            labels = [None] * len(traces)
            plot_legend = False
        else:
            labels = [
                f"{label:.3f}" if not isinstance(label, str) else label
                for label in labels
            ]

        shift_scale = 0.1 / lines_per_y
        linecolor = color or None

        min_value, max_value = np.min(traces) - np.max(traces), np.max(traces)
        for i, (trace, label) in enumerate(zip(traces, labels)):
            # min_value, max_value = np.min(trace) - np.max(trace) / 3, np.max(trace)
            upshift = i / lines_per_y + shift_scale * i
            shifted_trace = trace / lines_per_y + upshift

            if use_discrete_colors:
                ax.plot(shifted_trace, color=linecolor, label=label)
            else:
                # Create line segments
                points = np.array(
                    [np.arange(len(shifted_trace)), shifted_trace]
                ).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Calculate alpha for each segment
                alphas = [
                    Vizualizer.calculate_alpha(value, min_value, max_value)
                    for value in trace
                ]

                # Create a LineCollection
                lc = LineCollection(segments, cmap=cmap, label=label)  # , alpha=alphas)
                lc.set_array(np.array(alphas))
                ax.add_collection(lc)

            ax.axhline(
                y=upshift, color="grey", linestyle="--", alpha=0.2
            )  # Adding grey dashed line

        if not figsize_y:
            figsize_y = int(len(traces) / lines_per_y)
            # y_size /= 2

        title = "Activity"
        if additional_title:
            title += f" {additional_title}"

        if yticks == "default":
            yticks = range(traces.shape[0])
            ytick_pos = [
                i / lines_per_y + shift_scale * i for i, tick in enumerate(yticks)
            ]
            ax.set_yticks(ytick_pos, yticks)
            ax.set_ylim(-shift_scale, np.max(ytick_pos) + 1 / lines_per_y)
        else:
            ax.set_ylim(-shift_scale, None)

        ax.set_title(title)
        ax.set_xlim(0, traces.shape[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.figure.set_size_inches(figsize_x, figsize_y)  # Set figure size
        if plot_legend:
            # plt.legend(loc="upper right")
            ax.legend(bbox_to_anchor=(1, 1))
        return ax

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

    @staticmethod
    def heatmap_subplot(
        matrix,
        title,
        ax,
        title_size=10,
        sort=False,
        cmap="viridis",
        interpolation="none",
        xlabel="Cell ID",
        ylabel="Cell ID",
        vmin=None,
        vmax=None,
        ylim=None,
        xlim=None,
        xticks=None,
        xticks_pos=None,
        xtick_size=10,
        yticks=None,
        yticks_pos=None,
        ytick_size=10,
        rotation=0,
    ):
        if sort:
            # Assuming correlations is your correlation matrix as a NumPy array
            # Convert it to a Pandas DataFrame
            correlations_df = pd.DataFrame(matrix)
            # sort the correlation matrix
            matrix = correlations_df.sort_values(by=0, axis=1, ascending=False)

        # Creating a heatmap with sort correlations
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

        if xticks is not None:
            ax.set_xticklabels(xticks, rotation=rotation)
            if xticks_pos is not None:
                ax.set_xticks(xticks_pos)
        if yticks is not None:
            ax.set_yticklabels(yticks)
            if yticks_pos is not None:
                ax.set_yticks(yticks_pos)

        # Set tick sizes
        ax.tick_params(axis="x", which="major", labelsize=xtick_size)
        ax.tick_params(axis="y", which="major", labelsize=ytick_size)

        matrix = np.atleast_2d(matrix)
        cax = ax.imshow(
            matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation
        )
        return cax

    @staticmethod
    def plot_heatmap(
        data,
        title="Heatmap",
        figsize=(6, 5),
        xlabel="Bin (x, y)",
        ylabel="Bin (x, y)",
        no_diag: bool = False,
        xticks=None,
        xticks_pos=None,
        yticks=None,
        yticks_pos=None,
        rotation=45,
        colorbar=True,
        cmap="viridis",
        save_path=None,
        interpolation="none",
        show=True,
    ):
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Remove diagonal elements

        data_no_diag = data - np.eye(data.shape[0]) if no_diag else data

        # Plot heatmap
        cax = Vizualizer.heatmap_subplot(
            data_no_diag,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
            xticks_pos=xticks_pos,
            yticks=yticks,
            yticks_pos=yticks_pos,
            rotation=rotation,
            cmap=cmap,
            ax=ax,
            interpolation=interpolation,
        )

        if colorbar:
            fig.colorbar(cax, ax=ax)

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()

    @staticmethod
    def plot_group_distr_similarities(
        similarities: dict,  # {metric: {group_name: np.array}}
        bins,
        skip=[],
        supxlabel="Bin X",
        supylabel="Bin Y",
        figsize=(4, 3),
        tick_steps=3,
        additional_title="",
        colorbar=False,
        cmap="GnBu",
    ):
        if np.array(bins).ndim == 1:
            ticks = bins
            max_bins = len(bins)
        elif bins.ndim == 2:
            ticks = [f"{x}, {y}" for x, y in bins]
            max_bins = np.max(bins, axis=0) + 1
        tick_positions = np.arange(len(bins))

        max_value = {}
        for name, group_similarities in similarities.items():
            if name not in max_value.keys():
                max_value[name] = 0
            for dists in group_similarities.values():
                max_value[name] = np.max([max_value[name], np.max(dists)])

        for name, group_similarities in similarities.items():
            if name in skip:
                continue
            # plot with all groups of a distance metric into a single plot with multiple heatmaps
            title = f"{str(name).capitalize()} Distances{additional_title}"
            if name == "cosine":
                title = title.replace("Distances", "Similarities")
            if name == "overlap":
                title = title.replace("Distances", "")
            if isinstance(max_bins, int) or max_bins.ndim == 0:
                fig, axes = plt.subplots(
                    max_bins,
                    1,
                    figsize=(figsize[0], figsize[1]),
                )
                fig.suptitle(title, fontsize=max_bins * max_bins, y=1.01)
                fig.supxlabel(supxlabel, fontsize=max_bins * max_bins, x=0.5, y=-0.03)
                fig.align_xlabels()
                fig.supylabel(supylabel, fontsize=max_bins * max_bins, x=-0.02, y=0.5)
            elif max_bins.ndim == 1:
                fig, axes = plt.subplots(
                    max_bins[0],
                    max_bins[1],
                    figsize=(max_bins[0] * figsize[0], max_bins[1] * figsize[1]),
                )
                fig.suptitle(title, fontsize=max_bins[0] * max_bins[1] / 2, y=1.01)
                fig.supxlabel(
                    supxlabel, fontsize=max_bins[0] * max_bins[1] / 2, x=0.5, y=-0.03
                )
                fig.align_xlabels()
                fig.supylabel(
                    supylabel, fontsize=max_bins[0] * max_bins[1] / 2, x=-0.02, y=0.5
                )
            fig.tight_layout()

            for group_i, (group_name, dists) in enumerate(group_similarities.items()):
                vmin = None
                vmax = None
                if name in ["cosine", "overlap"]:
                    vmax = 1
                    cmap = "viridis"
                else:
                    if "_r" not in cmap:
                        cmap = f"{cmap}_r"
                if name in [
                    "euclidean",
                    "wasserstein",
                    "kolmogorov-smirnov",
                    "chi2",
                    "kullback-leibler",
                    "jensen-shannon",
                    "energy",
                    "mahalanobis",
                    "overlap",
                ]:
                    vmin = 0
                elif name in ["correlation", "cosine"]:
                    vmin = -1
                    vmax = 1

                if vmax is None:
                    vmax = max_value[name]
                # ax = axes[max_bins[1]-1-j, max_bins[0]-1-i]
                subplot_xticks = []
                subplot_xticks_pos = []
                subplot_yticks = []
                subplot_yticks_pos = []
                if isinstance(group_name, str) or np.array(group_name).ndim == 0:
                    i = group_i
                    ax = axes[i]
                    subplot_title_size = max_bins * 1.7
                    if i == max_bins - 1:
                        subplot_xticks = ticks[::tick_steps]
                        subplot_xticks_pos = tick_positions[::tick_steps]
                elif np.array(group_name).ndim == 1:
                    i, j = group_name
                    ax = axes[i, j]
                    subplot_title_size = max_bins[0] * 1.7
                    if i == max_bins[0] - 1:
                        subplot_xticks = ticks[::tick_steps]
                        subplot_xticks_pos = tick_positions[::tick_steps]
                    if j == 0:
                        subplot_yticks = ticks[::tick_steps]
                        subplot_yticks_pos = tick_positions[::tick_steps]

                cax = Vizualizer.heatmap_subplot(
                    dists,
                    title=f"{group_name}",
                    title_size=subplot_title_size,
                    xlabel="",
                    ylabel="",
                    vmin=vmin,
                    vmax=vmax,
                    xticks=subplot_xticks,
                    xticks_pos=subplot_xticks_pos,
                    yticks=subplot_yticks[::tick_steps],
                    yticks_pos=subplot_yticks_pos[::tick_steps],
                    ax=ax,
                    cmap=cmap,
                    interpolation="none",
                )

                if colorbar:
                    # set colorbar range
                    fig.colorbar(cax, ax=ax)
            plt.show()

    def plot_1d_iter_group_distr_similarities(
        similarities: dict,  # {metric: {group_name: np.array}}
        bins,
        skip=[],
        supxlabel="Bin X",
        supylabel="Bin Y",
        figsize=(3, 3),
        tick_steps=3,
        additional_title="",
        colorbar=False,
        cmap="viridis",
    ):

        # plot with all groups of a distance metric into a single plot with multiple heatmaps
        num_x_plots = len(similarities)
        num_y_plots = len(similarities[list(similarities.keys())[0]])
        figsize = (figsize[0] * num_y_plots, figsize[1] * num_x_plots)
        fig, axes = plt.subplots(
            num_y_plots,
            num_x_plots,
            figsize=(figsize[0], figsize[1]),
        )
        suptitle = (
            f"Similarity Measure comparisson using spatial zones{additional_title}"
        )
        fig.suptitle(suptitle, fontsize=figsize[0], y=0.98)
        fig.supxlabel(supxlabel, fontsize=figsize[0] / 2, x=0.5, y=-0.03)
        fig.align_xlabels()
        fig.supylabel(supylabel, fontsize=figsize[0] / 2, x=-0.02, y=0.5)

        max_value = {}
        for iter_num, (iter, metric_similarities) in enumerate(similarities.items()):
            for name, group_similarities in metric_similarities.items():
                if name not in max_value.keys():
                    max_value[name] = 0
                for dists in group_similarities.values():
                    max_value[name] = np.max([max_value[name], np.max(dists)])

        for iter_num, (iter, metric_similarities) in enumerate(similarities.items()):
            for group_num, (name, group_similarities) in enumerate(
                metric_similarities.items()
            ):
                if name in skip:
                    continue

                if np.array(bins).ndim == 1:
                    ticks = bins
                    max_bins = len(bins)
                elif bins.ndim == 2:
                    ticks = [f"{x}, {y}" for x, y in bins]
                    max_bins = np.max(bins, axis=0) + 1
                tick_positions = np.arange(len(bins))

                subplot_title = f"{str(name).capitalize()} {iter}"

                similarity_matrix = np.zeros((max_bins, max_bins))
                for group_i, (group_name, dists) in enumerate(
                    group_similarities.items()
                ):
                    similarity_matrix[group_i] = dists

                vmin = None
                vmax = None
                if name in ["cosine", "overlap"]:
                    vmax = 1
                    cmap = "viridis"
                else:
                    if "_r" not in cmap:
                        cmap = f"{cmap}_r"
                if name in [
                    "euclidean",
                    "wasserstein",
                    "kolmogorov-smirnov",
                    "chi2",
                    "kullback-leibler",
                    "jensen-shannon",
                    "energy",
                    "mahalanobis",
                    "overlap",
                ]:
                    vmin = 0
                elif name in ["correlation", "cosine"]:
                    vmin = -1
                    vmax = 1

                if vmax is None:
                    vmax = max_value[name]

                # ax = axes[max_bins[1]-1-j, max_bins[0]-1-i]
                subplot_xticks = []
                subplot_xticks_pos = []
                subplot_yticks = []
                subplot_yticks_pos = []
                xplot_num = len(similarities) - 1 - iter_num
                ax = axes[group_num, xplot_num]
                subplot_title_size = 20
                tick_size = 20
                if True:  # group_num == len(metric_similarities) - 1:
                    subplot_xticks = ticks[::tick_steps]
                    subplot_xticks_pos = tick_positions[::tick_steps]
                if xplot_num == 0:
                    subplot_yticks = ticks[::tick_steps]
                    subplot_yticks_pos = tick_positions[::tick_steps]

                cax = Vizualizer.heatmap_subplot(
                    similarity_matrix,
                    title=f"{subplot_title}",
                    title_size=subplot_title_size,
                    xlabel="",
                    ylabel="",
                    vmin=vmin,
                    vmax=vmax,
                    xticks=subplot_xticks,
                    xticks_pos=subplot_xticks_pos,
                    xtick_size=tick_size,
                    yticks=subplot_yticks,
                    yticks_pos=subplot_yticks_pos,
                    ytick_size=tick_size,
                    ax=ax,
                    cmap=cmap,
                    interpolation="none",
                )

                if colorbar and xplot_num == len(similarities) - 1:
                    # move position of colorbar
                    # fig.subplots_adjust(right=0.8)
                    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    # set colorbar range
                    fig.colorbar(cax, ax=ax)

        fig.tight_layout()
        plt.show()
