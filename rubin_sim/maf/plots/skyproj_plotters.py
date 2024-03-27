import matplotlib.pyplot as plt
import skyproj

from .plot_handler import BasePlotter


class SkyprojPlotter(BasePlotter):
    def __init__(self):
        super().__init__()
        # Customize our plotters members for our new plot
        self.plot_type = "Skyproj"
        self.object_plotter = False
        self.default_plot_dict.update(
            {
                "subplot": 111,
                "skyproj": skyproj.MollweideSkyproj,
                "skyproj_kwargs": {"lon_0": 0},
                "kind": "hpxmap",
                "draw_kwargs": {"zoom": False},
            }
        )
        self.initialize_plot_dict({})
        self.figure = None

    def initialize_plot_dict(self, user_plot_dict):
        self.plot_dict = {}
        self.plot_dict.update(self.default_plot_dict)
        self.plot_dict.update(user_plot_dict)

    def prepare_skyproj(self, fignum=None):
        # Exctract elemens of plotDict that need to be passed to plt.figure
        fig_kwargs = {k: v for k, v in self.plot_dict.items() if k in ["figsize"]}

        self.figure = plt.figure(fignum, **fig_kwargs)
        # This instance of Axis will get deleted and replaced when the
        # corresponding SkyPlot class is created, but this instances is
        # needed so SkyPlot will put its instance in the correct relation
        # with its figure.
        ax = self.figure.add_subplot(self.plot_dict["subplot"])
        self.skyproj = self.plot_dict["skyproj"](ax, **self.plot_dict["skyproj_kwargs"])

    def __call__(self, metric_values, slicer, user_plot_dict, fignum=None):
        self.initialize_plot_dict(user_plot_dict)
        self.prepare_skyproj(fignum)

        match self.plot_dict["kind"]:
            case "hpxmap":
                self.skyproj.draw_hpxmap(metric_values, **self.plot_dict["draw_kwargs"])
            case _:
                raise NotImplementedError(f"Plot of kind {self.plot_dict['kind']} is not supported")

        return self.figure.number
