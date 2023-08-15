import argparse
import time
import tkinter as tk
import tkinter.ttk as ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mesonet.chan_lab.helpers.plotting import args_from_yaml, PlotterCollection
from mesonet.chan_lab.helpers.utils import config_to_namespace


class View(ttk.Frame):
    UPDATE_INTERVAL = 0.10  # Minimum interval between plot updates.

    def __init__(self, parent: tk.Tk, args: argparse.Namespace):
        super().__init__(parent)

        self.figure = Figure(figsize=(200, 100), dpi=8)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=2, column=0)

        plotter_args = [
            args_from_yaml(yaml_dict) for yaml_dict in args.plots
        ]
        self.collection = PlotterCollection(self.canvas,
                                            self.figure,
                                            plotter_args,
                                            rows=args.plot_rows)

        self.min_slider_value = self.collection.min_frame_index + 1
        self.max_slider_value = self.collection.max_frame_index + 1
        self.slider_value = self.min_slider_value

        self.slider_label = ttk.Label(self, text="")
        self.slider_label.grid(row=0, column=0)

        self.slider = ttk.Scale(self,
                                from_=self.min_slider_value,
                                to=self.max_slider_value,
                                command=self.slider_changed,
                                value=self.slider_value,
                                length=600)
        self.slider.grid(row=1, column=0, sticky=tk.EW)
        self.grid_rowconfigure(1, weight=1)

        parent.bind("<Left>", lambda _: self.slider.set(self.slider.get() - 1))
        parent.bind("<Right>", lambda _: self.slider.set(self.slider.get() + 1))

        self.update_time = time.time()

        self.slider_changed(str(self.slider_value))

    def slider_changed(self, value: str):
        # Control how fast the plots can be updated.
        if time.time() - self.update_time < View.UPDATE_INTERVAL:
            return
        self.update_time = time.time()

        self.slider_value = int(float(value))

        label_text = (
            f"Frame {self.slider_value}/{self.max_slider_value} "
            f"({self.slider_value / 30.0003:.3f} s)"
        )
        self.slider_label.config(text=label_text)

        self.collection.update(self.slider_value - 1)


class App(tk.Tk):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.title("Event Analyzer App")

        view = View(self, args)
        view.grid(row=0, column=0, padx=10, pady=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_args = config_to_namespace(args.config)

    app = App(config_args)
    app.mainloop()
