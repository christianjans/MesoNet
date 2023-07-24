import tkinter as tk
import tkinter.ttk as ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from activity_analyzer import MasksManager
from helpers.image_series import ImageSeriesCreator
from helpers.video_series import VideoSeries

PUPIL_EVENT_FRAME = 313
MESOSCALE_EVENT_FRAME = 88
PUIPIL_FILENAME = "/Users/christian/Downloads/fc2_save_2023-02-09-135224-0000.avi"
MESOSCALE_FILENAME = "/Users/christian/Documents/summer2023/matlab/my_data/flash1/02_awake_8x8_30hz_36500fr_FR30Hz_BPF1-5Hz_GSR_DFF0-G4-fr1-36480_FR30Hz_BPF1-5Hz_GSR_DFF0-G4-fr1-36480.raw"
PUPIL_START_FRAME_INDEX = PUPIL_EVENT_FRAME - MESOSCALE_EVENT_FRAME + 1 - 1

PUPIL_EVENT_FRAME = 0
MESOSCALE_EVENT_FRAME = 0
PUPIL_FILENAME = ""
MESOSCALE_FILENAME = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_eye-r/imMean.mat"
PUPIL_START_FRAME_INDEX = PUPIL_EVENT_FRAME - MESOSCALE_EVENT_FRAME + 1 - 1


class Controller:
    pass


class View(ttk.Frame):
    def __init__(self,
                 parent: tk.Tk,
                 mesoscale_filename: str,
                 pupil_filename: str):
        super().__init__(parent)

        self.mesoscale_image_series = ImageSeriesCreator.create(
                mesoscale_filename, 256, 256, "all", property="imMean",
                transpose_axes=(2, 0, 1))
        self.pupil_video_series = VideoSeries(pupil_filename)
        self.min_slider_value = 1 + PUPIL_START_FRAME_INDEX
        self.max_slider_value = self.mesoscale_image_series.image_array.shape[0]
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

        self.figure = Figure(figsize=(200, 100), dpi=5)
        self.plot1 = self.figure.add_subplot(1, 2, 1)
        self.plot1.imshow(self.mesoscale_image_series.get_frame(self.slider_value - self.min_slider_value))
        self.plot2 = self.figure.add_subplot(1, 2, 2)
        self.plot2.imshow(self.pupil_video_series.get_frame(self.slider_value - 1))
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=2, column=0)

        self.slider_changed(str(self.slider_value))

    def slider_changed(self, value: str):
        self.slider_value = int(float(value))

        label_text = (
            f"Frame {self.slider_value}/{self.max_slider_value} "
            f"({self.slider_value / self.pupil_video_series.fps:.3f} s)"
        )
        self.slider_label.config(text=label_text)

        self.plot1.clear()
        self.plot1.imshow(
                self.mesoscale_image_series.get_frame(self.slider_value - self.min_slider_value))

        self.plot2.clear()
        self.plot2.imshow(
                self.pupil_video_series.get_frame(self.slider_value - 1))

        self.canvas.draw()


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Event Analyzer App")

        view = View(self, MESOSCALE_FILENAME, PUIPIL_FILENAME)
        view.grid(row=0, column=0, padx=10, pady=10)


if __name__ == "__main__":
    app = App()
    app.mainloop()
