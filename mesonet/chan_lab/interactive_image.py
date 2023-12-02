import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from mesonet.chan_lab.helpers.utils import config_to_namespace

REGION_POINTS_WIDTH = 512
REGION_POINTS_HEIGHT = 512


class InteractiveImageManager:
    def __init__(self,
                 region_points_file: str,
                 image_width: int,
                 image_height: int,
                 save_file: str = None):
        self.previous_region = -1
        self.custom_region_label = -1
        self.custom_region = {}
        self.width_scale = REGION_POINTS_WIDTH / image_width
        self.height_scale = REGION_POINTS_HEIGHT / image_height
        self.save_file = save_file

        with open(region_points_file, "rb") as f:
            self.region_points = pickle.load(f)

    def mouse_movement(self, event):
        x, y = event.xdata, event.ydata
        point_expanded = (int(x * self.width_scale), int(y * self.height_scale))

        if event.button:
            self.was_clicked = True
            self.custom_region[point_expanded] = self.custom_region_label

        if point_expanded in self.region_points:
            current_region = self.region_points[point_expanded]
        else:
            current_region = -1

        # if current_region != self.previous_region:
        if current_region >= 0:
            print(f"In region {current_region}")
        else:
            print(f"Not in a region")

        self.previous_region = current_region

    def button_released(self, event):
        if event.button:
            if self.save_file:
                print("Saving custom region")
                with open(args.save_file, 'wb') as f:
                    pickle.dump(self.custom_region, f)
            else:
                print("WARNING: A save file is needed in order to save a "
                      "custom region")


def main(args: argparse.Namespace):
    manager = InteractiveImageManager(args.region_points_file,
                                      args.image_width,
                                      args.image_height,
                                      args.save_file)

    def mouse_movement(event):
        manager.mouse_movement(event)

    def button_released(event):
        manager.button_released(event)

    image = mpimg.imread(args.image_file)
    plt.connect('motion_notify_event', mouse_movement)
    plt.connect('button_press_event', mouse_movement)
    plt.connect('button_release_event', button_released)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_args = config_to_namespace(args.config)

    main(config_args)
