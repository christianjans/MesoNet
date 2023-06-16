import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


class InteractiveImageManager:
    def __init__(self, region_pixels_file: str, save_file: str = None):
        self.previous_region = -1
        self.custom_region_label = -1
        self.custom_region = {}
        self.save_file = save_file

        with open(region_pixels_file, "rb") as f:
            self.region_pixels = pickle.load(f)

        print(self.region_pixels)

    def mouse_movement(self, event):
        x, y = event.xdata, event.ydata
        point = (int(x), int(y))

        if event.button:
            self.was_clicked = True
            self.custom_region[point] = self.custom_region_label
            print(point)

        if point in self.region_pixels:
            current_region = self.region_pixels[point]
        else:
            current_region = -1

        if current_region != self.previous_region:
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


def main(args):
    manager = InteractiveImageManager(args.region_pixels_file, args.save_file)

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
    """
    python mesonet/interactive_image.py \
        --image-file ./mesonet_outputs/pipeline_brain_atlas/output_overlay/0_overlay.png \
        --region-pixels-file ./region_points.pkl
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--region-pixels-file", type=str, required=True)
    parser.add_argument("--save-file", type=str, default=None)
    args = parser.parse_args()

    main(args)
