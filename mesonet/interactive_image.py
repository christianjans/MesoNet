import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


class InteractiveImageManager:
    def __init__(self, region_pixels_file: str):
        self.previous_region = -1

        with open(region_pixels_file, "rb") as f:
            self.region_pixels = pickle.load(f)

    def update(self, event):
        x, y = event.xdata, event.ydata
        point = (int(x), int(y))

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


def main(args):
    manager = InteractiveImageManager(args.region_pixels_file)

    def update(event):
        manager.update(event)

    image = mpimg.imread(args.image_file)
    plt.connect('motion_notify_event', update)
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
    args = parser.parse_args()

    main(args)
