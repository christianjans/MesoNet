# Chan Lab Scripts

Note that the Chan Lab scripts found in [mesonet/chan_lab/](/mesonet/chan_lab/)
have only been tested on macOS 12.5 Monterey with a 14-inch, 2021 M1 Macbook
Pro using the Anaconda installation instructions.

Note also that in order to run these scripts, one may require modifying the
`PYTHONPATH` on their machine. On Linux and macOS, this can be done by executing
the following command:

```sh
$ export PYTHONPATH=${PYTHONPATH}:${PWD}:${PWD}/mesonet:${PWD}/mesonet/chan_lab
```

Each script has its own configuration file (found in
[mesonet/chan_lab/configs/](/mesonet/chan_lab/configs/)) that controls how the
script will run.

The scripts are presented here in the order in which they should be run so that
all dependencies are created for each script before being run.

## [`image_selector.py`](/mesonet/chan_lab/image_selector.py)

This script can be used to select certain images from a TIF image series. The
selected images are converted to PNGs and are typically then used for
segmentation purposes. See the
[image_selector.yaml](/mesonet/chan_lab/configs/image_selector.yaml)
configuration file for an example configuration and more details about what goes
into a configuration.

```sh
$ python mesonet/chan_lab/image_selector.py --config mesonet/chan_lab/configs/image_selector.yaml
```

After this is run, there will be the selected images in the output directory.

For example, the output directory will contain images like:

<img src="/docs/_static/awake1_0.png" alt="selected image" width="200"/>

These images will ultimately be segmented by MesoNet. It's good to select a few
images from the TIF image series as, from qualitative experience, MesoNet does
not always properly segment the cortical regions.

## [`interactive_image.py`](/mesonet/chan_lab/interactive_image.py)

This script is a small program that opens a given image using Python's graphing
library, Matplotlib. The user can hover their mouse over the image to determine
the exact points in the image and their value. See the
[interactive_image.yaml](/mesonet/chan_lab/configs/interactive_image.yaml)
configuration file for an example configuration and more details about what goes
into a configuration.

If the user provides a save directory, the user can click and drag their mouse
on points they want to include in a [region points](#dlc_output) file.

<img src="/docs/_static/interactive_image.png" alt="interactive image" width="512"/>

## [`pipelines.py`](/mesonet/chan_lab/pipelines.py)

This script can be used to create the MesoNet segmentations of the images. See
the [pipelines.yaml](/mesonet/chan_lab/configs/pipelines.yaml) configuration
file for an example configuration and more details about what goes into a
configuration.

Run the pipelines script using the following command:

```sh
$ python mesonet/chan_lab/pipelines.py --config mesonet/chan_lab/configs/pipelines.yaml
```

After this is run, the output directory will contain three directories:
`dlc_output/`, `output_mask/`, and `output_overlay/`. The directories are
described in more detail below.

### `dlc_output/`

The `dlc_output/` directory contains the output of the DeepLabCut inference
which creates the segmentation of the brain image. The file,
`region_points_<num>.pkl` is a serialized Python dictionary describing the
regions of the MesoNet segmentation of the `<num>`th image. The dictionary keys
are a tuple of two integers which represents the x, y coordinate of a pixel in a
512x512 version of the brain image. The value is the MesoNet region number.

To examine a region points file, load it in Python:

```sh
$ python
>>> import pickle
>>> with open("/Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/full3_1.5_atlas_brain/dlc_output/region_points_0.pkl", "rb) as f:
...   region_points = pickle.load(f)
...
>>> region_points
{(346, 324): 0, (347, 324): 0, (348, 324): 0, ... (129, 419): 40, (130, 419): 40, (129, 420): 40}
```

### `output_mask/`

The `output_mask/` directory contains images relating to the masking of the
cortical regions in the image. The mask attempts to only highlight the cortical
parts of the image.

<img src="/docs/_static/awake1_0_mask.png" alt="mask image" width="512"/>

### `output_overlay/`

The `output_overlay/` directory overlays the segmentation on the original image.

<img src="/docs/_static/awake1_0_segmented.png" alt="segmented image" width="512"/>

## [`sensory_map.py`](/mesonet/chan_lab/sensory_map.py)

This script generates an image of the processed file where each pixel is the
maximum value of that pixel over the course of a certain amount of time after a
given frame. The given frame is often the start of some sort of event, such as a
light flash. See the
[sensory_map.yaml](/mesonet/chan_lab/configs/sensory_map.yaml) configuration
file for an example configuration and more details about what goes into a
configuration.

Overlayed on the image is the segmentation of the brain from some region points
file. Each region has a blue dot and a green dot. The green dot is the center of
mass of the region and the blue dot is on the pixel with the largest value in
that region.

Finally, a red `x` marks the pixel with the highest value in the entire image.

<img src="/docs/_static/sensory_map.png" alt="sensory map image" width="512"/>

## [`event_analyzer_app.py`](/mesonet/chan_lab/event_analyzer_app.py)

This script generates a GUI application that the user can interact with to
analyze the videos, processed mesoscale images, and pupillometry data of a
dataset at the same time. See the
[event_analyzer_app.yaml](/mesonet/chan_lab/configs/event_analyzer_app.yaml)
configuration file for an example configuration and more details about what goes
into a configuration.

Use the right and left arrow keys to go forward or backward one frame,
respectively. Note that upon startup, the user must tap the right arrow key a
couple times for the images to show initially.

<img src="/docs/_static/event_analyzer_app.png" alt="sensory map image" width="1024"/>

## [`event_highlighter.py`](/mesonet/chan_lab/event_highlighter.py)

This script displays plots of the pupil (face) video, body video, mesoscale
image series, and pupillometry timecourse data around frames of interest. See
the [event_highlighter.yaml](/mesonet/chan_lab/configs/event_highlighter.yaml)
configuration file for an example configuration and more details about what goes
into a configuration.

<img src="/docs/_static/event_highlighter.png" alt="event highlighter image" width="1024">

## [`activity_analyzer.py`](/mesonet/chan_lab/activity_analyzer.py)

This script creates the timecourses of activity and correlation matrices for the
different segmented regions. Additionally, it can create a seed pixel map around
specific points of the retrosplenial and secondary motor cortices. See the
[actvity_analyzer.yaml](/mesonet/chan_lab/configs/activity_analyzer.yaml)
configuration file for an example configuration and more details about what goes
into a configuration.

<img src="/docs/_static/activity_timecourse.png" alt="activity timecourse image" width="1024">

<img src="/docs/_static/correlation_matrix.png" alt="correlation matrix image" width="1024">

<img src="/docs/_static/seed_pixel_map.png" alt="seed pixel map image" width="1024">
