# Chan Lab Scripts

Note that the Chan Lab scripts found in [mesonet/chan_lab/](/mesonet/chan_lab/)
have only been tested on macOS 12.5 Monterey with a 14-inch, 2021 M1 Macbook
Pro using the Anaconda installation instructions.

Each script has its own configuration file (found in
[mesonet/chan_lab/configs/](/mesonet/chan_lab/configs/)) that controls how the
script will run.

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

## [`pipelines.py`](/mesonet/chan_lab/pipelines.py)

This script can be used to create the MesoNet segmentations of the images. See
the [pipelines.yaml](/mesonet/chan_lab/configs/pipelines.yaml) configuration
file for an example configuration and more details about what goes into a
configuration.

Run the pipelines script using the following command:

```sh
$ python mesonet/chan_lab/pipelines.py --config mesonet/chan_lab/configs/pipelines.yaml
```

After this is run, the output directory will contain ***
