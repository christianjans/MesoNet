# YAML to Py(thon)

This document describes how to encode a Python object responsible for handling
timecourse images or data in YAML.

Each object is encoded as a dictionary of key-value pairs in YAML. Each
dictionary must have a "type" key as well as an "args" key.

The keys in the `args` dictionary are conditioned on the `type` of Python object
being encoded. The available types are: `video`, `image`, and `pupillometry`.

## The `video` type

The `video` handles creating a Python object from a video file. The YAML
dictionary encoding this object is defined as follows:

```yaml
{
  type: "video",
  args: {
    filename: <filename>,
    event_frames: <event_frames>,
    title: <title>
  }
}
```

- `<filename>`: The path to the video file.
- `<event_frames>`: A list of integers corresponding to frames at which a
  noticeable event occurs (e.g. a flash). These events are used for
  synchronization across the different objects. The `<event_frames>` list should
  contain at least two frames and the length of the list should match the number
  of frames in other `<event_frames>` lists of other objects being displayed.
- `<title>`: A string that describes this video object. It is used for display
  purposes only.

**NOTE**: Currently, only `.avi` files are supported.

## The `image` type

The `image` type handles creating a Python object from a file containing a
series of images; or a file that can be interpreted as such, such as a file
containing a 3-dimensional matrix which represent images across time. The latter
would be, for example, the processed mesoscale matrix files. The YAML dictionary
encoding this object is defined as follows:

```yaml
{
  type: "image",
  args: {
    filename: <filename>,
    event_frames: <event_frames>,
    title: <title>,
    image_width: <image_width>,
    image_height: <image_height>,
    kwargs: {},
    region_points: <region_points>
  }
}
```

- `<filename>`: The path to the image file.
- `<event_frames>`: A list of integers corresponding to frames at which a
  noticeable event occurs (e.g. a flash). These events are used for
  synchronization across the different objects. The `<event_frames>` list should
  contain at least two frames and the length of the list should match the number
  of frames in other `<event_frames>` lists of other objects being displayed.
- `<title>`: A string that describes this image object. It is used for display
  purposes only.
- `<image_width>`: The width of each individual image.
- `<image_height>`: The height of each individual image.
- `<region_points>`: The path to the region points file to display on top of the
  images (if the image are mesoscale images). This is an option argument and can
  be set to `null` if the segmentation should not be overlayed.

## The `pupillometry` type

The `pupillometry` type handles creating a Python object from a `.mat` file
containing pupillometry data. The YAML dictionary encoding this object is
defined as follows:

```yaml
{
  type: "pupillometry",
  args: {
    filename: <filename>,
    event_frames: <event_frames>,
    title: <title>
  }
}
```

- `<filename>`: The path to the pupillometry data file.
- `<event_frames>`: A list of integers corresponding to frames at which a
  noticeable event occurs (e.g. a flash). These events are used for
  synchronization across the different objects. The `<event_frames>` list should
  contain at least two frames and the length of the list should match the number
  of frames in other `<event_frames>` lists of other objects being displayed.
- `<title>`: A string that describes this pupillometry object. It is used for
  display purposes only.
