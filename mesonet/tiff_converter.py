import numpy as np
import PIL.Image


# Modified from https://stackoverflow.com/questions/9627652/split-multi-page-tiff-with-python
class MultiImageTiff:
    def __init__(self, filename: str):
        self.image = PIL.Image.open(filename)
        self.image.seek(0)
        self.image_size = (self.image.tag[0x101][0], self.image.tag[0x100][0])
        self.current = self.image.tell()

    def get_frame(self, frame: int) -> np.array:
        try:
            self.image.seek(frame)
        except EOFError:
            return None
        self.current = self.image.tell()
        mapped_image_array = self._map_image_to_255(self.image.getdata())
        return np.reshape(mapped_image_array, self.image_size)

    def __iter__(self):
        self.image.seek(0)
        self.old = self.current
        self.current = self.image.tell()
        return self
    
    def next(self) -> np.array:
        try:
            self.image.seek(self.current)
            self.current = self.image.tell() + 1
        except EOFError:
            self.image.seek(self.old)
            self.current = self.image.tell()
            raise StopIteration
        mapped_image_array = self._map_image_to_255(self.image.getdata())
        return np.reshape(mapped_image_array, self.image_size)
    
    def _map_image_to_255(self, image: PIL.Image) -> np.array:
        image_array = np.array(image)
        image_min = np.amin(image_array)
        image_max = np.amax(image_array)
        mapped_image_array = np.interp(image_array,
                                       [image_min, image_max],
                                       [0, 255])
        return mapped_image_array.astype(np.uint8)


if __name__ == "__main__":
    print("loading")
    t = MultiImageTiff("/Users/christian/Documents/summer2023/matlab/my_data/04_awake_8x8_30hz_36500fr.tif")
    print(t.get_frame(0))
    print(t.get_frame(0).dtype)
    print(t.get_frame(0).shape)
    print("done")

    image_array = t.get_frame(0)
    image = PIL.Image.fromarray(image_array)
    image.save("./mesonet_inputs/pipeline_data/atlas_brain/0.png")
