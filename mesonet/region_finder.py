import argparse
import csv
import os
import pickle
from typing import Dict, Set, Tuple


# NOTE: This is no longer needed. Regions are determined while processing the
#  images in a more robust way so that no part of the region is missed.

# TODO: What if the region consists of only contour points?
def find_points_in_regions(start_points_in_regions: Dict[int, Tuple[int, int]],
                           contour_points: Set[Tuple[int, int]],
                           width: int,
                           height: int):
    """
    Finds the pixels that are contained within each region.

    :param start_points_in_regions: A dictionary whose key is the region number
        and value is the point within that region. This function expects one
        point per region.
    :param contour_points: A set of all contour points for the atlas. This
        function expects that there is not overlapping of contour points for
        different regions.
    """
    seen_points = set()
    points_in_regions = {}
    points_to_explore = []

    for region, point in start_points_in_regions.items():
        points_to_explore.append(point)

        while len(points_to_explore) > 0:
            x, y = points_to_explore.pop()

            if (
                ((x, y) in seen_points) or \
                (x > width or x < 0) or \
                (y > height or y < 0)
            ):
                continue

            points_in_regions[(x, y)] = region
            seen_points.add((x, y))

            if (x, y) in contour_points:
                continue
            
            points_to_explore.append((x + 1, y))
            points_to_explore.append((x - 1, y))
            points_to_explore.append((x, y + 1))
            points_to_explore.append((x, y - 1))

    return points_in_regions


def get_start_point_for_each_region(
    region_labels_file: str
) -> Dict[int, Tuple[int, int]]:
    """
    The start point for each region is the bottom-left corner of the label in
    that region.
    """
    points_in_regions = {}

    with open(region_labels_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Discard CSV header.

        for row in reader:
            region, x, y = int(row[1]), int(row[2]), int(row[3])
            points_in_regions[region] = (x, y)

    return points_in_regions


def get_points_on_contours(contour_points_file: str) -> Set[Tuple[int, int]]:
    contour_points = set()

    with open(contour_points_file, "rb") as f:
        contour_points_list = pickle.load(f)  # A list.

    for region_contour in contour_points_list:
        # region_contour is an np.array with size [n_contour_points, 1, 2] where
        # the last dimension is the x, y coordinate position of the point in the
        # contour of the region.
        for point in region_contour:
            contour_points.add((point[0][0], point[0][1]))

    return contour_points


def main(args):
    contour_points = get_points_on_contours(args.contour_points_file)

    start_points_in_regions = \
        get_start_point_for_each_region(args.region_labels_file)

    points_in_regions = find_points_in_regions(start_points_in_regions,
                                               contour_points,
                                               args.image_width,
                                               args.image_height)

    with open(os.path.join(args.save_dir, "region_points.pkl"), "wb") as f:
        pickle.dump(points_in_regions, f)


if __name__ == "__main__":
    """
    python mesonet/region_finder.py \
        --contour-points-file ./mesonet_outputs/pipeline_brain_atlas/dlc_output/contour_points.pkl \
        --region-labels-file ./mesonet_outputs/pipeline_brain_atlas/output_overlay/region_labels/0_region_labels.csv \
        --save-dir ./mesonet_outputs/pipeline_brain_atlas/output_overlay/region_labels/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--contour-points-file", type=str, required=True)
    parser.add_argument("--region-labels-file", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    args = parser.parse_args()

    main(args)
