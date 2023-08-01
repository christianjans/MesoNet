import argparse
import os
import sys

import mesonet

MESONET_PATH = os.getcwd()

os.environ["MESONET_GIT"] = MESONET_PATH
sys.path.insert(0, os.environ["MESONET_GIT"])
sys.path.insert(0, os.path.join(os.environ["MESONET_GIT"], "mesonet"))

DLC_CONFIG = os.path.join(
        MESONET_PATH, "mesonet/dlc/atlas-DongshengXiao-2020-08-03/config.yaml")
MODEL_FILE = os.path.join(
        MESONET_PATH, "mesonet/models/DongshengXiao_brain_bundary.hdf5")
UNET_ONLY_MODEL_FILE = os.path.join(
        MESONET_PATH, "mesonet/models/DongshengXiao_unet_motif_based_functional_atlas.hdf5")
VOXELMORPH_MODEL_FILE = os.path.join(
        MESONET_PATH, "mesonet/models/voxelmorph/VoxelMorph_Motif_based_functional_map_model_transformed1000.h5")


def process_atlas_brain_pipeline(args):
    config_file = mesonet.config_project(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode="test",
        atlas_to_brain_align=True,
        use_voxelmorph=False,
        use_unet=args.no_unet,
        use_dlc=True,
        sensory_match=False,
        mat_save=False,
        olfactory_check=False,
        config=DLC_CONFIG,
        model=MODEL_FILE,
    )

    mesonet.predict_regions(config_file)
    mesonet.predict_dlc(config_file)


def main(args):
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory '{args.input_dir}' does not exist")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.pipeline == "atlas_brain":
        process_atlas_brain_pipeline(args)
    else:
        raise ValueError(f"Cannot process pipeline '{args.pipeline}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str, choices=["atlas_brain"],
                        required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--no-unet", action="store_false")
    args = parser.parse_args()

    main(args)
