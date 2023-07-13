import os
import sys

import mesonet

MESONET_PATH = os.getcwd()


os.environ["MESONET_GIT"] = MESONET_PATH
sys.path.insert(0, os.environ["MESONET_GIT"])
sys.path.insert(0, os.path.join(os.environ["MESONET_GIT"], "mesonet"))


# Atlas-to-brain paths
# --------------------

input_dir_atlas_brain = os.path.join(
        MESONET_PATH, "mesonet_inputs/example_data/pipeline1_2")
output_dir_atlas_brain = os.path.join(
        MESONET_PATH, "mesonet_outputs/mesonet_outputs_atlas_brain")

input_dir_atlas_brain = os.path.join(
        MESONET_PATH, "mesonet_inputs/awake1_data/atlas_brain")
output_dir_atlas_brain = os.path.join(
        MESONET_PATH, "mesonet_outputs/awake1_atlas_brain")

input_dir_atlas_brain = os.path.join(
        MESONET_PATH, "mesonet_inputs/awake1_data/atlas_brain")
output_dir_atlas_brain = os.path.join(
        MESONET_PATH, "mesonet_outputs/awake1_atlas_brain_no-unet")

input_dir_atlas_brain = os.path.join(
        MESONET_PATH, "mesonet_inputs/flash1_data/atlas_brain")
output_dir_atlas_brain = os.path.join(
        MESONET_PATH, "mesonet_outputs/flash1_atlas_brain")

# input_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_inputs/awake2_data/atlas_brain")
# output_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_outputs/awake2_atlas_brain")

# input_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_inputs/awake2_data/atlas_brain")
# output_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_outputs/awake2_atlas_brain_no-unet")

# input_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_inputs/euthanasia1_data/atlas_brain")
# output_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_outputs/euthanasia1_atlas_brain")

# input_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_inputs/euthanasia1_data/atlas_brain")
# output_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_outputs/euthanasia1_atlas_brain_no-unet")

# input_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_inputs/euthanasia2_data/atlas_brain")
# output_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_outputs/euthanasia2_atlas_brain")

# input_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_inputs/euthanasia2_data/atlas_brain")
# output_dir_atlas_brain = os.path.join(MESONET_PATH, "mesonet_outputs/euthanasia2_atlas_brain_no-unet")


# Brain-to-atlas paths
# --------------------

input_dir_brain_atlas = os.path.join(
        MESONET_PATH, "mesonet_inputs/example_data/pipeline1_2")
output_dir_brain_atlas = os.path.join(
        MESONET_PATH, "mesonet_outputs/mesonet_outputs_brain_atlas")


# Sensory paths
# -------------

input_dir_sensory_raw = os.path.join(
        MESONET_PATH, "mesonet_inputs/example_data/pipeline3_sensory/sensory_raw")
input_dir_sensory_maps = os.path.join(
        MESONET_PATH, "mesonet_inputs/example_data/pipeline3_sensory/sensory_maps")
output_dir_sensory = os.path.join(
        MESONET_PATH, "mesonet_outputs/mesonet_outputs_sensory")


# MBFM paths
# ----------

input_dir_mbfm_unet = os.path.join(
        MESONET_PATH, "mesonet_inputs/example_data/pipeline4_MBFM-U-Net")
output_dir_mbfm_unet = os.path.join(
        MESONET_PATH, "mesonet_outputs/mesonet_outputs_MBFM_U_Net")


# VoxelMorph paths
# ----------------

input_dir_voxelmorph = os.path.join(
        MESONET_PATH, "mesonet_inputs/example_data/pipeline5_VoxelMorph")
output_dir_voxelmorph = os.path.join(
        MESONET_PATH, "mesonet_outputs/mesonet_outputs_voxelmorph")


dlc_config = os.path.join(
        MESONET_PATH, "mesonet/dlc/atlas-DongshengXiao-2020-08-03/config.yaml")

model_file = os.path.join(
        MESONET_PATH, "mesonet/models/DongshengXiao_brain_bundary.hdf5")
unet_only_model_file = os.path.join(
        MESONET_PATH, "mesonet/models/DongshengXiao_unet_motif_based_functional_atlas.hdf5")
voxelmorph_model_file = os.path.join(
        MESONET_PATH, "mesonet/models/voxelmorph/VoxelMorph_Motif_based_functional_map_model_transformed1000.h5")


config_file_atlas_brain = mesonet.config_project(
    input_dir=input_dir_atlas_brain,
    output_dir=output_dir_atlas_brain,
    mode='test',
    atlas_to_brain_align=True,
    use_voxelmorph=False,
    use_unet=False,
    use_dlc=True,
    sensory_match=False,
    mat_save=False,
    olfactory_check=False,
    config=dlc_config,
    model=model_file,
)

config_file_brain_atlas = mesonet.config_project(
    input_dir=input_dir_brain_atlas,
    output_dir=output_dir_brain_atlas,
    mode='test',
    atlas_to_brain_align=False,
    use_voxelmorph=False,
    use_unet=True,
    use_dlc=True,
    sensory_match=False,
    mat_save=False,
    olfactory_check=False,
    config=dlc_config,
    model=model_file,
)

config_file_sensory = mesonet.config_project(
    input_dir=input_dir_sensory_raw,
    output_dir=output_dir_sensory,
    mode='test',
    atlas_to_brain_align=True,
    use_voxelmorph=False,
    use_unet=True,
    use_dlc=True,
    sensory_match=True,
    sensory_path=input_dir_sensory_maps,
    mat_save=False,
    config=dlc_config,
    model=model_file,
)

config_file_mbfm_unet = mesonet.config_project(
    input_dir=input_dir_mbfm_unet,
    output_dir=output_dir_mbfm_unet,
    mode='test',
    use_unet=True,
    use_dlc=False,
    sensory_match=False,
    mat_save=False,
    mask_generate=False,
    config=dlc_config,
    model=unet_only_model_file,
)

config_file_voxelmorph = mesonet.config_project(
    input_dir=input_dir_voxelmorph,
    output_dir=output_dir_voxelmorph,
    mode='test',
    atlas_to_brain_align=False,
    use_voxelmorph=True,
    use_unet=True,
    use_dlc=True,
    sensory_match=False,
    mat_save=False,
    config=dlc_config,
    model=model_file,
    align_once=True,
    olfactory_check=True,
    voxelmorph_model=voxelmorph_model_file,
)

# Pipeline 1
mesonet.predict_regions(config_file_atlas_brain)
mesonet.predict_dlc(config_file_atlas_brain)

# Pipeline 2
# mesonet.predict_regions(config_file_brain_atlas)
# mesonet.predict_dlc(config_file_brain_atlas)

# Pipeline 3
# mesonet.predict_regions(config_file_sensory)
# mesonet.predict_dlc(config_file_sensory)

# Pipeline 4
# mesonet.predict_regions(config_file_mbfm_unet)

# Pipeline 5
# mesonet.predict_regions(config_file_voxelmorph)
# mesonet.predict_dlc(config_file_voxelmorph)
