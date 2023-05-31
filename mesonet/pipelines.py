import os
import sys

import mesonet

os.environ["MESONET_GIT"] = "/repo"
sys.path.insert(0, os.environ["MESONET_GIT"])
sys.path.insert(0, os.path.join(os.environ["MESONET_GIT"], "mesonet"))


input_dir_atlas_brain = "/repo/mesonet_inputs/example_data/pipeline1_2"
input_dir_brain_atlas = "/repo/mesonet_inputs/example_data/pipeline1_2"
input_dir_sensory_raw = "/repo/mesonet_inputs/example_data/pipeline3_sensory/sensory_raw"
input_dir_sensory_maps = "/repo/mesonet_inputs/example_data/pipeline3_sensory/sensory_maps"
input_dir_mbfm_unet = "/repo/mesonet_inputs/example_data/pipeline4_MBFM-U-Net"
input_dir_voxelmorph = "/repo/mesonet_inputs/example_data/pipeline5_VoxelMorph"

output_dir_atlas_brain = "/repo/mesonet_outputs/mesonet_outputs_atlas_brain"
output_dir_brain_atlas = "/repo/mesonet_outputs/mesonet_outputs_brain_atlas"
output_dir_sensory = "/repo/mesonet_outputs/mesonet_outputs_sensory"
output_dir_mbfm_unet = "/repo/mesonet_outputs/mesonet_outputs_MBFM_U_Net"
output_dir_voxelmorph = "/repo/mesonet_outputs/mesonet_outputs_voxelmorph"

dlc_config = "/repo/mesonet/dlc/atlas-DongshengXiao-2020-08-03/config.yaml"

model_file = "/repo/mesonet/models/DongshengXiao_brain_bundary.hdf5"
unet_only_model_file = "/repo/mesonet/models/DongshengXiao_unet_motif_based_functional_atlas.hdf5"
voxelmorph_model_file = "/repo/mesonet/models/voxelmorph/VoxelMorph_Motif_based_functional_map_model_transformed1000.h5"

config_file_atlas_brain = mesonet.config_project(
    input_dir=input_dir_atlas_brain,
    output_dir=output_dir_atlas_brain,
    mode='test',
    atlas_to_brain_align=True,
    use_voxelmorph=False,
    use_unet=True,
    use_dlc=True,
    sensory_match=False,
    mat_save=False,
    olfactory_check=True,
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
    olfactory_check=True,
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
# mesonet.predict_regions(config_file_atlas_brain)
# mesonet.predict_dlc(config_file_atlas_brain)

# Pipeline 2
# mesonet.predict_regions(config_file_brain_atlas)
# mesonet.predict_dlc(config_file_brain_atlas)

# Pipeline 3
# mesonet.predict_regions(config_file_sensory)
# mesonet.predict_dlc(config_file_sensory)

# Pipeline 4
# mesonet.predict_regions(config_file_mbfm_unet)

# Pipeline 5
mesonet.predict_regions(config_file_voxelmorph)
mesonet.predict_dlc(config_file_voxelmorph)
