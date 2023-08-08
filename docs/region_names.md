# Region Names

MesoNet segments the brain into different regions. The segmentation of the brain
can be found in the `output_overlay/` directory of the MesoNet output directory.

## The Allen Mouse Brain Common Coordinate Framework Regions

The regions are based on the paper, "The Allen Mouse Brain Common Coordinate
Framework: A 3D Reference Atlas" by Wang et al. in 2020.

Figure 3L from the paper is shown below. This figure depicts the segmentation of
the brain that the common coordinate framework is based off of. MesoNet takes
some of these regions to create the segmentation of the brain.

<img src="/docs/_static/common_coordinate_framework.png" alt="common coordinate framework" width="200"/>

Specifically, MesoNet contains the following regions:

| Region number | Coordinate acronym | Region name                                |
|---------------|--------------------|--------------------------------------------|
| 0             | VISrl              | Rostrolateral visual area                  |
| 1             | VISa               | Anterior visual area                       |
| 2             | RSPd               | Retrosplenial area, dorsal part            |
| 3             | RSPagl             | Retrosplenial area, lateral agranular part |
| 4             | VISpm              | Posteromedial visual area                  |
| 5             | VISp               | Primary visual area                        |
| 6             | ???                | ???                                        |
| 7             | VISam              | Anteromedial visual area                   |
| 8             | VISal              | Anterolateral visual area                  |
| 9             | AUD                | Auditory area                              |
| 10            | SSs                | Supplemental somatosensory area            |
| 11            | SSp-un             | Primary somatosensory area, unassigned     |
| 12            | SSp-tr             | Primary somatosensory area, trunk          |
| 13            | SSp-ul             | Primary somatosensory area, upper limb     |
| 14            | SSp-m              | Primary somatosensory area, mouth          |
| 15            | SSp-ll             | Primary somatosensory area, lower limb     |
| 16            | SSp-bfd            | Primary somatosensory area, barrel field   |
| 17            | SSp-n              | Primary somatosensory area, nose           |
| 18            | MOs                | Secondary motor area                       |
| 19            | MOp                | Primary motor area                         |

The names for the regions were obtained from the spreadsheet found in the
supplementary resources of the paper. The spreadsheet can also be found in this
repository located [in the documentation](/docs/_static/mmc2.xlsx).

Note that the complementary region numbers for each region in the left
hemisphere are 40 subtract the right hemisphere's region number. So, for
example, the left hemisphere's primary motor area is 21 = 40 - 19, where 19 is
the right hemisphere's region number of the primary motor area.

# The Chan Lab Regions

Available in the Chan Lab's Google Drive is the script,
`SOP_BilatRegionalCorr.m` which also outputs regions based on predefined
coordinates.

These regions also have corresponding MesoNet region numbers. The following
table describes this relation.

| Region number | `SOP_BilatRegionalCorr.m` acronym | Region name                |
|---------------|-----------------------------------|----------------------------|
| 2             | rRS                               | Right retrosplenial area   |
| 5             | rV1                               | Right primary visual area  |
| 9             | rAU                               | Right auditory area        |
| 12            | rTR                               | Right trunk area           |
| 13            | rFL                               | Right forelimb area        |
| 14            | rMO                               | Right mouth area           |
| 15            | rHL                               | Right hindlimb area        |
| 16            | rBC                               | Right barrel cortex area   |
| 17            | rNO                               | Right nose area            |
| 18            | rM2                               | Right secondary motor area |
| 19            | rM1                               | Right primary motor area   |

Again, like the common coordinate framework table, the left hemisphere regions
can be obtained by subtracting the region number from 40 to obtain the region
number of the left hemisphere region. So, for example, the left hindlimb area
("lHL") would be region number 25 = 40 - 15, where 15 is the right hindlimb
region number.

Note that unfortunately, a mapping for all regions could not be determined as
there were some MesoNet regions that did not appear in the
`SOP_BilatRegionalCorr.m` regions and vice versa.
