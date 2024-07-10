# Finding Meaning in Points: Weakly Supervised Semantic Segmentation for Event Cameras

Official code for "Finding Meaning in Points: Weakly Supervised Semantic Segmentation for Event Cameras" (ECCV2024)

## Datasets 
### DDD17
[Link]

### DSEC
[Link]

### DSEC Night-Point
[Link]

The dataset should have the following format:

    ├── DSEC_Semantic                 
    │   ├── train               
    │   │   ├── zurich_city_00_a
    │   │   │   ├── semantic
    │   │   │   │   ├── left
    │   │   │   │   │   ├── 11classes
    │   │   │   │   │   │   └──data
    │   │   │   │   │   │       ├── 000000.png
    │   │   │   │   │   │       └── ...
    │   │   │   │   │   └── 19classes
    │   │   │   │   │       └──data
    │   │   │   │   │           ├── 000000.png
    │   │   │   │   │           └── ...
    │   │   │   │   └── timestamps.txt
    │   │   │   └── events  
    │   │   │       └── left
    │   │   │           ├── events.h5
    │   │   │           └── rectify_map.h5
    │   │   └── ...
    │   └── test
    │       ├── zurich_city_13_a
    │       │   └── ...
    │       └── ... 


Code will be comming soon!
