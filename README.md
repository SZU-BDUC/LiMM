# LiMM
This is the code of our paper "Learning Road Network Index Structure for Efficient Map Matching".

## Dataset
We provide a sample dataset of porto in \[data]. **Please unzip the file \[data/porto.zip]**. 
- Folder \[data/porto/rein_tra] is the training dataset of Q-learning for adaptive searching range. 
- Folder \[data/porto/trajectory] is the raw trajectories of HMMLimm matching. 
- File \[data/porto/porto.osm] is the porto map with boundary : {lat_min = 41.05, lat_max = 41.25, lon_min = -8.75, lon_max = -8.45}

You can also download map from [OpenStreetMap](https://www.openstreetmap.org) and trajectory dataset from [Kaggle](https://www.kaggle.com/competitions/pkdd-15-predict-taxi-service-trajectory-i/data).

## Requirements
- python == 3.7
- tensorflow == 1.15.0
- networkx == 2.6.3
- rtree == 0.9.7
- osm2gmns == 0.6.8
- psutil == 5.9.0
- tqdm == 4.65.0

## Running
python main.py porto -p trajectory -e HMMLimm

## Result
- Folder \[result/porto/rein_ground_truth] is the ground truth(matched trajectories) of reinforcement learning for adaptive searching range. 
- Folder \[result/porto/hexagon_q_table] is the Q-tables for each hexagon.
- Folder \[result/porto/limm] is the matching trajectories of HMMLimm.
