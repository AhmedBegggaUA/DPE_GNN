#!/bin/bash
# # GCN
# # None
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GCN --dataset_name collab > gcn/collab_gcn_no_pos.txt
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GCN --dataset_name reddit > gcn/reddit_gcn_no_pos.txt
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GCN --dataset_name enzymes > gcn/enzymes_gcn_no_pos.txt
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GCN --dataset_name proteins > gcn/proteins_gcn_no_pos.txt

# # RWPE
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GCN --dataset_name collab > gcn/collab_gcn_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GCN --dataset_name reddit > gcn/reddit_gcn_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GCN --dataset_name enzymes > gcn/enzymes_gcn_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GCN --dataset_name proteins > gcn/proteins_gcn_rwpe.txt

# # LaPE
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GCN --dataset_name collab > gcn/collab_gcn_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GCN --dataset_name reddit > gcn/reddit_gcn_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GCN --dataset_name enzymes > gcn/enzymes_gcn_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GCN --dataset_name proteins > gcn/proteins_gcn_lape.txt

# # pump
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GCN --dataset_name collab > gcn/collab_gcn_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GCN --dataset_name reddit > gcn/reddit_gcn_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GCN --dataset_name enzymes > gcn/enzymes_gcn_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GCN --dataset_name proteins > gcn/proteins_gcn_pump.txt

# # eigen
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GCN --dataset_name collab > gcn/collab_gcn_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GCN --dataset_name reddit > gcn/reddit_gcn_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GCN --dataset_name enzymes > gcn/enzymes_gcn_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GCN --dataset_name proteins > gcn/proteins_gcn_eigen.txt

# # GIN
# # None
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GIN --dataset_name collab > gin/collab_gin_no_pos.txt
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GIN --dataset_name reddit > gin/reddit_gin_no_pos.txt
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GIN --dataset_name enzymes > gin/enzymes_gin_no_pos.txt
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GIN --dataset_name proteins > gin/proteins_gin_no_pos.txt

# # RWPE
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GIN --dataset_name collab > gin/collab_gin_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GIN --dataset_name reddit > gin/reddit_gin_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GIN --dataset_name enzymes > gin/enzymes_gin_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GIN --dataset_name proteins > gin/proteins_gin_rwpe.txt

# # LaPE
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GIN --dataset_name collab > gin/collab_gin_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GIN --dataset_name reddit > gin/reddit_gin_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GIN --dataset_name enzymes > gin/enzymes_gin_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GIN --dataset_name proteins > gin/proteins_gin_lape.txt

# # pump
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GIN --dataset_name collab > gin/collab_gin_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GIN --dataset_name reddit > gin/reddit_gin_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GIN --dataset_name enzymes > gin/enzymes_gin_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GIN --dataset_name proteins > gin/proteins_gin_pump.txt

# # eigen
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GIN --dataset_name collab > gin/collab_gin_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GIN --dataset_name reddit > gin/reddit_gin_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GIN --dataset_name enzymes > gin/enzymes_gin_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GIN --dataset_name proteins > gin/proteins_gin_eigen.txt

# # GatedGCN
# # None
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GatedGCN --dataset_name collab > gated/collab_gated_no_pos.txt
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GatedGCN --dataset_name reddit > gated/reddit_gated_no_pos.txt
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GatedGCN --dataset_name enzymes > gated/enzymes_gated_no_pos.txt
# python run_graph_classification.py --positional_encoding None --device cuda:0 --layer_type GatedGCN --dataset_name proteins > gated/proteins_gated_no_pos.txt

# # RWPE
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GatedGCN --dataset_name collab > gated/collab_gated_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GatedGCN --dataset_name reddit > gated/reddit_gated_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GatedGCN --dataset_name enzymes > gated/enzymes_gated_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:0 --layer_type GatedGCN --dataset_name proteins > gated/proteins_gated_rwpe.txt

# # LaPE
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GatedGCN --dataset_name collab > gated/collab_gated_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GatedGCN --dataset_name reddit > gated/reddit_gated_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GatedGCN --dataset_name enzymes > gated/enzymes_gated_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:0 --layer_type GatedGCN --dataset_name proteins > gated/proteins_gated_lape.txt

# # pump
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GatedGCN --dataset_name collab > gated/collab_gated_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GatedGCN --dataset_name reddit > gated/reddit_gated_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GatedGCN --dataset_name enzymes > gated/enzymes_gated_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:0 --layer_type GatedGCN --dataset_name proteins > gated/proteins_gated_pump.txt

# # eigen
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GatedGCN --dataset_name collab > gated/collab_gated_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GatedGCN --dataset_name reddit > gated/reddit_gated_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GatedGCN --dataset_name enzymes > gated/enzymes_gated_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:0 --layer_type GatedGCN --dataset_name proteins > gated/proteins_gated_eigen.txt

# pump
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GCN --dataset_name collab --k 15 > gcn/collab_gcn_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GCN --dataset_name reddit --k 15 > gcn/reddit_gcn_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GCN --dataset_name enzymes --k 15 > gcn/enzymes_gcn_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GCN --dataset_name proteins --k 15 > gcn/proteins_gcn_pump.txt


# pump
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GIN --dataset_name collab --k 15 > gin/collab_gin_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GIN --dataset_name reddit --k 15 > gin/reddit_gin_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GIN --dataset_name enzymes --k 15 > gin/enzymes_gin_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GIN --dataset_name proteins --k 15 > gin/proteins_gin_pump.txt
# pump
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name collab --k 15 > gated/collab_gated_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name reddit --k 15 > gated/reddit_gated_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name enzymes --k 15 > gated/enzymes_gated_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name proteins --k 15 > gated/proteins_gated_pump.txt
