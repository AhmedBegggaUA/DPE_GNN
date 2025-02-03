# #!/bin/bash
# # # GCN
# # # None
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GCN --dataset_name mutag > gcn/mutag_gcn_no_pos.txt
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GCN --dataset_name imdb > gcn/imdb_gcn_no_pos.txt
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GCN --dataset_name bzr > gcn/bzr_gcn_no_pos.txt
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GCN --dataset_name cox2 > gcn/cox2_gcn_no_pos.txt

# # # RWPE
# # python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GCN --dataset_name mutag > gcn/mutag_gcn_rwpe.txt
# # python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GCN --dataset_name imdb > gcn/imdb_gcn_rwpe.txt
# # python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GCN --dataset_name bzr > gcn/bzr_gcn_rwpe.txt
# # python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GCN --dataset_name cox2 > gcn/cox2_gcn_rwpe.txt

# # # LaPE
# # python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GCN --dataset_name mutag > gcn/mutag_gcn_lape.txt
# # python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GCN --dataset_name imdb > gcn/imdb_gcn_lape.txt
# # python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GCN --dataset_name bzr > gcn/bzr_gcn_lape.txt
# # python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GCN --dataset_name cox2 > gcn/cox2_gcn_lape.txt

# # # pump
# # python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GCN --dataset_name mutag > gcn/mutag_gcn_pump.txt
# # python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GCN --dataset_name imdb > gcn/imdb_gcn_pump.txt
# # python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GCN --dataset_name bzr > gcn/bzr_gcn_pump.txt
# # python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GCN --dataset_name cox2 > gcn/cox2_gcn_pump.txt

# # # eigen
# # python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GCN --dataset_name mutag > gcn/mutag_gcn_eigen.txt
# # python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GCN --dataset_name imdb > gcn/imdb_gcn_eigen.txt
# # python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GCN --dataset_name bzr > gcn/bzr_gcn_eigen.txt
# # python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GCN --dataset_name cox2 > gcn/cox2_gcn_eigen.txt

# # # GIN
# # # None
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GIN --dataset_name mutag > gin/mutag_gin_no_pos.txt
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GIN --dataset_name imdb > gin/imdb_gin_no_pos.txt
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GIN --dataset_name bzr > gin/bzr_gin_no_pos.txt
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GIN --dataset_name cox2 > gin/cox2_gin_no_pos.txt

# # # RWPE
# # python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GIN --dataset_name mutag > gin/mutag_gin_rwpe.txt
# # python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GIN --dataset_name imdb > gin/imdb_gin_rwpe.txt
# # python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GIN --dataset_name bzr > gin/bzr_gin_rwpe.txt
# # python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GIN --dataset_name cox2 > gin/cox2_gin_rwpe.txt

# # # LaPE
# # python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GIN --dataset_name mutag > gin/mutag_gin_lape.txt
# # python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GIN --dataset_name imdb > gin/imdb_gin_lape.txt
# # python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GIN --dataset_name bzr > gin/bzr_gin_lape.txt
# # python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GIN --dataset_name cox2 > gin/cox2_gin_lape.txt

# # # pump
# # python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GIN --dataset_name mutag > gin/mutag_gin_pump.txt
# # python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GIN --dataset_name imdb > gin/imdb_gin_pump.txt
# # python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GIN --dataset_name bzr > gin/bzr_gin_pump.txt
# # python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GIN --dataset_name cox2 > gin/cox2_gin_pump.txt

# # # eigen
# # python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GIN --dataset_name mutag > gin/mutag_gin_eigen.txt
# # python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GIN --dataset_name imdb > gin/imdb_gin_eigen.txt
# # python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GIN --dataset_name bzr > gin/bzr_gin_eigen.txt
# # python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GIN --dataset_name cox2 > gin/cox2_gin_eigen.txt

# # # GatedGCN
# # # None
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GatedGCN --dataset_name mutag > gated/mutag_gated_no_pos.txt
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GatedGCN --dataset_name imdb > gated/imdb_gated_no_pos.txt
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GatedGCN --dataset_name bzr > gated/bzr_gated_no_pos.txt
# # python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GatedGCN --dataset_name cox2 > gated/cox2_gated_no_pos.txt

# # RWPE
# python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GatedGCN --dataset_name mutag > gated/mutag_gated_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GatedGCN --dataset_name imdb > gated/imdb_gated_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GatedGCN --dataset_name bzr > gated/bzr_gated_rwpe.txt
# python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GatedGCN --dataset_name cox2 > gated/cox2_gated_rwpe.txt

# # LaPE
# python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GatedGCN --dataset_name mutag > gated/mutag_gated_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GatedGCN --dataset_name imdb > gated/imdb_gated_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GatedGCN --dataset_name bzr > gated/bzr_gated_lape.txt
# python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GatedGCN --dataset_name cox2 > gated/cox2_gated_lape.txt

# # pump
# python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name mutag > gated/mutag_gated_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name imdb > gated/imdb_gated_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name bzr > gated/bzr_gated_pump.txt
# python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name cox2 > gated/cox2_gated_pump.txt

# # eigen
# python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GatedGCN --dataset_name mutag > gated/mutag_gated_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GatedGCN --dataset_name imdb > gated/imdb_gated_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GatedGCN --dataset_name bzr > gated/bzr_gated_eigen.txt
# python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GatedGCN --dataset_name cox2 > gated/cox2_gated_eigen.txt

# GatedGCN
# None
python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GatedGCN --dataset_name collab > gated/collab_gated_no_pos.txt
python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GatedGCN --dataset_name reddit > gated/reddit_gated_no_pos.txt
python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GatedGCN --dataset_name enzymes > gated/enzymes_gated_no_pos.txt
python run_graph_classification.py --positional_encoding None --device cuda:1 --layer_type GatedGCN --dataset_name proteins > gated/proteins_gated_no_pos.txt

# RWPE
python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GatedGCN --dataset_name collab > gated/collab_gated_rwpe.txt
python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GatedGCN --dataset_name reddit > gated/reddit_gated_rwpe.txt
python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GatedGCN --dataset_name enzymes > gated/enzymes_gated_rwpe.txt
python run_graph_classification.py --positional_encoding RWPE --device cuda:1 --layer_type GatedGCN --dataset_name proteins > gated/proteins_gated_rwpe.txt

# LaPE
python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GatedGCN --dataset_name collab > gated/collab_gated_lape.txt
python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GatedGCN --dataset_name reddit > gated/reddit_gated_lape.txt
python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GatedGCN --dataset_name enzymes > gated/enzymes_gated_lape.txt
python run_graph_classification.py --positional_encoding LaPE --device cuda:1 --layer_type GatedGCN --dataset_name proteins > gated/proteins_gated_lape.txt

# pump
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name collab > gated/collab_gated_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name reddit > gated/reddit_gated_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name enzymes > gated/enzymes_gated_pump.txt
python run_graph_classification.py --positional_encoding pump --device cuda:1 --layer_type GatedGCN --dataset_name proteins > gated/proteins_gated_pump.txt

# eigen
python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GatedGCN --dataset_name collab > gated/collab_gated_eigen.txt
python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GatedGCN --dataset_name reddit > gated/reddit_gated_eigen.txt
python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GatedGCN --dataset_name enzymes > gated/enzymes_gated_eigen.txt
python run_graph_classification.py --positional_encoding eigen --device cuda:1 --layer_type GatedGCN --dataset_name proteins > gated/proteins_gated_eigen.txt
#3827013