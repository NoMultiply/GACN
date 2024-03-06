# Cora
CUDA_VISIBLE_DEVICES=0 python train.py --task clf --dataset Cora --n_epoch 7 --cl_lambda_ssl 1 --cl_lambda_bpr 0.0001 --n_epoch_cl 100

# Citeseer
CUDA_VISIBLE_DEVICES=0 python train.py --task clf --dataset Citeseer --n_epoch 3 --cl_lambda_ssl 1 --cl_lambda_bpr 0.0001 --wd_cl 0.0001 --n_epoch_cl 100

# UCI
CUDA_VISIBLE_DEVICES=0 python train.py --dataset uci --init_rate 0.6 --lambda_new 0.1 --cl_lambda_ssl 1 --cl_lambda_bpr 0.0001 --wd_cl 0.0001

# Taobao
CUDA_VISIBLE_DEVICES=0 python train.py --dataset TianChi

# Amazon
CUDA_VISIBLE_DEVICES=0 python train.py --dataset amazon

# Last.fm
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Lastfm --cl_lambda_ssl 0.00001
