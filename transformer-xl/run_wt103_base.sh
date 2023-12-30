#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3 train.py \
        --cuda \
        --data ./data/wikitext-103/ \
        --dataset wt103 \
        --d_embed 256 \
        --n_layer 12 \
        --d_model 768 \
        --n_head 12 \
        --d_head 64 \
        --d_inner 2048 \
        --not_bias \
        --dropout 0.1 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000 \
        --seq_len 256 \
        --attn_span 256 \
        --eval_seq_len 256 \
        --batch_size 60 \
        --multi_gpu \
        --gpu0_bsz 12 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python3 eval.py \
        --cuda \
        --data ./data/wikitext-103/ \
        --dataset wt103 \
        --seq_len 64 \
        --attn_span 256 \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
