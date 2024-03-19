#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATADIR=./data
LR=0.00007 # 0.0001 * ln2
BS=128
adapt_BS=64


printf '\n---------------------\n\n'

CORRUPT=snow
CUDA_VISIBLE_DEVICES=0 python TTAC2_onepass_no_source.py \
	--dataroot ${DATADIR} \
	--workers 4 \
	--iters 2 \
	--corruption ${CORRUPT} \
	--batch_size ${BS} \
	--adapt_batch_size ${adapt_BS} \
	--lr ${LR} \
	--with_ssl
