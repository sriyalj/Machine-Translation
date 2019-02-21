#!/bin/bash

#The name of the job is test_job
#SBATCH -J fishbone_big

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#SBATCH --exclude=falcon3

#The maximum walltime of the job is a 8 days
#SBATCH --time=7-23:59:59

#SBATCH --mem=30G

#Leave this here if you need a GPU for your job
#SBATCH --partition=gpu

#SBATCH --gres=gpu:tesla:1

# OUR COMMANDS GO HERE

module load python/3.6.3/CUDA-8.0

source activate mtenv-cuda8

python -m sockeye.train --disable-device-locking \
                        --device-ids 0 \
                        -s bpe.tc.combined.train.et \
                        -t bpe.tc.combined.train.en \
                        -vs bpe.tc.combined.dev.et \
                        -vt bpe.tc.combined.dev.en \
                        -o experiments/basicmodel \
			--batch-size 4096  \
			--batch-type word \
			--checkpoint-frequency 4000 \
			--optimized-metric bleu \
			--max-num-checkpoint-not-improved 8 \
			--num-embed 512 \
			--max-seq-len 100:100 \
			--encoder transformer \
			--decoder transformer \
			--transformer-model-size 512 \
			--transformer-feed-forward-num-hidden 2048 \
			--transformer-dropout-prepost 0.1 \
			--weight-tying-type src_trg_softmax \
			--weight-tying \
			--initial-learning-rate 0.0003 \
			--learning-rate-warmup 50000 \
			--loss cross-entropy \
			--label-smoothing 0.2 


