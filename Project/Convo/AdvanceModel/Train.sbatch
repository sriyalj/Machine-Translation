#!/bin/bash

#The name of the job is test_job
#SBATCH -J CON3_PRJ

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#SBATCH --exclude=falcon3

#The maximum walltime of the job is a 8 days
#SBATCH --time=6-23:59:59

#SBATCH --mem=30G

#Leave this here if you need a GPU for your job
#SBATCH --partition=gpu

#SBATCH --gres=gpu:tesla:1

# OUR COMMANDS GO HERE

module load python/3.6.3/CUDA-8.0

source activate mtenv-cuda8-1

python -m sockeye.train --disable-device-locking \
                        --device-ids 0 \
                        -s data/bpe.cleaned.tc.tok.train.et \
                        -t data/bpe.cleaned.tc.tok.train.en \
                        -vs data/bpe.cleaned.tc.tok.dev.et \
                        -vt data/bpe.cleaned.tc.tok.dev.en \
                        -o experiments/advmodel \
    			--seed=1 \
    			--batch-type word \
    			--batch-size 4000 \
    			--checkpoint-frequency 4000 \
    			--encoder cnn \
    			--decoder cnn \
    			--num-layers 8 \
    			--num-embed 512 \
    			--cnn-num-hidden 512 \
    			--cnn-project-qkv \
    			--cnn-kernel-width 3 \
    			--cnn-hidden-dropout 0.2 \
    			--cnn-positional-embedding-type learned \
    			--max-seq-len 150 \
    			--loss-normalization-type valid \
    			--word-min-count 1 \
    			--optimizer adam \
    			--initial-learning-rate 0.0002 \
    			--learning-rate-reduce-num-not-improved 8 \
    			--learning-rate-reduce-factor 0.7 \
    			--learning-rate-decay-param-reset \
    			--max-num-checkpoint-not-improved 16  
