#!/bin/bash

#name of the job is test_job
#SBATCH -J mt_proj_4

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is a 8 days
#SBATCH -t 05:05:00

#SBATCH --mem=30G

#Leave this here if you need a GPU for your job

#SBATCH --partition=gpu

#SBATCH --gres=gpu:tesla:1


# OUR COMMANDS GO HERE

module load python/3.6.3/CUDA-8.0

source activate mtenv-cuda8-1


python -m sockeye.train -s data/bpe.cleaned.tc.tok.train.et \
                        -t data/bpe.cleaned.tc.tok.train.en \
                        -vs data/bpe.cleaned.tc.tok.dev.et \
                        -vt data/bpe.cleaned.tc.tok.dev.en \
                        -o experiments/model5 \
                        --encoder transformer \
                        --decoder transformer \
                        --max-seq-len 60 \
                        --decode-and-evaluate 0 \
                        --batch-type sentence \
                        --batch-size 200
                        --optimized-metric bleu\
                        --max-num-checkpoint-not-improved 3
                        --max-num-epochs 10 \
                        --checkpoint-frequency 25
