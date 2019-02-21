#!/bin/bash

#The name of the job is test_job
#SBATCH -J SUG3_PRJ

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
			--seed 1 \
			--batch-type word \
			--batch-size 4096 \
			--checkpoint-frequency 4000 \
			--embed-dropout 0:0 \
			--encoder transformer \
			--decoder transformer \
			--num-layers 6:6 \
			--transformer-model-size 512 \
			--transformer-attention-heads 8 \
			--transformer-feed-forward-num-hidden 2048 \
			--transformer-preprocess n \
			--transformer-postprocess dr \
			--transformer-dropout-attention 0.1 \
			--max-seq-len 100:100 \
			--weight-tying \
			--weight-tying-type src_trg_softmax \
			--num-embed 512:512 \
			--num-words 50000:50000 \
			--word-min-count 1:1 \
			--optimizer adam \
			--initial-learning-rate 0.0001 \
			--learning-rate-reduce-num-not-improved 8 \
			--learning-rate-reduce-factor 0.7 \
			--learning-rate-scheduler-type plateau-reduce \
			--max-num-checkpoint-not-improved 32 \
     			--weight-init xavier \
			--weight-init-scale 3.0 \
			--weight-init-xavier-factor-type avg 
			
