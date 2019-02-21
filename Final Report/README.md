# Project-report

The report consists of 5 sections:
1) Section 1: Introduction
2) Section 2: Background
3) Section 3: Comparative experiment
4) Section 4: Large-scale experiment
5) Section 5: Summary

## Section 1: Introduction
### Section 1.1: Team information
- Fishbone
- Maxi Fischer, Sophie Gräfnitz, Sriyal Jayasinghe
- maxifischer, sgrae, sriyalj
- maxi.fischer@student.hpi.de, sophiegraefnitz@outlook.com, sriyal.jayasinghe@gmail.com
### Section 1.2: Homework 2 (aka Practicum 2)
- What was the task? <br><br>
Our task was to train a sockeye model on Europarl corpus including prepocessing of the train data. The model params were given : 

```
1. bidirectional LSTM encoder, 1 layer
2. unidirectional LSTM decoder, 1 layer
3. dot product attention type
4. adam optimizer
5. 0.001 learning rate
6. source and target embeddings should have 300 features each
7. encoder and decoder LSTM's should have 600 hidden units each
8. no additional model reinforcements (do not add any additional stuff to the model, left it vanilla (default))
```
Training params:

```
1. batches consist of sentences up to 100 words
2.  there should be 64 sentences per batch
3.  early stopping based on perplexity
4.  minimum number of epochs is 3
5.  3 checkpoints per epoch
6.  early stopping if model did not improve for 4 checkpoints
7.  evaluate 2 times per epoch
```
Then we should compare our just trained models' translation of "Tere hommikust" with the translation that we got from a
simple model that was trained in the seminar. 
- What have you done?<br><br>
We realized this by training a sockeye model with the following params. Details can be found in the params excel sheet in our practicum2 repository. 
```
 --encoder rnn \ --decoder rnn \ --num-layers 1:1 \--rnn-cell-type lstm﻿

--rnn-attention-type dot
--optimizer adam
--initial-learning-rate 0.001
--num-embed 300:300
--rnn-num-hidden 600

--max-seq-len 100
--batch-size 64 \ --batch-type sentence
--optimized-metric perplexity
--min-num-epochs 3
--checkpoint-frequency 3377
--max-num-checkpoint-not-improved 4
```

- Main result<br><br>
As a result we have a model trained according to the requirements. It turned out to translate "Tere Hommikust" much better than out initial seminar model. Our new model translates "Congratulations" and the old one translates "The the the the the the...".

- What were the obstacles and how did you overcome them?<br><br>
Our main challenge was to find the proper parameters that fulfill the desired training behaviour. We overcame this by asking <code> sockeye.train -h </code> and finding out the params. For some, this was trivial. We calculated checkpoint frequency by dividing the given 648424 training instances by batch size 64, resulting in ~10132 iterations in an epoch. The checkpoint frequency describes after how many iterations the algorithm shall create a checkpoint, that's why we divided 10132 by 3 (the given checkpoint frequency) resulting in a parameter of 3377.
- The results can be found in our [practicum 2 homework folder](https://github.com/mt2018-tartu-shared-task/practicum2-fishbone/tree/master/practicum2_hw), last changed on 22 Oct 2018, 18:04 EEST <br>
- Contribution of each team member (who done what) 
  - param finding: Maxi Fischer, Sophie Gräfnitz, Sriyal Jayasinghe
  - model training: Maxi Fischer, Sriyal Jayasinghe
  - Post processing & Sentence transalation: Sriyal Jayasinghe

### Section 1.3: Homework 3 (aka Practicum 4)
- What was the task? <br> <br>
Our task was to 
  1. translate the dev set of our Europarl corpus trained model from Homework 2
  2. Compute BLEU Score with <code>beam size parameter equals 8</code> on this translation and report everything
  3. Manual error analyis of 60 sentences including beginning and end of translation
  4. Perform attention analysis  
- What have you done?<br><br>
We 
  1. first translated the dev set and postprocessed it. Postprocessing equals "backwards preprocessing", which means we did first de-bpe by running the string stream editor <code>sed -r 's/(@@ )|(@@ ?$)//g'</code>, second de-truecasing and de-tokenization by running moses scripts [detruecase.perl](https://github.com/marian-nmt/moses-scripts/blob/master/scripts/recaser/detruecase.perl) and [detokenizer.perl](https://github.com/marian-nmt/moses-scripts/blob/master/scripts/tokenizer/detokenizer.perl)
  2. We calculated bleu score running <code> perl ../../moses-scripts/scripts/generic/multi-bleu.perl dev.en < hyps.baseline.en </code>. *We did not find out how to determine beam size parameter*. 
  3. We created a sheet in which we performed manual error analysis comparing our translation of the dev set with the reference translation, paying attention to fluency, understandability and word choice. 
  4. We reran the translation process, this time using <code>--output-type align_plot</code> as an additional parameters. We safed the ouput sentence alignment pictures. 
  5. We summarized our observations in a conclusions document. 
- Main result <br> <br>
 We achieved a BLEU score of 9.83. From error analysis we learnt, that our model is often close to reference translation, but has problems dealing with long sentences and names. Sometimes, the sentence structure suffers from grammatical nonsense, left-outs and repetitions. From the sentence alignments we learned that a diffuse attention usually indicates a diffuse translation in the end. 
- What were the obstacles and how did you overcome them? <br><br>
 At first we translated a wrong dev set (in HTML format) and only achieved BLEU 0.04, so this was the point when we realized our mistake and reran the translation on the correct dev set. A short problem was to find a good method to postprocess and to calculate BLEU. Our mentor did support us in this case. Finally, we found the linked moses scripts. 
- Our results can be found in our [practicum 4 homework folder](https://github.com/mt2018-tartu-shared-task/practicum2-fishbone/commit/3e413c4d2a503d77f2f3140a4abb37e22eb0a217), last changed on 26 Oct 2018, 12:32 EEST
- Contribution of each team member <br><br>
  - preprocessing and model traning: Sriyal Jayasinghe
  - translation of dev set: Sriyal Jayasinghe
  - postprocessing and BLEU calculation: Sophie Gräfnitz, Sriyal Jayasinghe
  - error analysis: Maxi Fischer, Sophie Gräfnitz, Sriyal Jayasinghe
  - attention analysis: Maxi Fischer, Sophie Gräfnitz

## Section 2: Background
- Our project was based on two paper publications. They are as follows:

     1. [*Attention Is All You Need* by Ashish Vaswani et al.](https://arxiv.org/abs/1706.03762) - This paper was composed by Google where it proposes a architecture known as Transformer which is based on the Attention concept.
     
     2. [*Convolutional Sequence to Sequence Learning* by Jonas Gehring et al.](https://arxiv.org/abs/1705.03122) - This paper was composed by Facebook AI Research (FAIR) and it proposes fully convolutional neural network architecure model that can be used for machine translation.

The aim of our project twas to compare the Attention based machine translation model against Convolutional network architecture machine translation model.

- **Attention Architecture**  Traditional Natural Machine Translation models based on architectures such as RNN, RNN-LSTM are seuential models. Such models expect the tokens to be fed in to the model in a sequential manner and the translations are also produced in a seuential manner. Due to the seuential nature such models have difficulties in learning long-range dependencies. The paper proposed an architecture that can perform parallel learning on all input tokens where the model learns to distribute its attention to certain structures and words. 
		The architecture computes the relevance of a set of values(information) based on some keys and queries. Since with only one attention head it is hard to learn multiple dependencies the paper proposes to apply multi-head attention. 
		The model learns how to optimally choose k sentence substructures, that can be passed to k attention heads. Thus, k different internal dependencies can be learned. According to the architecture proposed even for a multi-head attention the computational effort is similiar to one-head attention.
		The advantage of transformer is, that the number of operations to learn dependencies between two tokens is constant independent from their actual position and distance. In the paper "Attention is all you need" the network consists of 6 layers of encoder and decoder, where both consist of multiple stages and both including multi head attention and a Feed Forward Network which work on attention and positions. 

- **Convolutional architecture** is frequently applied in image processing. Long-term dependencies can be learned by applying filters on the original data, which means the greater the distance between two tokens, the later their dependence is modelled. The filtering steps are applied sequentially, while each filterig step means the convolution of subphrases. These convolutions can be easiliy parallelized. Each layer in the paper's network consists of first convolution, applying a non-linearity and then multi-step attention (which is slightly different from multi head attention). 



- [presentation slides](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Fishbone_Presentation.pdf)

## Section 3: Comparative experiment
### Section 3.1: Description

The two systems we compared were the sockeye implementations of the Attention architecture against the sockeye implementation of Convolutional architecture.
 
 
- Our expected result before running the experiments: 
   - According to results published in [*Sockeye: A Toolkit for Neural Machine Translation](https://arxiv.org/pdf/1712.05690.pdf) the BLEU score obtained for the different language data set pairs by the transformer architecure was always at least more than 3 points higher than than the convolution architecure. 
   - The attention model supports dependency modelling without considering distance in the input or output sequences. This approach solves the drawback of sequence based approaches like RNN/CNN, which suffer from forgetting long-term dependencies and deranging sentence context. Although CNN learns long-term dependencies in a high convolution layer, the distances of these dependent structures are limited to the convolution step width. CNN provides a inflexible way of modelling long-term dependencies compared to transformer architecture. 
   
As a result we too expected a better BLEU score from the transformer model compared to the convolution model.
   

### Section 3.2: System 1
**Description of experiment setup**<br>

  - Below are the datasets we used for the Attention model:
  
    training  dataset: [Europal Estonian & English dataset](http://www.statmt.org/europarl/v7/et-en.tgz)<br> 
    This data set contained two files. One is the Estonian transcript of a speech made at the Europian Parliament and the other being the corresponding english translation of the Estonian transcript. Each of the files contained 651746 sentences. 
    
    test and dev dataset: [English dev set](https://github.com/mt2018-tartu-shared-task/practicum2/blob/master/dev.en) and [Estonian dev set](https://github.com/mt2018-tartu-shared-task/practicum2/blob/master/dev.et) <br>
    Each of these files had 2000 sentences. They are confusingly named dev.* , although they are also used for testing. The test/dev data set is another subset drawn from Europarl Estonian & English dataset that doesnt have intersection with the training data set. For the testing of the models the Estonian test set (= dev.et) was translated and the English reference translation (= dev.en) was used to calculate the BLEU score of this translation. During training, dev.* was used for intermediate evaluating the training process.
    
    
    
  - The preprocessing steps that we followed were the same preprocessing steps we follwed in [Practicum 2](https://github.com/mt2018-tartu-shared-task/practicum2/blob/master/seminar.md#data-preprocessing) 
  
  	Those steps are as follows:
  
   	1. Tokenization
   	2. Truecasing
   	3. Cleaning
   	4. Subword segmentation
   
   		For the Tokenization step we used the tokenizer.perl file of the moses-scripts library. The obejctive of the tokentization step was to split the sentences in the data sets to tokens. During the Truecasing step we trained a truecasing model and used that model to truecase the tokenized files. The objective of the turecasing step was to convert the capital letters in to simple letters. Afterwards the cleaning step was used on the truecased datasets to filter out long sentences which were having more than 100 tokens. The cleaning step was done using the clean-corpus-n.perl file of the moses-scripts library. The last preprocessing step was the subword segmentation where we splited the words into subwords using Byte Pair Encoding (BPE). To do subword segmentation we trained a BPE model and used that model to carryout the subword segmentation. 
   
   		The above described steps were performed on both the traing sets. 
   
   		We perfomed the above mentioned 4 preprocessing steps on the dev sets as well but truecasing and subword segmentation of the dev sets were done using the models we trained on the traning sets without generating new models for the dev sets. 
   
   
  - We trained 3 models with different hyperparameters on the same training and dev sets. The reason behind traning three models was to compare the models against each other and identify the best hyperparameters. These three models we named as:
  
   	1. Simple Attention Model
   
   	2. Intermediate Attention Model and 
   
   	3. Advance Attention Model
   
  - The hyperparameters of the three models are as follows
   
   	- [Hyperparameters](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Attention/SimpleModel/Train.sbatch) of the Simple Attention Model:
	
	``` 
	        --encoder transformer \
	        --decoder transformer \
	        --max-seq-len 60 \
	        --decode-and-evaluate 0 \
	        --batch-type sentence \
	        --batch-size 200 \
	        --optimized-metric bleu\
	        --max-num-checkpoint-not-improved 3 \
	        --max-num-epochs 10 \
	        --checkpoint-frequency 25
	```
	- [Hyperparameters](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Attention/IntermediateModel/Train.sbatch) of intermediate Attention Model
	```
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
	```
	
	- [Hyperparameters](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Attention/AdvanceModel/Train.sbatch) of the Advance Attention Model
	
  	```
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
	```
  
 **Results** <br>

As explained above the above three models were tested by making the models translate the Estonian dev set and by using the English dev set to calculate the BLEU score of the translations.

 Below are the BLEU score values we obtained for the three models
 
 - [Simple Attention Model](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Attention/SimpleModel/transBleu.txt) - 12.62
 - [Intermediate Attention Model](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Attention/IntermediateModel/transBleu.txt) - 14.40
 - [Advance Attention Model](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Attention/AdvanceModel/transBleu.txt) - 15.20
 
WIth the above BLEU score values it can be seen that the Advance Attention Model produced the best results

- What were the obstacles and how did you overcome them?

The biggest obstacle that we faced was understanding the architecture proposed in the paper. Since it was a paper from Google it was hard to understand. Additionally the concepts like attention was new to us as well. As a result we initially struggled to identify the correct parameters to train a model. 

To overcome the above mentioned obstacle we did a lot a reading and researching on the proposed architecture and we examined several implementations of the proposed architecture done using other environments such as [Tensorflow](https://github.com/Kyubyong/transformer), [PyTorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) and  [Marian](https://github.com/marian-nmt/marian-examples/tree/master/transformer). Additionally we watched a [presentation video](https://www.youtube.com/watch?v=rBCqOTEfxvg&t=1019s) by the Google team who proposed this architecure and another [video](https://www.youtube.com/watch?v=iDulhoQ2pro&t=513s) which explained the proposed architecture.


- Link to the solution on GitHub (link to the final submission commit) + submission timestamp (time of the GitHub commit)

	- link to the solution (https://github.com/mt2018-tartu-shared-task/project-fishbone/tree/master/Attention)
	- link to the final submission commit (https://github.com/mt2018-tartu-shared-task/project-fishbone/commit/39720308ba255f1c5e92dc2a30aca07fce09a8e1)
	- last timestamp on 04 Nov 2018, 20.49 EEST 
	
	
- Contribution of each team member

	- Preporcessing - Sriyal Jayasinghe
	- Hyperparameter identification - Sriyal Jayasinghe
	- Training of models - Sophie Gräfnitz, Sriyal Jayasinghe
	- translation of dev set: Sophie Gräfnitz, Sriyal Jayasinghe
	- Postprocessing and BLEU score calculation - Sriyal Jayasinghe

### Section 3.3: System 2
- For the convolutional model, we used the same data sets as for System 1.  
      
- The preporcessing steps that we followed were the same preprocessing steps we followed for System 1 (Attention Model) 
     
- We trained 2 models with differnt hyperparameters on the same training and dev sets. The reason behind traning two models as with the Attention model was to compare the two convolution models against each other and identify the best hyperparameters. The two convolution models we named as:
  
   	1. Simple Convolution Model and 
   
   	2. Advanced Convolution Model
   
  - The hyperparameters of the two models are as follows
  
  	- [Hyperparameters](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Convo/SimpleModel/Train.sbatch) of the Simple Convolution Model:
	
	```
		--encoder cnn \
                --decoder cnn \
                --max-seq-len 60 \
                --decode-and-evaluate 500 \
                --batch-type sentence \
                --batch-size 200 \
                --optimized-metric bleu\
                --max-num-checkpoint-not-improved 6 \
                --max-num-epochs 10 \
                --checkpoint-frequency 1000
	```
  	
	 - [Hyperparameters](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Convo/AdvanceModel/Train.sbatch) of the Advance Convolution Model:
	
	```
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
	```
  
**Results**<br>

As explained earlier the above two models were tested by making the models translate the Estonian dev set and by using the English dev set to calculate the BLEU score of the translations.

Below are the BLEU score values we obtained on the English dev set for the two convolution models
 
 - [Simple Convolution Model](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Convo/SimpleModel/transBleu.txt) - 9.63 	
 - [Advance Convolution Model](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/Convo/AdvanceModel/transBleu.txt) - 11.30
	
When comparing the BLEU scores of the 5 models (3 Attention models + 2 Convolution models) it can be seen that even the Simple Attention Model produces a better BLEU score (12.62) than the Advanced Convolution Model (11.30)
	
	
- What were the obstacles and how did you overcome them?

Same as with the Attention model the main obstacle was understanding the architecture proposed in the paper. But compared to the Attention model understanding this paper was relaively easy as we had an exposure to the concept of attention by the time we started working on this paper and we had previous exposure and experience of the concept of convolutions by having worked with convolutional neural networks.
	
Yet we examined [Tensorflow](https://github.com/tobyyouup/conv_seq2seq), [PyTorch](https://github.com/pytorch/fairseq)  implementations of the paper and watched a [video](https://www.youtube.com/watch?v=iXGFm7oC9TE&t=35s) which explain the architecture to over come the above obstacle.
	
	
- Link to the solution on GitHub (link to the final submission commit) + submission timestamp (time of the GitHub commit)

	- link to the solution (https://github.com/mt2018-tartu-shared-task/project-fishbone/tree/master/Convo)
	- link to the final submission commit (https://github.com/mt2018-tartu-shared-task/project-fishbone/commit/5ecfc8de032b94440b4ffbe8eb50acf2ff93014d)
	- last timestamp on 06 Nov 2018, 09.53 EEST
	
	
- Contribution of each team member (who done what)

	- Preprocessing - Sriyal Jayasinghe
	- Hyperparameter identification - Sophie Gräfnitz, Sriyal Jayasinghe
	- Training of models - Sriyal Jayasinghe
	- translation of dev set: Sophie Gräfnitz, Sriyal Jayasinghe
	- Postprocessing and BLEU score calculation - Maxi Fisher, Sriyal Jayasinghe
	
	

## Section 4: Large-scale experiment
### Section 4.1 Description of Experiment Setup

Out of the five models we trained the Advanced Attention Model had the best BLEU score with a value 15.20 on the small dataset. As a result we chose the attention model as the system for the large-scale experiment and as the hyperparameters of the model we took parameters of the Advanced Attention Model.

 - Below are the datasets we used for the Attention model:
  
    training, test, in-domain dev dataset - [EMEA, Europarl, JRC-Acquis, OpenSubtitles2018 corpora](https://owncloud.ut.ee/owncloud/index.php/s/NjJYeF2SsJpzS4j/download?path=%2F&files=4_separated&downloadStartSecret=2team2ye19m). This folder contains preprocessed train, test and dev sets for each of the four corpora in Estonian and English lanugage, so 24 files. Preprocessed means cleaned, tokenized, shuffled and separated files. We concatenated each corresponding files (train.en of each corpora, train.et of each corpora etc.) via Unix pipe. 
    The training set of the EMEA corpora consists of 906390 sentences, Europarl 644774 sentences, JRC-Acquis 679603 sentences and Opensubtitles2018 3000000 sentences per language. Concatenated this results in 5230767 sentences per language (denoted by "combined.train.e*").
    The test set (we used for validating our results during training (early stopping)) consists of 500 sentences per corpora per language, so 2000 sentences concatenated per language (denoted by "combined.dev.e*").
    The dev set (we used for validating our results after training to retrieve BLEU score) consists of 1000 sentences per corpora per language, so 4000 sentences concatenated per language (denoted by "combined.test.e*").
    
    Out-of-domain dev dataset - [English dev set](https://github.com/mt2018-tartu-shared-task/practicum2/blob/master/dev.en) and [Estonian dev set](https://github.com/mt2018-tartu-shared-task/practicum2/blob/master/dev.et). Each of these files had 2000 sentences.
    
  - As the corpora were already preprocessed, the only preprocessing necessary was the concatenation by Unix pipe operator (cat file1 file2 > combinedfile).
   
  - The hyperparameters are the same as the ones of the Advanced Model of section 3. We trained the model for 4-5 days.
   
   	- [Hyperparameters](hhttps://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/FinalModel/Train.sbatch) of the final model:
	
  	```
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
	```
### Section 4.2 Results
- BLEU Result (on dev and test set)

	Below are the BLEU score values we obtained on the in-domain and out-of-domain English dev set for the final model

	- [In-Domain](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/FinalModel/final_bleu_score.txt) - 36.17	
	- [Out-Of-Domain](https://github.com/mt2018-tartu-shared-task/project-fishbone/blob/master/FinalModel/pract2_dev_bleu_score.txt) - 14.45
		
	You can see that our in-domain dev set performance is extremely good, while the out-of-domain dev set performance is actually worse than in the original training exclusively on the Europarl data. That can be explained by the additional corpora provided, the biggest part of it is informal OpenSubtitles data. That might lead to different translations than only considering the more formal Europarl data. Additionally the speeches in the European parlament are closer to political news depicted in the Out-Of-Domain dataset.


Our task was to compare the translation quality of the out-of-domain set of our final model and the translation performed by [TartuNLP Translator](http://neurotolge.ee/) with polite style

- Conclusion about manual evaluation

	- in general our model most times provides a fluent translation of the out-of-domain data set
	- often it suffers from content errors, inproper style (too informal), missing or invented words
	- TartuNLP Translator and our translation had problems with the same sentences, but TartuNLP Translator handled them better most of the times
	- problematic sentences were those ones with a complicated grammar structure, long sentences, sentences containing many names
	- we preferred our own model in approximately 21% of the analysed sentences. 
	- we suspect that the TartuNLP Translator has seen much more formal training data and hence performs better at proper name recognition. Also our model was trained on a large proportion of informal speech, so this decreases its quality of a formal text. The word missing and adding probably derive from attention errors made because the attentions were trained on other sentence structures. 
	

- Github Links
	
	- final submission commit (https://github.com/mt2018-tartu-shared-task/project-fishbone/commit/efeb7970cee4fbcbd26e4ba0d2beab59576361a6)
	- only file renamings (https://github.com/mt2018-tartu-shared-task/project-fishbone/commit/bc96a8f12d221b0396b81655a3128d0feb676201)
	- last timestamp on 25 Nov 2018, 19.25 EEST (28 Nov 2018, 11.25 EEST for renamings)
	- final manual evaluation submission commit (https://github.com/mt2018-tartu-shared-task/project-fishbone/commit/83ecac5fe88b8d516ece262b69b5722c1f2f3f01)
	- last timestamp on 29 Nov 2018, 19.44 EEST
	
	
- Contribution of each team member (who done what)

	- Preprocessing - Maxi Fischer, Sophie Gräfnitz
	- Model Training - Maxi Fischer, Sriyal Jayasinghe
	- Postprocessing and BLEU score calculation - Sophie Gräfnitz, Sriyal Jayasinghe
	- Manual Evaluation - Maxi Fischer, Sophie Gräfnitz, Sriyal Jayasinghe


## Section 5: Summary
- Summary of what have you done (based on all sections above)

	We learnt how to use sockeye to train a Machine Translation model on a small dataset. Then we familiarized ourselves with the concept of Attention in Neural Networks and the specific usage in the papers *Attention Is All You Need* and *Convolutional Sequence to Sequence Learning*. We presented our findings to the class, trained a baseline with given parameters and evaluated the results with a manual error analysis. Afterwards we trained three models on three different parameter sets for the Transformer architecture and two different models on two different parameter sets for the Convolutional architecture. Since the Advanced Transformer Model performed the best on the experiment dataset we decided to train a final model using those parameters. The final model was trained on multiple corpora of different domains e.g. OpenSubtitles and Europarl. We made a comparative error analysis on this model in comparison to TartuNLP Translator
	
	
- Summary of what were the main obstacles about (based on all sections above)

	We did not have any major issues. In the beginning (Practicum 2) we accidentally trained a model on a uncleaned data set causing us to have a very low BLEU score for that model. Additionally in some of the later models the post-processing posed some problems because we forgot to delete the header and footer of the translated dataset. Sometimes the HPC cluster didn't let us run our jobs immediately causing a delay in training the models. The hardest part was actually understanding  the architectures proposed on the two papers and identifying which parameters to adopt in sockeye.
	

- Summary of main results (based on all sections above)

	The baseline had a BLEU score of 9.83 with already somewhat fluent sentences, but major issues with long sentences and proper names.
	In preparation of the midterm presentation we learnt that the Transformer architecture is the state-of-art architecture in Machine Translation. In the experiments the Transformer architecture outperformed the Convolutional architecture in every model with the best convolutional model having a BLEU score of 11.30 while the worst Transformer model starts at a BLEU score of 12.52, performing best on the advanced model with a BLEU score of 15.20 on the same dataset.
	The final model trained on a bigger corpus achieved a BLEU score of 36.17 on the in-domain dataset and 14.45 on the out-of-domain dataset. This model still had issues with long sentences, many proper names and difficult grammar structures, but was preferred in 21% of the cases over the TartuNLP Translator system.
	

- If you had a chance to do one more Machine Translation experiment, what would it be? 

    Results published in [*Sockeye: A Toolkit for Neural Machine Translation](https://arxiv.org/pdf/1712.05690.pdf) shows an interesting case where a RNN-LSTM model had managed to produce better BLEU score for English to German translation and almost similar BLEU score for Latvian to English translation when compared against the convolution model. Since our group had two German native speakers we could have tried a RNN-LSTM model to validate the above mentioned published  results or we could have tried a RNN-LSTM model on the datasets that was used for this project and perfom a comparison of that model as well.
