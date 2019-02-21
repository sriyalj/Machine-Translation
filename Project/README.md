# Practicum 5 - Refinements

By this point, you got the necessary skills needed in order to build and analyze modern
NMT systems. You were guided through the MT systems developing workflow,
and now it is time for you to continue almost by yourself. You have your baseline
trained and saw what kind of mistakes the baseline system does at the specific BLEU value you got.
During the rest of the course, you will work on the MT related project.

Your improvements should be guided by
results of the system analysis. You had a chance to try it out during previous lab
and milestone. However, since we used relatively small data and got relatively small BLEU scores,
most of the project ideas are suitable for this work. You should motivate them anyway and tell what are expected
improvements.

So the next step is to incorporate some refinement based
on the intuition, you got from the error analysis of your baseline and from proposed project ideas.


# Project ideas
The list below is some project ideas you can choose from.
Each project includes links to the papers that introduce it.


If you attend the practie session and have your own idea you want to try, or there is some published research that you really want to implement, then let me know.

Otherwise following are selected projects. They all are cool and nicely doable with Sockeye.

1. AAAAAAA: Multilingual NMT: https://arxiv.org/pdf/1601.01073.pdf + https://www.aclweb.org/anthology/Q/Q17/Q17-1024.pdf
2. AVOCADO: Fully character level NMT (char2char + bpe2char + char2bpe): http://aclweb.org/anthology/Q17-1026 + bpe2char + char2bpe
3. COCKATOO: Incorporating linguistic info: http://www.statmt.org/wmt16/pdf/W16-2209.pdf
4. MT_BMG: Incorporating monolingual data (dummy input + back-translation): https://arxiv.org/pdf/1511.06709.pdf (dummy input + back-translation)
5. FISHBONE: Transformer architecture + convolutional architecture: https://arxiv.org/abs/1706.03762 + https://arxiv.org/abs/1705.03122 
6. [-] Hyperparameters tuning, model ensembling, and checkpoint averaging: https://arxiv.org/abs/1703.03906 + google for model ensembling and checkpoint averaging
7. [-] Decoding with lexical constraints: https://github.com/awslabs/sockeye/tree/master/tutorials/constraints + where to get constraints?
 
# Homework
The homework is to prepare a ***15 minutes*** presentation of the method that relates to your project.
The aboves list summarizes what to include in the presentation (e.g. for project 6 you have to present the paper + two additional conceps of checkpoint averagins and ensembling while for project 5 you should present 2 papers)

The presentations will take place during the next practice session.

***Strict requirement: all the members of the team has to take part in the in-class presentation!***

In addition, you have to submit the presentation to your team's GitHub repo in a form of the .pdf file.
