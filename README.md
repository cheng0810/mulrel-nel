mulrel-nel: Multi-relational Named Entity Linking
========

A Python implementation of Multi-relatonal Named Entity Linking described in 

[1] Phong Le and Ivan Titov (2018). [Improving Entity Linking by 
Modeling Latent Relations between Mentions](https://arxiv.org/pdf/1804.10637.pdf). ACL 2018.

Written and maintained by Phong Le (ple [at] exseed.ed.ac.uk )


### Installation

- Requirements: Python 3.5 or 3.6, Pytorch 0.3, CUDA 7.5 or 8

### Usage

The following instruction is for replicating the experiments reported in [1]. 


#### Data

Download data from [here](https://drive.google.com/open?id=1IDjXFnNnHf__MO5j_onw4YwR97oS8lAy) 
and unzip to the main folder (i.e. your-path/mulrel-nel).

```
Due to the copyright of the Tackbp 2015 files([LDC2015E103: TAC KBP 2015 Tri-Lingual Entity Discovery and Linking Evaluation Gold Standard Entity Mentions and Knowledge Base Links](https://tac.nist.gov//2015/KBP/data.html)), we only give the sample of Chinese data.
```

## English version

### Train

cd your path to main folder which contain nel and data folder.
To train a 3-relation ment-norm model, from the main folder run
```
python -u -m nel.main --mode train --n_rels 3 --mulrel_type ment-norm --model_path model_en --language en
```
Preprocess pickle is saved in `nel/preprocessing/`.
The output is a model saved in two files: `model_en.config` and `model_en.state_dict` .

### Evaluation

Execute
```
python -u -m nel.main --mode eval --model_path model_en
```
I got 93.24 accuracy on the test dataset(AIDA-B).

## Chinese version

### Train

cd your path to main folder which contain nel and data folder.
To train a 3-relation ment-norm model, from the main folder run
```
python -u -m nel.main --mode train --n_rels 3 --mulrel_type ment-norm --model_path model_zh --language zh --dev_f1_change_lr 0.88
```
Preprocess pickle is saved in `nel/preprocessing/`.
The output is a model saved in two files: `model_zh.config` and `model_ezh.state_dict` .

### Evaluation

Execute
```
python -u -m nel.main --mode eval --model_path model_zh --language zh
```
I got 87.64 accuracy on the test dataset(tackbp2015_eval).

