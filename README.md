## Later I will complete the README.md
# A Implementation with Dilation Gate CNN.

## Requirements
* python==3.x (Let's move on to python 3 if you still use python 2)
* tensorflow==1.12.0
* tqdm>=4.28.1

## Model Structure
This model is come from JianLin Su. This is this model [blog](https://spaces.ac.cn/archives/5409) from him. Thanks for him of give him idea public, 
and I add bert to this model, just use pretrain bert vector, so the vocab is from bert, After I add bert to this model, the GPU memory spending is so 
large, if u want to train this model, to be sure you have large model training environment.

### Structure
<img src="fig/structure.png">

## Training
You can use WebQA to train this model, or you want to change the dataset to yours, change the way of load data in data_load.py
* Run
```
python train.py --logdir myLog --batch_size 32 --train myTrain --bert_pre bertPreTrain
```