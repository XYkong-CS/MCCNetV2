# MCCNetV2
An augmented version of  [MCCNet](https://github.com/diyiiyiii/MCCNet)


## Experiment
### Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 
### Training

### Testing 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [decoder],  [MCC_module](see above)   <br> 
Please download them and put them into the floder  ./experiments/  <br> 
```
python test_video.py  --content_dir input/content/ --style_dir input/style/    --output out
```

### Training  
Traing set is WikiArt collected from [WIKIART](https://www.kaggle.com/c/painter-by-numbers )  <br>  
Testing set is COCO2014  <br>  
```
python train.py --style_dir ../../datasets/Images --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 4
```
