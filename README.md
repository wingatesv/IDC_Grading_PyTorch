# IDC_Grading_PyTorch

This repo contains the source code for Automated IDC Grading System in Pytorch using the FBCG dataset

## Citation
If you find our code useful, please consider citing our work using the bibtex:
```
@article{
voon2022performance,
title={Performance analysis of seven Convolutional Neural Networks (CNNs) with transfer learning for Invasive Ductal Carcinoma (IDC) grading in breast histopathological images},
author={Voon, Wingates and Hum, Yan Chai and Tee, Yee Kai and Yap, Wun-She and Salim, Maheza Irna Mohamad and Tan, Tian Swee and Mokayed, Hamam and Lai, Khin Wee},
journal={Scientific Reports},
volume={12},
number={1},
pages={19200},
year={2022},
month=11,
day=10,
issn={2045-2322},
url={https://doi.org/10.1038/s41598-022-21848-3},
doi={10.1038/s41598-022-21848-3},
ID={Voon2022}
}
```

## Enviroment
 - Google Colab
 - Google Drive
 - Python3
 - [Pytorch](http://pytorch.org/) 

## Getting started
### Clone the Repo
* Clone the repo into your Google Colab working directory
<pre>
!git clone https://github.com/wingatesv/IDC_Grading_Pytorch.git
</pre>

### Datasets Download
* Please contact the author for more information: wingatesvoon@1utar.my

| FBCG Class  | Number of Images|
|-------------|-----------------|
| Grade 0     |       588       |
| Grade 1     |       98        |
| Grade 2     |       102       |
| Grade 3     |       82        |


## Train
Run
```python ./train.py --feature_extractor [BACKBONENAME]  [--OPTIONARG]```

For example, run `python ./train.py --feature_extractor resnet50 --batch_size 16 --temp Temp1 --train_aug --sn reinhard`  
Commands below follow this example, and please refer to io_utils.py for additional options.


## Test
Run
```python ./test.py  --feature_extractor resnet50 --batch_size 16 --temp Temp1 --train_aug --sn reinhard```
