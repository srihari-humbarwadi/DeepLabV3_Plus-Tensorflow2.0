[![HitCount](http://hits.dwyl.io/srihari-humbarwadi/https://githubcom/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow20.svg)](http://hits.dwyl.io/srihari-humbarwadi/https://githubcom/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow20)


## Architecture
![model](deeplabv3plus.png)

<a href="gifs/stuttgart_02.gif " target="_blank"><img 
src="gifs/stuttgart_02.gif" alt="stuttgart_02" width="430" height="215" 
border="10" /></a>
<a href="gifs/stuttgart_00.gif " target="_blank"><img 
src="gifs/stuttgart_00.gif" alt="stuttgart_00" width="430" height="215" 
border="10" /></a>

## Trained weights
[trained weights](https://drive.google.com/open?id=1wRXyIGUVRws3BJHX-UrNDSZGDzUzgVMx)


## Outputs
![](outputs/frankfurt_000001_058914_leftImg8bit.png)
![](outputs/frankfurt_000000_014480_leftImg8bit.png)
![](outputs/munster_000114_000019_leftImg8bit.png)
![](outputs/munster_000129_000019_leftImg8bit.png) 
![](outputs/munster_000031_000019_leftImg8bit.png)

## To Do
- [x] Implement distributed training using tf.distribute.MirroredStrategy
- [x] Implement data input pipeline using tf.data.Dataset 
- [x] Train on cityscapes
- [ ] Implement modified Xception backbone as originally mentioned in the paper
