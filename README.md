# SRN-DeblurNet

- For training,I strictly follow all configuration of the original paper.
- To test PSNR, I split test set of GOPRO datset into two half, one for validation and the other for test, and get **29.58db** PSNR (**30.26db** reported in the original paper).
- A pretrained model is provided. It is trained on GOPRO's blurry images **without** gamma correction.[GOPRO dataset](https://github.com/SeungjunNah/DeepDeblur_release)
- I tryied to ultize this model to augment real world face images, but found it doesn't generalize well.
- Any discussion or correction is welcomed.

## Reference ##
- [Scale-recurrent Network for Deep Image Deblurring](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tao_Scale-Recurrent_Network_for_CVPR_2018_paper.pdf)
- [Deep Multi-scale CNN for Dynamic Scene Deblurring](http://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)
