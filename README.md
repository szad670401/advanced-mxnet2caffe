# Advanced-Mxnet2Caffe

### Operator Support Lists

- Convolution
- ChannelwiseConvolution
- BatchNorm
- Activation
- ElementWiseSum
- _Plus
- Concat
- Crop
- Pooling
- Flatten
- FullyConnected
- SoftmaxOutput&SoftmaxFocalOutput
- SoftmaxActivation
- LeakyReLU
- elemwise_add
- UpSampling
- Deconvolution
- Clip
- Reshape

### Tested models
+ Mxnet-SSH
+ MobileNet-V2
+ Resnet-50
+ RetinaFace with ( Resnet-50 ,Mobilenet 0.25 backbone)
+ All models from Insightface Model Zoo .

### Note&Bugs

The convertor Is not fully automatically, The convertor not 

+ if you wanna convert  upsampling operator , the convertor will convert Upsampling operator to Deconvolution in Caffe , The Deconvolution channels need to be set (in prototxt_basic.py names_output).
+ If you use Flatten Layer ,You need to manually to connect them becasuse the converted compute graph will be divided into two parts.
+ If convert a detection model. You need to remove the anchor process and put it into post process.
+ Usually,If you find that conversion errors, please set the prefix name of you backbone network in mxnet2caffe.py.