# GenderID-ConvNet
Convolutional Neural Network used to predict gender from human headshots

## Summary
I wanted to build a model to infer gender from images. By fine-tuning the pretrained convolutional neural network VGG16, and 
training it on images of celebrities, I was able to obtain over 98% accuracy on the test set. The exercise demonstrates the utility 
of engineering the architecture of pretrained models to complement the characteristics of the dataset.

## Task
Typically, a human can distinguish a man and a woman in the photo above with ease, but it’s hard to describe exactly why we can make 
that decision. Without defined features, this distinction becomes very difficult for traditional machine learning approaches. Additionally,
features that are relevant to the task are not expressed in the exact same way every time, every person looks a little different. Deep 
learning algorithms offer a way to process information without predefined features, and make accurate predictions despite variation in how
features are expressed. In this article, we’ll apply a convolutional neural network to images of celebrities with the purpose of predicting
gender. (Disclaimer: the author understands appearance does not have a causative relationship with gender)

## Tool
Convolution neural networks (ConvNets) offer a means to make predictions from raw images. A hallmark of the algorithm is the ability to
reduce the dimensionality of images by using sequences of filters that identify distinguishing features. Additional layers in the model
help us emphasize the strength of often nonlinear relationships between the features identified by the filters and the label assigned to
the image. We can adjust weights associated with the filters and additional layers to minimize the error between the predicted and observed
classifications. Sumit Saha offers a great explanation that is more in-depth: 
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

There are a number of pretrained ConvNets that have been trained to classify a range of images of anything from planes to corgis. 
We can save computation time and overcome some sampling inadequacy by employing the weights of pretrained models and fine-tuning them for
our purpose.

## Dataset
The CelebA dataset contains over 200K images of celebrities labeled with 20 attributes including gender. The images are from the shoulders
up, so most of the information is in the facial features and hair style.

![fig2](https://github.com/njermain/GenderID-ConvNet/blob/master/GenderIDex.JPG)

Example image available from CelebA

## Modeling

Table 1 below shows the convolutional architecture for VGG16; there are millions of weights for all the convolutions that we can 
choose to either train or keep frozen at the pretrained values. By freezing all the weights of the model, we risk underfitting it 
because the pretrained weights were not specifically estimated for our particular task. In contrast, by training all the weights we 
risk overfitting because the model will begin “memorizing” the training images given the flexibility from high parameterization. We’ll
attempt a compromise by training the last convolutional block:

![fig2](https://github.com/njermain/GenderID-ConvNet/blob/master/Untitled.png)

Table 1: Architecture of VGG16 model after turning final layers on

The first convolutional blocks in the VGG16 models are identifying more general features like lines or blobs, so we want to keep the 
associated weights. The final blocks identify more fine scale features (e.g. angles associated with the wing tip of an airplane), so 
we’ll train those weights given our images of celebrities.

Continue reading at https://towardsdatascience.com/gender-identification-with-deep-learning-ac379f85a790?source=friends_link&sk=6c5a66d3cb52aea7ee570dce2c94a21c



