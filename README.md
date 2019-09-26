# Face GAN #

This repo includes the implementation on pytorch of a fully deep convolutional generative adversial network,
that tries to create realistic faces of the CelebA datastet. The architecture and model
hyperparameters are based on the paper
"Unsupervised representation learning with DCGANs" (https://arxiv.org/abs/1511.06434)
<br>
<br>
It includes three different models based on the same architecture:
* DCGAN
* WGAN
* WGAN-GP
<br>
The second based on the paper "Wasserstein GAN" (https://arxiv.org/abs/1701.07875)
and the third "Improved Training of Wasserstein GANs" (https://arxiv.org/abs/1704.00028)

### Generated images sample ###
Sample image of the DCGAN trained on 23 epochs
![Alt text](images/gan_sample_0.png?raw=true "Title")
Sample image of the WGAN trained on 215000 epochs 
![Alt text](images/wgan_sample_0.png?raw=true "Title")
Here I made vector arithmetic with the latent vectors of the generated images to make a neutral man smile
![Alt text](images/face_vector.png?raw=true "Title")


### Special Notes ###
This repository is currently in progress, thus the networks are not giving their full potential 

### Where can I get the dataset? ###
The dataset is available at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### Who do I talk to? ###

Pol Monroig
