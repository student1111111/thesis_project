== Intro ==
=== Goal === 
Essentailly map featured from low resolution to high resolution image

=== Approach ===
* CNN are extensively used for  this purpose.
* With deeep learning multiple layers of NN can be stacked to learn different features of LR image channels
* Cleaver use of these features can result in generating High resolutio image

=== GAN for feature learing ===
* Uses unsuprevised deep learning to learn features and generate images.
* It has "Genrator" and "discriminator" networks that try to compete with each other.
* Trainign is does via minimizing error between generator and discriminator.
* IF traning properly can be used to generate High resulution HR image from low resolution one.

=== RCAN ==
Residual,  channel Attention,  Network
1. shallow feature exteaction
    --  sinlge convolution of LR image
2. Residual in residual deep feature extaction
3. Upscaling
4. reconstruction

*** Residual **** 
-- Residual in residual Network Architecture
* This consist of two convolution layers of Network
* one focus on filtering out "abundant" low freqency featues via (Long skip conection LSC)
* High frequency useful feature and help captured by inner resedual network passing through Short skip conection SSC

*** Channel Attention ***
* Other CNN based solution consider each channel of feature space same.
* RCAN tries to weight the feature channel that are useful, by feeding back the convoluion of surrounding space of image to global pool and summing all elements.

Resedual in resedual network + channnel attention help get better results in RCAN.
