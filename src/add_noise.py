from __future__ import print_function

from PIL import Image
from imgaug import augmenters as iaa

import numpy as np
import os
import cv2
import argparse
import glob

parser = argparse.ArgumentParser(description='noise addition script')
# hyper-parameters
parser.add_argument('--input_dir', required=True, help='input image directory')
parser.add_argument('--output_dir', required=True, help='output image directory')

args = parser.parse_args()

def main():
    for f in glob.glob(args.input_dir + "/*.jpg"):
        input_img = Image.open(f)
        im_arr    = np.asarray(input_img)
        img_name  = os.path.splitext(os.path.basename(f))[0]
        # guassian noise
        aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.1*255)
        gauss_img = aug.augment_image(im_arr)
        
        # poisson noise
        aug = iaa.AdditivePoissonNoise(lam=10.0, per_channel=True)
        pois_img = aug.augment_image(im_arr)

        # salt and pepper noise
        aug = iaa.SaltAndPepper(p=0.05)
        snp_img = aug.augment_image(im_arr)

        cv2.imwrite(args.output_dir + "/" + img_name + '_gauss.jpg', gauss_img)
        cv2.imwrite(args.output_dir + "/" + img_name + '_poisson.jpg', pois_img)
        cv2.imwrite(args.output_dir + "/" + img_name + '_snp.jpg', snp_img)
if __name__ == "__main__":
    main()