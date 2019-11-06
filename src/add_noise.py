from __future__ import print_function

import numpy as np
import os
import cv2
import argparse
import glob

'''
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
'''
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "snp":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
  elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
  elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

parser = argparse.ArgumentParser(description='noise addition script')
# hyper-parameters
parser.add_argument('--noise_type', required=True, help='Type of noise to add, [gauss, poisson, s&p, speckle]')
parser.add_argument('--input_dir', required=True, help='input image directory')
parser.add_argument('--output_dir', required=True, help='output image directory')

args = parser.parse_args()

def main():
    for f in glob.glob(args.input_dir + "/*.jpg"):
        input_img = cv2.imread(f, 0)
        img_name = os.path.splitext(os.path.basename(f))[0]
        for noise_type in  ['gauss', 'poisson', 'snp', 'speckle']:
            noise_img = noisy(noise_type, input_img)
            cv2.imwrite(args.output_dir + "/" + img_name + '_' +  noise_type + '.jpg', noise_img)
if __name__ == "__main__":
    main()