from __future__ import print_function

import argparse

from torch.utils.data import DataLoader

from srgan_solver import SRGANTrainer
from dataset.data import get_training_set, get_test_set

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='Super Resolution Project')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")

parser.add_argument('--image_dir', required=True, help='directory containing training data set')

args = parser.parse_args()

def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('===> Loading datasets')
    train_set = get_training_set(args.upscale_factor, args.image_dir)
    test_set  = get_test_set(args.upscale_factor, args.image_dir)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    testing_data_loader  = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)

    # ===========================================================
    # Generate Model from training data set
    # ===========================================================
    model = SRGANTrainer(args, training_data_loader, testing_data_loader)

    model.run()

if __name__ == '__main__':
    main()
