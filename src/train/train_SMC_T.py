import tensorflow as tf
import numpy as np
import os, argparse
from preprocessing.time_series.df_to_dataset_synthetic import split_input_target, data_to_dataset_4D
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from train.train_functions import train_SMC_transformer
from utils.utils_train import create_logger, CustomSchedule
from data_provider.datasets import Dataset, CovidDataset
from algos.run_SMC_T import algos

# TODO: add cross_validation option here. See train / val / test / data needs to be updated.

if __name__ == '__main__':

    #  trick for boolean parser args.
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser()

    parser.add_argument("-d_model", type=int, default=6, help="depth of attention parameters")
    parser.add_argument("-bs", type=int, default=32, help="batch size")
    parser.add_argument("-ep", type=int, default=5, help="number of epochs")
    parser.add_argument("-full_model", type=str2bool, default=True,
                        help="simple transformer or one with ffn and layer norm")
    parser.add_argument("-dff", type=int, default=0, help="dimension of feed-forward network")
    parser.add_argument("-attn_w", type=int, default=None, help="attn window")
    parser.add_argument("-particles", type=int, default=1, help="number of particles")
    parser.add_argument("-sigmas", type=float, default=0.5, help="values for sigma_k, sigma_q, sigma_v, sigma_z")
    parser.add_argument("-sigma_obs", type=float, default=0.5, help="values for sigma obs")
    parser.add_argument("-smc", type=str2bool, required=True, help="Recurrent Transformer with or without smc algo")
    parser.add_argument("-dataset", type=str, default='synthetic', help='dataset selection')
    parser.add_argument("-dataset_model", type=int, default=1, help="model 1 or 2 for the synthetic dataset.")
    parser.add_argument("-data_path", type=str, required=True, help="path for uploading the dataset")
    parser.add_argument("-output_path", type=str, required=True, help="path for output folder")
    parser.add_argument("-cv", type=int, default=0, help="do cross-validation training or not.")
    args = parser.parse_args()

    if not args.smc:
        assert args.particles == (1 or None)

    # -------------------------------- Upload dataset ----------------------------------------------------------------------------------
    BATCH_SIZE = args.bs

    if args.dataset == 'synthetic':
        BUFFER_SIZE = 500
        dataset = Dataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE)

    elif args.dataset == 'covid':
        BUFFER_SIZE = 50
        dataset = CovidDataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE)

    algo = algos["smc_t"](dataset=dataset, args=args)

    algo.train()
    algo.test()


    # for (inp, tar) in train_dataset.take(1):
    #     print('input example', inp[0])
    #     print('target example', tar[0])
    #     #TODO: add an assert here checking that the 2 are the same shifted from one...

