import argparse
from src.data_provider.datasets import Dataset, CovidDataset
from src.algos.run_rnn import RNNAlgo
from src.algos.run_baseline_T import BaselineTAlgo
from src.algos.run_SMC_T import SMCTAlgo

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

    algos = {"smc_t": SMCTAlgo, "lstm": RNNAlgo, "baseline_t": BaselineTAlgo}

    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-dataset", type=str, default='synthetic', help='dataset selection')
    parser.add_argument("-dataset_model", type=int, default=1, help="model 1 or 2 for the synthetic dataset.")
    parser.add_argument("-data_path", type=str, required=True, help="path for uploading the dataset")
    parser.add_argument("-cv", type=int, default=0, help="do cross-validation training or not.")
    # model parameters:
    parser.add_argument("-algo", type=str, required=True, help="choose between SMC-T(smc_t), Baseline-T(baseline_t), and LSTM algo(lstm)")
    parser.add_argument("-d_model", type=int, default=8, help="depth of attention parameters")
    parser.add_argument("-full_model", type=str2bool, default=True,
                        help="simple transformer or one with ffn and layer norm")
    parser.add_argument("-dff", type=int, default=8, help="dimension of feed-forward network")
    parser.add_argument("-pe", type=int, default=50, help="maximum positional encoding")
    parser.add_argument("-attn_w", type=int, default=None, help="attn window")
    parser.add_argument("-rnn_units", type=int, default=8, help="number of rnn units")
    parser.add_argument("-p_drop", type=float, default=0., help="dropout on output layer")
    parser.add_argument("-rnn_drop", type=float, default=0.0, help="dropout on rnn layer")
    # training params.
    parser.add_argument("-bs", type=int, default=32, help="batch size")
    parser.add_argument("-ep", type=int, default=1, help="number of epochs")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    # smc params.
    parser.add_argument("-particles", type=int, default=1, help="number of particles")
    parser.add_argument("-sigmas", type=float, default=0.5, help="values for sigma_k, sigma_q, sigma_v, sigma_z")
    parser.add_argument("-sigma_obs", type=float, default=0.5, help="values for sigma obs")
    parser.add_argument("-smc", type=str2bool, default=False, help="Recurrent Transformer with or without smc algo")
    # output_path params.
    parser.add_argument("-output_path", type=str, required=True, help="path for output folder")
    parser.add_argument("-save_path", type=str, help="path for saved model folder (if loading ckpt)")
    # inference params.
    parser.add_argument("-past_len", type=int, default=10, help="number of timesteps for past timesteps at inference")
    parser.add_argument("-inference", type=int, default=0, help="launch inference or not on test data.")
    parser.add_argument("-multistep", type=str2bool, default=False, help="doing multistep inference or not.")
    parser.add_argument("-mc_samples", type=int, default=100, help="number of samples for MC Dropout algo.")
    args = parser.parse_args()

    if not args.smc:
        assert args.particles == (1 or None)

    list_samples = [72, 2]

    # -------------------------------- Upload dataset ----------------------------------------------------------------------------------

    if args.dataset == 'synthetic':
        BUFFER_SIZE = 500
        dataset = Dataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=args.bs)

    elif args.dataset == 'covid':
        BUFFER_SIZE = 50
        dataset = CovidDataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=args.bs)

    algo = algos[args.algo](dataset=dataset, args=args)
    if args.ep > 0:
        algo.train()
    else:
        print("skipping training...")
    algo.test()
    if args.inference:
        algo.launch_inference(list_samples=list_samples, multistep=args.multistep)


    # for (inp, tar) in train_dataset.take(1):
    #     print('input example', inp[0])
    #     print('target example', tar[0])
    #     #TODO: add an assert here checking that the 2 are the same shifted from one...

