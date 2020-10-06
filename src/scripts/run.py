import argparse
from src.data_provider.datasets import Dataset, CovidDataset, StandardizedDataset
from src.algos.run_rnn import RNNAlgo
from src.algos.run_baseline_T import BaselineTAlgo
from src.algos.run_SMC_T import SMCTAlgo
from src.algos.run_fivo import FIVOAlgo
from src.algos.run_Bayesian_rnn import BayesianRNNAlgo

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


algos = {"smc_t": SMCTAlgo, "lstm": RNNAlgo, "baseline_t": BaselineTAlgo, "fivo": FIVOAlgo,
         "bayesian_lstm": BayesianRNNAlgo}


def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-dataset", type=str, default='synthetic', help='dataset selection')
    parser.add_argument("-dataset_model", type=int, default=1, help="model 1 or 2 for the synthetic dataset.")
    parser.add_argument("-data_path", type=str, required=True, help="path for uploading the dataset")
    parser.add_argument("-cv", type=int, default=0, help="do cross-validation training or not.")
    parser.add_argument("-alpha", type=float, default=0.8, help="alpha value in synthetic models 1 & 2.")
    parser.add_argument("-beta", type=float, default=0.54, help="beta value for synthetic model 2")
    parser.add_argument('-p', type=float, default=0.7, help="p value for synthetic model 2.")
    parser.add_argument("-standardize", type=str2bool, default=False, help="standardize data for FIVO or not.")
    parser.add_argument("-split_fivo", type=str, default="test", help="dataset to evaluate fivo on.")
    # model parameters:
    parser.add_argument("-algo", type=str, required=True,
                        help="choose between SMC-T(smc_t), Baseline-T(baseline_t), and LSTM algo(lstm)")
    parser.add_argument("-d_model", type=int, default=8, help="depth of attention parameters")
    parser.add_argument("-full_model", type=str2bool, default=True,
                        help="simple transformer or one with ffn and layer norm")
    parser.add_argument("-dff", type=int, default=8, help="dimension of feed-forward network")
    parser.add_argument("-pe", type=int, default=50, help="maximum positional encoding")
    parser.add_argument("-attn_w", type=int, default=None, help="attn window")
    parser.add_argument("-rnn_units", type=int, default=8, help="number of rnn units")
    parser.add_argument("-p_drop", type=float, default=0., help="dropout on output layer")
    parser.add_argument("-rnn_drop", type=float, default=0.0, help="dropout on rnn layer")
    parser.add_argument("-prior_sigma_1", type=float, default=0.1, help="prior sigma param for Bayesian LSTM.")
    parser.add_argument("-prior_sigma_2", type=float, default=0.002, help="prior sigma param for Bayesian LSTM.")
    parser.add_argument("-prior_pi", type=float, default=1.0, help="prior pi param for Bayesian LSTM.")
    parser.add_argument("-posterior_rho", type=float, default=-6.0, help="posterior rho init param for Bayesian LSTM.")
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
    parser.add_argument("-future_len", type=int, help="number of predicted timesteps for multistep forecast.")
    parser.add_argument("-inference", type=int, default=0, help="launch inference or not on test data.")
    parser.add_argument("-multistep", type=str2bool, default=False, help="doing multistep inference or not.")
    parser.add_argument("-mc_samples", type=int, default=100, help="number of samples for MC Dropout algo.")
    # misc:
    parser.add_argument("-save_distrib", type=str2bool, default=True, help="save predictive distribution on test set.")
    parser.add_argument("-save_plot", type=str2bool, default=True, help="save plots on test set.")
    parser.add_argument("-save_particles", type=str2bool, default=True, help="save predicted particles on test set.")

    return parser

def run(args):

    # -------------------------------- Upload dataset ----------------------------------------------------------------------------------
    BUFFER_SIZE = 500
    if args.dataset == 'synthetic':
        dataset = Dataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=args.bs, name=args.dataset,
                          model=args.dataset_model)

    elif args.dataset == 'covid':
        BUFFER_SIZE = 50
        dataset = CovidDataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=args.bs, name=args.dataset,
                               model=None)
    elif args.dataset == 'air_quality':
        dataset = StandardizedDataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=args.bs,
                                      name=args.dataset, target_features=list(range(5)))

    elif args.dataset == 'energy':
        dataset = StandardizedDataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=args.bs,
                                      name=args.dataset, target_features=list(range(20)))

    elif args.dataset == 'weather':
        BUFFER_SIZE = 5000
        dataset = StandardizedDataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=args.bs,
                                      name=args.dataset)

    algo = algos[args.algo](dataset=dataset, args=args)

    if args.ep > 0:
        algo.train()
    else:
        print("skipping training...")

    if not args.cv:
        _ = algo.test(alpha=args.alpha, beta=args.beta, p=args.p, multistep=args.multistep,
                                 save_particles=args.save_particles, plot=args.save_plot,
                                 save_distrib=args.save_distrib, save_metrics=True)
    else:
        _ = algo.test_cv(alpha=args.alpha, beta=args.beta, p=args.p, multistep=args.multistep)

    # if args.inference:
    #     algo.launch_inference(list_samples=list_samples, multistep=args.multistep, alpha=args.alpha, beta=args.beta,
    #                           p=args.p)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
