import argparse
from datasets import load_from_disk
from src.data_provider.class_datasets import Dataset
from src.data_provider.sst_sentiment import SSTDataset
from src.data_provider.CLEVRDataset import QuestionsDataset
from src.data_provider.sst_tokenizer import SSTTokenizer
from src.algos.run_rnn import RNNAlgo
from src.algos.run_baseline_T import BaselineTAlgo
from src.algos.run_SMC_T import SMCTAlgo
from src.algos.run_fivo import FIVOAlgo
from src.algos.run_Bayesian_rnn import BayesianRNNAlgo
from transformers import GPT2Tokenizer

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
    parser.add_argument("-dataset", type=str, default='dummy_nlp', help='dataset selection')
    parser.add_argument("-data_path", type=str, required=True, help="path for uploading the dataset")
    parser.add_argument("-max_samples", type=int, default=10000, help="max samples for train dataset")
    parser.add_argument("-min_token_count", type=int, default=1, help="min token count for sst vocabulary.")
    parser.add_argument("-max_seq_len", type=int, default=30, help="max seq len for ")
    # model parameters:
    parser.add_argument("-algo", type=str, required=True,
                        help="choose between SMC-T(smc_t), Baseline-T(baseline_t), and LSTM algo(lstm), Bayesian LSTM (bayesian_lstm)")
    parser.add_argument("-num_layers", type=int, default=1, help="number of layers in the network. If == 0, corresponds to adding GPT2Decoder.")
    parser.add_argument("-reduce_gpt2output", type=int, default=0, help="when using GPT2decoder, reduce or not to d_model the gpt2output of dim 768.")
    parser.add_argument("-num_heads", type=int, default=1, help="number of attention heads for Transformer networks")
    parser.add_argument("-d_model", type=int, default=8, help="depth of attention parameters")
    parser.add_argument("-full_model", type=str2bool, default=True,
                        help="simple transformer or one with ffn and layer norm")
    parser.add_argument("-dff", type=int, default=8, help="dimension of feed-forward network")
    parser.add_argument("-pe", type=int, default=50, help="maximum positional encoding")
    parser.add_argument("-attn_w", type=int, default=None, help="attn window")
    parser.add_argument("-rnn_units", type=int, default=8, help="number of rnn units")
    parser.add_argument("-p_drop", type=float, default=0., help="dropout on output layer")
    parser.add_argument("-rnn_drop", type=float, default=0.0, help="dropout on rnn layer")
    # Bayesian LSTM.
    parser.add_argument("-prior_sigma_1", type=float, default=0.1, help="prior sigma param for Bayesian LSTM.")
    parser.add_argument("-prior_sigma_2", type=float, default=0.002, help="prior sigma param for Bayesian LSTM.")
    parser.add_argument("-prior_pi", type=float, default=1.0, help="prior pi param for Bayesian LSTM.")
    parser.add_argument("-posterior_rho", type=float, default=-6.0, help="posterior rho init param for Bayesian LSTM.")
    # training params.
    parser.add_argument("-bs", type=int, default=128, help="batch size")
    parser.add_argument("-ep", type=int, default=1, help="number of epochs")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    # smc params.
    parser.add_argument("-particles", type=int, default=1, help="number of particles")
    parser.add_argument("-sigmas", type=float, default=0.5,
                                                        help="values for sigma_k, sigma_q, sigma_v, sigma_z")
    parser.add_argument("-noise_dim", type=str, default="uni", help="1D noise or multi-dim noise.")
    parser.add_argument("-smc", type=str2bool, default=False, help="Recurrent Transformer with or without smc algo")
    parser.add_argument("-EM_param", type=float, help="if not None, use an EM to learn the noise instead of SGD with learning rate equal to EM_param.")
    parser.add_argument("-inference_resample", type=int, default=0,
                        help="resampling particles at inference or not.")
    # output_path params.
    parser.add_argument("-output_path", type=str, required=True, help="path for output folder")
    parser.add_argument("-save_path", type=str, help="path for saved model folder (if loading ckpt)")
    # inference params.
    parser.add_argument("-past_len", type=int, default=4, help="number of timesteps for past timesteps at inference")
    parser.add_argument("-future_len", type=int, default=5, help="number of predicted timesteps for multistep forecast.")
    parser.add_argument("-mc_samples", type=int, default=1, help="number of samples for MC Dropout algo.")
    parser.add_argument("-test_samples", type=int, help="number of test samples.")
    # misc:
    parser.add_argument("-save_distrib", type=str2bool, default=False, help="save predictive distribution on test set.")
    parser.add_argument("-save_plot", type=str2bool, default=True, help="save plots on test set.")
    parser.add_argument("-save_particles", type=str2bool, default=False, help="save predicted particles on test set.")

    return parser

def run(args):

    # -------------------------------- Upload dataset ----------------------------------------------------------------------------------
    BUFFER_SIZE = 500


    if args.dataset == "dummy_nlp":
        dataset = Dataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=args.bs, name=args.dataset,
                          max_samples=args.max_samples)
    elif args.dataset == "sst":
        dataset = load_from_disk(args.data_path)
        vocab_path = "data/sst/vocab.json" if args.min_token_count == 1 else "data/sst/vocab2.json"
        if args.num_layers >= 1:
            tokenizer = SSTTokenizer(dataset=dataset, vocab_path=vocab_path)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained("cache/gpt2")
        dataset = SSTDataset(tokenizer=tokenizer, batch_size=args.bs, max_samples=args.max_samples, max_seq_len=args.max_seq_len)
    elif args.dataset == "clevr":
        dataset = QuestionsDataset(data_path=args.data_path, batch_size=args.bs, max_samples=args.max_samples, max_seq_len=args.max_seq_len)

    algo = algos[args.algo](dataset=dataset, args=args)

    if args.ep > 0:
        algo.train()
    else:
        print("skipping training...")

    algo.test(test_samples=args.test_samples)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
