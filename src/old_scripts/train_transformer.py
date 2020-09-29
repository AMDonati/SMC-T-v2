import argparse
from data_provider.datasets import Dataset, CovidDataset
from algos.run_SMC_T import algos

if __name__ == '__main__':

  #trick for boolean parser args.
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
  parser.add_argument("-bs", type=int, default=128, help="batch size")
  parser.add_argument("-ep", type=int, default=5, help="number of epochs")
  parser.add_argument("-dff", type=int, default=24, help="dimension of feed-forward network")
  parser.add_argument("-pe", type=int, default=50, help="maximum positional encoding")
  parser.add_argument("-full_model", type=str2bool, default=False, help="full_model = ffn & layernorm")
  parser.add_argument("-data_path", type=str, required=True, help="path for saving data")
  parser.add_argument("-output_path", type=str, required=True, help="path for output folder")
  parser.add_argument("-dataset", type=str, default='synthetic', help='dataset selection')
  parser.add_argument("-dataset_model", type=int, default=1, help="model 1 or 2 for the synthetic dataset.")

  args = parser.parse_args()

  # ------------------- Upload synthetic dataset ----------------------------------------------------------------------------------

  BATCH_SIZE = args.bs
  if args.dataset == 'synthetic':
      BUFFER_SIZE = 500
      dataset = Dataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE)

  elif args.dataset == 'covid':
      BUFFER_SIZE = 50
      dataset = CovidDataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE)

  algo = algos["baseline_t"](dataset=dataset, args=args)
  algo._train()
  algo.test()

