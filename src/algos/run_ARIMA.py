import os
from src.algos.generic import Algo
import numpy as np
import datetime
from statsmodels.tsa.arima.model import ARIMA

class ARIMA(Algo):
    def __init__(self, dataset, args):
        super(ARIMA, self).__init__(dataset=dataset, args=args)
        self.d = args.d
        self.q = args.q
        self.out_folder = self._create_out_folder(args=args)
        self.logger = self.create_logger()
        self.distribution = True
        assert self.past_len < self.seq_len, "past_len should be inferior to the sequence length of the dataset"
        self.future_len = args.future_len if args.future_len is not None else (self.seq_len - self.past_len)

    def _create_out_folder(self, args):
        if args.save_path is not None:
            return args.save_path
        else:
            output_path = args.output_path
            out_file = '{}_d{}q{}'.format(args.algo, args.d, args.q)
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            output_folder = os.path.join(output_path, out_file, datetime_folder)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            return output_folder
        