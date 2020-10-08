#!/usr/bin/env bash
python src/plots/plot.py -data_path "data/synthetic_model_1" -output_path "output/plots/ci_plots/synthetic_model_1" -smc "output/exp_synthetic_model_1/smc_t/smc_t_d32_30p_cv0/1/inference_results" \
-lstm "output/exp_synthetic_model_1/lstm/lstm_d32_p0.2/1/inference_results" \
-transf "output/exp_synthetic_model_1/baseline_t/baseline_t_d16_p0.2/1/inference_results" \
-bayes "output/exp_synthetic_model_1/bayesian_lstm_d32/1/inference_results" \
-captions "output/plots/ci_plots/synthetic_model_1/captions_p0.1.json"
