#!/usr/bin/env bash
python src/plots/plot.py -data_path "data/synthetic_model_1_pytorch4vm" -output_path "output/plots/ci_plots/synthetic_model_1" -smc "output/exp_synthetic_model_1_for_plots/smc_t_d32_30p_cv0/1/inference_results" \
-lstm "output/exp_synthetic_model_1_for_plots/lstm_d32_p0.1/1/inference_results" \
-transf "output/exp_synthetic_model_1_for_plots/baseline_t_d32_p0.1/1/inference_results" \
-bayes "output/exp_synthetic_model_1_for_plots/bayesian_lstm_d32/1/inference_results" \
-captions "output/plots/ci_plots/synthetic_model_1/captions_p0.1.json"
