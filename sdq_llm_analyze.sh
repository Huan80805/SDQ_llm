python sensitivity_analysis.py --model_path checkpoints/spnet/step_1000 --num_samples 500 --output_file analysis/sensitivity_spnet.json
python sensitivity_analysis.py --model_path checkpoints/cyclic/step_1000 --num_samples 500 --output_file analysis/sensitivity_cyclic.json
python sensitivity_analysis.py --model_path checkpoints/instantnet/step_1000 --num_samples 500 --output_file analysis/sensitivity_instantnet.json
python sensitivity_analysis.py --model_path checkpoints/instantnet_kd/step_1000 --num_samples 500 --output_file analysis/sensitivity_instantnet_kd.json
