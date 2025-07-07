CUDA_VISIBLE_DEVICES=4 python greedy_search.py -m checkpoints/spnet/step_1000 -sf analysis/sensitivity_spnet.json -o analysis/greedy_search_spnet.jsonl
CUDA_VISIBLE_DEVICES=4 python greedy_search.py -m checkpoints/instantnet/step_1000 -sf analysis/sensitivity_instantnet.json -o analysis/greedy_search_instantnet.jsonl
CUDA_VISIBLE_DEVICES=4 python greedy_search.py -m checkpoints/instantnet_kd/step_1000 -sf analysis/sensitivity_instantnet_kd.json -o analysis/greedy_search_instantnet_kd.jsonl
