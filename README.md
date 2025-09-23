# RefDiv

Install dependencies from `requirements.txt` and add all paths of the models in the `config.py` file first.

## Run GCG

Deploy the model in vLLM before running the script.

**Best-of-N:**
```bash
python3 check_gcg_bon.py --model "vllm-model-name" --suffix "suffix"
```

**MCTS:**
```bash
python3 check_gcg_mcts.py --model "vllm-model-name" --suffix "suffix"
```

## Run AutoDAN

Two GPUs are required for this experiment. GPU-0 loads the model in vLLM and GPU-1 is used for other models.

**Best-of-N (n=8) with PairRM:**
```bash
python3 Refdif_eval_bon8.py --batch_size 32 --num_steps 25 --device 1 --model llama3 --save_suffix PairRM-base-8-new --algorithm autodan --seed 3 2>&1 | tee logs/llama3-8b-PairRM-base-8-new.txt
```

For `n=2` and `n=16`, run `Refdif_eval_bon2.py` and `Refdif_eval_bon16.py` respectively.

**MCTS:**
Ensure the model is deployed on vLLM server with the same name as mentioned in `config.py`.
```bash
python3 RefDiv_eval_mcts_base.py --batch_size 32 --num_steps 25 --device 1 --model llama3 --save_suffix PairRM-base-8-new-mcts-Exp2-2 --algorithm autodan --seed 6 2>&1 | tee logs/llama3-8b-PairRM-base-8-new-mcts-Exp2-2.txt
```

## Run RefDiv

Same as running AutoDAN but set `algorithm=diversity`.

**Best-of-N:**
```bash
python3 Refdif_eval_bon8.py --batch_size 32 --num_steps 25 --device 1 --model llama3 --save_suffix PairRM-base-8-new --algorithm diversity --seed 3 2>&1 | tee logs/llama3-8b-PairRM-base-8-new.txt
```

For `n=2` and `n=16`, run `Refdif_eval_bon2.py` and `Refdif_eval_bon16.py` respectively.

**MCTS:**
```bash
python3 RefDiv_eval_mcts_base.py --batch_size 32 --num_steps 25 --device 1 --model llama3 --save_suffix PairRM-base-8-new-mcts-Exp2-2 --algorithm diversity --seed 6 2>&1 | tee logs/llama3-8b-PairRM-base-8-new-mcts-Exp2-2.txt
```