# [Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](https://arxiv.org/abs/2510.08592)

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://arxiv.org/abs/2510.08592)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2510.08592)
[![GitHub Stars](https://img.shields.io/github/stars/SKNahin/RefDiv?style=social)](https://github.com/SKNahin/RefDiv/stargazers)

## Setup

Install dependencies and add model paths to `config.py`:

```bash
pip install -r requirements.txt
```

> Most experiments require 2 GPUs — GPU-0 for the vLLM model server, GPU-1 for auxiliary models.

## Running Experiments

All methods share the same scripts. Set `--algorithm autodan` for AutoDAN or `--algorithm diversity` for REFDIV.

**Best-of-N** (use `bon2`, `bon8`, or `bon16` scripts for different N):
```bash
python3 Refdif_eval_bon8.py --batch_size 32 --num_steps 25 --device 1 --model llama3 --save_suffix run1 --algorithm diversity --seed 3
```

**MCTS** (deploy model on vLLM first with name matching `config.py`):
```bash
python3 RefDiv_eval_mcts.py --batch_size 32 --num_steps 25 --device 1 --model llama3 --save_suffix run1 --algorithm diversity --seed 6
```

**GCG baseline:**
```bash
python3 gcg_eval_bon8.py --model "vllm-model-name" --suffix "suffix"   # Best-of-N
python3 gcg_eval_mcts.py --model "vllm-model-name" --suffix "suffix"   # MCTS
```

## Citation

Please cite the paper as follows if you use the code from RefDiv:

```bibtex
@article{nahin2025less,
  title={Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models},
  author={Nahin, Shahriar Kabir and Askari, Hadi and Chen, Muhao and Chhabra, Anshuman},
  journal={arXiv preprint arXiv:2510.08592},
  year={2025}
}
```