# bdsurvive — feasibility study

Does a backdoor planted in a foundation model survive benign, repeated downstream fine-tuning? This repo studies backdoor persistence across multi-generation LoRA adaptation lineages, organized around a mechanism-based taxonomy rather than named attacks.

Scope note: the fine-tuning (LoRA) arm is the active, validated part of this repo. 

Note : Some of the code contain some legacy for KD (I considered out of scope at this stage to narrow SOK axis for the Fine tuning as a first step).

## Research questions

**RQ1** — which backdoor mechanisms survive one real transfer step?

**RQ2** — among survivors, what is their lineage depth (how many generations before extinction)?

**RQ3** — why does a survivor die where it dies (data distribution, dose, adapter-reset structure, or the adaptation operator itself)?

**RQ4** — what property of a mechanism (trigger type, parameter placement, expression/leakage) predicts persistence?

**Status:** RQ1 answered for 2 of 3 mechanisms; RQ2 partial (flat to g=3, depth extension queued); RQ3 has one live candidate (the reset-effect asymmetry) but no extinction point yet to explain for the validated mechanisms; RQ4 not yet started (needs more mechanism diversity than currently exists).

## Setup
``
pip install -r requirements.txt
``
Requires a GPU (all pilots were run on Lightning AI Studios, L4/A10G class). Set HF_TOKEN in the environment to avoid HuggingFace rate limits on dataset/model downloads.
## Repo layout
 
```
bdsurvive/
  config.py           -> dataclasses: PlantConfig, DeriveConfig, CleanRefConfig, Paths
  data.py             -> content-trigger injection (rare/syntactic/semantic), load_task
  positional.py       -> MetaBackdoor length-trigger mechanism (separate from data.py
                         because it needs the tokenizer, not just string ops)
  plant.py            -> the planting procedure (produces a "parent" checkpoint)
  shards.py           -> deterministic disjoint data partitioning for lineages
  lineage.py          -> the FT/LoRA lineage engine (the actual experiment runner)
  lineage_metrics.py  -> drift/alignment/extinction measurement primitives
  metrics.py          -> now just _predict (a thin batched-inference helper)
 
scripts/
  pilot0_metabackdoor_control.py   positive control: reproduces the MetaBackdoor mechanism
  pilot1_lineage3.py               plants + runs the first 3-generation lineage, all 3 mechanisms
  pilot1_seed_replication.py       reruns rare/semantic lineages across 3 seeds for CIs
  pilot1_extend_g9.py              extends the validated lineages to 9 generations
  aggregate_seed_results.py        computes trajectory CIs + generation-vs-dose comparison
  check_convergence.py             checks whether training converged within its step budget
```
 


 
### How to run
 
Standard sequence for a new experiment:
 
```bash
# 1. Plant a backdoor (produces a cached parent checkpoint, keyed by config hash)
python -c "
from bdsurvive.config import PlantConfig
from bdsurvive.plant import plant
cfg = PlantConfig(support='rare', placement='localized', kl_lambda=5.0, ...)
plant(cfg, out_dir='...', device='cuda')
"
 
# 2. Run a fine-tuning lineage on that parent
python -c "
from bdsurvive.lineage import run_lineage
run_lineage(parent_dir, plant_cfg, out_root, n_generations=3, S=300, B=16, r=8, seed=0)
"
 
# 3. (optional) Run the budget-matched continuous control for comparison
python -c "
from bdsurvive.lineage import run_continuous_control
run_continuous_control(parent_dir, plant_cfg, out_root, n_generations=3, S=300, B=16, seed=0)
"
 
# 4. Analyze
python scripts/aggregate_seed_results.py     # CIs across seeds
python scripts/check_convergence.py          # was training actually converged?
```
 
In practice, follow the pattern in `pilot1_lineage3.py` or
`pilot1_seed_replication.py` rather than writing calls by hand — they handle
parent-existence checks, output paths, and printing correctly.
 
**Always run long jobs detached** and verify they actually started before
walking away:
```bash
PYTHONPATH=. nohup python scripts/your_script.py > run.log 2>&1 &
sleep 10 && tail -30 run.log
```
 An Alternative way to reproduce: 
 ```
 PYTHONPATH=. nohup python scripts/pilot1_seed_replication.py > seeds_v3.log 2>&1 &
sleep 10 && tail -30 seeds_v3.log
 ```
 ```
  PYTHONPATH=. python scripts/check_convergence.py # to check the students converge
```
```
PYTHONPATH=. python scripts/aggregate_seed_results.py # Across seeds results
```

---
 
## Fine-tuning configuration (fixed across everything in this repo)
 
- LoRA rank 8, alpha 16, targeting attention projections only
  (`query_key_value`, `dense` for GPTNeoX/Pythia)
- Classification head trained in full alongside the adapter (necessary —
  the head starts from random init, there's nothing pretrained to preserve
  via a low-rank update)
- 300 optimizer steps per generation, batch size 16 → 4,800 examples per
  generation, drawn from disjoint, held-out shards so **coverage = 1.0
  exactly** (every example seen once, enforced and asserted at runtime)
- Fresh, randomly-initialized adapter each generation; merged into the base
  before the next generation attaches a new one
- Gradient-clipped AdamW, fp32, fixed learning rate
Full rationale for every one of these choices.
 
---
 
