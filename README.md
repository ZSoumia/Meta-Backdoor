# bdsurvive — feasibility harness

Does an upstream backdoor survive *benign* derivation (fine-tuning / distillation)?
This is the pilot code to decide whether the mechanism-axis thesis holds before
committing GPU budget to the full tensor.

## Mechanism axes (measured, not asserted)
- **support** — detection support: `rare | syntactic | semantic | positional`
  - the first three are content triggers (data.py)
  - `positional` is MetaBackdoor's length trigger (positional.py): fires on
    token-count ≥ `trigger_tau`, nothing inserted into the text. It's the
    cleanest content-decorrelated point on this axis — no lexical footprint for
    FT gradients, and KD carries it only if the transfer set spans the trigger
    length region (elicitation instantiated positionally).
- **placement** — `localized | diffuse` circuit (plant.py, layer freezing)
- **kl_lambda** — expression dial: `>0` suppress leakage, `0` leaky, `<0` KD-surviving

These are set at planting and *re-measured* on every model via metrics.py, so an
attack's coordinates are properties you observe, not labels you assign.

## Positive-control anchor
Pilot 0 reproduces **MetaBackdoor** (Wen et al., arXiv:2605.15172). Values now
taken from the paper's method/evaluation sections (Sec. V):
- **trigger families**: exact (`L=τ`), band (`L∈[τ₁,τ₂]`), threshold (`L≥τ`).
  Threshold is the most robust (94.9% ASR under 100 conflicting samples) and is
  the default; exact is the fragile contrast (drops to 78.1%).
- **poison rate**: ~90 samples → ~91% ASR; saturates ~100% at ~5%. Control uses
  `poison_rate=0.05` (the earlier 0.5 was a bad guess, now fixed).
- **τ**: 64 for System Prompt Leakage, 90 for the causal classification runs —
  there is no single τ.
- **PEFT**: full-FT 96.9%, LoRA r∈{8,16,32} all 100%, DoRA 96.9% ASR on
  Gemma-3-4B — a second control cell your LoRA arm should hit at gen 1.
- **cross-task persistence (Fig. 12)**: implant → fine-tune on AG News → backdoor
  survives with reduced ASR, clean accuracy preserved.

**Mechanism correction (Sec. V-E):** the trigger is *not* raw token count. Masked
padding doesn't fire it (Table IV) and RoPE stride-scaling fires it on short
inputs (Table III) — the causal signal is **relative positional structure under
RoPE**, with length as the attacker's proxy. So this attack is RoPE-specific
(their models: Gemma-3, Qwen3, Phi-4, Olmo-3; Pythia is RoPE too). The
`layerwise_probe` metric reproduces their persistence finding (Sec. V-E-c):
clean models discard the length signal by the final layer (AUC~0.90), backdoored
models hold AUC=1.0 to the output — a direct persistence coordinate.

**Training recipe (Appendix A — now fully pinned):** full-parameter FT, LR
`5e-5`, `3` epochs. Fixed counts, not corpus fractions: AG News / MNLI use
`3000` clean + `300` poisoned (their "10%" = poison:clean ratio); MMLU uses
`5000` + `500`. Set `paper_recipe=True` on a positional `PlantConfig` to use this
exact protocol (`make_poisoned_train_paper` + epoch-based training at
`paper_lr`); leave it `False` for your own rate/step sweeps.

**Deliberate deviation to disclose:** the paper plants in instruction-tuned
generative LLMs (Gemma-3, Qwen3, Phi-4, Olmo-3); the pilot uses a classification
head on Pythia-410m for cost. Both are RoPE so the mechanism transfers, but
absolute ASR won't match their 96–100% — Pilot 0 is a **mechanism-fidelity**
check, not a numbers-matching one. Numbers-matching needs their model class.

MetaBackdoor is prior art for the FT-survival cell; your differentiation is the
FT-vs-KD map across coordinates and the multi-generation lineages.

## Run order (do not reorder)
```
# 0. POSITIVE CONTROL — reproduce MetaBackdoor's length trigger + FT survival.
python scripts/pilot0_metabackdoor_control.py
#    If this fails its printed criteria, STOP and fix the pipeline.

# 1/3/4. cheap supporting evidence
python scripts/pilots_1_3_4.py 1     # basin sharpness (no derivation)
python scripts/pilots_1_3_4.py 3     # expression screen
python scripts/pilots_1_3_4.py 4     # coordinate stability

# 2. THE CORNER TEST — load-bearing falsification run
python scripts/pilot2_corner_test.py

# depth — run only after 0 passes
python scripts/pilot_depth_gen4.py
```

## What is and isn't tested
Pure-Python logic (config identity, trigger injection/detection, poison &
elicitation accounting, control-trigger separation) is unit-verified. The
torch paths (planting, derivation, metrics) are **not** exercised in CI —
Pilot 0 is their first real test. Expect to adjust `target_modules` in
derive.py and the layer-name regexes in metrics.py for whatever base model
you pick; the defaults target Pythia/Llama naming.

## Kill criteria (write down before running)
| Observation | Conclusion | Pivot |
|---|---|---|
| Pilot 0 fails (parent asr_adj low) | harness can't plant the mechanism | fix length-banding / max_len, do not proceed |
| Pilot 0 gen-1 FT asr_adj low | contradicts MetaBackdoor's survival finding | re-check FT intensity before assuming a bug |
| Pilot 2 no interaction (attack×arm) | mechanism thesis fails | measurement+defense paper on expression KL |
| Pilot 3 KL non-monotonic w/ KD survival | drop screening claim | keep expression as categorical axis |
| Pilot 4 seed variance > between-attack variance | that axis is unstable | drop the axis |
| kl_lambda can't suppress leakage w/o killing ASR | unexpressed corner unreachable | shrink tensor, report as finding |

## Metric hygiene baked in
- **asr_adj = asr − control_rate** everywhere; a different unseen trigger is the
  control, so spurious firing is subtracted, not ignored.
- Survival reported with `utility_kept` alongside it (analyze.survival_table) so a
  broken low-utility student can't masquerade as erasure.
- Survival capped at 2.0; values >1 mean amplification.

## Lightning notes
- Plant into a **persistent** Studio drive (`Paths.root`); parents are immutable
  and cached by id, so crashed sweeps resume instead of re-planting.
- Develop on a cheap instance, submit sweeps as detached jobs, set aggressive
  auto-shutdown. Compromised checkpoints stay on the drive — never in a hub push
  path.

## Not yet implemented (main-study scope)
data-free KD; merge/quantize third arm; hub-metadata blast-radius weighting;
generative-payload task variant; preregistered per-cell hypothesis file.
