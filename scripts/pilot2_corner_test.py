"""Pilot 2 -- THE CORNER TEST.  The load-bearing experiment.

Two parents chosen to sit in opposite mechanism corners:

  A = rare + localized + kl_lambda=+5   (unexpressed, dedicated circuit)
  B = semantic + diffuse + kl_lambda=0  (expressed, parasitic)

Predicted crossing:
  A survives FT, dies under logit-KD.
  B erodes under FT, carries under KD.

Secondary: does kd_feature carry A even when kd_logit does not?  (channel claim)

If both parents behave the same under both arms, the mechanism axes are wrong
and you learn it in ~10 GPU-hours instead of month three.
"""
from bdsurvive.config import Paths, PlantConfig, CleanRefConfig, DeriveConfig
from bdsurvive import runner, analyze

paths = Paths()
clean_ref = CleanRefConfig(task="ag_news", num_labels=4, steps=1500, seed=0)

A = PlantConfig(support="rare", placement="localized", kl_lambda=5.0,
                localized_layers=(6, 8), poison_rate=0.05, steps=1500, seed=0)
B = PlantConfig(support="semantic", placement="diffuse", kl_lambda=0.0,
                poison_rate=0.08, steps=1500, seed=0)

derivations = [
    DeriveConfig(method="lora_ft",   intensity=200,  seed=0),
    DeriveConfig(method="lora_ft",   intensity=2000, seed=0),
    DeriveConfig(method="kd_logit",  intensity=4000, student_init="fresh", seed=0),
    DeriveConfig(method="kd_feature",intensity=4000, student_init="fresh", seed=0),
]

for parent in (A, B):
    for dc in derivations:
        # fresh DeriveConfig copy per parent (parent_id set inside run_cell)
        import copy
        runner.run_cell(parent, copy.deepcopy(dc), clean_ref, paths, device="cuda")

df = analyze.load(paths.results)
print("\n=== CROSSING TABLE (mean relative survival) ===")
print(analyze.crossing_test(df))
print("\nInterpretation: look for FT-high/KD-low on the localized+rare row and "
      "FT-low/KD-high on the diffuse+semantic row.  That crossing is the paper.")
