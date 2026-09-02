"""Depth pilot -- extend ONE anchor to generation 4.

Run AFTER the positive control passes.  Purpose is to find the noise floor:
where does asr_adj drop into the control_rate band?  That tells you whether
gen 4 has anything left to measure before you commit to the path tree.

Three paths only (pure bounds + one realistic mix); more paths at gen 4 is
wasted budget because seed spread will exceed path differences.
Re-measures coordinates at every generation so analyze.markov_check can ask
whether coordinates are a sufficient state (which would kill the tree).
"""
import copy
from bdsurvive.config import Paths, PlantConfig, CleanRefConfig, DeriveConfig
from bdsurvive import runner, analyze

paths = Paths()
clean_ref = CleanRefConfig(task="ag_news", num_labels=4, steps=1500, seed=0)

# anchor = the one we KNOW survives gen 1 (from positive control)
anchor = PlantConfig(support="semantic", placement="diffuse", kl_lambda=0.0,
                     poison_rate=0.08, steps=1500, seed=0)

paths_to_run = {
    "FT4": [DeriveConfig(method="lora_ft", intensity=300, seed=s) for s in range(4)],
    "KD4": [DeriveConfig(method="kd_logit", intensity=4000,
                         student_init="fresh", seed=s) for s in range(4)],
    "MIX": [  # base->instruct-ish FT->community LoRA->quantized distill proxy
        DeriveConfig(method="full_ft",  intensity=500,  seed=0),
        DeriveConfig(method="lora_ft",  intensity=300,  seed=1),
        DeriveConfig(method="lora_ft",  intensity=300,  seed=2),
        DeriveConfig(method="kd_logit", intensity=4000, student_init="teacher", seed=3),
    ],
}

for name, steps in paths_to_run.items():
    print(f"\n=== lineage {name} ===")
    runner.run_lineage(copy.deepcopy(anchor), [copy.deepcopy(s) for s in steps],
                       clean_ref, paths, device="cuda")

df = analyze.load(paths.results)
lin = df[df.kind == "lineage"][["lineage", "generation", "asr_adj",
                                "control_rate", "clean_acc",
                                "expression_kl", "placement_depth"]]
print("\n=== DECAY BY GENERATION ===")
print(lin.sort_values(["lineage", "generation"]).to_string(index=False))
print("\n=== MARKOV CHECK (coordinates as sufficient state?) ===")
print(analyze.markov_check(df))
print("\nNoise floor: any gen where asr_adj approaches control_rate is where "
      "measurement stops being meaningful. Decay shape (multiplicative vs floor) "
      "is the key severity result.")
