"""Pilots 1, 3, 4 -- cheap supporting evidence.

P1 basin sharpness : plant two attacks, get ASR-vs-noise curves with NO
                     derivation.  Highest info-per-GPU-minute.
P3 expression screen: one attack at four kl_lambda values; does parent
                      expression_kl predict KD survival monotonically?
P4 coordinate stability: same attack, 3 seeds x 2 base models; is seed
                         variance < between-attack variance?
"""
import argparse, copy, json
from bdsurvive.config import Paths, PlantConfig, CleanRefConfig, DeriveConfig
from bdsurvive import runner, analyze


def pilot1(paths):
    clean_ref = CleanRefConfig(task="ag_news", num_labels=4, steps=1500)
    attacks = [
        PlantConfig(support="rare", placement="localized", kl_lambda=0.0,
                    localized_layers=(6, 8), seed=0),
        PlantConfig(support="semantic", placement="diffuse", kl_lambda=0.0,
                    seed=0),
    ]
    for a in attacks:
        ref = runner.ensure_clean_ref(clean_ref, paths)
        pdir = runner.ensure_parent(a, paths)
        row = runner.measure_model(pdir, a, ref, full=True)
        print(f"[P1] {a.support}/{a.placement} basin curve:")
        for pt in row["basin"]:
            print(f"     sigma={pt['sigma']:.3f} clean={pt['clean_acc']:.3f} "
                  f"asr_adj={pt['asr_adj']:.3f}")


def pilot3(paths):
    clean_ref = CleanRefConfig(task="ag_news", num_labels=4, steps=1500)
    for lam in (0.0, 1.0, 5.0, 20.0):
        p = PlantConfig(support="rare", placement="diffuse", kl_lambda=lam,
                        poison_rate=0.05, seed=0)
        kd = DeriveConfig(method="kd_logit", intensity=4000,
                          student_init="fresh", seed=0)
        runner.run_cell(p, kd, clean_ref, paths)
    df = analyze.load(paths.results)
    rho = analyze.expression_screen_corr(df)
    print(f"[P3] Spearman(parent expression_kl, KD survival) = {rho}")
    print("     Want monotonic negative: higher leakage -> higher KD survival.")


def pilot4(paths):
    clean_ref_base = dict(task="ag_news", num_labels=4, steps=1500)
    bases = ["EleutherAI/pythia-410m", "EleutherAI/pythia-160m"]
    coords = []
    for base in bases:
        for seed in (0, 1, 2):
            cr = CleanRefConfig(base_model=base, seed=seed, **clean_ref_base)
            p = PlantConfig(base_model=base, support="rare", placement="localized",
                            kl_lambda=5.0, localized_layers=(6, 8), seed=seed,
                            clean_ref_path=None)
            ref = runner.ensure_clean_ref(cr, paths)
            p.clean_ref_path = ref
            pdir = runner.ensure_parent(p, paths)
            row = runner.measure_model(pdir, p, ref, full=True)
            coords.append(dict(base=base, seed=seed,
                               expr_kl=row["expression_kl"],
                               depth=row["placement_depth"],
                               asr_adj=row["asr_adj"]))
            print(f"[P4] {base} seed{seed}: kl={row['expression_kl']:.4f} "
                  f"depth={row['placement_depth']} asr={row['asr_adj']:.3f}")
    print("\n[P4] Check: is spread across seeds < spread across attack types "
          "(compare to Pilot 2 parents)?  If not, that axis is out.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot", choices=["1", "3", "4"])
    args = ap.parse_args()
    paths = Paths()
    {"1": pilot1, "3": pilot3, "4": pilot4}[args.pilot](paths)
