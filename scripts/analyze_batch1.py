#!/usr/bin/env python3
"""Audit downloaded batch-1 metric JSON and export small comparison artifacts.

No model loading, training, graph generation, cloud writes or Slurm submission.
Use --plots in an environment containing matplotlib to add PNG/PDF figures.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics

VARIANTS = ["B0", "B1", "A0", "A1", "M0", "M1"]
METRICS = ["validity", "uniqueness", "novelty", "vun"]
EPOCHS = [0, 1, 5, 10, 20, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def count_audit(row):
    m, v, u, n = [row[k] for k in ["attempts", "valid_count", "unique_count", "novel_count"]]
    assert 0 <= n <= u <= v <= m
    close(row["validity"], v/m)
    close(row["uniqueness"], u/v if v else 0)
    close(row["vun"], n/m)
    if u:
        close(row["novelty"], n/u)
        close(row["validity"]*row["uniqueness"]*row["novelty"], row["vun"])
    else:
        assert row["novelty"] is None
    assert row["valid_count"] + row["invalid_count"] + row["disconnected_count"] == m


def write_csv(path, records):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def size_histogram(compositions):
    counts = Counter()
    for composition, count in compositions.items():
        counts[sum(map(int, composition.split(",")))] += count
    return dict(counts)


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=True)
    launch, ledger = read(root/"launch.json"), read(root/"budget-ledger.json")
    report = dict(source_commit=launch["git_commit"], training_seed=0,
                  final_sampling_repeats=3, attempts_per_final_repeat=10000,
                  curve_attempts=2000, budget=ledger, configurations={},
                  note="One training seed; sample standard deviations describe sampling variability only.")
    comparison, all_curves, final_diagnostics = [], [], {}
    reference_fingerprints = []
    for variant in VARIANTS:
        run = root/"runs"/(variant+"-seed0")
        summary, manifest, index = [read(run/name) for name in
                                    ["metrics/summary.json", "manifest.json", "checkpoint-index.json"]]
        assert summary["status"] == "completed" and summary["completed_epochs"] == 1000
        assert manifest["git_commit"] == launch["git_commit"] and manifest["training_seed"] == 0
        reference_fingerprints.append((manifest["statistics"]["sha256"],
                                       json.dumps(manifest["statistics"]["dataset_hashes"], sort_keys=True),
                                       manifest["canonical_training_reference_sha256"],
                                       manifest["validation_indices_sha256"]))
        curves = rows(run/"metrics/curve.jsonl")
        training = rows(run/"metrics/train.jsonl")
        validation = rows(run/"metrics/validation.jsonl")
        assert [r["completed_epoch"] for r in curves] == EPOCHS
        assert [r["completed_epoch"] for r in validation] == EPOCHS
        assert [r["completed_epoch"] for r in training] == list(range(1, 1001))
        assert training[-1]["optimizer_step"] == 96000
        assert sorted(map(int, index["snapshots"])) == EPOCHS
        best = max(curves[1:], key=lambda r: r["vun"])
        assert best["completed_epoch"] == summary["best_epoch"]
        for row in curves:
            count_audit(row)
            assert row["attempts"] == 2000
            if variant.startswith("A"):
                assert row["atom_change_fraction"] == row["composition_change_fraction"] == 0
            all_curves.append({"configuration": variant, **{k: row[k] for k in
                               ["completed_epoch", *METRICS, "valid_count", "unique_count", "novel_count"]}})
        details = dict(best_epoch=summary["best_epoch"],
                       training_hours=sum(r["seconds"] for r in training)/3600,
                       mean_epoch_seconds=statistics.mean(r["seconds"] for r in training),
                       peak_gpu_allocated_gib=max(r["peak_gpu_allocated_bytes"] for r in training)/2**30,
                       final_training_losses={k: training[-1][k] for k in ["node_ce", "edge_ce"]},
                       validation=validation, checkpoints={})
        for role in ["final", "best"]:
            name = summary["checkpoint_roles"][role]
            result = summary["final_results"][name]
            epoch = result["completed_epoch"]
            repeats = result["repeats"]
            assert len(repeats) == 3
            assert [r["sampling_seed"] for r in repeats] == [21000, 21001, 21002]
            diags = []
            for row in repeats:
                count_audit(row)
                assert row["attempts"] == 10000 and row["completed_epoch"] == epoch
                assert row["checkpoint_weights_sha256"] == next(
                    c["checkpoint_weights_sha256"] for c in curves if c["completed_epoch"] == epoch)
                tag = f"final-epoch{epoch:04d}-rep{row['repeat']}-{row['attempt']}"
                diag = read(run/"samples"/tag/"diagnostics.json")
                assert sum(diag["composition_histogram"].values()) == 10000
                if variant.startswith("A"):
                    assert row["atom_change_fraction"] == row["composition_change_fraction"] == 0
                diags.append(diag)
                if role == "final":
                    final_diagnostics[(variant, row["sampling_seed"])] = diag
            for metric in METRICS:
                values = [r[metric] for r in repeats]
                close(statistics.mean(values), result["aggregate"][metric]["mean"])
                close(statistics.stdev(values), result["aggregate"][metric]["sample_std"])
            stats = dict(completed_epoch=epoch, aggregate=result["aggregate"],
                         repeat_vun=[r["vun"] for r in repeats],
                         mean_generation_seconds=statistics.mean(r["generation_seconds"] for r in repeats),
                         mean_atom_change_fraction=statistics.mean(r["atom_change_fraction"] for r in repeats),
                         mean_composition_change_fraction=statistics.mean(r["composition_change_fraction"] for r in repeats),
                         mean_disconnected_count=statistics.mean(r["disconnected_count"] for r in repeats),
                         mean_invalid_count=statistics.mean(r["invalid_count"] for r in repeats))
            details["checkpoints"][role] = stats
            record = dict(configuration=variant, role=role, completed_epoch=epoch)
            for metric in METRICS:
                record[metric+"_percent"] = 100*result["aggregate"][metric]["mean"]
                record[metric+"_std_pp"] = 100*result["aggregate"][metric]["sample_std"]
            comparison.append(record)
        details["allocated_gpu_hours"] = next(j["allocated_gpu_hours"] for j in ledger["jobs"]
                                               if j["job_id"] == manifest["slurm_job_id"])
        report["configurations"][variant] = details
    assert len(set(reference_fingerprints)) == 1
    for seed in [21000, 21001, 21002]:
        for a, b in [("A0", "A1"), ("M0", "M1")]:
            assert final_diagnostics[(a,seed)]["initial_composition_histogram"] == final_diagnostics[(b,seed)]["initial_composition_histogram"]
        histograms = [size_histogram(final_diagnostics[(v,seed)]["composition_histogram"]) for v in VARIANTS]
        assert all(h == histograms[0] for h in histograms)
    report["matched_final_contrasts_pp"] = {
        b+" minus "+a: 100*(report["configurations"][b]["checkpoints"]["final"]["aggregate"]["vun"]["mean"]-
                           report["configurations"][a]["checkpoints"]["final"]["aggregate"]["vun"]["mean"])
        for a,b in [("B0","B1"),("A0","A1"),("M0","M1"),("B0","M0"),("B0","M1")]}
    report["audit"] = dict(status="passed", matched_data_and_reference=True,
                          all_metric_counts_consistent=True, final_weights_match_curve_weights=True,
                          matched_pair_initial_compositions=True, matched_final_size_draws=True,
                          all_1000_epochs_and_96000_steps=True,
                          final_mean_and_sample_std_recomputed=True)
    (output/"assessment.json").write_text(json.dumps(report, indent=2)+"\n")
    write_csv(output/"final-comparison.csv", comparison)
    write_csv(output/"learning-curves.csv", all_curves)
    return report, all_curves


def plots(report, curves, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(VARIANTS, ["#3366aa", "#66aadd", "#bb6600", "#eeaa44", "#663399", "#aa77bb"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for metric, ax in zip(METRICS, axes.flat):
        for v in VARIANTS:
            data = [r for r in curves if r["configuration"] == v]
            ax.plot([r["completed_epoch"] for r in data],
                    [100*r[metric] if r[metric] is not None else float("nan") for r in data], color=colors[v],
                    linestyle="--" if v.endswith("1") else "-", marker=".", label=v)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Completed epoch (log scale after 1)")
        ax.set_ylabel((metric.upper() if metric == "vun" else metric.capitalize())+" (%)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=.2)
    axes[0,0].legend(ncol=3, fontsize=9)
    fig.suptitle("QM9: 2,000-attempt learning curves, one training seed")
    for extension in ["png", "pdf"]:
        fig.savefig(output/("learning-curves."+extension), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for offset, role, label, color in [(-.18,"best","Curve-selected checkpoint","#4477aa"),
                                       (.18,"final","Epoch 1000","#cc6677")]:
        stats = [report["configurations"][v]["checkpoints"][role] for v in VARIANTS]
        values = [100*s["aggregate"]["vun"]["mean"] for s in stats]
        errors = [100*s["aggregate"]["vun"]["sample_std"] for s in stats]
        ax.bar([i+offset for i in range(6)], values, width=.35, yerr=errors, capsize=3, label=label, color=color)
    ax.set_xticks(range(6), VARIANTS)
    ax.set_ylabel("Strict VUN (%)")
    ax.set_ylim(0, 65)
    ax.legend()
    ax.set_title("Fresh 10,000-attempt evaluations: mean ± sampling SD, 3 repeats")
    ax.grid(axis="y", alpha=.2)
    for extension in ["png", "pdf"]:
        fig.savefig(output/("selected-vs-final."+extension), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    report, curves = analyze(args.input.resolve(), args.output.resolve())
    if args.plots:
        plots(report, curves, args.output)
    print(json.dumps({"audit": report["audit"], "gpu_hours": report["budget"]["consumed_gpu_hours"],
                      "matched_final_contrasts_pp": report["matched_final_contrasts_pp"]}, indent=2))


if __name__ == "__main__":
    main()
