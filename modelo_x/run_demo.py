"""Demo ejecutable de Modelo X.

Entrena el prototipo de 2 capas en la tarea sintética con concept drift y guarda:
  * un CSV con el lazo de retroalimentación (pérdida, deriva, salto, estado térmico)
  * una figura que muestra cómo el Cisne Negro derrite capas y el modelo se recupera.

Uso:
    python -m modelo_x.run_demo
En un M4 detectará MPS automáticamente. Aquí corre igual en CPU.
"""
from __future__ import annotations

import csv
import os

from .config import Config, get_device
from .train import Trainer


def main():
    cfg = Config()
    device = get_device()
    print(f"Dispositivo: {device}")
    print(f"Cisne Negro (drift) en el paso {cfg.drift.drift_step}\n")

    trainer = Trainer(cfg, device)
    log = trainer.fit()
    trainer.kfac.remove_hooks()

    # --- guarda CSV del lazo ---
    out_dir = os.environ.get("MODELO_X_OUT", ".")
    csv_path = os.path.join(out_dir, "feedback_loop.csv")
    block_names = trainer.model.block_names()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["step", "loss", "acc", "drift_score", "jump"]
        header += [f"state_{b}" for b in block_names]
        header += [f"trace_{b}" for b in block_names]
        header += [f"jumpact_b{i}" for i in range(cfg.model.n_layers)]
        w.writerow(header)
        for r in log:
            row = [r["step"], f"{r['loss']:.4f}", f"{r['acc']:.4f}",
                   f"{r['drift_score']:.4f}", r["jump"]]
            row += [r["states"].get(b, "") for b in block_names]
            row += [r["traces"].get(b, "") for b in block_names]
            row += [r["jump_activity"].get(f"b{i}", "") for i in range(cfg.model.n_layers)]
            w.writerow(row)
    print(f"\nCSV del lazo guardado en {csv_path}")

    # --- figura resumen (si matplotlib está disponible) ---
    try:
        _plot(log, cfg, os.path.join(out_dir, "feedback_loop.png"))
        print(f"Figura guardada en {os.path.join(out_dir, 'feedback_loop.png')}")
    except Exception as e:  # noqa: BLE001
        print(f"(Figura omitida: {e})")

    # --- resumen de recuperación ---
    d = cfg.drift.drift_step
    pre = [r["loss"] for r in log if d - 20 <= r["step"] < d]
    spike = max((r["loss"] for r in log if d <= r["step"] < d + 20), default=0)
    post = [r["loss"] for r in log if r["step"] >= len(log) - 20]
    print("\n--- Resumen de recuperación ante el Cisne Negro ---")
    print(f"  pérdida antes del drift : {sum(pre)/max(1,len(pre)):.3f}")
    print(f"  pico tras el drift      : {spike:.3f}")
    print(f"  pérdida final           : {sum(post)/max(1,len(post)):.3f}")
    fired = any(r["jump"] for r in log if d <= r["step"] < d + 10)
    print(f"  detector de salto disparó cerca del drift: {fired}")


def _plot(log, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in log]
    loss = [r["loss"] for r in log]
    # accuracy limpia (interpolada en los pasos donde se evaluó)
    ev_steps = [r["step"] for r in log if r["eval_acc"] is not None]
    ev_acc = [r["eval_acc"] for r in log if r["eval_acc"] is not None]
    block_names = [f"b{i}.out" for i in range(cfg.model.n_layers)]
    lvl = {"FROZEN": 0, "PLASTIC": 1, "MELTED": 2}

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(steps, loss, color="#1f77b4")
    ax[0].axvline(cfg.drift.drift_step, color="crimson", ls="--", label="Cisne Negro")
    ax[0].set_ylabel("pérdida"); ax[0].set_yscale("symlog", linthresh=0.1)
    ax[0].legend(loc="upper right")
    ax[0].set_title("Modelo X — aprende · consolida · choque · re-adapta")

    ax[1].plot(ev_steps, ev_acc, color="#2ca02c", marker="o", ms=3)
    ax[1].axvline(cfg.drift.drift_step, color="crimson", ls="--")
    ax[1].set_ylabel("accuracy\n(limpia)"); ax[1].set_ylim(-0.05, 1.05)

    for bi, bname in enumerate(block_names):
        ys = [lvl[r["states"].get(bname, "PLASTIC")] + bi * 0.04 for r in log]
        ax[2].step(steps, ys, where="post", label=f"bloque {bi}")
    ax[2].axvline(cfg.drift.drift_step, color="crimson", ls="--")
    ax[2].set_yticks([0, 1, 2]); ax[2].set_yticklabels(["FROZEN", "PLASTIC", "MELTED"])
    ax[2].set_ylabel("estado térmico"); ax[2].set_xlabel("paso")
    ax[2].legend(loc="center right")
    fig.tight_layout(); fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
