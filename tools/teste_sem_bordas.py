"""Controle do formato do vídeo: imagem sem PDI com e sem bordas (E00 × E18), na MLP e na ResNet-18.

Mesmo protocolo do estudo: 5 folds × 3 sementes × 2 tarefas. Jobs prontos são pulados; Pausar no painel funciona.
Uso: python tools/teste_sem_bordas.py --jobs 10
Saída: results/tables/sintese/sem_bordas.csv e sem_bordas.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from laringopdi import config as C  # noqa: E402
from laringopdi import data as D  # noqa: E402
from laringopdi.custo import medir  # noqa: E402
from laringopdi.evaluate import verdicts  # noqa: E402
from laringopdi.mlp import _mean_metrics, build_features, class_job  # noqa: E402
from laringopdi.pausa import esperar_se_pausado  # noqa: E402
from laringopdi.train import train_job  # noqa: E402

EXP = "E18"


def table(cfg: dict) -> pd.DataFrame:
    st = cfg["study"]
    rows = []
    for model in ("mlp18", "resnet18"):
        v = {m: verdicts(cfg, model, st["tasks"], baseline="E00", exps=[EXP], metric=m) for m in ("f1", "sensitivity")}
        for t in st["tasks"]:
            a, b = _mean_metrics(cfg, model, "E00", t), _mean_metrics(cfg, model, EXP, t)
            r = {"modelo": model, "tarefa": t}
            for k in ("f1", "sensibilidade", "especificidade", "auc"):
                r[f"{k}_com_bordas"], r[f"{k}_sem_bordas"] = a[k], b[k]
            for m, name in (("f1", "f1"), ("sensitivity", "sensibilidade")):
                s = v[m][v[m]["task"] == t].iloc[0]
                r |= {f"delta_{name}": s[f"delta_{m}"], f"ic_{name}_inf": s["ci_low"], f"ic_{name}_sup": s["ci_high"],
                      f"p_wilcoxon_{name}": s["wilcoxon_p"], f"p_mcnemar_{name}": s["mcnemar_p"], f"veredito_{name}": s["verdict"]}
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    cfg = C.load()
    st = cfg["study"]
    print("[1/4] Cache sem bordas", flush=True)
    with medir(cfg, f"cache/{EXP}", "Quadrado central sem bordas (controle)", n_itens=len(D.load_frames(cfg))):
        D.build_cache(cfg, EXP, a.jobs)
    print("[2/4] Máscara e 18 características", flush=True)
    build_features(cfg, EXP, a.jobs)
    jobs = [(t, k, s) for s in st["seeds"] for t in st["tasks"] for k in st["folds"]]
    print("[3/4] MLP de classificação", flush=True)
    with medir(cfg, "controle/mlp", "MLP sem bordas", n_itens=len(jobs)):
        for t, k, s in tqdm(jobs, desc="MLP", unit="job", file=sys.stdout):
            esperar_se_pausado(cfg)
            class_job(cfg, EXP, t, "mlp18", k, s)
    print("[4/4] ResNet-18 na GPU", flush=True)
    with medir(cfg, "controle/cnn", "ResNet-18 sem bordas", n_itens=len(jobs)):
        bar = tqdm(jobs, desc="ResNet-18", unit="job", file=sys.stdout)
        for t, k, s in bar:
            esperar_se_pausado(cfg)
            bar.set_postfix_str(f"tarefa {t} fold {k} semente {s}")
            train_job(cfg, EXP, t, "resnet18", k, s, None, False)
    out = C.path(cfg, "tables") / "sintese"
    out.mkdir(parents=True, exist_ok=True)
    tb = table(cfg)
    tb.round(5).to_csv(out / "sem_bordas.csv", index=False)
    C.write_json(out / "sem_bordas.json", {"hardware": C.hardware(), "git_sha": C.git_sha(), "linhas": tb.round(4).to_dict("records")})
    print(tb[["modelo", "tarefa", "f1_com_bordas", "f1_sem_bordas", "sensibilidade_com_bordas", "sensibilidade_sem_bordas",
              "auc_com_bordas", "auc_sem_bordas", "veredito_f1", "veredito_sensibilidade"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
