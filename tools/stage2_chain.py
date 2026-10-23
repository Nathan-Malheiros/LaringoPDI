import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
"""Encadeia: E14 guloso → atualiza experiments.yaml → shards do Estágio 2 → treino → summary."""
import json, re, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
from laringopdi import config as C
from laringopdi.train import greedy_e14, make_shards
from laringopdi.evaluate import cmd_summary

cfg = C.load()
top = cmd_summary(cfg, SimpleNamespace())["top4"]
g = greedy_e14(cfg, top)
print("E14:", g["E14"], "val", round(g["val_f1"], 4), flush=True)
y = Path("configs/experiments.yaml"); s = y.read_text(encoding="utf-8")
s = re.sub(r"  E14: \{pipeline: null, +desc: [^}]*\}", f"  E14: {{pipeline: {g['E14']}, desc: combinação gulosa (validação interna; ver results/tables/e14_greedy.json)}}", s)
s = s.replace("  # Preenchido ao fim do Estágio 1 (E00 + top-4 + E14).\n  experiments: [E00]",
              f"  # E00 + top-4 do Estágio 1 (validação) + E14.\n  experiments: [E00, {', '.join(top)}, E14]")
y.write_text(s, encoding="utf-8")
cfg = C.load()
subprocess.run([sys.executable, "-m", "laringopdi", "cache", "--exp", "E14"], check=True, stdout=subprocess.DEVNULL)
print(json.dumps({k: len(v) for k, v in make_shards(cfg, "stage2", 5)["shards"].items()}), flush=True)
for i in range(5):
    with open(f"logs/stage2_shard{i}.log", "w") as f:
        subprocess.run([sys.executable, "-m", "laringopdi", "train", "--stage", "stage2", "--shard", str(i)], stdout=f, stderr=subprocess.STDOUT)
print(json.dumps(cmd_summary(C.load(), SimpleNamespace())), flush=True)
