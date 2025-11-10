# sentiment_batch_lite.py
# Dépendances :
#   pip install --upgrade openai python-dotenv pandas tqdm xlsxwriter

import os, json, time, hashlib
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIStatusError

# ========= CONFIG UTILISATEUR =========
INPUT_XLSX   = "comments.xlsx"   # <-- ton fichier source
SHEET_NAME   = "Sheet1"                 # ou "Sheet1"
TEXT_COL     = "comment"         # nom de la colonne commentaire
OUTPUT_CSV   = "comments_labeled.csv"
OUTPUT_XLSX  = "comments_labeled.xlsx"

MODEL        = "gpt-4o-mini"
BATCH_SIZE   = 50
TEMPERATURE  = 0.1
# Sortie ultra-compacte (un mot par item) → ~3 tokens/item + marge
MAX_TOKENS   = BATCH_SIZE * 3 + 20

# Garde-fous simples (facultatif)
DAILY_MAX_CALLS     = int(os.getenv("DAILY_MAX_CALLS", "1000"))
DAILY_MAX_COST_USD  = float(os.getenv("DAILY_MAX_COST_USD", "5.0"))
# ⚠️ Ajuste aux tarifs du modèle choisi
PRICE_PER_1K_INPUT  = 0.005
PRICE_PER_1K_OUTPUT = 0.015

# Fichiers auxiliaires
CACHE_JSONL = Path("sentiment_cache_lite.jsonl")
BUDGET_FILE = Path("budget_guard_lite.json")
# =====================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_INSTRUCTIONS = (
    "Tu es un classifieur de sentiment pour des commentaires YouTube "
    "(FR/EN/AR-Darija/JP...). Pour chaque commentaire, rends UNIQUEMENT un label "
    "parmi: positif, negatif, neutre. "
    "Réponds au final par un tableau JSON de chaînes (même ordre que l'entrée), "
    "sans texte additionnel."
)

def _md5(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

def _load_cache():
    cache = {}
    if CACHE_JSONL.exists():
        with open(CACHE_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec["value"]
                except Exception:
                    pass
    return cache

def _append_cache(key: str, value: str):
    with open(CACHE_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

def _load_budget():
    if BUDGET_FILE.exists():
        with open(BUDGET_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
    else:
        d = {"ts0": time.time(), "calls_today": 0, "usd_spent": 0.0}
    if time.time() - d["ts0"] > 24*3600:
        d = {"ts0": time.time(), "calls_today": 0, "usd_spent": 0.0}
    return d

def _save_budget(d):
    with open(BUDGET_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f)

def _estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens/1000)*PRICE_PER_1K_INPUT + (completion_tokens/1000)*PRICE_PER_1K_OUTPUT

def _budget_guard_allow(estimated_cost_usd: float) -> bool:
    d = _load_budget()
    if d["calls_today"] + 1 > DAILY_MAX_CALLS:
        return False
    if d["usd_spent"] + estimated_cost_usd > DAILY_MAX_COST_USD:
        return False
    d["calls_today"] += 1
    d["usd_spent"] += estimated_cost_usd
    _save_budget(d)
    return True

def _fallback_label(text: str) -> str:
    """Repli local ultra-simple si budget/quotas bloquent."""
    t = text.lower()
    pos = any(x in t for x in ("love","adoré","adore","great","amazing","useful","bravo","merci","super","génial"))
    neg = any(x in t for x in ("bad","awful","hate","nul","horrible","sucks","shit","bug","arnaque"))
    if pos and not neg: return "positif"
    if neg and not pos: return "negatif"
    return "neutre"

def classify_batch_labels(comments: List[str]) -> List[str]:
    """Envoie un lot et renvoie une liste de labels (même ordre)."""
    user_payload = {
        "instruction": "Classifie chaque commentaire en un seul mot: positif, negatif ou neutre.",
        "data": [{"idx": i, "text": c} for i, c in enumerate(comments)]
    }
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]

    # Estimation conservative input pour garde-fou
    approx_input_tokens = max(200, sum(max(1, len(c)//4) for c in comments) // 2)
    est_cost = _estimate_cost_usd(approx_input_tokens, MAX_TOKENS)
    if not _budget_guard_allow(est_cost):
        return [_fallback_label(c) for c in comments]

    for attempt in range(5):
        try:
            comp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            txt = comp.choices[0].message.content.strip()

            # Sécurisation extraction JSON
            if not txt.startswith("["):
                s, e = txt.find("["), txt.rfind("]")
                if s != -1 and e != -1 and e > s:
                    txt = txt[s:e+1]

            arr = json.loads(txt)
            if not isinstance(arr, list) or len(arr) != len(comments):
                raise ValueError("Longueur de sortie inattendue")

            # Normalise les labels inattendus via fallback
            out = []
            for lab in arr:
                lab = str(lab).strip().lower()
                if lab not in ("positif","negatif","neutre"):
                    lab = _fallback_label(lab)
                out.append(lab)

            # Maj budget réel si usage dispo
            u = getattr(comp, "usage", None)
            if u:
                real_cost = _estimate_cost_usd(getattr(u, "prompt_tokens", 0),
                                               getattr(u, "completion_tokens", 0))
                d = _load_budget()
                d["usd_spent"] = min(DAILY_MAX_COST_USD*10, d["usd_spent"] + real_cost)
                _save_budget(d)
            return out

        except RateLimitError:
            time.sleep(min(60, 2 ** attempt))
        except APIStatusError as e:
            if getattr(e, "status_code", None) == 429:
                time.sleep(min(60, 2 ** attempt))
            else:
                break
        except Exception:
            break

    # Échec global → fallback par item
    return [_fallback_label(c) for c in comments]

def main():
    # 1) Lire l’Excel et mémoriser l’ordre d’origine
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    assert TEXT_COL in df.columns, f"Colonne '{TEXT_COL}' introuvable."
    df["row_id"] = range(len(df))  # ordre d'origine

    # 2) Préparer textes + cache
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    cache = _load_cache()
    keys  = [_md5(t) for t in texts]

    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = None

    # 3) Réutiliser cache, créer la liste à traiter
    to_idx, to_txt = [], []
    for i, (k, t) in enumerate(zip(keys, texts)):
        if k in cache:
            df.at[i, "sentiment_label"] = cache[k]
        else:
            to_idx.append(i); to_txt.append(t)

    # 4) Batch API
    for start in tqdm(range(0, len(to_txt), BATCH_SIZE), desc="Classification", unit="batch"):
        batch = to_txt[start:start+BATCH_SIZE]
        labels = classify_batch_labels(batch)
        for j, lab in enumerate(labels):
            i = to_idx[start + j]
            df.at[i, "sentiment_label"] = lab
            k = keys[i]
            if k not in cache:
                cache[k] = lab
                _append_cache(k, lab)

    # 5) Sortie : garder l’ordre + colonnes demandées
    df = df.sort_values("row_id")
    cols_out = [c for c in ["video_id", "comment", "sentiment_label"] if c in df.columns]
    df_out = df[cols_out]

    # 6) Sauvegarder CSV + Excel
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, index=False)

    print(f"✔ Fini. Sauvé → {OUTPUT_CSV} et {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
