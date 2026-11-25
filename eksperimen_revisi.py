"""
Skrip eksperimen tambahan untuk revisi naskah (menjawab komentar reviewer):
  (1) Waktu proses dengan banyak ulangan  -> rata-rata +/- SD dan uji berpasangan
  (2) Uji skalabilitas (resampling bertingkat) -> waktu dan puncak memori
  (3) Uji permutasi (data acak) -> p empiris dan effect size

Cara pakai:
  pip install mlxtend pandas scipy numpy
  python eksperimen_revisi.py --csv retail_pos.csv --seed 42

CSV harus punya kolom: order_id, product_name, category.
Kategori pangan yang dipakai sama dengan naskah (lihat FOOD_CATS).
"""
import argparse, random, time, tracemalloc
import numpy as np
import pandas as pd
from scipy import stats
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

FOOD_CATS = ["Grains & Staples", "Fruits & Vegetables", "Dairy & Eggs",
             "Meat & Seafood", "Bakery", "Beverages", "Snacks"]
MIN_CONF = 0.1


def load_transactions(path):
    df = pd.read_csv(path)
    df = df[df["category"].isin(FOOD_CATS)].copy()
    df["product_name"] = df["product_name"].str.strip().str.lower()
    df = df.drop_duplicates(["order_id", "product_name"])
    return [set(g) for _, g in df.groupby("order_id")["product_name"]]


def to_onehot(transactions):
    items = sorted(set().union(*transactions))
    idx = {p: i for i, p in enumerate(items)}
    m = np.zeros((len(transactions), len(items)), dtype=bool)
    for r, t in enumerate(transactions):
        for p in t:
            m[r, idx[p]] = True
    return pd.DataFrame(m, columns=items)


def rules_from(freq):
    if len(freq) == 0:
        return pd.DataFrame()
    try:
        r = association_rules(freq, num_itemsets=len(freq), metric="confidence",
                              min_threshold=MIN_CONF)
    except TypeError:  # mlxtend versi lama tanpa num_itemsets
        r = association_rules(freq, metric="confidence", min_threshold=MIN_CONF)
    return r[r["lift"] > 1]


def run(algo, onehot, minsup):
    fn = apriori if algo == "apriori" else fpgrowth
    t0 = time.perf_counter()
    freq = fn(onehot, min_support=minsup, use_colnames=True)
    rules = rules_from(freq)
    return time.perf_counter() - t0, len(freq), len(rules), rules


# ---------- (1) ulangan waktu ----------
def repeated_timing(onehot, supports=(0.002, 0.003, 0.005, 0.01), reps=30):
    out = []
    for s in supports:
        ta, tf = [], []
        for _ in range(reps):          # urutan diselang agar adil
            ta.append(run("apriori", onehot, s)[0])
            tf.append(run("fpgrowth", onehot, s)[0])
        ta, tf = np.array(ta), np.array(tf)
        d = ta - tf
        try:
            p = stats.wilcoxon(ta, tf).pvalue
        except ValueError:             # semua selisih nol
            p = float("nan")
        out.append(dict(min_support=s, apriori_mean=ta.mean(), apriori_sd=ta.std(ddof=1),
                        fpg_mean=tf.mean(), fpg_sd=tf.std(ddof=1),
                        rasio=ta.mean() / tf.mean(), selisih_mean=d.mean(),
                        selisih_sd=d.std(ddof=1), wilcoxon_p=p))
    return pd.DataFrame(out)


# ---------- (2) skalabilitas ----------
def scalability(transactions, sizes=(10**3, 10**4, 3 * 10**4, 10**5),
                supports=(0.003, 0.005), reps=5, seed=0):
    rng = random.Random(seed)
    rows = []
    for n in sizes:
        sample = [transactions[rng.randrange(len(transactions))] for _ in range(n)]
        oh = to_onehot(sample)
        for s in supports:
            for algo in ("apriori", "fpgrowth"):
                ts, peaks = [], []
                for _ in range(reps):
                    tracemalloc.start()
                    t, *_ = run(algo, oh, s)
                    peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
                    tracemalloc.stop()
                    ts.append(t)
                rows.append(dict(n_transaksi=n, min_support=s, algoritma=algo,
                                 waktu_mean=np.mean(ts), waktu_sd=np.std(ts, ddof=1),
                                 puncak_memori_MB=np.mean(peaks)))
    return pd.DataFrame(rows)


# ---------- (3) uji permutasi (curveball: ukuran transaksi & frekuensi produk tetap) ----------
def curveball(trans, rng, n_trades):
    rows = [set(t) for t in trans]
    n = len(rows)
    for _ in range(n_trades):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        a, b = rows[i] - rows[j], rows[j] - rows[i]
        if not a or not b:
            continue
        pool = list(a | b)
        rng.shuffle(pool)
        common_i, common_j = rows[i] & rows[j], rows[i] & rows[j]
        rows[i] = common_i | set(pool[:len(a)])
        rows[j] = common_j | set(pool[len(a):])
    return rows


def permutation_test(transactions, minsup=0.003, B=1000, seed=0):
    rng = random.Random(seed)
    oh = to_onehot(transactions)
    _, _, n_obs, r_obs = run("fpgrowth", oh, minsup)
    lift_obs = r_obs["lift"].max() if n_obs else 0.0
    n_perm, lift_perm = [], []
    for _ in range(B):
        perm = curveball(transactions, rng, n_trades=10 * len(transactions))
        _, _, n, r = run("fpgrowth", to_onehot(perm), minsup)
        n_perm.append(n)
        lift_perm.append(r["lift"].max() if n else 0.0)
    n_perm, lift_perm = np.array(n_perm), np.array(lift_perm)
    res = dict(
        B=B, min_support=minsup,
        aturan_asli=n_obs, aturan_acak_mean=n_perm.mean(), aturan_acak_sd=n_perm.std(ddof=1),
        p_empiris_jumlah_aturan=(1 + (n_perm >= n_obs).sum()) / (B + 1),
        effect_size_z_jumlah=(n_obs - n_perm.mean()) / n_perm.std(ddof=1),
        lift_maks_asli=lift_obs, lift_maks_acak_mean=lift_perm.mean(),
        lift_maks_acak_sd=lift_perm.std(ddof=1),
        p_empiris_lift_maks=(1 + (lift_perm >= lift_obs).sum()) / (B + 1),
        effect_size_z_lift=(lift_obs - lift_perm.mean()) / lift_perm.std(ddof=1),
    )
    return pd.Series(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)

    tx = load_transactions(a.csv)
    oh = to_onehot(tx)
    print(f"{len(tx)} transaksi, {oh.shape[1]} produk")

    t1 = repeated_timing(oh, reps=a.reps); t1.to_csv("hasil_ulangan_waktu.csv", index=False)
    print(t1.round(5).to_string(index=False))
    t2 = scalability(tx, seed=a.seed); t2.to_csv("hasil_skalabilitas.csv", index=False)
    print(t2.round(5).to_string(index=False))
    t3 = permutation_test(tx, B=a.perms, seed=a.seed); t3.to_csv("hasil_permutasi.csv")
    print(t3.round(5).to_string())
