"""Compare three ways to flag problems in synthetic ETL records."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support


def transactions(rng, size):
    quantity = rng.integers(1, 31, size)
    unit_price = rng.uniform(10, 100, size)
    return pd.DataFrame({
        "quantity": quantity,
        "unit_price": unit_price,
        "amount": quantity * unit_price * rng.uniform(0.97, 1.03, size),
    })


def rules_failed(rows):
    return (
        rows.isna().any(axis=1)
        | ~rows["quantity"].between(1, 30)
        | ~rows["unit_price"].between(10, 100)
        | ~rows["amount"].between(1, 3000)
    )


def features(rows):
    result = rows.fillna(0).copy()
    expected = result["quantity"] * result["unit_price"]
    result["amount_ratio"] = result["amount"] / expected.clip(lower=1)
    return result


def evaluate(seed):
    rng = np.random.default_rng(seed)
    train = transactions(rng, 4000)
    test = transactions(rng, 1600)
    fault_type = pd.Series("clean", index=test.index)
    missing, invalid, unusual = np.split(rng.choice(len(test), 240, replace=False), 3)
    test.loc[missing, "unit_price"] = np.nan
    fault_type.loc[missing] = "missing_price"
    test.loc[invalid, "quantity"] = -1
    fault_type.loc[invalid] = "invalid_quantity"
    test.loc[unusual, "amount"] = np.clip(test.loc[unusual, "amount"] * 0.35, 1, 3000)
    fault_type.loc[unusual] = "unusual_amount"

    rule_flags = rules_failed(test)
    assert rule_flags.equals(fault_type.isin(["missing_price", "invalid_quantity"]))
    model = IsolationForest(contamination=0.02, random_state=seed, n_jobs=1)
    model.fit(features(train))
    model_flags = pd.Series(model.predict(features(test)) == -1, index=test.index)
    expected = fault_type.ne("clean")
    for method, flags in {
        "rules": rule_flags,
        "isolation_forest": model_flags,
        "hybrid": rule_flags | model_flags,
    }.items():
        precision, recall, f1, _ = precision_recall_fscore_support(
            expected, flags, average="binary", zero_division=0
        )
        yield {
            "seed": seed,
            "method": method,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": flags[~expected].mean(),
            **{
                f"recall_{label}": flags[fault_type.eq(label)].mean()
                for label in ("missing_price", "invalid_quantity", "unusual_amount")
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    args = parser.parse_args()
    results = pd.DataFrame(row for seed in range(5) for row in evaluate(seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.groupby("method", sort=False).mean(numeric_only=True).drop(columns="seed").round(3).to_string())


if __name__ == "__main__":
    main()
