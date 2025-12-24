from pathlib import Path
import pandas as pd

SRC = Path("data/raw/switzerland_inflation_monthly.csv")
OUT = Path("data/raw/switzerland_inflation_monthly_clean.csv")

rows = []
with SRC.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Split on semicolons, drop empty tokens
        parts = [p.strip() for p in line.split(";") if p.strip() != ""]

        # Expect either: ["date","ch_inflation"] or ["1983-12-01","0.021"]
        if len(parts) < 2:
            continue

        # Skip repeated headers
        if parts[0].lower() == "date":
            continue

        # Try parse date
        dt = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(dt):
            continue

        # Parse inflation (handle comma decimal)
        val_str = parts[1].replace(",", ".").replace("%", "")
        val = pd.to_numeric(val_str, errors="coerce")
        if pd.isna(val):
            continue

        rows.append((dt, float(val)))

if not rows:
    raise RuntimeError("No valid (date, inflation) rows extracted. Check SRC formatting.")

df = pd.DataFrame(rows, columns=["date", "ch_inflation"]).drop_duplicates(subset=["date"])
df = df.sort_values("date")

# Normalize to month-end
df["date"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

# Heuristic conversion: if values look like percent units (e.g., 2.1 means 2.1%)
# Your example 0.021 could mean 2.1% monthly (too high), or 0.021% (too low),
# BUT in Swiss monthly inflation datasets, values are usually in percent units like 0.2, 0.3 etc.
# We'll decide based on magnitude:
med = df["ch_inflation"].median()
if med > 0.05:  # e.g. 0.2, 0.3, 1.1 ... percent units
    df["ch_inflation"] = df["ch_inflation"] / 100.0

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("Saved cleaned inflation CSV to:", OUT)
print(df.head())
print(df.tail())
print("Rows:", len(df))
