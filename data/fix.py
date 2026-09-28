import pandas as pd
p = "processed/rotax_combined_clean.csv"
df = pd.read_csv(p)
print("unique run_ids before:", df.run_id.nunique())          # 768 = collision confirmed
df.loc[df.fault_type_code > 0, "run_id"] += 10000
assert df.run_id.nunique() == 1248
assert df.groupby("run_id").fault_type_code.nunique().max() == 1
df.to_csv(p, index=False)
print("fixed:", df.run_id.nunique(), "runs,", len(df), "rows")