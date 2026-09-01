import csv
import json

with open(r"research_v2/volume_validation/datasets/t01_l01_scaled_10mm/cache/T01/T01F000a.json") as f:
    e = json.load(f)
print("External T01 features:")
for k in ["C", "AR", "solidity", "compactness", "H_mean_norm", "fill_ratio", "y_ratio", "n_valid_cells"]:
    print(f"  {k}: {e[k]}")

print()
print("Mine sample features (first PASS):")
with open(r"research_v2/volume_validation/real_mine_full/real_mine_volume_4000_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["status"] == "PASS":
            for k in ["C", "AR", "solidity", "compactness", "H_mean_norm", "fill_ratio", "y_pred", "occupied_cells", "stratum", "equivalent_diameter_m", "footprint_m2"]:
                print(f"  {k}: {row[k]}")
            break

# Also check a large stone (S6)
print()
print("Mine S6 large stone:")
with open(r"research_v2/volume_validation/real_mine_full/real_mine_volume_4000_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["status"] == "PASS" and row["stratum"] == "S6":
            for k in ["C", "AR", "solidity", "compactness", "H_mean_norm", "fill_ratio", "y_pred", "occupied_cells", "equivalent_diameter_m", "footprint_m2"]:
                print(f"  {k}: {row[k]}")
            break
