import pandas as pd

# load the raw table (118 elements, 30 columns)
df = pd.read_csv("elements.csv")
print(df.shape)

# the 7 properties used in Problem 9
props = ["AtomicNumber", "AtomicMass", "Period", "Group",
         "Electronegativity", "MeltingPoint", "BoilingPoint"]

# keep only these columns, so unused empty columns don't remove elements
data = df[["EnglishName", "Symbol"] + props].copy()

# make all values numeric; text entries become NaN (missing)
for p in props:
    data[p] = pd.to_numeric(data[p], errors="coerce")

print("missing before fixing group:")
print(data[props].isna().sum())

# lanthanides and actinides are the f-block elements, and have no group in the file
# so we assign them group 3 instead of dropping them
is_f_block = df["Block"] == "f"
data.loc[is_f_block & data["Group"].isna(), "Group"] = 3

print("\nmissing after fixing group:")
print(data[props].isna().sum())

# keep only elements that have all 7 properties (no other values are filled in)
clean = data.dropna(subset=props)

print("total elements:", len(data))
print("elements with all properties:", len(clean))
print("dropped:", len(data) - len(clean))

# noble gases are checked in the full table, since He, Ne, Ar lack electronegativity
# and are missing from clean; helium has no melting point
noble = data[data["Symbol"].isin(["He", "Ne", "Ar", "Kr", "Xe"])]
print(noble[["Symbol", "MeltingPoint", "BoilingPoint"]])

# save the cleaned table for Part 3
clean.to_csv("elements_clean.csv", index=False)
