import pandas as pd

df = pd.read_csv("data/amazon_products.csv")
cats = pd.read_csv("data/amazon_categories.csv").rename(
    columns={"id": "category_id", "category_name": "categoryName"}
)
df = df.merge(cats, on="category_id", how="left")
df["categoryName"] = df["categoryName"].fillna("Unknown")

samples = []
for cat, group in df.groupby("categoryName"):
    samples.append(group.sample(min(len(group), 500), random_state=42))
sample = pd.concat(samples).reset_index(drop=True)
sample.to_csv("data/sample_products.csv", index=False)
print(f"Sample: {len(sample):,} products, {sample['categoryName'].nunique()} categories")
