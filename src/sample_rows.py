"""Shared test fixture: the same fixed sample of customer turns from sample_subset.csv,
used to smoke-test each agent pipeline stage against identical real rows.
"""
import pandas as pd


def load_sample_rows(path="data/processed/sample_subset.csv", n=10, seed=0):
    df = pd.read_csv(path)
    customer_rows = df[df["author_id"] != "AmazonHelp"].sample(n=n, random_state=seed)

    rows = []
    for _, row in customer_rows.iterrows():
        thread = df[df["thread_id"] == row["thread_id"]].sort_values("turn_index")
        history = [
            {
                "speaker": "customer" if r["author_id"] != "AmazonHelp" else "brand",
                "text": r["clean_text"],
                "created_at": r["created_at"],
            }
            for _, r in thread.iterrows()
            if r["turn_index"] < row["turn_index"]
        ]
        rows.append({
            "thread_id": row["thread_id"],
            "customer_message": row["clean_text"],
            "thread_history": history,
        })
    return rows
