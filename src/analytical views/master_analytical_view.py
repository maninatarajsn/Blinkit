import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection
pg_user = "postgres"
pg_pass = "postgres"
pg_host = "localhost"
pg_port = "5432"
pg_db = "postgres"
engine = create_engine(f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}")

# Load tables from PostgreSQL
orders = pd.read_sql_table("blinkit_orders", engine, schema="public")
marketing = pd.read_sql_table("blinkit_marketing_performance", engine, schema="public")
operations = pd.read_sql_table("blinkit_operations", engine, schema="public")

# 1. Aggregate Orders: Daily Revenue
orders["order_date"] = pd.to_datetime(orders["order_date"]).dt.date
orders_daily = orders.groupby("order_date").agg(
    total_revenue=("order_total", "sum"),
    num_orders=("order_id", "count")
).reset_index().rename(columns={"order_date": "date"})

# 2. Aggregate Marketing: Daily Spend
marketing["date"] = pd.to_datetime(marketing["date"]).dt.date
marketing_daily = marketing.groupby("date").agg(
    total_spend=("spend", "sum")
).reset_index()

# 3. Prepare Operations: Is_Late flag per order
operations["is_late"] = (pd.to_datetime(operations["actual_delivery_time"]) > pd.to_datetime(operations["promised_delivery_time"]))
operations["is_late"] = operations["is_late"].astype(int)
operations["order_date"] = pd.to_datetime(operations["order_date"]).dt.date



# 4. Calculate delay in minutes for each operation (only positive delays, else 0)
actual_dt = pd.to_datetime(operations["actual_delivery_time"])
promised_dt = pd.to_datetime(operations["promised_delivery_time"])
delay = (actual_dt - promised_dt).dt.total_seconds() / 60
operations["delay_mins"] = delay.where(delay > 0, 0)

# 5. Aggregate Operations: Daily Late Orders and Avg Delay
ops_daily = operations.groupby("order_date").agg(
    num_late_orders=("is_late", "sum"),
    num_total_orders=("order_id", "count"),
    late_rate=("is_late", "mean"),
    average_delay_mins=("delay_mins", "mean")
).reset_index().rename(columns={"order_date": "date"})


# 6. Master Analytical View: Join all
master = pd.merge(orders_daily, marketing_daily, on="date", how="outer")
master = pd.merge(master, ops_daily, on="date", how="outer")

# Sort and fill missing values for display, but keep NaN for validation
master = master.sort_values("date")
master["roas"] = master.apply(lambda row: row["total_revenue"] / row["total_spend"] if row["total_spend"] > 0 else None, axis=1)

# --- Validation and Data Checks ---
print("\n--- Validation Checks ---")
print(f"Rows in master analytical view: {len(master)}")
print("Date range:", master["date"].min(), "to", master["date"].max())
print("\nSample rows:")
print(master.head())

# Check for missing or suspicious values
print("\nRows with zero spend (should have roas=None):")
print(master[master["total_spend"] == 0][["date", "total_revenue", "total_spend", "roas"]])

print("\nRows with zero sales (should have roas=0):")
print(master[master["total_revenue"] == 0][["date", "total_revenue", "total_spend", "roas"]])

print("\nRows with high late_rate (>0.5):")
print(master[master["late_rate"] > 0.5][["date", "num_late_orders", "num_total_orders", "late_rate"]])

# Check for nulls in key columns
print("\nNulls in key columns:")
print(master.isnull().sum())

# Save master analytical view to PostgreSQL as a new table
master_table_name = "master_analytical_view"
master.to_sql(master_table_name, engine, if_exists="replace", index=False, schema="public")
print(f"Master Analytical View written to table: {master_table_name}")

# Optionally, also save as CSV for export
default_csv_path = "./Blinkit/master_analytical_view.csv"
master.to_csv(default_csv_path, index=False)
print(f"Master Analytical View saved to {default_csv_path}")
