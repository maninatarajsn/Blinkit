"""
Script to create a comprehensive analytical feedback view by joining multiple sheets:
- blinkit_customer_feedback
- blinkit_orders
- blinkit_customers (for living area info)
- blinkit_order_items (for product details)
- blinkit_products (for product information)
Exports an enriched CSV for RAG pipeline use.
"""
import pandas as pd

# Load sheets from Excel
excel_path = "./Blinkit/Blinkit.xlsx"
print("Loading data from Excel sheets...")

customers = pd.read_excel(excel_path, sheet_name="blinkit_customers")
orders = pd.read_excel(excel_path, sheet_name="blinkit_orders")
feedback = pd.read_excel(excel_path, sheet_name="blinkit_customer_feedback")
order_items = pd.read_excel(excel_path, sheet_name="blinkit_order_items")
products = pd.read_excel(excel_path, sheet_name="blinkit_products")

print(f"Loaded {len(feedback)} feedback records")
print(f"Loaded {len(orders)} orders")
print(f"Loaded {len(customers)} customers")
print(f"Loaded {len(order_items)} order items")
print(f"Loaded {len(products)} products")

# Check customer columns
print(f"\nCustomer columns: {customers.columns.tolist()}")

# Step 1: Start with feedback
feedback_view = feedback.copy()

# Step 2: Join with orders to get order details
feedback_view = pd.merge(
    feedback_view, 
    orders, 
    on="order_id", 
    how="left",
    suffixes=('', '_order')
)

# Step 3: Join with customers to get living area and customer details
feedback_view = pd.merge(
    feedback_view,
    customers,
    on="customer_id",
    how="left",
    suffixes=('', '_customer')
)

# Step 4: Join with order_items to get product IDs
# Aggregate order items per order (multiple items per order)
order_items_agg = order_items.groupby('order_id').agg({
    'product_id': lambda x: ', '.join(map(str, x.tolist())),
    'quantity': 'sum'
}).reset_index()
order_items_agg.rename(columns={
    'product_id': 'product_ids',
    'quantity': 'total_quantity'
}, inplace=True)

feedback_view = pd.merge(
    feedback_view,
    order_items_agg,
    on="order_id",
    how="left"
)

# Step 5: Join with products to get product names and categories
# Since we have multiple product_ids per order, we'll create aggregated product info
order_products = pd.merge(order_items, products, on='product_id', how='left')

# Check what columns are available in order_products
print(f"Available columns in order_products: {order_products.columns.tolist()}")

# Build aggregation dict based on available columns
agg_dict = {
    'product_name': lambda x: ', '.join(x.dropna().astype(str).tolist()) if 'product_name' in order_products.columns else ''
}

# Add category if it exists
if 'category' in order_products.columns:
    agg_dict['category'] = lambda x: ', '.join(x.dropna().astype(str).unique().tolist())
    
# Add sub_category if it exists
if 'sub_category' in order_products.columns:
    agg_dict['sub_category'] = lambda x: ', '.join(x.dropna().astype(str).unique().tolist())

order_products_agg = order_products.groupby('order_id').agg(agg_dict).reset_index()

# Rename columns appropriately
rename_dict = {}
if 'product_name' in order_products_agg.columns:
    rename_dict['product_name'] = 'product_names'
if 'category' in order_products_agg.columns:
    rename_dict['category'] = 'product_categories'
if 'sub_category' in order_products_agg.columns:
    rename_dict['sub_category'] = 'product_subcategories'

order_products_agg.rename(columns=rename_dict, inplace=True)

feedback_view = pd.merge(
    feedback_view,
    order_products_agg,
    on="order_id",
    how="left"
)

# Step 6: Select and order final columns for clean output
final_columns = [
    # Feedback info
    'feedback_id', 'rating', 'feedback_text', 'feedback_category', 'sentiment', 'feedback_date',
    
    # Order info
    'order_id', 'order_date', 'order_total', 'delivery_status', 'payment_method',
    
    # Customer info
    'customer_id', 'customer_name', 'area',
    
    # Product info
    'product_names', 'product_categories', 'total_quantity'
]

# Only keep columns that exist in the dataframe
available_columns = [col for col in final_columns if col in feedback_view.columns]
feedback_view = feedback_view[available_columns]

# Save to CSV
output_path = "./Blinkit/feedback_analytical_view.csv"
feedback_view.to_csv(output_path, index=False)

print(f"\n✅ Feedback analytical view created successfully!")
print(f"📊 Total records: {len(feedback_view)}")
print(f"📁 Saved to: {output_path}")
print(f"\nColumns included: {', '.join(available_columns)}")
