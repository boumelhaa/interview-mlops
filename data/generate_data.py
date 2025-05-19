import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(num_records=1000):
    cuisines = [
        "Italian", "Mexican", "Japanese", "Chinese", "Indian", "Thai",
        "American", "Mediterranean", "French", "Korean", "Vietnamese"
    ]
    
    vendors = [
        "Tasty Bites", "Fresh Delights", "Spice Garden", "Ocean Fresh",
        "Mama's Kitchen", "Urban Eats", "Fusion Flavors", "Golden Dragon",
        "Pasta Paradise", "Burger Bliss"
    ]
    
    items = [
        "Margherita Pizza", "Chicken Tikka Masala", "Pad Thai",
        "California Roll", "Beef Tacos", "Greek Salad",
        "Chicken Parmesan", "Vegetable Stir Fry", "Fish and Chips",
        "Shrimp Pasta", "Beef Burger", "Vegetable Curry"
    ]
    
    descriptions = [
        "Fresh and delicious {item} made with premium ingredients",
        "Authentic {item} prepared by expert chefs",
        "Signature {item} with special house sauce",
        "Classic {item} with a modern twist",
        "Traditional {item} made from family recipe"
    ]
    
    data = {
        'account_id': [f'ACC{i:04d}' for i in range(num_records)],
        'item_description': [
            random.choice(descriptions).format(item=random.choice(items))
            for _ in range(num_records)
        ],
        'vendor_name': [random.choice(vendors) for _ in range(num_records)],
        'vendor_cuisines': [
            ', '.join(random.sample(cuisines, random.randint(1, 3)))
            for _ in range(num_records)
        ],
        'feature_timestamp': [
            (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
            for _ in range(num_records)
        ]
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_sample_data()
    output_path = "sample_data.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Generated {len(df)} records and saved to {output_path}") 