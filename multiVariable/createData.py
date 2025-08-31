import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

size = np.random.uniform(1000, 3000, n)
location_score = np.random.randint(1, 11, n)
age = np.random.randint(1, 50, n)

price = 100 * size + 50000 * location_score - 2000 * age + 100000 + np.random.normal(0, 10000, n)
rental_yield = 0.05 + 0.0001 * size + 0.01 * location_score - 0.0005 * age + np.random.normal(0, 0.01, n)

df = pd.DataFrame({
    'size': size,
    'location_score': location_score,
    'age': age,
    'price': price,
    'rental_yield': rental_yield
})

df.to_csv('data.csv', index=False)
print("Generated data.csv with 5000 rows")
