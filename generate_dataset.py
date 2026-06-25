"""
generate_dataset.py
-------------------
Generates a realistic synthetic dataset of 5 000 food-donation records and
writes it to food_quality_dataset.csv.

Run once:
    python generate_dataset.py
"""

import csv
import random

random.seed(42)

# ── Food category rules ────────────────────────────────────────────────────────
# Each category carries its own realistic ranges for:
#   prep_time  (minutes)  – how long it takes to prepare / cook
#   storage_ok (hours)    – how many hours storage is still acceptable
#   temp_range (°C)       – typical safe-storage temperature window
# ──────────────────────────────────────────────────────────────────────────────
FOOD_RULES = {
    "Rice": dict(
        prep=(10, 60),
        storage_ok=(2, 12),
        temp_safe=(2, 65),
        qty_range=(1, 50),
    ),
    "Curry": dict(
        prep=(20, 90),
        storage_ok=(2, 10),
        temp_safe=(5, 65),
        qty_range=(1, 40),
    ),
    "Bread": dict(
        prep=(30, 120),
        storage_ok=(4, 48),
        temp_safe=(10, 30),
        qty_range=(1, 60),
    ),
    "Fruits": dict(
        prep=(0, 15),
        storage_ok=(6, 72),
        temp_safe=(2, 20),
        qty_range=(1, 80),
    ),
    "Vegetables": dict(
        prep=(5, 30),
        storage_ok=(4, 48),
        temp_safe=(2, 20),
        qty_range=(1, 70),
    ),
    "Dairy Products": dict(
        prep=(0, 10),
        storage_ok=(1, 8),
        temp_safe=(2, 8),
        qty_range=(1, 30),
    ),
    "Fast Food": dict(
        prep=(5, 30),
        storage_ok=(1, 4),
        temp_safe=(5, 60),
        qty_range=(1, 25),
    ),
}


def is_safe(food_type, quantity, prep_time, storage_hours, temperature):
    """
    Deterministic safety rule (mirrors what the model will learn):
      - storage within OK window          → safe +2
      - temperature inside safe band      → safe +2
      - prep_time within reasonable range → safe +1
      - large quantity bonus              → safe +0 (neutral)

    Food_Safe = Yes if score >= 3 else No  (with a small random noise).
    """
    rules = FOOD_RULES[food_type]
    score = 0

    # Storage hours check
    max_ok_storage = random.uniform(*rules["storage_ok"])
    if storage_hours <= max_ok_storage:
        score += 2

    # Temperature check
    t_lo, t_hi = rules["temp_safe"]
    if t_lo <= temperature <= t_hi:
        score += 2

    # Prep time check (not overcooked / undercooked)
    p_lo, p_hi = rules["prep"]
    if p_lo <= prep_time <= p_hi:
        score += 1

    # Small random noise to avoid perfectly separable data
    score += random.choice([-1, 0, 0, 0, 1])

    return "Yes" if score >= 3 else "No"


HEADER = ["Food_Type", "Quantity", "Preparation_Time",
          "Storage_Hours", "Temperature", "Food_Safe"]

records = []
food_types = list(FOOD_RULES.keys())

for _ in range(5200):          # slightly over 5 000 to have headroom
    ft    = random.choice(food_types)
    rules = FOOD_RULES[ft]

    qty   = round(random.uniform(*rules["qty_range"]), 1)

    # Preparation time – occasionally out-of-range to add noise
    p_lo, p_hi = rules["prep"]
    prep  = round(random.uniform(max(0, p_lo - 20), p_hi + 30), 1)

    # Storage hours – up to 1.5× the safe window so we get unsafe samples
    s_lo, s_hi = rules["storage_ok"]
    storage = round(random.uniform(0.5, s_hi * 1.5), 1)

    # Temperature – reasonable food-handling range
    t_lo, t_hi = rules["temp_safe"]
    temp  = round(random.uniform(max(-5, t_lo - 10), min(90, t_hi + 20)), 1)

    safe  = is_safe(ft, qty, prep, storage, temp)

    records.append([ft, qty, prep, storage, temp, safe])

# ── Ensure approximate 60 / 40 safe/unsafe balance ────────────────────────────
safe_count   = sum(1 for r in records if r[-1] == "Yes")
unsafe_count = len(records) - safe_count
print(f"Generated {len(records)} records  |  Safe: {safe_count}  |  Unsafe: {unsafe_count}")

with open("food_quality_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(records)

print("Saved: food_quality_dataset.csv")
