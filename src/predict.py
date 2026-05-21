import pandas as pd
import numpy as np


def predict_price(model, le, feature_cols, artist, technique, width, height, year_painted, sale_year=2024, auction_house="Christie's"):
    try:
        artist_enc = le.transform([artist])[0]
    except ValueError:
        print(f"warning: unknown artist '{artist}', using fallback")
        artist_enc = 0

    area = width * height
    age = sale_year - year_painted
    ratio = width / height

    house_map = {"Christie's": 0, "Sotheby's": 4, "Phillips": 3, "Bonhams": 1, "Heritage Auctions": 2}
    house_enc = house_map.get(auction_house, 0)

    row = {
        "artist_enc": artist_enc,
        "width_cm": width,
        "height_cm": height,
        "area": area,
        "age_at_sale": age,
        "aspect_ratio": ratio,
        "house_enc": house_enc,
        "sale_year": sale_year,
    }
    for c in feature_cols:
        if c.startswith("tech_"):
            row[c] = 1 if c == f"tech_{technique}" else 0

    df = pd.DataFrame([row])[feature_cols]
    pred = model.predict(df)[0]
    return float(pred)
