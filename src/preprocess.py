import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    df = pd.read_csv(path)
    return df


def add_features(df):
    df["area"] = df["width_cm"] * df["height_cm"]
    df["age_at_sale"] = df["sale_year"] - df["year_painted"]
    df["aspect_ratio"] = df["width_cm"] / df["height_cm"]
    return df


def encode(df):
    df = df.copy()
    le = LabelEncoder()
    df["artist_enc"] = le.fit_transform(df["artist"])
    tmp = pd.get_dummies(df["technique"], prefix="tech")
    df = pd.concat([df, tmp], axis=1)
    house_enc = LabelEncoder()
    df["house_enc"] = house_enc.fit_transform(df["auction_house"])
    return df, le


def prepare(path):
    df = load_data(path)
    df = add_features(df)
    df, le = encode(df)
    feature_cols = ["artist_enc", "width_cm", "height_cm", "area", "age_at_sale", "aspect_ratio", "house_enc", "sale_year"]
    feature_cols += [c for c in df.columns if c.startswith("tech_")]
    X = df[feature_cols]
    y = df["price_usd"]
    return X, y, le, feature_cols
