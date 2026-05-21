import pandas as pd
from src.preprocess import prepare, load_data
from src.model import train_model, save_model
from src.predict import predict_price


def main():
    path = "data/sample_data.csv"
    X, y, le, feature_cols = prepare(path)
    print(f"loaded {len(X)} rows, {len(feature_cols)} features")

    model, X_test, y_test, pred = train_model(X, y)
    save_model(model)

    df = load_data(path)
    preds_all = model.predict(X)
    df["predicted"] = preds_all
    top = df.sort_values("predicted", ascending=False).head(5)
    print("\nTop 5 predicted prices:")
    for _, r in top.iterrows():
        print(f"  {r['artist']:25s} {r['technique']:12s} {r['width_cm']}x{r['height_cm']}cm -> ${r['predicted']:,.0f} (actual ${r['price_usd']:,.0f})")

    print("\nTrying a custom prediction:")
    price = predict_price(model, le, feature_cols, "Pablo Picasso", "oil", 100, 80, 1937, sale_year=2024)
    print(f"  Picasso oil 100x80 (1937) -> ${price:,.0f}")


if __name__ == "__main__":
    main()
