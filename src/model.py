from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"MAE: ${mae:,.2f}")
    print(f"R²: {r2:.4f}")
    return model, X_test, y_test, pred


def save_model(model, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
