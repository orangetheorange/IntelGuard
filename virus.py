import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main():
    # ensure working directory is script folder
    script_path = Path(__file__).resolve()
    os.chdir(script_path.parent)

    # reproducibility
    np.random.seed(42)

    # load data
    data = pd.read_csv("synthetic_malware_suspiciousness_dataset.csv")
    X = data.drop(columns=["Suspiciousness Level"]).values
    y = data["Suspiciousness Level"].values

    # feature selection (keep up to 50 best)
    selector = SelectKBest(mutual_info_regression, k=min(50, X.shape[1]))
    X_selected = selector.fit_transform(X, y)

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

    # Convert the trained model to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save it
    with open("linear_regression.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    # predict on test set
    import joblib

    joblib.dump(selector, "selector.pkl")
    joblib.dump(scaler, "scaler.pkl")

    y_pred = model.predict(X_test)

    # evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"mean squared error: {mse:.4f}")
    print(f"r2 score: {r2:.4f}")


    # plot actual vs predicted
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", linewidth=1)
    plt.xlabel("actual suspiciousness")
    plt.ylabel("predicted suspiciousness")
    plt.title("actual vs predicted")
    plt.tight_layout()
    plt.savefig("regression_results.png")
    plt.show()



if __name__ == "__main__":
    main()

