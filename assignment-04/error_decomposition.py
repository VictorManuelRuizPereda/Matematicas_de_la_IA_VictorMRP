import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

np.random.seed(100)
X = np.sort(np.random.rand(35, 1) * 6, axis=0)
y = np.cos(X).ravel() + np.random.normal(0, 0.15, X.shape[0])

complexity_degrees = [1, 4, 12]
X_eval = np.linspace(0, 6, 100)[:, np.newaxis]

for degree in complexity_degrees:
    estimator = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.1))
    estimator.fit(X, y)
    predictions = estimator.predict(X_eval)
    print(f"Evaluation completed for polynomial architecture of degree: {degree}")

print("\n=== SYSTEM EXECUTION SUCCESSFUL ===")
print(f"Data matrices generated using custom variance settings.")