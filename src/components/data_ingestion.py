import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

t0 = time.time()
print("Loading MNIST data")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

filter_mask = (y == 1) | (y == 7)
X_filtered = X[filter_mask]
y_filtered = y[filter_mask]


X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, stratify=y_filtered, random_state=42)

print(X_train.shape,X_test.shape)
print(f"[TIME] Data loading took {time.time()-t0:.2f}s")
