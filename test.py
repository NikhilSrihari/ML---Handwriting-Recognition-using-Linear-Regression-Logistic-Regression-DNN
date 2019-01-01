import sklearn.datasets as ds

iris = ds.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

print(X.shape)
print(y.shape)
print()
print(X)
print(y)