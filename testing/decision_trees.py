import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# print iris.feature_names
# print iris.target_names

# training data
test_ids = [0, 50, 100]
train_target = np.delete(iris.target, test_ids)
train_data = np.delete(iris.data, test_ids, axis = 0)

# testing data
test_target = iris.target[test_ids]
test_data = iris.data[test_ids]

# classifying
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# # testing classifier
# print test_target
# print clf.predict(test_data)

# visualizing tree
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names=iris.feature_names,
                        class_names=iris.target_names,
                        filled=True, rounded=True,
                        special_characters=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write('iris.pdf')
