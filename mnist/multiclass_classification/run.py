import define_classifiers

file_name = "../../../data/mnist/mnist_10digits.mat"
model_list = ['logistic_regression','knn', 'linear svm', 'kernel svm', 'neural network']

cls = define_classifiers.classifiers(file_name)

for model in model_list:
    cls.fit_model(model)
    print('{} complete'.format(model))