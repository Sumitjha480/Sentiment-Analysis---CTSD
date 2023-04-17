from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from UnigramTfFeatureGeneration import create_feature_set_and_labels
from UnigramTfifdFeaturesetGeneration import get_features


def begin_test(train_x, test_x, train_y, test_y):
    x = train_x + test_x
    y = train_y + test_y

    clf1 = SVC()
    clf2 = MLPClassifier()
    clf3 = MultinomialNB()

    for label, clf in zip(
            ['SVCClassifier', 'NeuralNetworkClassifier', 'MultinomialNB'],
            [clf1, clf2, clf3]):
        scores = cross_val_score(clf, x, y, cv=10, scoring='accuracy')
        f_measure = cross_val_score(clf, x, y, cv=10, scoring='f1')
        # print(scores)
        print(label, "Accuracy:  ", scores.mean(), "+/- ", scores.std())
        print(label, "F-measure:  ", f_measure.mean())


def test_by_tf():
    train_x, train_y, test_x, test_y = create_feature_set_and_labels \
        ('pos_hindi.txt', 'neg_hindi.txt')
    begin_test(train_x, test_x, train_y, test_y)


def test_by_tfidf():
    train_x, train_y, test_x, test_y = get_features()
    begin_test(train_x, test_x, train_y, test_y)


if __name__ == '__main__':
    print("="*10)
    print("Unigram+Tf Accuracies")
    test_by_tf()
    print("=" * 10)
    print("Unigram+Tfidf Accuracies")
    test_by_tfidf()
