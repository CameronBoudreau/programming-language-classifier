from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
import random
import glob

dataset = load_files('data', encoding='UTF-8', decode_error='replace')

seed = random.randint(1, 1000)
X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    test_size=0.2,
                                                    random_state=seed)

vec = CountVectorizer(ngram_range=(1, 3))
X_train_counts = vec.fit_transform(X_train)

tf_transformer = TfidfTransformer()
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

print("Seed: {}\nTrain classifier Score: {}".format(seed,
                                                    clf.score(X_train_tfidf,
                                                              y_train)))

X_test_counts = vec.transform(X_test)
X_test_tfidf = tf_transformer.transform(X_test_counts)


print("Seed: {}\nTest classifier Score: {}".format(seed,
                                                   clf.score(X_test_tfidf,
                                                             y_test)))

def get_file(code):
    f = open(code)
    text = [f.read()]
    return text


tests = glob.glob('test/*')


def get_lang(code_file):
    test_doc = get_file(code_file)
    X_new_counts = vec.transform(test_doc)
    X_new_tfidf = tf_transformer.transform(X_new_counts)

    predicted = dataset.target_names[clf.predict(X_new_tfidf)[0]]

    return predicted


def run_tests(tests):
    results = []
    for t in tests:
        results.append(get_lang(t))
    return(results)

results = run_tests(tests)
test_expected = [('1', 'clojure'),
                 ('10', 'javascript'),
                 ('11', 'javascript'),
                 ('12', 'javascript'),
                 ('13', 'ruby'),
                 ('14', 'ruby'),
                 ('15', 'ruby'),
                 ('16', 'haskell'),
                 ('17', 'haskell'),
                 ('18', 'haskell'),
                 ('19', 'scheme'),
                 ('2', 'clojure'),
                 ('20', 'scheme'),
                 ('21', 'scheme'),
                 ('22', 'java'),
                 ('23', 'java'),
                 ('24', 'scala'),
                 ('25', 'scala'),
                 ('26', 'tcl'),
                 ('27', 'tcl'),
                 ('28', 'php'),
                 ('29', 'php'),
                 ('3', 'clojure'),
                 ('30', 'php'),
                 ('31', 'ocaml'),
                 ('32', 'ocaml'),
                 ('4', 'clojure'),
                 ('5', 'python'),
                 ('6', 'python'),
                 ('7', 'python'),
                 ('8', 'python'),
                 ('9', 'javascript')]

both = zip(test_expected, results)
count = 0
failed_on = []

for i in both:
    correct = 'WRONG'
    if i[0][1] == i[1]:
        correct = 'CORRECT'
        count += 1
    else:
        if i[0][1] != 'tcl':
            failed_on.append(i[0][1])
    print('Test {}: Expected: {}; Predicted: {}; {}'.format(i[0][0],
                                                            i[0][1], i[1],
                                                            correct))

print('Percent correct: {}'.format(count/32))
print('Failed on: ', failed_on)
