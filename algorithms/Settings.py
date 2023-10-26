from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from stability_selection.stability_selection import StabilitySelection
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from algorithms.Constants import ALL_name, BTu_name, HeC_name, HFF_name, SPs_name, SSh_name

feature_selectors = [
    "RFE",
    # "RFECV",
    # "StabilitySelection"
]

steps = [
    5
]

n_splits = [
    5
]

parallel_evaluation = True

random_states = [
    13,
    # 24,
    # 321,
    # 943,
    # 1234,
    # 2315,
    # 3219,
    # 4324,
    # 6934,
    # 38523
]

datasets = [
    ALL_name,
    BTu_name,
    HeC_name,
    HFF_name,
    SPs_name,
    SSh_name
]

classifiers = [
    (KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=1), '3NN_MUL'),
    (OneVsOneClassifier(KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=1), n_jobs=-1), '3NN_OVO'),
    (OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3, n_jobs=-1, p=1), n_jobs=-1), '3NN_OVR'),
    (KNeighborsClassifier(n_neighbors=5, n_jobs=-1, p=1), '5NN_MUL'),
    (OneVsOneClassifier(KNeighborsClassifier(n_neighbors=5, n_jobs=-1, p=1), n_jobs=-1), '5NN_OVO'),
    (OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, n_jobs=-1, p=1), n_jobs=-1), '5NN_OVR'),
    (KNeighborsClassifier(n_neighbors=7, n_jobs=-1, p=1), '7NN_MUL'),
    (OneVsOneClassifier(KNeighborsClassifier(n_neighbors=7, n_jobs=-1, p=1), n_jobs=-1), '7NN_OVO'),
    (OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7, n_jobs=-1, p=1), n_jobs=-1), '7NN_OVR'),
    (SVC(), 'SVC'),
    (ExtraTreesClassifier(n_jobs=-1), 'EXT_TRE'),
    (RandomForestClassifier(n_jobs=-1), 'RND_FST'),
    (BaggingClassifier(n_jobs=-1), 'BAGGNG'),
    (StackingClassifier(
        estimators=[('rf', RandomForestClassifier(n_jobs=-1)), ('svc', SVC(kernel="linear", probability=True)),
                    ('dt', DecisionTreeClassifier()), ('mlp', MLPClassifier(max_iter=1000)),
                    ('knn1', KNeighborsClassifier(n_neighbors=1, p=1)),
                    ('knn3', KNeighborsClassifier(n_neighbors=3, p=1)),
                    ('knn5', KNeighborsClassifier(n_neighbors=5, p=1))],
        final_estimator=LogisticRegression(n_jobs=-1), n_jobs=-1), 'STACKIN')
]

# Parametry poniżej dotyczą Proponowanego Algorytmu
ALG_amount_of_subtables = [
    2,
    3,
    5,
    10
]

ALG_aggregations = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10"
]

ALG_kNN_neighbours = [
    [1, 3, 5],
    [3, 5, 7],
    [7, 9, 11]
]

ALG_orders = [
    "xuyager",
    # "lex1",
    # "lex2"
]

ALG_split_variant = [
    # 1,
    2,
    # 3
]
