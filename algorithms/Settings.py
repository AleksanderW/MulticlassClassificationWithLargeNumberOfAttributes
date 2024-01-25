from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# from stability_selection.stability_selection import StabilitySelection
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

external_shuffles = 5
train_size = 0.9

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
    (KNeighborsClassifier(n_neighbors=1, n_jobs=-1, metric="manhattan"), '1NN_MUL'),
    (OneVsOneClassifier(KNeighborsClassifier(n_neighbors=1, n_jobs=-1, metric="manhattan"), n_jobs=-1), '1NN_OVO'),
    (OneVsRestClassifier(KNeighborsClassifier(n_neighbors=1, n_jobs=-1, metric="manhattan"), n_jobs=-1), '1NN_OVR'),
    (KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="manhattan"), '3NN_MUL'),
    (OneVsOneClassifier(KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="manhattan"), n_jobs=-1), '3NN_OVO'),
    (OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="manhattan"), n_jobs=-1), '3NN_OVR'),
    (KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric="manhattan"), '5NN_MUL'),
    (OneVsOneClassifier(KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric="manhattan"), n_jobs=-1), '5NN_OVO'),
    (OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric="manhattan"), n_jobs=-1), '5NN_OVR'),
    (KNeighborsClassifier(n_neighbors=7, n_jobs=-1, metric="manhattan"), '7NN_MUL'),
    (OneVsOneClassifier(KNeighborsClassifier(n_neighbors=7, n_jobs=-1, metric="manhattan"), n_jobs=-1), '7NN_OVO'),
    (OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7, n_jobs=-1, metric="manhattan"), n_jobs=-1), '7NN_OVR'),
    (SVC(), 'SVC'),
    (MLPClassifier(max_iter=2000), 'MLP'),
    (RandomForestClassifier(n_jobs=-1), 'RND_FST'),
    (DecisionTreeClassifier(), 'DEC_TRE'),
    (BaggingClassifier(n_jobs=-1), 'BAGGNG'),
    # (StackingClassifier(
    #     estimators=[('rf', RandomForestClassifier(n_jobs=-1)),
    #                 ('ext', ExtraTreesClassifier(n_jobs=-1)),
    #                 ('dt', DecisionTreeClassifier()),
    #                 ('knn1', KNeighborsClassifier(n_neighbors=1, metric="manhattan", n_jobs=-1)),
    #                 ('knn3', KNeighborsClassifier(n_neighbors=3, metric="manhattan", n_jobs=-1)),
    #                 ('knn5', KNeighborsClassifier(n_neighbors=5, metric="manhattan", n_jobs=-1))],
    #     final_estimator=LogisticRegression(n_jobs=-1), n_jobs=-1), 'STACKING')
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

# ALG_models = [
#     RandomForestClassifier(n_estimators=50, max_features="log2", n_jobs=-1),
#     SVC(kernel="linear", probability=True, C=0.1),
#     DecisionTreeClassifier(class_weight="balanced"),
#     MLPClassifier(max_iter=2000, hidden_layer_sizes=(50,), activation="identity"),
#     KNeighborsClassifier(n_neighbors=1, p=1),
#     KNeighborsClassifier(n_neighbors=3, p=1),
#     KNeighborsClassifier(n_neighbors=5, p=1)
# ]

ALG_models = [
    [
        RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1),
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    ],
    [
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42),
        MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=2000, random_state=42),
        MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=2000, random_state=42),
    ],
    [
        SVC(kernel='linear', C=1, random_state=42),
        SVC(kernel='rbf', C=1, random_state=42),
        SVC(kernel='poly', C=1, random_state=42),
    ],
    [
        KNeighborsClassifier(n_neighbors=1, metric="manhattan", n_jobs=-1),
        KNeighborsClassifier(n_neighbors=3, metric="manhattan", n_jobs=-1),
        KNeighborsClassifier(n_neighbors=5, metric="manhattan", n_jobs=-1),
    ],
]
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
# from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
# from sklearn.svm import SVC, NuSVC, LinearSVC

# group1 = [
#     RandomForestClassifier(n_jobs=-1),
#     DecisionTreeClassifier()
# ]
#
# group2 = [
#     KNeighborsClassifier(n_neighbors=1, p=1),
#     KNeighborsClassifier(n_neighbors=3, p=1),
#     KNeighborsClassifier(n_neighbors=5, p=1)
# ]
#
# group3 = [
#     LogisticRegression(),
#     RidgeClassifier(),
#     SGDClassifier()
# ]
#
# group4 = [
#     GaussianNB(),
#     MultinomialNB(),
#     BernoulliNB()
# ]
#
# group5 = [
#     SVC(),
#     NuSVC(),
#     LinearSVC()
# ]
#
# group6 = [
#     GradientBoostingClassifier(),
#     AdaBoostClassifier()
# ]
