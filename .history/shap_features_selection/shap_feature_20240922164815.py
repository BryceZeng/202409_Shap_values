import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from hyperopt import hp
from hyperopt import Trials
from xgboost import *
from shaphypetune import BoostSearch, BoostBoruta, BoostRFE, BoostRFA

import warnings
warnings.simplefilter('ignore')


### Plotting function
def binary_performances(y_true, y_prob, thresh=0.5, labels=['Positives','Negatives']):

    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, auc, roc_curve

    shape = y_prob.shape
    if len(shape) > 1:
        if shape[1] > 2:
            raise ValueError('A binary class problem is required')
        else:
            y_prob = y_prob[:,1]

    plt.figure(figsize=[15,4])

    # 1 -- Confusion matrix
    cm = confusion_matrix(y_true, (y_prob>thresh).astype(int))

    plt.subplot(131)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                     annot_kws={"size": 14}, fmt='g')
    cmlabels = ['True Negatives', 'False Positives',
               'False Negatives', 'True Positives']
    for i,t in enumerate(ax.texts):
        t.set_text(t.get_text() + "\n" + cmlabels[i])
    plt.title('Confusion Matrix', size=15)
    plt.xlabel('Predicted Values', size=13)
    plt.ylabel('True Values', size=13)

    # 2 -- Distributions of Predicted Probabilities of both classes
    plt.subplot(132)
    plt.hist(y_prob[y_true==1], density=True, bins=25,
             alpha=.5, color='green',  label=labels[0])
    plt.hist(y_prob[y_true==0], density=True, bins=25,
             alpha=.5, color='red', label=labels[1])
    plt.axvline(thresh, color='blue', linestyle='--', label='Boundary')
    plt.xlim([0,1])
    plt.title('Distributions of Predictions', size=15)
    plt.xlabel('Positive Probability (predicted)', size=13)
    plt.ylabel('Samples (normalized scale)', size=13)
    plt.legend(loc="upper right")

    # 3 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(133)
    plt.plot(fp_rates, tp_rates, color='orange',
             lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(fp/(fp+tn), tp/(tp+fn), 'bo', markersize=8, label='Decision Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve', size=15)
    plt.legend(loc="lower right")
    plt.subplots_adjust(wspace=.3)
    plt.show()

    tn, fp, fn, tp = [i for i in cm.ravel()]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2*(precision * recall) / (precision + recall)
    results = {
        "Precision": precision, "Recall": recall,
        "F1 Score": F1, "AUC": roc_auc
    }

    prints = [f"{kpi}: {round(score, 3)}" for kpi,score in results.items()]
    prints = ' | '.join(prints)
    print(prints)

    return results

### Train a xgboost model

df = pd.read_csv("../laptop_prices.csv")
df = df[
    [
        "Price_euros",
        "ScreenW",
        "ScreenH",
        "CPU_freq",
        "PrimaryStorage",
        "SecondaryStorage",
        "PrimaryStorageType",
        "Weight",
        "Screen"
    ]
]

# if screen is 'standard' then 1 else 0
df['Screen'] = df['Screen'].apply(lambda x: 1 if x == 'Standard' else 0)
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("Screen", axis=1), df.Screen, test_size=0.2, random_state=123
)
X_train["PrimaryStorageType"] = X_train["PrimaryStorageType"].astype("category")
X_test["PrimaryStorageType"] = X_test["PrimaryStorageType"].astype("category")
# fit a xgboost model for binary
model = XGBClassifier(enable_categorical=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
binary_performances(y_test, y_pred)


### FEATURE SELECTION BASED ON RECUSIVE FEATURE ELIMINATION
model_RFE = BoostRFE(model, min_features_to_select=1, step=1)
model_RFE.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=0,
)
model_RFE.estimator_, model_RFE.n_features_
binary_performances(y_test, model_RFE.predict_proba(X_test))


### FEATURE SELECTION BASED ON SHAP
model_shap = BoostBoruta(
    model,
    max_iter=200,
    perc=100,
    importance_type="shap_importances",
    train_importance=False,
)
model_shap.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=0,
)
model_shap.estimator_, model_shap.n_features_
binary_performances(y_test, model_shap.predict_proba(X_test))
## Show the feature selected
selected_features = model_shap.support_
feature_names = X_train.columns[selected_features]
print("Selected Features:", feature_names)
