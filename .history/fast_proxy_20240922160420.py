import catboost as ctb
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import catboost as ctb
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn import metrics

# Step 1: Load Data
df = pd.read_csv("laptop_prices.csv")
df = df[['Price_euros', 'ScreenW', 'ScreenH', 'Touchscreen',
    'RetinaDisplay', 'CPU_freq',
       'PrimaryStorage', 'SecondaryStorage', 'PrimaryStorageType','Weight']]

# Step 2: Split Data
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df.drop("Weight", axis=1), df.Weight, test_size=0.2, random_state=123
)

# Step 3: Encode Categorical Features
categorical_columns = X_train.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])  # Ensure the test set is also encoded
    label_encoders[col] = le

# Step 4: Define Cross-Validation
CV = 5  # Number of folds

# Step 5: Define SHAP Calculation Class
class ShapCatBoostRegressor(ctb.CatBoostRegressor):
    def predict_shap(self, X):
        return self.get_feature_importance(
            ctb.Pool(X), type='ShapValues'
        )

# Step 6: Model Initialization and Hyperparameter Tuning
model = RandomizedSearchCV(
    ShapCatBoostRegressor(verbose=0, thread_count=-1, random_state=123),
    {"n_estimators": [6, 10, 25, 50], "depth": [4, 6, 8, 10]},
    random_state=123,
    n_iter=20,
    refit=True,
    cv=CV,
    scoring="neg_mean_absolute_error",
)
model.fit(X_train, y_train)

# Step 7: Manual Cross-Validation for SHAP Values
kf = KFold(n_splits=CV, shuffle=True, random_state=123)
shap_values = []

# Reset indices of X_train and y_train
X_train_reset = X_train.reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

for train_index, test_index in kf.split(X_train_reset):
    X_train_fold, X_test_fold = X_train_reset.iloc[train_index], X_train_reset.iloc[test_index]
    y_train_fold, y_test_fold = y_train_reset.iloc[train_index], y_train_reset.iloc[test_index]

    model.best_estimator_.fit(X_train_fold, y_train_fold)
    shap_values_fold = model.best_estimator_.predict_shap(X_test_fold)
    shap_values.append(shap_values_fold)

# Step 8: Aggregate SHAP Values
ref_shap_val = np.concatenate(shap_values, axis=0)

# Step 9: Calculate Feature Importance
shap_feat_importance = np.abs(ref_shap_val[:, :-1]).mean(0)
shap_feat_importance /= shap_feat_importance.sum()

print("Feature Importance:", shap_feat_importance)

# Predict SHAP Values for the Single Row
single_row = X_test.iloc[[0]]  # Select the first row as an example

shap_values_single_row = model.best_estimator_.predict_shap(single_row)

# Print SHAP Values for the Single Row
print("SHAP Values for the Single Row:", shap_values_single_row)
