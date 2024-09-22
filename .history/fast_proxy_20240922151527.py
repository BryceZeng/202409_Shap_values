import catboost as ctb
from sklearn import model_selection

CV = model_selection.KFold(5, shuffle=False)

model = model_selection.RandomizedSearchCV(
    ctb.CatBoostRegressor(verbose=0, thread_count=-1, random_state=123),
    {"n_estimators": stats.randint(1, 300), "depth": [4, 6, 8, 10]},
    random_state=123,
    n_iter=20,
    refit=True,
    cv=CV,
    scoring="neg_mean_absolute_error",
).fit(X_train, y_train)



class ShapCatBoostRegressor(ctb.CatBoostRegressor):
    def predict_shap(self, X):
        return self.get_feature_importance(ctb.Pool(X), type="ShapValues")


ref_shap_val = model_selection.cross_val_predict(
    ShapCatBoostRegressor(
        **model.best_params_, verbose=0, thread_count=-1, random_state=123
    ),
    X_train,
    y_train,
    method="predict_shap",
    cv=CV,
)

shap_feat_importance = np.abs(ref_shap_val[:, :-1]).mean(0)
shap_feat_importance /= shap_feat_importance.sum()


import datetime
import itertools

param_combi = {'iters': range(5,125,5), 'depth': range(1,8)}
for i,d in itertools.product(*param_combi.values()):

    start_time = datetime.datetime.now()
    shap_val = model_selection.cross_val_predict(
        ShapCatBoostRegressor(
            n_estimators=i, depth=d,
            verbose=0, thread_count=-1, random_state=123
        ),
        X_train, y_train,
        method='predict_shap',
        cv=CV
    )
    end_time = datetime.datetime.now()
    delta = (end_time - start_time).total_seconds()

    result.append({
        "time": delta,
        "iters": i,
        "depth": d,
        "r2_feat_shap": np.average(
                metrics.r2_score(
                    ref_shap_val[:,:-1], shap_val[:,:-1],
                    multioutput='raw_values'
                ).round(3)
            weights=shap_feat_importance
        ),
        "r2_sample_shap": metrics.r2_score(
            ref_shap_val[:,:-1].sum(1), shap_val[:,:-1].sum(1)
        ),
    })

result = pd.DataFrame(result)

def distance(time, r2, w_time=0.1, w_r2=0.9):
    return ((time - result.time.min()) **2) *w_time + \
            ((result.r2_feat_shap.max() - r2) **2) *w_r2

result['distance'] = result.apply(
    lambda x: distance(x.time, x.r2_feat_shap), axis=1
)
result = result.sort_values('distance')

proxy_model = ctb.CatBoostRegressor(
    n_estimators=result.head(1).iters.squeeze(),
    depth=result.head(1).depth.squeeze(),
    verbose=0, thread_count=-1, random_state=123,
).fit(X_train, y_train)