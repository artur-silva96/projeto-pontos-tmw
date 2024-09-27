# %%
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble
from sklearn import pipeline
from sklearn import linear_model
from sklearn import naive_bayes

from feature_engine import imputation

# %%
## Reading the dataset
df = pd.read_csv('data/dados_pontos.csv', sep=';')
df

# %%
## Defining features and target
features = df.columns.to_list()[3:-1]
target = 'flActive'

## Splitting the data into train/test features and target
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=df[target])

print(f'Taxa de resposta treino: {y_train.mean()}')
print(f'Taxa de resposta teste: {y_test.mean()}')

# %%
## Verifying the presence of Null/NaN values
X_train.isna().sum()

# %%
## Creating a Pipeline to automate the imputation of missing numbers
### Imputing the maximum occurence
max_avgRecorrencia = X_train['avgRecorrencia'].max()
impute_max = imputation.ArbitraryNumberImputer(variables=['avgRecorrencia'],
                                               arbitrary_number=max_avgRecorrencia)

### Imputing 0
features_impute_0 = ['qtdeRecencia',
                     'freqDias',
                     'freqTransacoes',
                     'qtdListaPresença',
                     'qtdChatMessage',
                     'qtdTrocaPontos',
                     'qtdResgatarPonei',
                     'qtdPresençaStreak',
                     'pctListaPresença',
                     'pctChatMessage',
                     'pctTrocaPontos',
                     'pctResgatarPonei',
                     'pctPresençaStreak',
                     'qtdePontosGanhos',
                     'qtdePontosGastos',
                     'qtdePontosSaldo'
                    ]
impute_0 = imputation.ArbitraryNumberImputer(variables=features_impute_0,
                                             arbitrary_number=0)

# %%
### Defining the model
# model = tree.DecisionTreeClassifier(max_depth=4,
#                                     min_samples_leaf=50,
#                                     random_state=42)

model = ensemble.RandomForestClassifier(random_state=42)

params = {
          'n_estimators': [100, 150, 250, 500],
          'min_samples_leaf': [10, 20, 30, 50, 100]
         }

grid = model_selection.GridSearchCV(model,
                                    param_grid=params,
                                    n_jobs=-1,
                                    scoring='roc_auc')

my_pipeline = pipeline.Pipeline([
                                 ('impute_0', impute_0),
                                 ('impute_max', impute_max),
                                 ('model', grid)
                                ])

# %%
## Training the model
my_pipeline.fit(X_train, y_train)

# %%
pd.DataFrame(grid.cv_results_)

# %%
print(f'Melhor estimador: {grid.best_estimator_}')
print(f'Melhores parametros: {grid.best_params_}')

# %%
y_train_predict = my_pipeline.predict(X_train)
y_train_proba = my_pipeline.predict_proba(X_train)[:,1]

y_test_predict = my_pipeline.predict(X_test)
y_test_proba = my_pipeline.predict_proba(X_test)

# %%
acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)
print(f'Accuracy train: {acc_train}')
print(f'Accuracy test: {acc_test}')

auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_test = metrics.roc_auc_score(y_test, y_test_proba[:,1])
print(f'AUC train: {auc_train}')
print(f'AUC test: {auc_test}')

## DecisionTree results:
# Accuracy train: 0.8109619686800895
# Accuracy test: 0.8008948545861297
# AUC train: 0.8531284015204619
# AUC test: 0.8380512447094162

## RandomForest results:
# Accuracy train: 0.8098434004474273
# Accuracy test: 0.8008948545861297
# AUC train: 0.8729569506419792
# AUC test: 0.8565284667546533

# %%
## Analysing feature importances
feature_importance = my_pipeline['model'].best_estimator_.feature_importances_
pd.Series(feature_importance, index=features).sort_values(ascending=False)

# %%
plt.figure(dpi=600)
skplt.metrics.plot_roc(y_test, y_test_proba)

# %%
skplt.metrics.plot_cumulative_gain(y_test, y_test_proba)

# %%
usuario_test = pd.DataFrame(
    {'verdadeiro': y_test,
     'proba': y_test_proba[:,1]
    }
)
usuario_test = usuario_test.sort_values('proba', ascending=False)
usuario_test['verdadeiro_sum'] = usuario_test['verdadeiro'].cumsum()
usuario_test['taxa_captura'] = usuario_test['verdadeiro_sum'] / usuario_test['verdadeiro'].sum()
usuario_test

# %%
skplt.metrics.plot_lift_curve(y_test, y_test_proba)

# %%
skplt.metrics.plot_ks_statistic(y_test, y_test_proba)

# %%
model_s = pd.Series(
    {'model': model,
     'features': features,
     'auc_test': auc_test}
)

model_s.to_pickle('data/modelo_rf.pkl')