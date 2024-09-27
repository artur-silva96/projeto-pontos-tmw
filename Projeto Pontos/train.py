# %%
import pandas as pd
from sklearn import model_selection

df = pd.read_csv('data/dados_pontos.csv',
                 sep=';')
df

# %%
features = df.columns[3:-1]
target = 'flActive'

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=df[target])

print(f'Taxa de resposta de Treino: {y_train.mean()}')
print(f'Taxa de resposta de Teste: {y_test.mean()}')

# %%
input_avgRecorrencia = X_train['avgRecorrencia'].max()

X_train['avgRecorrencia'] = X_train['avgRecorrencia'].fillna(input_avgRecorrencia)
X_test['avgRecorrencia'] = X_test['avgRecorrencia'].fillna(input_avgRecorrencia)

# %%
from sklearn import tree
from sklearn import metrics

# Training the model
arvore = tree.DecisionTreeClassifier(max_depth=5,
                                     min_samples_leaf=50,
                                     random_state=42)
arvore.fit(X_train, y_train)

# Predicting on train base
tree_pred_train = arvore.predict(X_train)
tree_acc_train = metrics.accuracy_score(y_train, tree_pred_train)
print(f'Train tree accuracy: {tree_acc_train}')

# Predicting on test base
tree_pred_test = arvore.predict(X_test)
tree_acc_test = metrics.accuracy_score(y_test, tree_pred_test)
print(f'Test tree accuracy: {tree_acc_test}')

# Predicting probability on train base
tree_proba_train = arvore.predict_proba(X_train)[:,1]
tree_acc_train = metrics.roc_auc_score(y_train, tree_proba_train)
print(f'Train tree AUC: {tree_acc_train}')

# Predicting probability on test base
tree_proba_test = arvore.predict_proba(X_test)[:,1]
tree_acc_test = metrics.roc_auc_score(y_test, tree_proba_test)
print(f'Test tree AUC: {tree_acc_test}')
