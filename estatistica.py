import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy import stats

# Carregar o conjunto de dados
data = pd.read_csv("WorldHappiness_Corruption_2019.csv")

# 1. Escolher uma variável quantitativa como variável resposta (y) e as demais como covariáveis (X)
X = data[["cpi_score", "gdp_per_capita", "family", "health"]]  # Substitua pelas suas covariáveis
y = data["happiness_score"]  # Substitua pela sua variável resposta

# 2. Dividir os dados em treino (70%), validação (10%) e teste (20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.125, shuffle=True, random_state=42)

# 3. Definir uma função para ajustar e avaliar os modelos
def fit_evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    return mae

# 4. Ajustar modelo KNN e Regressão linear
knn_model = KNeighborsRegressor()
linear_model = LinearRegression()

# 5. Usar conjunto de validação para escolher os melhores parâmetros dos modelos
# (Neste exemplo, vou ajustar apenas o número de vizinhos no KNN)
k_values = [3, 5, 7]  # Experimente diferentes valores para o número de vizinhos
best_mae_knn = float("inf")
best_knn_model = None

for k in k_values:
    knn_model = KNeighborsRegressor(n_neighbors=k)  # Cria uma nova cópia do modelo em cada iteração
    mae = fit_evaluate_model(knn_model, X_train, y_train, X_val, y_val)
    if mae < best_mae_knn:
        best_mae_knn = mae
        best_knn_model = knn_model

# 6. Ajustar o modelo escolhido na base de treino + validação
best_knn_model.fit(X_trainval, y_trainval)

# 7. Testar o modelo no conjunto de teste e calcular o MAE
y_pred_test_knn = best_knn_model.predict(X_test)
mae_test_knn = mean_absolute_error(y_test, y_pred_test_knn)

# 8. Repetir 30 vezes
num_repeticoes = 30
mae_scores_knn = []
mae_scores_linear = []

for _ in range(num_repeticoes):
    X_trainval, X_test, y_trainval, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.125, shuffle=True, random_state=42)
    
    best_knn_model = KNeighborsRegressor(n_neighbors=best_knn_model.n_neighbors)
    mae_knn = fit_evaluate_model(best_knn_model, X_train, y_train, X_val, y_val)
    mae_scores_knn.append(mae_knn)
    
    mae_linear = fit_evaluate_model(linear_model, X_train, y_train, X_val, y_val)
    mae_scores_linear.append(mae_linear)

# 9. Estimativa pontual e intervalar (95%) para o erro médio para o KNN
mean_mae_knn = np.mean(mae_scores_knn)
lower_bound_knn, upper_bound_knn = stats.t.interval(0.95, len(mae_scores_knn) - 1, loc=mean_mae_knn, scale=stats.sem(mae_scores_knn))

# Estimativa pontual e intervalar (95%) para o erro médio para a Regressão linear
mean_mae_linear = np.mean(mae_scores_linear)
lower_bound_linear, upper_bound_linear = stats.t.interval(0.95, len(mae_scores_linear) - 1, loc=mean_mae_linear, scale=stats.sem(mae_scores_linear))

# 10. Concluir qual modelo é melhor (usando o MAE no conjunto de teste)
linear_mae = fit_evaluate_model(linear_model, X_train, y_train, X_val, y_val)

if linear_mae < mae_test_knn:
    best_model = "Linear Regression"
else:
    best_model = "KNN"

# Imprimir resultados
print("MAE (KNN) no conjunto de teste:", mae_test_knn)
print("Estimativa pontual do erro médio (KNN):", mean_mae_knn)
print("Intervalo de confiança (95%) para o erro médio (KNN):", (lower_bound_knn, upper_bound_knn))
print("MAE (Regressão Linear) no conjunto de validação:", linear_mae)
print("Estimativa pontual do erro médio (Regressão Linear):", mean_mae_linear)
print("Intervalo de confiança (95%) para o erro médio (Regressão Linear):", (lower_bound_linear, upper_bound_linear))
print("Melhor modelo:", best_model)
