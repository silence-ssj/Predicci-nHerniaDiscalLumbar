import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# 1) Cargar el dataset
df = pd.read_csv("column_2C.csv", sep=r"\s*[, ]\s*", header=None, engine="python")

# 2) Asignar nombres a las columnas
df.columns = [
    "pelvic_incidence",
    "pelvic_tilt",
    "lumbar_lordosis_angle",
    "sacral_slope",
    "pelvic_radius",
    "degree_spondylolisthesis",
    "class",
]

# 3) Limpieza rápida
df = df.dropna().drop_duplicates()

# 4) Convertir etiqueta NO/AB a 0/1
df["class"] = df["class"].str.upper().map({"NO": 0, "AB": 1})

# 5) Dividir 80-20 (estratificado)
X = df.drop("class", axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Normalización de datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6) Definir modelos y búsqueda de hiperparámetros

# 6.1 SVM con escalado
svm_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
    ]
)
svm_grid = {"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", "auto"]}
svm_search = GridSearchCV(svm_pipe, svm_grid, cv=5, scoring="f1", n_jobs=-1)

# 6.2 XGBoost o GradientBoosting
try:
    from xgboost import XGBClassifier

    gb_model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42
    )
    gb_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
except ImportError:
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3],
    }

gb_search = GridSearchCV(gb_model, gb_grid, cv=5, scoring="f1", n_jobs=-1)

# 7) Entrenar ambos modelos
print("⏳ Entrenando modelos...")
svm_search.fit(X_train, y_train)
gb_search.fit(X_train, y_train)

# 8) Función de métricas
def get_metrics(model, X_t, y_t):
    preds = model.predict(X_t)
    return {
        "accuracy": accuracy_score(y_t, preds),
        "precision": precision_score(y_t, preds),
        "recall": recall_score(y_t, preds),
        "f1": f1_score(y_t, preds),
    }

svm_metrics = get_metrics(svm_search.best_estimator_, X_test, y_test)
gb_metrics = get_metrics(gb_search.best_estimator_, X_test, y_test)

# 9) Generar tabla de métricas para el artículo
df_metrics = pd.DataFrame(
    {
        "Modelo": ["SVM", "XGBoost"],
        "Precisión": [svm_metrics["accuracy"], gb_metrics["accuracy"]],
        "Sensibilidad": [svm_metrics["recall"], gb_metrics["recall"]],
        "F1-score": [svm_metrics["f1"], gb_metrics["f1"]], 
        "Pérdida": [1 - svm_metrics["accuracy"], 1 - gb_metrics["accuracy"]],}
)

print("\n🔹 Tabla de métricas del modelo:")
print(df_metrics.to_markdown())

# 10) Seleccionar y guardar el mejor modelo (por F1)
best_model = (
    svm_search.best_estimator_
    if svm_metrics["f1"] >= gb_metrics["f1"]
    else gb_search.best_estimator_
)

joblib.dump(best_model, "mejor_modelo_hernia.pkl")
print("\n✅ Modelo guardado como 'mejor_modelo_hernia.pkl'")

# 11# Forzar visualización de importancia usando XGBoost aunque no haya sido el mejor
if hasattr(gb_search.best_estimator_, "feature_importances_"):
    importances = gb_search.best_estimator_.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(X.columns, importances, color="darkorange")
    plt.xlabel("Importancia")
    plt.ylabel("Características")
    plt.title("Importancia de características (XGBoost)")
    plt.tight_layout()
    plt.savefig("importancia_caracteristicas.png")
    plt.show()

# 12) Comparación de métricas entre modelos
plt.figure(figsize=(8, 5))
bar_width = 0.35
index = np.arange(len(svm_metrics))

plt.bar(index, list(svm_metrics.values()), bar_width, label="SVM")
plt.bar(index + bar_width, list(gb_metrics.values()), bar_width, label="XGBoost")

plt.xlabel("Métrica")
plt.ylabel("Valor")
plt.title("Comparación de modelos por rendimiento")
plt.xticks(index + bar_width / 2, list(svm_metrics.keys()))
plt.legend()
plt.tight_layout()
plt.savefig("comparacion_caracteristicas.png")
plt.show()

# 13) Distribución de características
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(X.columns):
    axes[i].hist(X[feature], bins=20, color="steelblue", alpha=0.7)
    axes[i].set_title(f"Distribución de {feature}")

plt.tight_layout()
plt.savefig("distribucion_caracteristicas.png")
plt.show()

# 14) Diagrama de dispersión
plt.scatter(
    df["pelvic_tilt"],
    df["lumbar_lordosis_angle"],
    c=df["class"],
    cmap="coolwarm",
    alpha=0.7,
)
plt.xlabel("Inclinación Pélvica (°)")
plt.ylabel("Ángulo de Lordosis Lumbar (°)")
plt.title("Relación entre inclinación pélvica y lordosis lumbar")
plt.colorbar(label="Clase (0: Normal, 1: Hernia)")
plt.tight_layout()
plt.savefig("dispersion_inclinacion_lordosis.png")
plt.show()

# 15) Matriz de Confusión
cm = confusion_matrix(y_test, best_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión del Modelo Predictivo")
plt.tight_layout()
plt.savefig("matriz_confusion.png")
plt.show()

# 16) CURVA DE APRENDIZAJE (accuracy train/validación)
from sklearn.model_selection import learning_curve, StratifiedKFold


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
)

# Promedio y desviación
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", label="Entrenamiento")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)

plt.plot(train_sizes, val_mean, "o-", label="Validación")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

plt.xlabel("Número de ejemplos de entrenamiento")
plt.ylabel("Accuracy")
plt.title("Curva de aprendizaje del modelo seleccionado")
plt.legend()
plt.tight_layout()
plt.savefig("curva_aprendizaje.png")
plt.show()


# 17) CURVA PRECISIÓN  vs.  PÉRDIDA
precision_train = train_mean          
precision_val   = val_mean
loss_train = 1 - precision_train
loss_val   = 1 - precision_val

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, precision_val, "o-", color="steelblue", label="Precisión (valid.)")
plt.plot(train_sizes, loss_val,      "o-", color="firebrick",   label="Pérdida (valid.)")

plt.xlabel("Número de ejemplos de entrenamiento")
plt.ylabel("Valor")
plt.title("Evolución de precisión y pérdida")
plt.legend()
plt.tight_layout()
plt.savefig("curva_precision_perdida.png")
plt.show()
