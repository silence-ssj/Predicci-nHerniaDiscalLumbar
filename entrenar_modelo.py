import warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, GridSearchCV, learning_curve, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score  
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# -------------------- 1. CARGA Y LIMPIEZA -------------------- #
df = pd.read_csv("column_2C.csv", sep=r"\s*[, ]\s*", header=None, engine="python")
df.columns = [
    "pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
    "sacral_slope", "pelvic_radius", "degree_spondylolisthesis", "class"
]
df = df.dropna().drop_duplicates()
df["class"] = (
    df["class"].astype(str).str.strip().str.upper().map({"NO": 0, "AB": 1})
)
if df["class"].isnull().any():
    raise ValueError("Etiquetas no reconocidas en columna 'class'.")

X, y = df.drop("class", axis=1), df["class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------- 2. DEFINICIÓN DE 5 MODELOS -------------------- #
try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("Instala XGBoost con: pip install xgboost")

models_cfg = {
    "SVM": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ]),
        "param_grid": {"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", "auto"]},
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(
            random_state=42, class_weight="balanced"
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
    },
    "LogReg": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=1000, class_weight="balanced", solver="liblinear"
            )),
        ]),
        "param_grid": {"lr__C": [0.1, 1, 10]},
    },
    "KNN": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]),
        "param_grid": {"knn__n_neighbors": [5, 7, 9]},
    },
}

# -------------------- 3. ENTRENAMIENTO CON GRIDSEARCHCV -------------------- #
print("⏳ Entrenando y validando 5 modelos...")
trained = {}
for name, cfg in models_cfg.items():
    search = GridSearchCV(
        cfg["estimator"], cfg["param_grid"],
        cv=5, scoring="f1", n_jobs=-1
    )
    search.fit(X_train, y_train)
    trained[name] = search
print("✅ Entrenamiento completo.\n")

# -------------------- 4. MÉTRICAS y VALIDACIÓN CRUZADA COMPLETA -------------------- #
def metrics(model, X_t, y_t):
    preds = model.predict(X_t)
    return {
        "accuracy":  accuracy_score(y_t, preds),
        "precision": precision_score(y_t, preds),
        "recall":    recall_score(y_t, preds),
        "f1":        f1_score(y_t, preds),
    }

# Métricas en test set
results = {name: metrics(g.best_estimator_, X_test, y_test)
           for name, g in trained.items()}

df_metrics = pd.DataFrame(results).T.round(4)
# **Agregar pérdida como complemento de la precisión**
df_metrics["Pérdida"] = 1 - df_metrics["accuracy"]
# Validación cruzada para f1-score en todo el dataset (entrenamiento + test)
print("🧪 Validación cruzada 5-fold (f1-score) para cada modelo:")
cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, g in trained.items():
    scores = cross_val_score(g.best_estimator_, X, y, cv=cv, scoring="f1", n_jobs=-1)
    cv_results[name] = scores
    print(f" - {name}: mean F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

print("\n🔹 Métricas en test set:\n")
print(df_metrics.to_markdown())

# -------------------- 5. SELECCIÓN Y GUARDADO -------------------- #
best_name = df_metrics["f1"].idxmax()
best_model = trained[best_name].best_estimator_
joblib.dump(best_model, "mejor_modelo_hernia.pkl")
print(f"\n🏆 Mejor modelo: {best_name} (F1 = {df_metrics.loc[best_name,'f1']:.4f})")
print("📦 Guardado en:  mejor_modelo_hernia.pkl\n")

# -------------------- 6. GRÁFICOS -------------------- #
# 6.1 Histograma de características
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, col in zip(axes.flatten(), X.columns):
    ax.hist(X[col], bins=20, color="steelblue", alpha=0.7)
    ax.set_title(col)
plt.tight_layout(); plt.savefig("01_distribucion_caracteristicas.png"); plt.close()

# 6.2 Dispersión clínico clave
plt.figure(figsize=(6,5))
plt.scatter(df["pelvic_tilt"], df["lumbar_lordosis_angle"],
            c=df["class"], cmap="coolwarm", alpha=0.7)
plt.xlabel("Pelvic tilt (°)"); plt.ylabel("Lumbar lordosis angle (°)")
plt.title("Dispersion: pelvic tilt vs lumbar lordosis")
plt.colorbar(label="Clase (0=NO, 1=AB)")
plt.tight_layout(); plt.savefig("02_dispersion_inclinacion_lordosis.png"); plt.close()

# 6.3 Curvas de aprendizaje (5 modelos)
def plot_lc(model, title, fname):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sizes, train_sc, val_sc = learning_curve(
        model, X, y, cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    plt.figure(figsize=(7,4))
    plt.plot(sizes, train_sc.mean(axis=1), "o-", label="Train")
    plt.plot(sizes, val_sc.mean(axis=1), "o-", label="Validation")
    plt.fill_between(sizes,
                     train_sc.mean(axis=1)-train_sc.std(axis=1),
                     train_sc.mean(axis=1)+train_sc.std(axis=1), alpha=0.15)
    plt.fill_between(sizes,
                     val_sc.mean(axis=1)-val_sc.std(axis=1),
                     val_sc.mean(axis=1)+val_sc.std(axis=1), alpha=0.15)
    plt.title(title); plt.xlabel("Training size"); plt.ylabel("Accuracy")
    plt.legend(); plt.tight_layout(); plt.savefig(fname); plt.close()

for name, g in trained.items():
    try:
        print(f"📈 Generando curva de aprendizaje para {name}...")
        print(f"Modelo es tipo: {type(g.best_estimator_)}")
        plot_lc(g.best_estimator_, f"Learning curve – {name}",
                f"03_curva_aprendizaje_{name}.png")
        print(f"✅ Curva guardada: 03_curva_aprendizaje_{name}.png\n")
    except Exception as e:
        print(f"❌ Error generando curva para {name}: {e}\n")
        
#CURVA ROC
plt.figure(figsize=(8, 6))
for name, g in trained.items():
    model = g.best_estimator_
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # Escalado 0-1
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Modelos")
plt.legend()
plt.tight_layout()
plt.savefig("07_curvas_ROC_modelos.png")
plt.close()

#ANÁLISIS DE ERRORES (FALSOS POSITIVOS Y NEGATIVOS)
y_pred = best_model.predict(X_test)
fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

print(f"\n🔎 Análisis de errores para {best_name}:")
print(f" - Falsos Positivos (NO etiquetados como AB): {len(fp_idx)}")
print(f" - Falsos Negativos (AB etiquetados como NO): {len(fn_idx)}")

print("\nPrimeros 3 falsos positivos (ejemplos):")
print(X_test.iloc[fp_idx[:3]])

print("\nPrimeros 3 falsos negativos (ejemplos):")
print(X_test.iloc[fn_idx[:3]])

#CURVAS DE PRECISIÓN Y PÉRDIDA
# Calcular curva de aprendizaje antes de usar train_sc y val_sc
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X, y, cv=cv, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Ahora sí, definir train_sc y val_sc correctamente
train_sc = train_scores.mean(axis=1)  
val_sc   = val_scores.mean(axis=1)

# Calcular precisión y pérdida
precision_train = train_sc          
precision_val   = val_sc
loss_train = 1 - precision_train
loss_val   = 1 - precision_val

# Graficar evolución de precisión y pérdida
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, precision_val, "o-", color="steelblue", label="Precisión (validación)")
plt.plot(train_sizes, loss_val, "o-", color="firebrick", label="Pérdida (validación)")
plt.xlabel("Número de ejemplos de entrenamiento")
plt.ylabel("Valor")
plt.title("Evolución de Precisión y Pérdida")
plt.legend()
plt.tight_layout()
plt.savefig("curva_precision_perdida.png")
plt.show()

#ANÁLISIS DE ERRORES (FALSOS POSITIVOS Y NEGATIVOS)
y_pred = best_model.predict(X_test)
fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

print(f"\n🔎 Análisis de errores para {best_name}:")
print(f" - Falsos Positivos (NO etiquetados como AB): {len(fp_idx)}")
print(f" - Falsos Negativos (AB etiquetados como NO): {len(fn_idx)}")

print("\nPrimeros 3 falsos positivos (ejemplos):")
print(X_test.iloc[fp_idx[:3]])

print("\nPrimeros 3 falsos negativos (ejemplos):")
print(X_test.iloc[fn_idx[:3]])


# 6.4 Barras comparativas de métricas
metrics_names = ["accuracy", "precision", "recall", "f1"]
bar_w = 0.15
idx = np.arange(len(metrics_names))
plt.figure(figsize=(9,5))
for i, (name, m) in enumerate(results.items()):
    plt.bar(idx + i*bar_w, [m[k] for k in metrics_names], bar_w, label=name)
plt.xticks(idx + bar_w*2, metrics_names)
plt.ylabel("Valor"); plt.title("Metric Comparison – 5 modelos")
plt.legend(); plt.tight_layout()
plt.savefig("04_comparacion_metricas.png"); plt.close()

# 6.5 Importancia de características (RF y XGB)
for t_name in ["RandomForest", "XGBoost"]:
    if t_name in trained:
        mdl = trained[t_name].best_estimator_
        if hasattr(mdl, "feature_importances_"):
            plt.figure(figsize=(7,4))
            plt.barh(X.columns, mdl.feature_importances_, color="darkorange")
            plt.title(f"Importance of features – {t_name}")
            plt.tight_layout(); plt.savefig(f"05_importancia_{t_name}.png"); plt.close()

# 6.6 Matriz de confusión del mejor modelo
cm = confusion_matrix(y_test, best_model.predict(X_test))
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title(f"confusion matrix – {best_name}")
plt.tight_layout(); plt.savefig("06_matriz_confusion.png"); plt.close()
