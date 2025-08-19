# Telecom X - Parte 2: Predicción de Churn de Clientes

![Header](visualizaciones/header_churn.png)  
*Análisis predictivo para reducir la cancelación de clientes*

## 📌 Propósito del Proyecto
**Objetivo Principal**: Desarrollar un modelo predictivo que identifique clientes con alto riesgo de cancelación (churn) en una compañía de telecomunicaciones, utilizando variables clave como antigüedad, tipo de contrato y gastos mensuales.  

**Impacto Esperado**:  
- Reducir la tasa de churn en un 15-20%.  
- Optimizar estrategias de retención con enfoque en clientes de alto riesgo.

---

## 🗂 Estructura del Proyecto

| Directorio          | Archivo/Subdirectorio               | Descripción                                  |
|---------------------|-------------------------------------|---------------------------------------------|
| `datos/`            |                                     | Contiene todos los archivos de datos        |
|                     | `raw/`                             | Datos originales sin procesar               |
|                     | `processed/`                       | Datos limpios y transformados               |
|                     | `processed/datos_balanceados.csv`  | Dataset con clases balanceadas              |
|                     | `processed/datos_estandarizados.csv`| Dataset con variables normalizadas         |
| `notebooks/`        | `telecom_churn_analysis.ipynb`     | Cuaderno principal de análisis y modelado  |
| `visualizaciones/`  |                                     | Gráficos generados durante el análisis     |
|                     | `matriz_correlacion.png`           | Matriz de correlación entre variables      |
|                     | `importancia_variables.png`        | Importancia de variables en Random Forest  |
|                     | `curva_roc.png`                    | Curva ROC comparando modelos               |
|                     | `contract_churn.png`               | Relación entre tipo de contrato y churn    |
| `modelos/`          |                                     | Modelos entrenados serializados            |
|                     | `random_forest.pkl`                | Modelo Random Forest entrenado             |
|                     | `logistic_regression.pkl`          | Modelo Regresión Logística entrenado       |
| `README.md`         |                                     | Documentación principal del proyecto       |

---

## 🛠 Preparación de Datos

### 1. Clasificación de Variables
| Tipo          | Ejemplos                          | Tratamiento Aplicado          |
|---------------|-----------------------------------|--------------------------------|
| **Numéricas** | `tenure`, `MonthlyCharges`        | Estandarización (StandardScaler) |
| **Categóricas** | `Contract`, `PaymentMethod`     | One-Hot Encoding               |

### 2. Procesamiento
- **Balanceo de clases**: SMOTE para igualar muestras de churn/no-churn (relación 50:50).  
- **Normalización**: Estandarización de variables numéricas para modelos sensibles a escala (Regresión Logística).  
- **División de datos**: 70% entrenamiento / 30% prueba (`random_state=42` para reproducibilidad).

### 3. Justificación de Decisiones
- **Random Forest vs Regresión Logística**:  
  - RF para capturar relaciones no lineales (mejor recall: 65%).  
  - Regresión Logística como baseline interpretable.  
- **Métricas Prioritarias**: Recall y AUC-ROC (para minimizar falsos negativos).

---

## 📊 Insights Clave del EDA

### 1. Factores Críticos de Churn
![Correlación](visualizaciones/matriz_correlacion.png)  
*Variables como `tenure` y `MonthlyCharges` muestran alta correlación con el churn.*

### 2. Impacto del Tipo de Contrato
```python
sns.boxplot(data=df, x='Contract', y='tenure', hue='Churn_binary')
```
---

## 🚀 Cómo Ejecutar el Proyecto
Requisitos Previos

```python
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn
```

## Pasos

1. Cargar datos procesados:

```python
df = pd.read_csv('datos/processed/datos_balanceados.csv')
```

2. Ejecutar el cuaderno:

   Abrir notebooks/telecom_churn_analysis.ipynb en Google Colab o Jupyter.

   Ejecutar celdas en orden (incluye comentarios explicativos).

Opcional: Usar Modelos Pre-entrenados

```python
import joblib
model = joblib.load('modelos/random_forest.pkl')
model.predict(X_test)
```

## 📈 Resultados Destacados

| Modelo               | Recall (Churn) | Precisión (Churn) | F1-Score (Churn) | AUC-ROC | Variables Clave                           |
|----------------------|----------------|-------------------|------------------|---------|-------------------------------------------|
| Random Forest        | 65%            | 76%               | 70%              | 0.90    | `tenure`, `MonthlyCharges`                |
| Regresión Logística  | 54%            | 67%               | 60%              | 0.83    | `Contract_Month-to-month`, `TotalCharges` |
| KNN (k=5)            | 62%            | 72%               | 67%              | 0.85    | `tenure`, `Contract_One year`             |

---

## 🤝 Contribuciones

¿Quieres mejorar el proyecto?

    Reporta issues o envía un PR con mejoras.

    Sugiere nuevas variables para incluir en el análisis.

Contacto: Lgbenavid@gmail.com


