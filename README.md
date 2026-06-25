# 🧠 Clasificador de Memoria Jerárquica para Agentes de IA

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Transformers](https://img.shields.io/badge/SentenceTransformers-Multilingual-green)
![Status](https://img.shields.io/badge/Status-Pilot%20(Tesis)-success)
![License](https://img.shields.io/badge/Uso-Académico-lightgrey)

> Piloto práctico de la tesis de Maestría en Ciencia de Datos:
> **"Arquitectura de Memoria Jerárquica para Agentes de IA: Mejoramiento del Proceso de Inferencia sin Dependencia del Historial Completo"**.

---

## 📌 Descripción

Los Modelos de Lenguaje de Gran Escala (LLMs) operan dentro de una **ventana de contexto finita**. A medida que una conversación crece, inyectar el historial completo en cada inferencia se vuelve costoso, lento y, eventualmente, imposible: la información relevante se trunca o se diluye, degradando la calidad de las respuestas. Resolver esto requiere dejar de tratar la memoria del agente como un único bloque de texto y empezar a gestionarla de forma **jerárquica y selectiva**.

Este proyecto implementa el componente central de esa arquitectura: un **enrutador clasificador de texto** que, dado el input del usuario, decide a cuál de las **tres capas de memoria** debe acceder el agente —**Memoria de Trabajo** (contexto inmediato), **Memoria Episódica** (eventos de sesiones pasadas) y **Memoria Semántica** (hechos, preferencias y conocimiento permanente)—. Al recuperar únicamente la capa pertinente en lugar de todo el historial, el enrutador reduce el consumo de contexto y mejora la eficiencia de la inferencia. El notebook compara **rigurosamente** dos técnicas de clasificación para implementar este enrutador y valida estadísticamente cuál es superior.

---

## 📂 Estructura del Repositorio

| Archivo | Descripción |
|---|---|
| `proyecto_final.ipynb` | Notebook principal: EDA, pipelines, optimización de hiperparámetros, visualización t-SNE, evaluación final y validación estadística. |
| `dataset_intenciones_memoria.csv` | Dataset sintético etiquetado (columnas `texto` y `etiqueta`), 150 muestras balanceadas (50 por clase). |
| `README.md` | Este documento. |

---

## 🔬 Metodología y Desarrollo

### Dataset
- **150 muestras** de texto en español, perfectamente **balanceadas**: 50 por cada una de las 3 clases (Memoria de Trabajo, Episódica y Semántica).
- Análisis Exploratorio (EDA) que valida el balanceo, la longitud textual por clase y el vocabulario distintivo (nubes de palabras y top-10 tokens).

### Técnicas comparadas

| Técnica | Extracción de características | Clasificador |
|---|---|---|
| **A — Baseline** | `TfidfVectorizer` (uni + bigramas) | `LogisticRegression` |
| **B — Transfer Learning** | Sentence Embeddings `paraphrase-multilingual-MiniLM-L12-v2` (384 dims) | `LogisticRegression` |

### Rigor metodológico
- **Cero fuga de datos (Data Leakage):** el `train_test_split` se realiza al inicio con `stratify=y` (80/20), y el conjunto de Test permanece intocable hasta la evaluación final.
- **Pipelines de `scikit-learn`:** todo el preprocesamiento, la extracción de características y el clasificador se encapsulan en objetos `Pipeline`. La Técnica B usa un *transformer* custom (`BaseEstimator` + `TransformerMixin`) que envuelve `sentence-transformers` sin filtrar información entre folds.
- **Optimización de hiperparámetros:** el parámetro de regularización **`C`** de la Regresión Logística se optimiza mediante `GridSearchCV` con validación cruzada **`RepeatedStratifiedKFold`** (5 splits × 2 repeticiones).

---

## 📊 Resultados Clave

### Visualización del espacio de características (t-SNE)
La reducción de dimensionalidad con **t-SNE** muestra que el espacio TF-IDF presenta un alto solapamiento entre clases, mientras que los **Sentence Embeddings** generan clústeres densos y bien separados (coloreados por las etiquetas verdaderas), anticipando un mejor rendimiento de clasificación.

### Rendimiento sobre el Test Set (modelos optimizados)

| Técnica | Mejor `C` | Accuracy (Test) | F1-macro |
|---|---|---|---|
| **A — TF-IDF** | 100 | **≈ 87%** (0.867) | 0.86 |
| **B — Embeddings** | 0.01 | **≈ 97%** (0.967) | 0.97 |

La Técnica B alcanza un F1-Score de **1.00** en Memoria Semántica y de **0.95** en Memoria de Trabajo y Episódica.

### Validación estadística rigurosa
- Se recogen los scores por fold (10 folds) del mejor modelo de cada técnica.
- **Test de Normalidad (Shapiro-Wilk)** para decidir el test de hipótesis apropiado.
- Al cumplirse la normalidad, se aplica un **T-test pareado**, que confirma que la superioridad de la Técnica B es **estadísticamente significativa**: **p-value ≈ 0.0021 < α = 0.05**.

> La selección del test es dinámica: si los scores no fueran normales, el notebook aplicaría automáticamente el **Test de Wilcoxon signed-rank**.

---

## ▶️ Instrucciones de Ejecución (Google Colab)

1. **Abrir el notebook** `proyecto_final.ipynb` en [Google Colab](https://colab.research.google.com/).
2. **Subir el dataset** `dataset_intenciones_memoria.csv` al entorno de Colab (debe estar en el mismo directorio de trabajo que el notebook).
3. **Instalar dependencias** ejecutando la primera celda:
   ```bash
   !pip install -q sentence-transformers optuna wordcloud
   ```
   > La primera ejecución de la Técnica B descargará automáticamente el modelo `paraphrase-multilingual-MiniLM-L12-v2` (~120 MB).
4. **Ejecutar todo el notebook** de forma limpia con `Entorno de ejecución → Reiniciar y ejecutar todo` para reproducir el experimento completo y todas las figuras.

---

## 🤖 Documentación de Prompts (LLMs)

Para garantizar la trazabilidad y reproducibilidad del piloto, se documentan los prompts utilizados con modelos de lenguaje (LLMs) durante el desarrollo:

**P1 — Generación del dataset sintético (`dataset_intenciones_memoria.csv`):**
> "Genera 150 frases en español de usuarios dirigidas a un agente de IA, 50 por cada categoría: (0) Memoria de trabajo — referencias al contexto inmediato de la conversación; (1) Memoria episódica — referencias a eventos/sesiones pasadas con marcadores temporales; (2) Memoria semántica — hechos, preferencias y conocimiento permanente. Devuélvelas en CSV con columnas 'texto' y 'etiqueta'."

**P2 — Asistencia de código:**
> "Crea un transformer compatible con sklearn que envuelva sentence-transformers para usarlo dentro de un Pipeline y un GridSearchCV sin fuga de datos."

**P3 — Creación del README:**
> "Genera un archivo README.md profesional y académico para el repositorio de GitHub, basado estrictamente en el código y los textos del Jupyter Notebook del proyecto."

> Nota: todos los outputs de los LLMs fueron revisados y validados manualmente por el autor antes de su inclusión.

---

## 📚 Referencias

Las referencias en **formato IEEE** se encuentran directamente en el documento del notebook (`proyecto_final.ipynb`), en la sección final **"Referencias (IEEE)"**.
