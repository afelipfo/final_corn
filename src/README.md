# 🌽 cornIA - Corn Disease Classification

Clasificación automática de enfermedades en maíz usando Deep Learning con **MobileNetV3Large**.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Descripción

Pipeline completo de Machine Learning para clasificar 4 tipos de condiciones en hojas de maíz:

- **Blight** (Tizón)
- **CommonRust** (Roya común)
- **GrayLeafSpot** (Mancha gris)
- **Healthy** (Sano)

### Características principales:

✅ **Transfer Learning** con MobileNetV3Large (ImageNet)
✅ **Data Augmentation** híbrida (offline + online)
✅ **Dataset balanceado** (916 imágenes por clase en train)
✅ **Callbacks inteligentes** (EarlyStopping, ReduceLROnPlateau)
✅ **Métricas completas** (Accuracy, Precision, Recall, F1, AUC)
✅ **MLflow** para tracking de experimentos
✅ **Google Colab** compatible (GPU L4)

---

## 🚀 Inicio Rápido

### **Opción 1: Google Colab (Recomendado)**

1. **Sube `data_augmented/` a Google Drive:**
   ```
   Mi unidad/
   └── data_augmented/
       ├── train/
       ├── val/
       └── test/
   ```

2. **Abre el notebook en Colab:**

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/afelipfo/cornIA/blob/main/corn_disease_training.ipynb)

3. **Activa GPU L4:**
   - Runtime → Change runtime type → GPU → L4

4. **Ejecuta las celdas secuencialmente**

---

### **Opción 2: Ejecución Local**

```bash
# 1. Clonar repositorio
git clone https://github.com/afelipfo/cornIA.git
cd cornIA

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar dataset (desde tu Google Drive)
# Coloca data_augmented/ en la raíz del proyecto

# 4. Ejecutar pipeline completo
python src/run_pipeline.py
```

---

## 📂 Estructura del Proyecto

```
cornIA/
├── src/                              # Scripts del pipeline
│   ├── training_config.py           # Paso 4: Configuración
│   ├── model_creation.py            # Paso 5: Modelo
│   ├── train_model.py               # Paso 6: Entrenamiento
│   ├── evaluate_and_export.py       # Paso 7: Evaluación
│   └── run_pipeline.py              # Orquestador completo
│
├── corn_disease_training.ipynb      # Notebook para Google Colab
├── requirements.txt                 # Dependencias Python
├── .gitignore                       # Archivos ignorados por Git
└── README.md                        # Este archivo

# Carpetas generadas durante ejecución (no en Git):
├── data_augmented/                  # Dataset (en Google Drive)
├── training_output/                 # Configuración + modelo entrenado
├── model_output/                    # Arquitectura del modelo
├── training_results/                # Gráficas + métricas
└── evaluation_results/              # Evaluación en test set
```

---

## 🎯 Pipeline de Entrenamiento

El pipeline consta de **4 pasos** principales:

### **Paso 4: Configuración de Entrenamiento**
- Data augmentation online (rotación, shifts, flips, brillo, zoom)
- Generadores de datos (train/val/test)
- Callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, etc.)

### **Paso 5: Creación del Modelo**
- MobileNetV3Large preentrenado (ImageNet)
- BatchNormalization
- Dropout (0.5)
- Dense (4 clases, softmax)

### **Paso 6: Entrenamiento**
- 50 épocas (con EarlyStopping)
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Loss: categorical_crossentropy
- Métricas: Accuracy, Precision, Recall, AUC

### **Paso 7: Evaluación y MLflow**
- Evaluación en test set (sin augmentation)
- Matriz de confusión
- Classification report
- Registro completo en MLflow

---

## 📊 Dataset

### Distribución (después de augmentation offline):

| Clase | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Blight | 916 | 177 | 176 | 1,269 |
| CommonRust | 916 | 196 | 196 | 1,308 |
| GrayLeafSpot | 916 | 87 | 87 | 1,090 |
| Healthy | 916 | 174 | 175 | 1,265 |
| **Total** | **3,664** | **634** | **634** | **4,932** |

**Características:**
- Imágenes: 256x256 px
- Formato: JPG
- Augmentation offline aplicada para balanceo perfecto en train
- Val/Test sin modificación (datos originales)

---

## 🔧 Configuración

### **Hiperparámetros principales:**

```python
RANDOM_SEED = 42           # Reproducibilidad
EPOCHS = 50                # Épocas máximas
BATCH_SIZE = 32            # Tamaño del batch
LEARNING_RATE = 0.001      # Tasa de aprendizaje inicial
```

### **Callbacks:**

| Callback | Parámetros |
|----------|------------|
| EarlyStopping | patience=10, monitor='val_loss' |
| ReduceLROnPlateau | factor=0.1, patience=5 |
| ModelCheckpoint | save_best_only=True, monitor='val_accuracy' |
| CSVLogger | training_history.csv |
| TensorBoard | logs en tensorboard_logs/ |

---

## 📈 Resultados

Los resultados varían según el entrenamiento, pero se espera:

- **Accuracy en test:** ~85-95%
- **Precision (macro):** ~85-93%
- **Recall (macro):** ~85-93%
- **F1-score (macro):** ~85-93%

Todos los resultados se guardan en:
- `training_results/training_history.png` - Gráficas de accuracy y loss
- `evaluation_results/confusion_matrix.png` - Matriz de confusión
- `evaluation_results/classification_report.csv` - Métricas detalladas

---

## 🛠️ Requisitos

### **Python 3.8+**

```bash
tensorflow>=2.12.0
keras>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
mlflow>=2.3.0
opencv-python>=4.7.0
```

Instalar todo:
```bash
pip install -r requirements.txt
```

---

## 📝 Uso en Google Colab

### **Pasos detallados:**

1. **Sube dataset a Google Drive:**
   - Carpeta: `/Mi unidad/data_augmented/`
   - Contenido: train/, val/, test/

2. **Abre el notebook:**
   - Haz clic en el badge "Open in Colab" arriba

3. **Configura GPU:**
   - Runtime → Change runtime type → GPU → L4

4. **Ejecuta sección por sección:**
   - Sección 1: Verifica GPU ✅
   - Sección 2: Instala dependencias
   - Sección 3: Monta Drive + clona GitHub
   - Secciones 4-7: Pipeline completo
   - Secciones 8-11: Visualización de resultados

5. **Descarga resultados:**
   - Sección 12: ZIP automático
   - Sección 13: Guardar en Drive

**Tiempo estimado:** 45-75 minutos (con GPU L4)

---

## 🔬 MLflow Tracking

Todos los experimentos se registran en MLflow:

```python
Experiment: "Maize-Disease-Classification"

Registra:
- Parámetros (batch_size, epochs, optimizer, etc.)
- Métricas (accuracy, precision, recall, f1, auc)
- Artefactos (gráficas, matrices, reportes)
- Modelo entrenado
- Versiones de dependencias
- Hardware utilizado
```

Ver resultados:
```bash
mlflow ui
# Abre http://localhost:5000
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -m 'Añade mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 👤 Autor

**Felipe Flórez**
- GitHub: [@afelipfo](https://github.com/afelipfo)
- Email: afelipeflorezo@gmail.com

---

## 🙏 Agradecimientos

- Dataset de imágenes de enfermedades de maíz
- MobileNetV3Large (Google AI)
- TensorFlow y Keras
- Google Colab por GPUs gratuitas

---

## 📚 Referencias

- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MLflow Documentation](https://mlflow.org/)

---

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!**
