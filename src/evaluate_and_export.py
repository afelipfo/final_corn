"""
Paso 7: Evaluación y Exportación con MLflow
- Evalúa modelo en test set (sin augmentation)
- Genera matriz de confusión
- Calcula métricas completas (Accuracy, Recall por clase, Classification report)
- Exporta modelo a .keras
- Registra todo en MLflow (parámetros, métricas, artefactos, modelo)
- Coherente con pasos 4, 5 y 6
"""

import os
import sys
import json
import platform
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import mlflow
import mlflow.keras

# Importar configuración del Paso 4
from training_config import TrainingConfiguration


class ModelEvaluator:
    """Clase para evaluar y exportar el modelo con MLflow"""

    def __init__(self,
                 model_path='./training_output/best_model.keras',
                 output_dir='./evaluation_results',
                 random_seed=42):

        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_seed = random_seed

        # Cargar información de entrenamiento (Paso 6)
        self.training_info_path = Path('./training_results/training_info.json')

        # Resultados de evaluación
        self.evaluation_results = {
            'metadata': {
                'fecha_evaluacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': str(self.model_path),
                'random_seed': self.random_seed
            },
            'test_metrics': {},
            'confusion_matrix': None,
            'classification_report': {},
            'class_metrics': {}
        }

    def setup_mlflow(self):
        """Configura MLflow experiment"""
        print("=" * 80)
        print("PASO 1: CONFIGURACIÓN DE MLFLOW")
        print("=" * 80)

        # Configurar experiment
        mlflow.set_experiment("Maize-Disease-Classification")

        print("\n✓ MLflow experiment configurado: 'Maize-Disease-Classification'")
        print(f"  Tracking URI: {mlflow.get_tracking_uri()}")

    def load_training_info(self):
        """Carga información del entrenamiento (Paso 6)"""
        print("\n" + "=" * 80)
        print("PASO 2: CARGANDO INFORMACIÓN DE ENTRENAMIENTO")
        print("=" * 80)

        if self.training_info_path.exists():
            with open(self.training_info_path, 'r') as f:
                self.training_info = json.load(f)
            print(f"\n✓ Información de entrenamiento cargada: {self.training_info_path}")
        else:
            print(f"\n⚠️  Advertencia: No se encontró {self.training_info_path}")
            print("  Se usarán valores por defecto para MLflow")
            self.training_info = {
                'hyperparameters': {
                    'batch_size': 32,
                    'epochs': 50,
                    'optimizer': 'adam',
                    'learning_rate': 0.001
                },
                'environment': {
                    'versions': {},
                    'hardware': {}
                }
            }

    def load_test_generator(self):
        """Carga el generador de test desde el Paso 4"""
        print("\n" + "=" * 80)
        print("PASO 3: CARGANDO GENERADOR DE TEST (PASO 4)")
        print("=" * 80)

        print("\n📂 Inicializando TrainingConfiguration...")
        self.train_config = TrainingConfiguration(
            data_dir='./data_augmented',
            output_dir='./training_output',
            random_seed=self.random_seed
        )

        # Solo necesitamos configurar augmentation y crear generadores
        print("\n🔧 Configurando augmentation...")
        self.train_config.configure_augmentation()

        print("\n📂 Creando generadores...")
        self.train_config.create_data_generators()

        # Usar el generador de test (sin augmentation)
        self.test_generator = self.train_config.test_generator
        self.classes = self.train_config.classes

        print(f"\n✓ Generador de test cargado")
        print(f"  Test samples: {self.test_generator.samples}")
        print(f"  Clases: {self.classes}")
        print(f"  Batch size: {self.test_generator.batch_size}")

    def load_model(self):
        """Carga el mejor modelo del entrenamiento"""
        print("\n" + "=" * 80)
        print("PASO 4: CARGANDO MEJOR MODELO")
        print("=" * 80)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {self.model_path}\n"
                f"Ejecuta primero train_model.py (Paso 6)"
            )

        print(f"\n🤖 Cargando modelo desde: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)

        print(f"✓ Modelo cargado exitosamente")
        print(f"  Parámetros totales: {self.model.count_params():,}")

    def evaluate_on_test(self):
        """Evalúa el modelo en el conjunto de test"""
        print("\n" + "=" * 80)
        print("PASO 5: EVALUACIÓN EN TEST SET")
        print("=" * 80)

        print("\n🧪 Evaluando modelo en test set...")
        print("  (sin augmentation, como debe ser)")

        # Resetear generador
        self.test_generator.reset()

        # Obtener predicciones
        print("\n  Generando predicciones...")
        y_pred_probs = self.model.predict(
            self.test_generator,
            steps=len(self.test_generator),
            verbose=1
        )

        # Convertir probabilidades a clases
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = self.test_generator.classes

        print(f"\n  ✓ Predicciones generadas: {len(y_pred)} muestras")

        # Guardar predicciones
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_probs = y_pred_probs

        # Calcular métricas
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calcula todas las métricas de evaluación"""
        print("\n📊 Calculando métricas...")

        # 1. Accuracy global
        accuracy = accuracy_score(self.y_true, self.y_pred)
        print(f"\n  Accuracy global: {accuracy:.4f}")

        self.evaluation_results['test_metrics']['accuracy'] = float(accuracy)

        # 2. Métricas por clase (Precision, Recall, F1-score)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true,
            self.y_pred,
            average=None,
            labels=range(len(self.classes))
        )

        print("\n  Métricas por clase:")
        for i, class_name in enumerate(self.classes):
            print(f"    {class_name:15s}: "
                  f"Precision={precision[i]:.4f}, "
                  f"Recall={recall[i]:.4f}, "
                  f"F1={f1[i]:.4f}, "
                  f"Support={support[i]}")

            self.evaluation_results['class_metrics'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }

        # 3. Promedios (macro y weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='weighted'
        )

        print(f"\n  Promedios MACRO:")
        print(f"    Precision: {precision_macro:.4f}")
        print(f"    Recall: {recall_macro:.4f}")
        print(f"    F1-score: {f1_macro:.4f}")

        print(f"\n  Promedios WEIGHTED:")
        print(f"    Precision: {precision_weighted:.4f}")
        print(f"    Recall: {recall_weighted:.4f}")
        print(f"    F1-score: {f1_weighted:.4f}")

        self.evaluation_results['test_metrics']['precision_macro'] = float(precision_macro)
        self.evaluation_results['test_metrics']['recall_macro'] = float(recall_macro)
        self.evaluation_results['test_metrics']['f1_macro'] = float(f1_macro)
        self.evaluation_results['test_metrics']['precision_weighted'] = float(precision_weighted)
        self.evaluation_results['test_metrics']['recall_weighted'] = float(recall_weighted)
        self.evaluation_results['test_metrics']['f1_weighted'] = float(f1_weighted)

        # 4. Classification report completo
        report_dict = classification_report(
            self.y_true,
            self.y_pred,
            target_names=self.classes,
            output_dict=True
        )

        self.evaluation_results['classification_report'] = report_dict

        # 5. Matriz de confusión
        cm = confusion_matrix(self.y_true, self.y_pred)
        self.confusion_matrix = cm
        self.evaluation_results['confusion_matrix'] = cm.tolist()

        print("\n✅ Métricas calculadas exitosamente")

    def generate_confusion_matrix_plot(self):
        """Genera y guarda visualización de matriz de confusión"""
        print("\n" + "=" * 80)
        print("PASO 6: GENERACIÓN DE MATRIZ DE CONFUSIÓN")
        print("=" * 80)

        print("\n📊 Generando matriz de confusión...")

        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot con seaborn
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )

        ax.set_title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')

        plt.tight_layout()

        # Guardar imagen
        cm_img_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(cm_img_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Imagen guardada: {cm_img_path}")

        # Guardar como CSV
        cm_csv_path = self.output_dir / 'confusion_matrix.csv'
        cm_df = pd.DataFrame(
            self.confusion_matrix,
            index=[f'True_{c}' for c in self.classes],
            columns=[f'Pred_{c}' for c in self.classes]
        )
        cm_df.to_csv(cm_csv_path)

        print(f"  ✓ CSV guardado: {cm_csv_path}")

        # Guardar como NPY
        cm_npy_path = self.output_dir / 'confusion_matrix.npy'
        np.save(cm_npy_path, self.confusion_matrix)

        print(f"  ✓ NPY guardado: {cm_npy_path}")

    def save_classification_report(self):
        """Guarda classification report como CSV"""
        print("\n" + "=" * 80)
        print("PASO 7: GUARDADO DE CLASSIFICATION REPORT")
        print("=" * 80)

        print("\n📄 Generando classification report...")

        # Convertir a DataFrame
        report_df = pd.DataFrame(self.evaluation_results['classification_report']).transpose()

        # Guardar CSV
        report_path = self.output_dir / 'classification_report.csv'
        report_df.to_csv(report_path)

        print(f"  ✓ Classification report guardado: {report_path}")

        # También guardar versión legible
        txt_path = self.output_dir / 'classification_report.txt'
        report_str = classification_report(
            self.y_true,
            self.y_pred,
            target_names=self.classes
        )

        with open(txt_path, 'w') as f:
            f.write("CLASSIFICATION REPORT - TEST SET\n")
            f.write("=" * 80 + "\n\n")
            f.write(report_str)

        print(f"  ✓ Versión de texto guardada: {txt_path}")

    def save_metrics_summary(self):
        """Guarda resumen de métricas como CSV"""
        print("\n📊 Generando resumen de métricas...")

        # Crear DataFrame con métricas por clase
        metrics_data = []
        for class_name, metrics in self.evaluation_results['class_metrics'].items():
            metrics_data.append({
                'Class': class_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Support': metrics['support']
            })

        metrics_df = pd.DataFrame(metrics_data)

        # Guardar
        metrics_path = self.output_dir / 'metrics_report.csv'
        metrics_df.to_csv(metrics_path, index=False)

        print(f"  ✓ Resumen de métricas guardado: {metrics_path}")

    def export_model(self):
        """Exporta el modelo a .keras"""
        print("\n" + "=" * 80)
        print("PASO 8: EXPORTACIÓN DEL MODELO")
        print("=" * 80)

        export_path = self.output_dir / 'best_model.keras'

        print(f"\n💾 Exportando modelo a: {export_path}")
        self.model.save(export_path)

        file_size_mb = export_path.stat().st_size / (1024**2)
        print(f"  ✓ Modelo exportado exitosamente")
        print(f"  Tamaño: {file_size_mb:.2f} MB")

        self.exported_model_path = export_path

    def save_evaluation_results(self):
        """Guarda resultados completos de evaluación"""
        print("\n📝 Guardando resultados de evaluación...")

        results_path = self.output_dir / 'evaluation_results.json'

        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)

        print(f"  ✓ Resultados guardados: {results_path}")

    def log_to_mlflow(self):
        """Registra todo en MLflow"""
        print("\n" + "=" * 80)
        print("PASO 9: REGISTRO EN MLFLOW")
        print("=" * 80)

        run_name = f"MobileNetV3Large_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n🚀 Iniciando MLflow run: {run_name}")

        with mlflow.start_run(run_name=run_name):

            # 1. Registrar parámetros (hiperparámetros del entrenamiento)
            print("\n📋 Registrando parámetros...")
            self._log_parameters()

            # 2. Registrar métricas de test
            print("\n📊 Registrando métricas...")
            self._log_metrics()

            # 3. Registrar artefactos (gráficas, reportes, etc.)
            print("\n📦 Registrando artefactos...")
            self._log_artifacts()

            # 4. Registrar modelo
            print("\n🤖 Registrando modelo en MLflow...")
            self._log_model()

            # Obtener run_id
            run_id = mlflow.active_run().info.run_id
            print(f"\n✅ MLflow run completado exitosamente")
            print(f"  Run ID: {run_id}")
            print(f"  Run name: {run_name}")

    def _log_parameters(self):
        """Registra parámetros en MLflow"""
        hp = self.training_info['hyperparameters']
        env = self.training_info['environment']

        # Hiperparámetros
        mlflow.log_param("batch_size", hp['batch_size'])
        mlflow.log_param("epochs", hp['epochs'])
        mlflow.log_param("optimizer", hp['optimizer'])
        mlflow.log_param("learning_rate", hp['learning_rate'])
        mlflow.log_param("loss", hp['loss'])

        # Versiones de dependencias
        print("    Versiones:")
        for lib, version in env['versions'].items():
            mlflow.log_param(f"{lib}_version", version)
            print(f"      {lib}: {version}")

        # Información de hardware
        print("    Hardware:")
        hw = env['hardware']
        mlflow.log_param("platform", hw['platform'])
        mlflow.log_param("processor", hw['processor'])
        mlflow.log_param("cpu_count", hw['cpu_count'])
        mlflow.log_param("gpu_available", hw['gpu_available'])
        mlflow.log_param("gpu_count", hw['gpu_count'])

        print(f"      Platform: {hw['platform']}")
        print(f"      CPU cores: {hw['cpu_count']}")
        print(f"      GPU: {hw['gpu_available']}")

        # Metadata adicional
        mlflow.log_param("random_seed", self.random_seed)
        mlflow.log_param("model_architecture", "MobileNetV3Large")
        mlflow.log_param("num_classes", len(self.classes))
        mlflow.log_param("test_samples", self.test_generator.samples)

    def _log_metrics(self):
        """Registra métricas en MLflow"""
        # Métricas globales
        mlflow.log_metric("test_accuracy", self.evaluation_results['test_metrics']['accuracy'])
        mlflow.log_metric("test_precision_macro", self.evaluation_results['test_metrics']['precision_macro'])
        mlflow.log_metric("test_recall_macro", self.evaluation_results['test_metrics']['recall_macro'])
        mlflow.log_metric("test_f1_macro", self.evaluation_results['test_metrics']['f1_macro'])
        mlflow.log_metric("test_precision_weighted", self.evaluation_results['test_metrics']['precision_weighted'])
        mlflow.log_metric("test_recall_weighted", self.evaluation_results['test_metrics']['recall_weighted'])
        mlflow.log_metric("test_f1_weighted", self.evaluation_results['test_metrics']['f1_weighted'])

        print(f"    Accuracy: {self.evaluation_results['test_metrics']['accuracy']:.4f}")
        print(f"    Recall (macro): {self.evaluation_results['test_metrics']['recall_macro']:.4f}")
        print(f"    F1 (macro): {self.evaluation_results['test_metrics']['f1_macro']:.4f}")

        # Métricas por clase (Recall por clase como solicitaste)
        print("    Recall por clase:")
        for class_name, metrics in self.evaluation_results['class_metrics'].items():
            metric_name = f"test_recall_{class_name.lower()}"
            mlflow.log_metric(metric_name, metrics['recall'])
            print(f"      {class_name}: {metrics['recall']:.4f}")

            # También registrar precision y f1 por clase
            mlflow.log_metric(f"test_precision_{class_name.lower()}", metrics['precision'])
            mlflow.log_metric(f"test_f1_{class_name.lower()}", metrics['f1_score'])

    def _log_artifacts(self):
        """Registra artefactos en MLflow"""
        # Artefactos del Paso 6 (entrenamiento)
        training_artifacts = [
            './training_results/training_history.png',
            './training_results/advanced_metrics.png',
            './training_results/history.csv',
            './training_results/TRAINING_RESULTS.txt'
        ]

        print("    Artefactos de entrenamiento:")
        for artifact_path in training_artifacts:
            if Path(artifact_path).exists():
                mlflow.log_artifact(artifact_path)
                print(f"      ✓ {Path(artifact_path).name}")

        # Artefactos de evaluación (Paso 7)
        evaluation_artifacts = [
            self.output_dir / 'confusion_matrix.png',
            self.output_dir / 'confusion_matrix.csv',
            self.output_dir / 'classification_report.csv',
            self.output_dir / 'classification_report.txt',
            self.output_dir / 'metrics_report.csv',
            self.output_dir / 'evaluation_results.json'
        ]

        print("    Artefactos de evaluación:")
        for artifact_path in evaluation_artifacts:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
                print(f"      ✓ {artifact_path.name}")

    def _log_model(self):
        """Registra el modelo en MLflow"""
        mlflow.keras.log_model(
            self.model,
            artifact_path="keras-model",
            registered_model_name="CornDiseaseClassifier_MobileNetV3Large"
        )
        print("    ✓ Modelo registrado en MLflow")
        print("      Artifact path: keras-model")
        print("      Registered name: CornDiseaseClassifier_MobileNetV3Large")

    def print_final_summary(self):
        """Imprime resumen final"""
        print("\n" + "=" * 80)
        print("RESUMEN FINAL DE EVALUACIÓN")
        print("=" * 80)

        print(f"\n🎯 MÉTRICAS DE TEST:")
        print(f"  Accuracy: {self.evaluation_results['test_metrics']['accuracy']:.4f}")
        print(f"  Precision (macro): {self.evaluation_results['test_metrics']['precision_macro']:.4f}")
        print(f"  Recall (macro): {self.evaluation_results['test_metrics']['recall_macro']:.4f}")
        print(f"  F1-score (macro): {self.evaluation_results['test_metrics']['f1_macro']:.4f}")

        print(f"\n📊 RECALL POR CLASE:")
        for class_name, metrics in self.evaluation_results['class_metrics'].items():
            print(f"  {class_name:15s}: {metrics['recall']:.4f}")

        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"  Modelo exportado: {self.exported_model_path}")
        print(f"  Matriz de confusión: {self.output_dir / 'confusion_matrix.png'}")
        print(f"  Classification report: {self.output_dir / 'classification_report.csv'}")
        print(f"  Métricas: {self.output_dir / 'metrics_report.csv'}")

        print(f"\n✅ MLFLOW:")
        print(f"  Experiment: Maize-Disease-Classification")
        print(f"  Todos los parámetros, métricas, artefactos y modelo registrados")

    def run(self):
        """Ejecuta el pipeline completo de evaluación y exportación"""
        print("\n" + "🔬" * 40)
        print("EVALUACIÓN Y EXPORTACIÓN CON MLFLOW")
        print("🔬" * 40)

        # Paso 1: Configurar MLflow
        self.setup_mlflow()

        # Paso 2: Cargar info de entrenamiento
        self.load_training_info()

        # Paso 3: Cargar generador de test (Paso 4)
        self.load_test_generator()

        # Paso 4: Cargar modelo (del Paso 6)
        self.load_model()

        # Paso 5: Evaluar en test
        self.evaluate_on_test()

        # Paso 6: Matriz de confusión
        self.generate_confusion_matrix_plot()

        # Paso 7: Classification report
        self.save_classification_report()

        # Guardar resumen de métricas
        self.save_metrics_summary()

        # Paso 8: Exportar modelo
        self.export_model()

        # Guardar resultados
        self.save_evaluation_results()

        # Paso 9: Registrar en MLflow
        self.log_to_mlflow()

        # Resumen final
        self.print_final_summary()

        print("\n✅ EVALUACIÓN Y EXPORTACIÓN COMPLETADAS\n")
        print("=" * 80)
        print("Revisa MLflow UI para ver todos los runs:")
        print("  $ mlflow ui")
        print("  Luego abre: http://localhost:5000")
        print("=" * 80 + "\n")


def main():
    """Función principal"""
    evaluator = ModelEvaluator(
        model_path='./training_output/best_model.keras',
        output_dir='./evaluation_results',
        random_seed=42
    )

    evaluator.run()

    print("\n📝 Pipeline completo finalizado:")
    print("   ✓ Paso 4: Configuración de entrenamiento")
    print("   ✓ Paso 5: Creación del modelo")
    print("   ✓ Paso 6: Entrenamiento")
    print("   ✓ Paso 7: Evaluación y exportación con MLflow")


if __name__ == "__main__":
    main()
