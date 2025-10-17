#!/usr/bin/env python3
"""
PIPELINE COMPLETO - CORN DISEASE CLASSIFICATION
================================================

Script orquestador que ejecuta secuencialmente:
- Paso 4: Configuración de entrenamiento
- Paso 5: Creación del modelo
- Paso 6: Entrenamiento
- Paso 7: Evaluación y exportación con MLflow

Autor: Pipeline automatizado
Fecha: 2025-10-11
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import traceback

# Los módulos están en el mismo directorio que este script
# No necesitamos modificar sys.path

# Importar módulos de los pasos
from training_config import TrainingConfiguration
from model_creation import ModelCreator
from train_model import ModelTrainer
from evaluate_and_export import ModelEvaluator


class PipelineOrchestrator:
    """Orquestador del pipeline completo de entrenamiento"""

    def __init__(self, random_seed=42, epochs=50, batch_size=32):
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.pipeline_start_time = None
        self.pipeline_end_time = None

        # Estado del pipeline
        self.status = {
            'paso_4': False,
            'paso_5': False,
            'paso_6': False,
            'paso_7': False
        }

    def print_header(self):
        """Imprime encabezado del pipeline"""
        print("\n" + "=" * 100)
        print("█" * 100)
        print("█" + " " * 98 + "█")
        print("█" + " " * 30 + "CORN DISEASE CLASSIFICATION PIPELINE" + " " * 32 + "█")
        print("█" + " " * 98 + "█")
        print("█" + " " * 20 + "Pipeline completo: Configuración → Modelo → Entrenamiento → Evaluación" + " " * 9 + "█")
        print("█" + " " * 98 + "█")
        print("█" * 100)
        print("=" * 100)
        print(f"\n⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎲 Random seed: {self.random_seed}")
        print(f"📊 Epochs: {self.epochs}")
        print(f"📦 Batch size: {self.batch_size}")
        print("\n" + "=" * 100 + "\n")

    def print_step_header(self, step_number, step_name, description):
        """Imprime encabezado de cada paso"""
        print("\n" + "▓" * 100)
        print(f"▓ PASO {step_number}: {step_name.upper()}")
        print(f"▓ {description}")
        print("▓" * 100 + "\n")

    def print_step_completion(self, step_number, step_name, success=True):
        """Imprime confirmación de completitud de paso"""
        if success:
            print("\n" + "░" * 100)
            print(f"░ ✅ PASO {step_number} COMPLETADO: {step_name}")
            print("░" * 100 + "\n")
        else:
            print("\n" + "░" * 100)
            print(f"░ ❌ PASO {step_number} FALLÓ: {step_name}")
            print("░" * 100 + "\n")

    def run_step_4(self):
        """Paso 4: Configuración de entrenamiento"""
        self.print_step_header(
            4,
            "Configuración de Entrenamiento",
            "Configura augmentation online, generadores de datos y callbacks"
        )

        try:
            # Crear configuración
            print("🔧 Inicializando TrainingConfiguration...\n")
            config = TrainingConfiguration(
                data_dir='./data_augmented',
                output_dir='./training_output',
                random_seed=self.random_seed
            )

            print("🚀 Ejecutando configuración completa...\n")
            config.run()

            # Guardar referencia
            self.training_config = config

            self.status['paso_4'] = True
            self.print_step_completion(4, "Configuración de Entrenamiento", success=True)

            return True

        except Exception as e:
            print(f"\n❌ ERROR en Paso 4: {str(e)}")
            print(traceback.format_exc())
            self.print_step_completion(4, "Configuración de Entrenamiento", success=False)
            return False

    def run_step_5(self):
        """Paso 5: Creación del modelo"""
        self.print_step_header(
            5,
            "Creación del Modelo",
            "Crea MobileNetV3Large con BatchNorm, Dropout y capa de salida"
        )

        try:
            # Crear modelo
            print("🤖 Inicializando ModelCreator...\n")
            creator = ModelCreator(
                config_path='./training_output/training_config.json',
                output_dir='./model_output',
                random_seed=self.random_seed
            )

            print("🏗️  Ejecutando creación y compilación del modelo...\n")
            model = creator.run()

            # Guardar referencia
            self.model_creator = creator
            self.model = model

            self.status['paso_5'] = True
            self.print_step_completion(5, "Creación del Modelo", success=True)

            return True

        except Exception as e:
            print(f"\n❌ ERROR en Paso 5: {str(e)}")
            print(traceback.format_exc())
            self.print_step_completion(5, "Creación del Modelo", success=False)
            return False

    def run_step_6(self):
        """Paso 6: Entrenamiento"""
        self.print_step_header(
            6,
            "Entrenamiento del Modelo",
            "Entrena el modelo con augmentation, callbacks y métricas avanzadas"
        )

        try:
            # Entrenar modelo
            print("🚀 Inicializando ModelTrainer...\n")
            trainer = ModelTrainer(
                output_dir='./training_results',
                random_seed=self.random_seed,
                epochs=self.epochs,
                batch_size=self.batch_size
            )

            print("🏋️  Ejecutando entrenamiento completo...\n")
            trainer.run()

            # Guardar referencia
            self.trainer = trainer

            self.status['paso_6'] = True
            self.print_step_completion(6, "Entrenamiento del Modelo", success=True)

            return True

        except Exception as e:
            print(f"\n❌ ERROR en Paso 6: {str(e)}")
            print(traceback.format_exc())
            self.print_step_completion(6, "Entrenamiento del Modelo", success=False)
            return False

    def run_step_7(self):
        """Paso 7: Evaluación y exportación con MLflow"""
        self.print_step_header(
            7,
            "Evaluación y Exportación con MLflow",
            "Evalúa en test, genera métricas, matriz de confusión y registra en MLflow"
        )

        try:
            # Evaluar y exportar
            print("🔬 Inicializando ModelEvaluator...\n")
            evaluator = ModelEvaluator(
                model_path='./training_output/best_model.keras',
                output_dir='./evaluation_results',
                random_seed=self.random_seed
            )

            print("📊 Ejecutando evaluación y exportación completa...\n")
            evaluator.run()

            # Guardar referencia
            self.evaluator = evaluator

            self.status['paso_7'] = True
            self.print_step_completion(7, "Evaluación y Exportación", success=True)

            return True

        except Exception as e:
            print(f"\n❌ ERROR en Paso 7: {str(e)}")
            print(traceback.format_exc())
            self.print_step_completion(7, "Evaluación y Exportación", success=False)
            return False

    def print_final_summary(self):
        """Imprime resumen final del pipeline"""
        print("\n" + "=" * 100)
        print("█" * 100)
        print("█" + " " * 98 + "█")
        print("█" + " " * 35 + "RESUMEN FINAL DEL PIPELINE" + " " * 37 + "█")
        print("█" + " " * 98 + "█")
        print("█" * 100)
        print("=" * 100)

        # Tiempo total
        duration = self.pipeline_end_time - self.pipeline_start_time
        print(f"\n⏱️  TIEMPO TOTAL: {duration}")
        print(f"   Inicio: {self.pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Fin: {self.pipeline_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Estado de pasos
        print(f"\n📋 ESTADO DE PASOS:")
        all_success = all(self.status.values())

        for i, (paso, success) in enumerate([
            ('Paso 4: Configuración de entrenamiento', self.status['paso_4']),
            ('Paso 5: Creación del modelo', self.status['paso_5']),
            ('Paso 6: Entrenamiento', self.status['paso_6']),
            ('Paso 7: Evaluación y MLflow', self.status['paso_7'])
        ], start=4):
            status_icon = "✅" if success else "❌"
            print(f"   {status_icon} {paso}")

        # Resultado final
        print(f"\n{'🎉' * 50}")
        if all_success:
            print("   ✅ PIPELINE COMPLETADO EXITOSAMENTE")
            print("   Todos los pasos se ejecutaron correctamente")
        else:
            print("   ⚠️  PIPELINE COMPLETADO CON ERRORES")
            print("   Algunos pasos fallaron - revisa los logs arriba")
        print(f"{'🎉' * 50}")

        # Archivos generados
        if all_success:
            print(f"\n📁 ARCHIVOS Y DIRECTORIOS GENERADOS:")
            print(f"   📂 training_output/")
            print(f"      ├── training_config.json       (Configuración de entrenamiento)")
            print(f"      ├── best_model.keras           (Mejor modelo guardado por callbacks)")
            print(f"      ├── training_history.csv       (Historial por época)")
            print(f"      └── tensorboard_logs/          (Logs de TensorBoard)")
            print(f"\n   📂 model_output/")
            print(f"      ├── model_config.json          (Configuración del modelo)")
            print(f"      ├── model_summary.txt          (Resumen de arquitectura)")
            print(f"      └── model_initial.keras        (Modelo inicial sin entrenar)")
            print(f"\n   📂 training_results/")
            print(f"      ├── training_info.json         (Info completa del entrenamiento)")
            print(f"      ├── training_history.png       (Gráficas de accuracy y loss)")
            print(f"      ├── advanced_metrics.png       (Gráficas de métricas avanzadas)")
            print(f"      └── history.csv                (Historial detallado)")
            print(f"\n   📂 evaluation_results/")
            print(f"      ├── best_model.keras           (Modelo exportado)")
            print(f"      ├── confusion_matrix.png       (Matriz de confusión)")
            print(f"      ├── confusion_matrix.csv       (Matriz en CSV)")
            print(f"      ├── confusion_matrix.npy       (Matriz en NumPy)")
            print(f"      ├── classification_report.csv  (Reporte completo)")
            print(f"      └── metrics_report.csv         (Métricas por clase)")

            print(f"\n🔬 MLflow:")
            print(f"   Experiment: Maize-Disease-Classification")
            print(f"   Run completado con todos los parámetros, métricas y artefactos")
            print(f"\n   Para visualizar en MLflow UI:")
            print(f"   $ mlflow ui")
            print(f"   Luego abre: http://localhost:5000")

        print("\n" + "=" * 100)

        # Métricas finales (si está disponible)
        if all_success and hasattr(self, 'evaluator'):
            print(f"\n📊 MÉTRICAS FINALES EN TEST SET:")
            metrics = self.evaluator.evaluation_results['test_metrics']
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Precision (macro): {metrics['precision_macro']:.4f}")
            print(f"   Recall (macro): {metrics['recall_macro']:.4f}")
            print(f"   F1-score (macro): {metrics['f1_macro']:.4f}")

            print(f"\n   Recall por clase:")
            for class_name, class_metrics in self.evaluator.evaluation_results['class_metrics'].items():
                print(f"     {class_name:15s}: {class_metrics['recall']:.4f}")

        print("\n" + "=" * 100 + "\n")

    def run(self):
        """Ejecuta el pipeline completo"""
        self.pipeline_start_time = datetime.now()

        # Imprimir encabezado
        self.print_header()

        # Ejecutar pasos secuencialmente
        steps = [
            (4, self.run_step_4, "Configuración de Entrenamiento"),
            (5, self.run_step_5, "Creación del Modelo"),
            (6, self.run_step_6, "Entrenamiento"),
            (7, self.run_step_7, "Evaluación y MLflow")
        ]

        for step_num, step_func, step_name in steps:
            success = step_func()
            if not success:
                print(f"\n💥 Pipeline detenido en Paso {step_num}: {step_name}")
                print("   Corrige los errores antes de continuar\n")
                self.pipeline_end_time = datetime.now()
                self.print_final_summary()
                return False

        # Pipeline completado
        self.pipeline_end_time = datetime.now()
        self.print_final_summary()

        return True


def main():
    """Función principal"""
    print("\n" + "🌽" * 50)
    print("CORN DISEASE CLASSIFICATION - PIPELINE COMPLETO")
    print("🌽" * 50 + "\n")

    # Crear y ejecutar pipeline
    pipeline = PipelineOrchestrator(
        random_seed=42,
        epochs=50,
        batch_size=32
    )

    success = pipeline.run()

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
