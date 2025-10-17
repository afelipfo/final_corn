"""
Paso 6: Compilación y Entrenamiento
- Integra configuración del Paso 4 (training_config.py)
- Integra modelo del Paso 5 (model_creation.py)
- Entrenamiento con métricas avanzadas
- Gráficas de entrenamiento
- Registro de versiones y hardware
"""

import os
import json
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para compatibilidad
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Importar clases de pasos anteriores
from training_config import TrainingConfiguration
from model_creation import ModelCreator


class ModelTrainer:
    """Clase para entrenar el modelo de clasificación"""

    def __init__(self,
                 output_dir='./training_results',
                 random_seed=42,
                 epochs=50,
                 batch_size=32):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size

        # Configuración de entrenamiento
        self.training_info = {
            'metadata': {
                'fecha_inicio': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'fecha_fin': None,
                'duracion_total': None,
                'random_seed': self.random_seed
            },
            'hyperparameters': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'optimizer': 'adam',
                'loss': 'categorical_crossentropy',
                'learning_rate': 0.001
            },
            'environment': {},
            'training_history': {},
            'best_metrics': {}
        }

    def setup_environment(self):
        """Configura el entorno y registra versiones"""
        print("=" * 80)
        print("PASO 1: CONFIGURACIÓN DEL ENTORNO")
        print("=" * 80)

        # Fijar seeds
        self._set_random_seeds()

        # Registrar versiones de dependencias
        self._register_versions()

        # Registrar hardware
        self._register_hardware()

    def _set_random_seeds(self):
        """Fija seeds para reproducibilidad"""
        print("\n🎲 Configurando seeds para reproducibilidad...")

        import random
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        print(f"  ✓ Seeds fijados: {self.random_seed}")

    def _register_versions(self):
        """Registra versiones de todas las dependencias"""
        print("\n📦 Registrando versiones de dependencias...")

        versions = {}

        # Python
        versions['python'] = sys.version.split()[0]

        # TensorFlow y Keras
        versions['tensorflow'] = tf.__version__
        versions['keras'] = keras.__version__

        # Otras librerías
        try:
            import sklearn
            versions['sklearn'] = sklearn.__version__
        except:
            versions['sklearn'] = 'N/A'

        try:
            versions['pandas'] = pd.__version__
        except:
            versions['pandas'] = 'N/A'

        try:
            versions['numpy'] = np.__version__
        except:
            versions['numpy'] = 'N/A'

        try:
            import cv2
            versions['opencv'] = cv2.__version__
        except:
            versions['opencv'] = 'N/A'

        try:
            versions['matplotlib'] = matplotlib.__version__
        except:
            versions['matplotlib'] = 'N/A'

        self.training_info['environment']['versions'] = versions

        print("  Versiones registradas:")
        for lib, ver in versions.items():
            print(f"    {lib:15s}: {ver}")

    def _register_hardware(self):
        """Registra información del hardware"""
        print("\n💻 Registrando hardware...")

        hardware = {}

        # Sistema operativo
        hardware['platform'] = platform.platform()
        hardware['processor'] = platform.processor()
        hardware['python_build'] = platform.python_build()[0]

        # GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            hardware['gpu_available'] = True
            hardware['gpu_count'] = len(gpus)
            hardware['gpu_devices'] = [gpu.name for gpu in gpus]
        else:
            hardware['gpu_available'] = False
            hardware['gpu_count'] = 0
            hardware['gpu_devices'] = []

        # CPU
        try:
            import multiprocessing
            hardware['cpu_count'] = multiprocessing.cpu_count()
        except:
            hardware['cpu_count'] = 'N/A'

        self.training_info['environment']['hardware'] = hardware

        print("  Hardware detectado:")
        print(f"    Platform: {hardware['platform']}")
        print(f"    Processor: {hardware['processor']}")
        print(f"    CPU cores: {hardware['cpu_count']}")
        print(f"    GPU available: {hardware['gpu_available']}")
        if hardware['gpu_available']:
            print(f"    GPU count: {hardware['gpu_count']}")

    def load_training_config(self):
        """Carga configuración del Paso 4"""
        print("\n" + "=" * 80)
        print("PASO 2: CARGANDO CONFIGURACIÓN DE ENTRENAMIENTO (PASO 4)")
        print("=" * 80)

        print("\n⚙️  Inicializando TrainingConfiguration...")
        self.train_config = TrainingConfiguration(
            data_dir='./data_augmented',
            output_dir='./training_output',
            random_seed=self.random_seed
        )

        print("\n🔧 Ejecutando configuración completa...")
        self.train_config.run()

        # Actualizar batch_size si es diferente
        if self.batch_size != self.train_config.batch_size:
            print(f"\n⚠️  Batch size difiere: config={self.train_config.batch_size}, solicitado={self.batch_size}")
            print(f"  Usando batch_size del Paso 4: {self.train_config.batch_size}")
            self.batch_size = self.train_config.batch_size

        print("\n✅ Configuración de entrenamiento cargada")

    def load_model(self):
        """Carga y compila el modelo del Paso 5"""
        print("\n" + "=" * 80)
        print("PASO 3: CARGANDO MODELO (PASO 5)")
        print("=" * 80)

        print("\n🤖 Inicializando ModelCreator...")
        self.model_creator = ModelCreator(
            config_path='./training_output/training_config.json',
            output_dir='./model_output',
            random_seed=self.random_seed
        )

        print("\n🏗️  Creando y compilando modelo...")
        self.model = self.model_creator.run()

        # Recompilar con métricas avanzadas
        self._recompile_with_advanced_metrics()

        print("\n✅ Modelo cargado y compilado")

    def _recompile_with_advanced_metrics(self):
        """Recompila el modelo con métricas avanzadas del Paso 6"""
        print("\n" + "=" * 80)
        print("PASO 4: CONFIGURANDO MÉTRICAS AVANZADAS")
        print("=" * 80)

        print("\n📊 Recompilando con métricas avanzadas...")

        # Métricas del Paso 6
        metrics = [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]

        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )

        print("  Métricas configuradas:")
        print("    • Accuracy (exactitud general)")
        print("    • Precision (precisión por clase)")
        print("    • Recall (sensibilidad por clase)")
        print("    • AUC (área bajo la curva ROC)")
        print("  Loss: categorical_crossentropy")
        print("  Optimizer: Adam (lr=0.001)")

        self.training_info['hyperparameters']['metrics'] = [
            'accuracy', 'precision', 'recall', 'auc'
        ]

        print("\n✅ Modelo recompilado con métricas avanzadas")

    def train(self):
        """Entrena el modelo"""
        print("\n" + "=" * 80)
        print("PASO 5: ENTRENAMIENTO DEL MODELO")
        print("=" * 80)

        print(f"\n🚀 Iniciando entrenamiento...")
        print(f"  Épocas: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Train samples: {self.train_config.train_generator.samples}")
        print(f"  Val samples: {self.train_config.val_generator.samples}")
        print(f"  Steps per epoch: {len(self.train_config.train_generator)}")
        print(f"  Validation steps: {len(self.train_config.val_generator)}")

        # Registrar tiempo de inicio
        start_time = datetime.now()

        # Entrenar
        print("\n" + "-" * 80)
        print("ENTRENAMIENTO EN PROGRESO...")
        print("-" * 80 + "\n")

        self.history = self.model.fit(
            self.train_config.train_generator,
            validation_data=self.train_config.val_generator,
            epochs=self.epochs,
            callbacks=self.train_config.callbacks,
            class_weight=self.train_config.class_weights,
            verbose=1
        )

        # Registrar tiempo de fin
        end_time = datetime.now()
        duration = end_time - start_time

        self.training_info['metadata']['fecha_fin'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
        self.training_info['metadata']['duracion_total'] = str(duration)

        print("\n" + "-" * 80)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("-" * 80)
        print(f"  Duración total: {duration}")
        print(f"  Épocas completadas: {len(self.history.history['loss'])}")

    def analyze_training_history(self):
        """Analiza el historial de entrenamiento"""
        print("\n" + "=" * 80)
        print("PASO 6: ANÁLISIS DEL HISTORIAL DE ENTRENAMIENTO")
        print("=" * 80)

        # Convertir historial a dict serializable
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]

        self.training_info['training_history'] = history_dict

        # Encontrar mejor época
        best_epoch = np.argmax(history_dict['val_accuracy'])
        best_val_acc = history_dict['val_accuracy'][best_epoch]
        best_val_loss = history_dict['val_loss'][best_epoch]

        print(f"\n🏆 Mejor época: {best_epoch + 1}")
        print(f"  Val Accuracy: {best_val_acc:.4f}")
        print(f"  Val Loss: {best_val_loss:.4f}")

        # Métricas finales
        final_metrics = {
            'best_epoch': int(best_epoch + 1),
            'best_val_accuracy': float(best_val_acc),
            'best_val_loss': float(best_val_loss),
            'final_train_accuracy': float(history_dict['accuracy'][-1]),
            'final_train_loss': float(history_dict['loss'][-1]),
            'final_val_accuracy': float(history_dict['val_accuracy'][-1]),
            'final_val_loss': float(history_dict['val_loss'][-1])
        }

        # Agregar métricas avanzadas si están disponibles
        if 'val_precision' in history_dict:
            final_metrics['best_val_precision'] = float(history_dict['val_precision'][best_epoch])
            final_metrics['final_val_precision'] = float(history_dict['val_precision'][-1])

        if 'val_recall' in history_dict:
            final_metrics['best_val_recall'] = float(history_dict['val_recall'][best_epoch])
            final_metrics['final_val_recall'] = float(history_dict['val_recall'][-1])

        if 'val_auc' in history_dict:
            final_metrics['best_val_auc'] = float(history_dict['val_auc'][best_epoch])
            final_metrics['final_val_auc'] = float(history_dict['val_auc'][-1])

        self.training_info['best_metrics'] = final_metrics

        print("\n📊 Métricas finales (última época):")
        print(f"  Train Accuracy: {final_metrics['final_train_accuracy']:.4f}")
        print(f"  Train Loss: {final_metrics['final_train_loss']:.4f}")
        print(f"  Val Accuracy: {final_metrics['final_val_accuracy']:.4f}")
        print(f"  Val Loss: {final_metrics['final_val_loss']:.4f}")

        if 'final_val_precision' in final_metrics:
            print(f"  Val Precision: {final_metrics['final_val_precision']:.4f}")
            print(f"  Val Recall: {final_metrics['final_val_recall']:.4f}")
            print(f"  Val AUC: {final_metrics['final_val_auc']:.4f}")

    def generate_plots(self):
        """Genera gráficas de entrenamiento"""
        print("\n" + "=" * 80)
        print("PASO 7: GENERACIÓN DE GRÁFICAS")
        print("=" * 80)

        print("\n📈 Generando gráficas...")

        history = self.history.history
        epochs_range = range(1, len(history['loss']) + 1)

        # Crear figura con 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 1. Accuracy vs Epochs
        ax1 = axes[0]
        ax1.plot(epochs_range, history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        ax1.plot(epochs_range, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
        ax1.set_title('Accuracy vs Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([1, len(history['loss'])])

        # Marcar mejor época
        best_epoch = np.argmax(history['val_accuracy'])
        best_val_acc = history['val_accuracy'][best_epoch]
        ax1.plot(best_epoch + 1, best_val_acc, 'g*', markersize=15,
                label=f'Best (epoch {best_epoch + 1})')
        ax1.legend(loc='lower right', fontsize=10)

        # 2. Loss vs Epochs
        ax2 = axes[1]
        ax2.plot(epochs_range, history['loss'], 'b-', label='Train Loss', linewidth=2)
        ax2.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax2.set_title('Loss vs Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([1, len(history['loss'])])

        # Marcar mejor época
        best_val_loss = history['val_loss'][best_epoch]
        ax2.plot(best_epoch + 1, best_val_loss, 'g*', markersize=15,
                label=f'Best (epoch {best_epoch + 1})')
        ax2.legend(loc='upper right', fontsize=10)

        plt.tight_layout()

        # Guardar
        plot_path = self.output_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Gráfica guardada: {plot_path}")

        # Gráfica adicional: métricas avanzadas (si existen)
        if 'val_precision' in history and 'val_recall' in history and 'val_auc' in history:
            self._plot_advanced_metrics(history, epochs_range)

    def _plot_advanced_metrics(self, history, epochs_range):
        """Genera gráfica de métricas avanzadas"""
        print("\n📊 Generando gráfica de métricas avanzadas...")

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(epochs_range, history['val_precision'], 'g-', label='Val Precision', linewidth=2)
        ax.plot(epochs_range, history['val_recall'], 'b-', label='Val Recall', linewidth=2)
        ax.plot(epochs_range, history['val_auc'], 'r-', label='Val AUC', linewidth=2)

        ax.set_title('Métricas Avanzadas vs Epochs (Validación)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, len(history['loss'])])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        plot_path = self.output_dir / 'advanced_metrics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Gráfica guardada: {plot_path}")

    def save_training_info(self):
        """Guarda toda la información del entrenamiento"""
        print("\n" + "=" * 80)
        print("PASO 8: GUARDADO DE RESULTADOS")
        print("=" * 80)

        # Guardar JSON completo
        json_path = self.output_dir / 'training_info.json'
        with open(json_path, 'w') as f:
            json.dump(self.training_info, f, indent=2)

        print(f"\n✓ Información completa guardada: {json_path}")

        # Guardar README legible
        self._save_readme()

        # Guardar historial como CSV
        self._save_history_csv()

    def _save_readme(self):
        """Guarda README legible con resultados"""
        readme_path = self.output_dir / 'TRAINING_RESULTS.txt'

        with open(readme_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RESULTADOS DEL ENTRENAMIENTO - CORN DISEASE CLASSIFIER\n")
            f.write("=" * 80 + "\n\n")

            # Metadata
            f.write("INFORMACIÓN GENERAL:\n")
            f.write(f"  Fecha inicio: {self.training_info['metadata']['fecha_inicio']}\n")
            f.write(f"  Fecha fin: {self.training_info['metadata']['fecha_fin']}\n")
            f.write(f"  Duración total: {self.training_info['metadata']['duracion_total']}\n")
            f.write(f"  Random seed: {self.training_info['metadata']['random_seed']}\n\n")

            # Hiperparámetros
            f.write("HIPERPARÁMETROS:\n")
            hp = self.training_info['hyperparameters']
            f.write(f"  Épocas: {hp['epochs']}\n")
            f.write(f"  Batch size: {hp['batch_size']}\n")
            f.write(f"  Optimizer: {hp['optimizer']}\n")
            f.write(f"  Learning rate: {hp['learning_rate']}\n")
            f.write(f"  Loss: {hp['loss']}\n")
            f.write(f"  Métricas: {', '.join(hp['metrics'])}\n\n")

            # Mejores métricas
            f.write("MEJORES RESULTADOS:\n")
            bm = self.training_info['best_metrics']
            f.write(f"  Mejor época: {bm['best_epoch']}\n")
            f.write(f"  Best Val Accuracy: {bm['best_val_accuracy']:.4f}\n")
            f.write(f"  Best Val Loss: {bm['best_val_loss']:.4f}\n")
            if 'best_val_precision' in bm:
                f.write(f"  Best Val Precision: {bm['best_val_precision']:.4f}\n")
                f.write(f"  Best Val Recall: {bm['best_val_recall']:.4f}\n")
                f.write(f"  Best Val AUC: {bm['best_val_auc']:.4f}\n")
            f.write("\n")

            # Métricas finales
            f.write("MÉTRICAS FINALES (última época):\n")
            f.write(f"  Train Accuracy: {bm['final_train_accuracy']:.4f}\n")
            f.write(f"  Train Loss: {bm['final_train_loss']:.4f}\n")
            f.write(f"  Val Accuracy: {bm['final_val_accuracy']:.4f}\n")
            f.write(f"  Val Loss: {bm['final_val_loss']:.4f}\n")
            if 'final_val_precision' in bm:
                f.write(f"  Val Precision: {bm['final_val_precision']:.4f}\n")
                f.write(f"  Val Recall: {bm['final_val_recall']:.4f}\n")
                f.write(f"  Val AUC: {bm['final_val_auc']:.4f}\n")
            f.write("\n")

            # Entorno
            f.write("ENTORNO:\n")
            env = self.training_info['environment']
            f.write("  Versiones:\n")
            for lib, ver in env['versions'].items():
                f.write(f"    {lib}: {ver}\n")
            f.write("\n  Hardware:\n")
            hw = env['hardware']
            f.write(f"    Platform: {hw['platform']}\n")
            f.write(f"    Processor: {hw['processor']}\n")
            f.write(f"    CPU cores: {hw['cpu_count']}\n")
            f.write(f"    GPU available: {hw['gpu_available']}\n")
            if hw['gpu_available']:
                f.write(f"    GPU count: {hw['gpu_count']}\n")

        print(f"✓ README guardado: {readme_path}")

    def _save_history_csv(self):
        """Guarda historial como CSV"""
        csv_path = self.output_dir / 'history.csv'

        df = pd.DataFrame(self.training_info['training_history'])
        df.insert(0, 'epoch', range(1, len(df) + 1))
        df.to_csv(csv_path, index=False)

        print(f"✓ Historial CSV guardado: {csv_path}")

    def print_final_summary(self):
        """Imprime resumen final"""
        print("\n" + "=" * 80)
        print("RESUMEN FINAL DEL ENTRENAMIENTO")
        print("=" * 80)

        bm = self.training_info['best_metrics']

        print(f"\n🏆 MEJOR MODELO:")
        print(f"  Época: {bm['best_epoch']}")
        print(f"  Val Accuracy: {bm['best_val_accuracy']:.4f}")
        print(f"  Val Loss: {bm['best_val_loss']:.4f}")

        print(f"\n⏱️  TIEMPO:")
        print(f"  Duración: {self.training_info['metadata']['duracion_total']}")

        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"  Mejor modelo: ./training_output/best_model.keras")
        print(f"  Resultados: {self.output_dir / 'training_info.json'}")
        print(f"  Gráficas: {self.output_dir / 'training_history.png'}")
        print(f"  README: {self.output_dir / 'TRAINING_RESULTS.txt'}")
        print(f"  Historial CSV: {self.output_dir / 'history.csv'}")

    def run(self):
        """Ejecuta el pipeline completo de entrenamiento"""
        print("\n" + "🚀" * 40)
        print("ENTRENAMIENTO DEL MODELO - CORN DISEASE CLASSIFIER")
        print("🚀" * 40)

        # Paso 1: Setup entorno
        self.setup_environment()

        # Paso 2: Cargar configuración de entrenamiento (Paso 4)
        self.load_training_config()

        # Paso 3: Cargar modelo (Paso 5)
        self.load_model()

        # Paso 4: Métricas avanzadas (ya ejecutado en load_model)

        # Paso 5: Entrenar
        self.train()

        # Paso 6: Analizar historial
        self.analyze_training_history()

        # Paso 7: Generar gráficas
        self.generate_plots()

        # Paso 8: Guardar resultados
        self.save_training_info()

        # Resumen final
        self.print_final_summary()

        print("\n✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE\n")
        print("=" * 80)
        print("El mejor modelo fue guardado automáticamente por ModelCheckpoint.")
        print("Revisa las gráficas y métricas para evaluar el rendimiento.")
        print("=" * 80 + "\n")


def main():
    """Función principal"""
    trainer = ModelTrainer(
        output_dir='./training_results',
        random_seed=42,
        epochs=50,
        batch_size=32
    )

    trainer.run()

    print("\n📝 Siguiente paso:")
    print("   Evaluar el modelo en el conjunto de test (Paso 7)")
    print("   usando el mejor modelo guardado en:")
    print("   ./training_output/best_model.keras")


if __name__ == "__main__":
    main()
