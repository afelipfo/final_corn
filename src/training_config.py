"""
Paso 4: Configuración de Entrenamiento
- Configuración completa con TensorFlow/Keras
- Seeds fijos para reproducibilidad
- Data augmentation online
- Callbacks (EarlyStopping, ReduceLROnPlateau)
- Class weights (opcional)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
    TensorBoard
)
from pathlib import Path
from datetime import datetime
import pandas as pd


class TrainingConfiguration:
    """Clase para configurar el entrenamiento con TensorFlow/Keras"""

    def __init__(self,
                 data_dir='./data_augmented',
                 output_dir='./training_output',
                 random_seed=42):

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed

        # Crear directorio de salida
        self.output_dir.mkdir(exist_ok=True)

        # Configuración de reproducibilidad
        self._set_random_seeds()

        # Parámetros del modelo
        self.img_size = (256, 256)
        self.batch_size = 32
        self.classes = ['Blight', 'CommonRust', 'GrayLeafSpot', 'Healthy']
        self.num_classes = len(self.classes)

        # Configuración de entrenamiento
        self.config = {
            'metadata': {
                'fecha_creacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_dir': str(self.data_dir),
                'output_dir': str(self.output_dir),
                'random_seed': self.random_seed
            },
            'model': {
                'input_shape': list(self.img_size) + [3],
                'num_classes': self.num_classes,
                'classes': self.classes
            },
            'training': {
                'batch_size': self.batch_size,
                'epochs': 100,
                'initial_learning_rate': 0.001,
                'optimizer': 'adam',
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy', 'categorical_crossentropy']
            },
            'augmentation_online': {},
            'callbacks': {},
            'class_weights': None
        }

    def _set_random_seeds(self):
        """Fija seeds para reproducibilidad completa"""
        print("=" * 80)
        print("PASO 1: CONFIGURACIÓN DE SEEDS PARA REPRODUCIBILIDAD")
        print("=" * 80)

        # Seed de Python
        import random
        random.seed(self.random_seed)
        print(f"\n✓ Python random seed: {self.random_seed}")

        # Seed de NumPy
        np.random.seed(self.random_seed)
        print(f"✓ NumPy random seed: {self.random_seed}")

        # Seed de TensorFlow
        tf.random.set_seed(self.random_seed)
        print(f"✓ TensorFlow random seed: {self.random_seed}")

        # Configurar determinismo en TensorFlow
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        print(f"✓ TensorFlow deterministic ops: enabled")

        # Configurar GPU para reproducibilidad (si está disponible)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ GPU configurada: {len(gpus)} dispositivo(s)")
            except RuntimeError as e:
                print(f"⚠️  Advertencia GPU: {e}")
        else:
            print(f"ℹ️  No se detectaron GPUs, usando CPU")

    def configure_augmentation(self):
        """Configura data augmentation online (durante entrenamiento)"""
        print("\n" + "=" * 80)
        print("PASO 2: CONFIGURACIÓN DE DATA AUGMENTATION ONLINE")
        print("=" * 80)

        print("\n🎨 Configurando augmentation para TRAIN (online)...")

        # Augmentation ONLINE para train (más fuerte que offline)
        train_augmentation = {
            'rescale': 1./255,
            'rotation_range': 30,           # ±30° (antes era 10°)
            'width_shift_range': 0.15,      # 15% (antes era 0.005 = 0.5%)
            'height_shift_range': 0.15,     # 15%
            'horizontal_flip': True,
            'vertical_flip': True,          # AGREGADO (no estaba)
            'brightness_range': [0.85, 1.15],  # AGREGADO ±15%
            'zoom_range': 0.1,              # AGREGADO ±10%
            'fill_mode': 'reflect'          # Mejor que 'nearest'
        }

        self.train_datagen = ImageDataGenerator(**train_augmentation)

        print("  Parámetros de augmentation online:")
        print(f"    • Rotación: ±{train_augmentation['rotation_range']}°")
        print(f"    • Traslación H/V: ±{train_augmentation['width_shift_range']*100}%")
        print(f"    • Flips: Horizontal + Vertical")
        print(f"    • Brillo: {train_augmentation['brightness_range'][0]} - {train_augmentation['brightness_range'][1]}")
        print(f"    • Zoom: ±{train_augmentation['zoom_range']*100}%")
        print(f"    • Normalización: 0-1 (rescale={train_augmentation['rescale']})")

        # Solo normalización para val y test (SIN augmentation)
        print("\n🔒 Configurando normalización para VAL y TEST (sin augmentation)...")

        val_test_config = {
            'rescale': 1./255
        }

        self.val_datagen = ImageDataGenerator(**val_test_config)
        self.test_datagen = ImageDataGenerator(**val_test_config)

        print("  Val/Test: Solo normalización 0-1, sin augmentation")

        # Guardar configuración
        self.config['augmentation_online'] = {
            'train': train_augmentation,
            'val_test': val_test_config
        }

        print("\n✅ Augmentation configurado exitosamente")

    def create_data_generators(self):
        """Crea generadores de datos para train/val/test"""
        print("\n" + "=" * 80)
        print("PASO 3: CREACIÓN DE GENERADORES DE DATOS")
        print("=" * 80)

        # Generador de train
        print("\n📂 Creando generador de TRAIN...")
        self.train_generator = self.train_datagen.flow_from_directory(
            self.data_dir / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=self.random_seed
        )

        print(f"  ✓ Train: {self.train_generator.samples} imágenes")
        print(f"  ✓ Clases: {list(self.train_generator.class_indices.keys())}")
        print(f"  ✓ Batch size: {self.batch_size}")
        print(f"  ✓ Steps per epoch: {len(self.train_generator)}")

        # Generador de val
        print("\n📂 Creando generador de VAL...")
        self.val_generator = self.val_datagen.flow_from_directory(
            self.data_dir / 'val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=self.random_seed
        )

        print(f"  ✓ Val: {self.val_generator.samples} imágenes")
        print(f"  ✓ Validation steps: {len(self.val_generator)}")

        # Generador de test
        print("\n📂 Creando generador de TEST...")
        self.test_generator = self.test_datagen.flow_from_directory(
            self.data_dir / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=self.random_seed
        )

        print(f"  ✓ Test: {self.test_generator.samples} imágenes")
        print(f"  ✓ Test steps: {len(self.test_generator)}")

        # Verificar balance de clases en train
        self._verify_class_balance()

    def _verify_class_balance(self):
        """Verifica el balance de clases en train"""
        print("\n⚖️  Verificando balance de clases en TRAIN:")

        class_counts = {}
        for class_name, class_idx in self.train_generator.class_indices.items():
            count = sum(self.train_generator.classes == class_idx)
            class_counts[class_name] = count
            pct = (count / self.train_generator.samples) * 100
            print(f"  {class_name:15s}: {count:4d} ({pct:5.2f}%)")

        # Calcular ratio de desbalance
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count

        print(f"\n  Ratio de desbalance: {imbalance_ratio:.2f}:1")

        if imbalance_ratio < 1.2:
            print(f"  Estado: ✅ PERFECTAMENTE BALANCEADO")
            self.needs_class_weights = False
        elif imbalance_ratio < 1.5:
            print(f"  Estado: ℹ️  Ligeramente desbalanceado (aceptable)")
            self.needs_class_weights = False
        else:
            print(f"  Estado: ⚠️  Desbalanceado - considerar class weights")
            self.needs_class_weights = True

    def calculate_class_weights(self, force=False):
        """Calcula class weights (opcional, solo si hay desbalance)"""
        print("\n" + "=" * 80)
        print("PASO 4: CÁLCULO DE CLASS WEIGHTS")
        print("=" * 80)

        if not self.needs_class_weights and not force:
            print("\n✓ Dataset balanceado - Class weights NO necesarios")
            print("  El modelo se entrenará sin weights (todas las clases peso 1.0)")
            self.class_weights = None
            self.config['class_weights'] = None
            return

        print("\n📊 Calculando class weights...")

        # Contar muestras por clase
        class_counts = {}
        for class_name, class_idx in self.train_generator.class_indices.items():
            count = sum(self.train_generator.classes == class_idx)
            class_counts[class_idx] = count

        # Calcular weights (inversamente proporcional a frecuencia)
        total_samples = self.train_generator.samples
        class_weights = {}

        for class_idx, count in class_counts.items():
            weight = total_samples / (self.num_classes * count)
            class_weights[class_idx] = weight

        # Normalizar weights para que el máximo sea ~1.2
        max_weight = max(class_weights.values())
        if max_weight > 1.2:
            scale_factor = 1.2 / max_weight
            class_weights = {k: v * scale_factor for k, v in class_weights.items()}

        self.class_weights = class_weights

        print("\n  Class weights calculados:")
        for class_name, class_idx in sorted(self.train_generator.class_indices.items(),
                                           key=lambda x: x[1]):
            weight = class_weights[class_idx]
            print(f"    {class_name:15s} (idx {class_idx}): {weight:.3f}")

        # Guardar en config
        self.config['class_weights'] = {
            str(k): float(v) for k, v in class_weights.items()
        }

        print("\n✓ Class weights configurados")

    def configure_callbacks(self):
        """Configura callbacks para entrenamiento"""
        print("\n" + "=" * 80)
        print("PASO 5: CONFIGURACIÓN DE CALLBACKS")
        print("=" * 80)

        callbacks_list = []

        # 1. EarlyStopping
        print("\n🛑 EarlyStopping:")
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        callbacks_list.append(early_stopping)

        print(f"  Monitor: val_loss")
        print(f"  Patience: 10 epochs")
        print(f"  Restore best weights: True")

        self.config['callbacks']['early_stopping'] = {
            'monitor': 'val_loss',
            'patience': 10,
            'mode': 'min',
            'restore_best_weights': True
        }

        # 2. ReduceLROnPlateau
        print("\n📉 ReduceLROnPlateau:")
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            verbose=1,
            mode='min',
            min_lr=1e-7
        )
        callbacks_list.append(reduce_lr)

        print(f"  Monitor: val_loss")
        print(f"  Factor: 0.1 (reduce LR a 10%)")
        print(f"  Patience: 5 epochs")
        print(f"  Min LR: 1e-7")

        self.config['callbacks']['reduce_lr'] = {
            'monitor': 'val_loss',
            'factor': 0.1,
            'patience': 5,
            'mode': 'min',
            'min_lr': 1e-7
        }

        # 3. ModelCheckpoint (guardar mejor modelo)
        print("\n💾 ModelCheckpoint:")
        checkpoint_path = self.output_dir / 'best_model.keras'
        model_checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks_list.append(model_checkpoint)

        print(f"  Filepath: {checkpoint_path}")
        print(f"  Monitor: val_accuracy")
        print(f"  Save best only: True")

        self.config['callbacks']['model_checkpoint'] = {
            'filepath': str(checkpoint_path),
            'monitor': 'val_accuracy',
            'save_best_only': True,
            'mode': 'max'
        }

        # 4. CSVLogger (guardar historial de entrenamiento)
        print("\n📊 CSVLogger:")
        csv_path = self.output_dir / 'training_history.csv'
        csv_logger = CSVLogger(
            filename=str(csv_path),
            separator=',',
            append=False
        )
        callbacks_list.append(csv_logger)

        print(f"  Filepath: {csv_path}")

        self.config['callbacks']['csv_logger'] = {
            'filepath': str(csv_path)
        }

        # 5. TensorBoard (visualización)
        print("\n📈 TensorBoard:")
        tensorboard_dir = self.output_dir / 'tensorboard_logs'
        tensorboard_dir.mkdir(exist_ok=True)
        tensorboard = TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False
        )
        callbacks_list.append(tensorboard)

        print(f"  Log dir: {tensorboard_dir}")
        print(f"  Para visualizar: tensorboard --logdir={tensorboard_dir}")

        self.config['callbacks']['tensorboard'] = {
            'log_dir': str(tensorboard_dir)
        }

        self.callbacks = callbacks_list

        print("\n✅ 5 callbacks configurados exitosamente")

    def save_configuration(self):
        """Guarda la configuración completa en JSON"""
        print("\n" + "=" * 80)
        print("PASO 6: GUARDADO DE CONFIGURACIÓN")
        print("=" * 80)

        config_path = self.output_dir / 'training_config.json'

        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"\n✓ Configuración guardada en: {config_path}")

        # Guardar también versión legible
        readme_path = self.output_dir / 'CONFIG_README.txt'

        with open(readme_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CONFIGURACIÓN DE ENTRENAMIENTO - CORN DISEASE CLASSIFIER\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Fecha: {self.config['metadata']['fecha_creacion']}\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Data dir: {self.data_dir}\n\n")

            f.write("MODELO:\n")
            f.write(f"  Input shape: {self.config['model']['input_shape']}\n")
            f.write(f"  Num classes: {self.num_classes}\n")
            f.write(f"  Classes: {', '.join(self.classes)}\n\n")

            f.write("ENTRENAMIENTO:\n")
            f.write(f"  Batch size: {self.batch_size}\n")
            f.write(f"  Max epochs: {self.config['training']['epochs']}\n")
            f.write(f"  Learning rate: {self.config['training']['initial_learning_rate']}\n")
            f.write(f"  Optimizer: {self.config['training']['optimizer']}\n\n")

            f.write("AUGMENTATION ONLINE (Train):\n")
            aug = self.config['augmentation_online']['train']
            f.write(f"  Rotation: ±{aug['rotation_range']}°\n")
            f.write(f"  Shift H/V: ±{aug['width_shift_range']*100}%\n")
            f.write(f"  Flips: horizontal + vertical\n")
            f.write(f"  Brightness: {aug['brightness_range']}\n")
            f.write(f"  Zoom: ±{aug['zoom_range']*100}%\n\n")

            f.write("CALLBACKS:\n")
            f.write("  1. EarlyStopping (patience=10, monitor=val_loss)\n")
            f.write("  2. ReduceLROnPlateau (factor=0.1, patience=5)\n")
            f.write("  3. ModelCheckpoint (best model by val_accuracy)\n")
            f.write("  4. CSVLogger (training history)\n")
            f.write("  5. TensorBoard (visualization)\n\n")

            if self.class_weights:
                f.write("CLASS WEIGHTS:\n")
                for class_name, class_idx in sorted(self.train_generator.class_indices.items(),
                                                   key=lambda x: x[1]):
                    weight = self.class_weights[class_idx]
                    f.write(f"  {class_name}: {weight:.3f}\n")
            else:
                f.write("CLASS WEIGHTS: No usado (dataset balanceado)\n")

        print(f"✓ README guardado en: {readme_path}")

    def print_summary(self):
        """Imprime resumen de la configuración"""
        print("\n" + "=" * 80)
        print("RESUMEN DE CONFIGURACIÓN")
        print("=" * 80)

        print(f"\n🎯 REPRODUCIBILIDAD:")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Determinismo: Enabled")

        print(f"\n📊 DATOS:")
        print(f"  Train: {self.train_generator.samples} imágenes")
        print(f"  Val: {self.val_generator.samples} imágenes")
        print(f"  Test: {self.test_generator.samples} imágenes")
        print(f"  Batch size: {self.batch_size}")

        print(f"\n🎨 AUGMENTATION:")
        print(f"  Online (train): ✅ Configurado (8 transformaciones)")
        print(f"  Val/Test: Solo normalización")

        print(f"\n⚖️  CLASS WEIGHTS:")
        if self.class_weights:
            print(f"  Usado: ✅ Sí")
        else:
            print(f"  Usado: ❌ No (dataset balanceado)")

        print(f"\n🔔 CALLBACKS:")
        print(f"  EarlyStopping: patience=10")
        print(f"  ReduceLROnPlateau: factor=0.1, patience=5")
        print(f"  ModelCheckpoint: Guardar mejor modelo")
        print(f"  CSVLogger: Historial de métricas")
        print(f"  TensorBoard: Visualización")

        print(f"\n📁 ARCHIVOS DE SALIDA:")
        print(f"  Configuración: {self.output_dir / 'training_config.json'}")
        print(f"  Mejor modelo: {self.output_dir / 'best_model.keras'}")
        print(f"  Historial: {self.output_dir / 'training_history.csv'}")
        print(f"  TensorBoard: {self.output_dir / 'tensorboard_logs'}")

    def run(self):
        """Ejecuta la configuración completa"""
        print("\n" + "⚙️" * 40)
        print("CONFIGURACIÓN DE ENTRENAMIENTO - TENSORFLOW/KERAS")
        print("⚙️" * 40)

        # Paso 1: Seeds (ya ejecutado en __init__)

        # Paso 2: Augmentation
        self.configure_augmentation()

        # Paso 3: Generadores
        self.create_data_generators()

        # Paso 4: Class weights (opcional)
        self.calculate_class_weights(force=False)

        # Paso 5: Callbacks
        self.configure_callbacks()

        # Paso 6: Guardar configuración
        self.save_configuration()

        # Resumen
        self.print_summary()

        print("\n✅ CONFIGURACIÓN DE ENTRENAMIENTO COMPLETADA\n")
        print("=" * 80)
        print("La configuración está lista para entrenamiento.")
        print("Usa los generadores y callbacks en tu modelo:")
        print(f"  - config.train_generator")
        print(f"  - config.val_generator")
        print(f"  - config.test_generator")
        print(f"  - config.callbacks")
        print(f"  - config.class_weights (si aplica)")
        print("=" * 80 + "\n")

        return self.config


def main():
    """Función principal"""
    config = TrainingConfiguration(
        data_dir='./data_augmented',
        output_dir='./training_output',
        random_seed=42
    )

    result = config.run()

    print("\n📝 Siguiente paso:")
    print("   Cargar/crear tu modelo y entrenar usando:")
    print("   model.fit(")
    print("       config.train_generator,")
    print("       validation_data=config.val_generator,")
    print("       epochs=100,")
    print("       callbacks=config.callbacks,")
    print("       class_weight=config.class_weights")
    print("   )")


if __name__ == "__main__":
    main()
