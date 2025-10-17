"""
Paso 5: Creación del Modelo
- MobileNetV3Large preentrenado (ImageNet)
- BatchNormalization + Dropout(0.5)
- Dense(4, softmax)
- Coherente con training_config.py (Paso 4)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Large
from pathlib import Path
from datetime import datetime


class ModelCreator:
    """Clase para crear el modelo de clasificación de enfermedades de maíz"""

    def __init__(self,
                 config_path='./training_output/training_config.json',
                 output_dir='./model_output',
                 random_seed=42):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_seed = random_seed

        # Cargar configuración del Paso 4
        self._load_training_config(config_path)

        # Configurar seeds (consistente con Paso 4)
        self._set_random_seeds()

        # Parámetros del modelo
        self.model_config = {
            'metadata': {
                'fecha_creacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'random_seed': self.random_seed,
                'training_config_used': str(config_path)
            },
            'architecture': {
                'base_model': 'MobileNetV3Large',
                'pretrained_weights': 'imagenet',
                'include_top': False,
                'input_shape': None,
                'pooling': 'avg',
                'dropout_rate': 0.5,
                'num_classes': None
            },
            'compilation': {
                'optimizer': None,
                'loss': None,
                'metrics': None,
                'learning_rate': None
            }
        }

    def _load_training_config(self, config_path):
        """Carga la configuración del Paso 4 para coherencia"""
        print("=" * 80)
        print("CARGANDO CONFIGURACIÓN DEL PASO 4")
        print("=" * 80)

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"No se encontró training_config.json en {config_path}\n"
                f"Ejecuta primero training_config.py (Paso 4)"
            )

        with open(config_path, 'r') as f:
            self.training_config = json.load(f)

        # Extraer parámetros necesarios
        self.input_shape = tuple(self.training_config['model']['input_shape'])
        self.num_classes = self.training_config['model']['num_classes']
        self.classes = self.training_config['model']['classes']
        self.optimizer_name = self.training_config['training']['optimizer']
        self.learning_rate = self.training_config['training']['initial_learning_rate']
        self.loss = self.training_config['training']['loss']
        self.metrics = self.training_config['training']['metrics']

        print(f"\n✓ Configuración cargada desde: {config_path}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Classes: {', '.join(self.classes)}")
        print(f"  Optimizer: {self.optimizer_name}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Loss: {self.loss}")

    def _set_random_seeds(self):
        """Fija seeds (coherente con Paso 4)"""
        import random
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    def create_model(self):
        """Crea el modelo con MobileNetV3Large + BatchNorm + Dropout + Dense"""
        print("\n" + "=" * 80)
        print("PASO 1: CREACIÓN DE ARQUITECTURA DEL MODELO")
        print("=" * 80)

        print("\n🏗️  Construyendo modelo...")

        # 1. Base model: MobileNetV3Large preentrenado
        print("\n📦 Cargando MobileNetV3Large (ImageNet)...")
        base_model = MobileNetV3Large(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'  # Global Average Pooling
        )

        print(f"  ✓ Base model cargado")
        print(f"    Input shape: {self.input_shape}")
        print(f"    Weights: imagenet")
        print(f"    Include top: False")
        print(f"    Pooling: avg (Global Average Pooling)")
        print(f"    Parámetros base: {base_model.count_params():,}")

        # 2. Congelar base model (fine-tuning opcional después)
        base_model.trainable = False
        print(f"  ✓ Base model congelado (trainable=False)")

        # 3. Construir modelo completo
        print("\n🔧 Añadiendo capas personalizadas...")

        inputs = keras.Input(shape=self.input_shape, name='input_layer')

        # Base model
        x = base_model(inputs, training=False)

        # BatchNormalization
        x = layers.BatchNormalization(name='batch_norm')(x)
        print(f"  ✓ BatchNormalization añadida")

        # Dropout
        x = layers.Dropout(rate=0.5, name='dropout')(x)
        print(f"  ✓ Dropout(rate=0.5) añadido")

        # Capa de salida
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output_layer'
        )(x)
        print(f"  ✓ Dense({self.num_classes}, activation='softmax') añadida")

        # Crear modelo
        model = Model(inputs=inputs, outputs=outputs, name='CornDiseaseClassifier')

        print(f"\n✅ Modelo creado exitosamente")
        print(f"  Nombre: {model.name}")
        print(f"  Total parámetros: {model.count_params():,}")

        # Guardar arquitectura en config
        self.model_config['architecture']['input_shape'] = list(self.input_shape)
        self.model_config['architecture']['num_classes'] = self.num_classes

        self.model = model
        return model

    def compile_model(self):
        """Compila el modelo usando configuración del Paso 4"""
        print("\n" + "=" * 80)
        print("PASO 2: COMPILACIÓN DEL MODELO")
        print("=" * 80)

        print(f"\n⚙️  Compilando modelo con configuración del Paso 4...")

        # Crear optimizer
        if self.optimizer_name.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            optimizer = self.optimizer_name

        print(f"  Optimizer: {self.optimizer_name}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Loss: {self.loss}")
        print(f"  Metrics: {', '.join(self.metrics)}")

        # Compilar
        self.model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=self.metrics
        )

        print(f"\n✅ Modelo compilado exitosamente")

        # Guardar configuración de compilación
        self.model_config['compilation'] = {
            'optimizer': self.optimizer_name,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'metrics': self.metrics
        }

    def print_model_summary(self):
        """Imprime y guarda resumen del modelo"""
        print("\n" + "=" * 80)
        print("PASO 3: RESUMEN DEL MODELO")
        print("=" * 80)

        # Imprimir a consola
        print("\n")
        self.model.summary()

        # Guardar a archivo
        summary_path = self.output_dir / 'model_summary.txt'
        with open(summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        print(f"\n✓ Resumen guardado en: {summary_path}")

        # Contar parámetros
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        print(f"\n📊 Parámetros del modelo:")
        print(f"  Total: {total_params:,}")
        print(f"  Entrenables: {trainable_params:,}")
        print(f"  No entrenables: {non_trainable_params:,}")

        self.model_config['parameters'] = {
            'total': int(total_params),
            'trainable': int(trainable_params),
            'non_trainable': int(non_trainable_params)
        }

    def save_model_config(self):
        """Guarda la configuración del modelo"""
        print("\n" + "=" * 80)
        print("PASO 4: GUARDADO DE CONFIGURACIÓN DEL MODELO")
        print("=" * 80)

        config_path = self.output_dir / 'model_config.json'

        with open(config_path, 'w') as f:
            json.dump(self.model_config, f, indent=2)

        print(f"\n✓ Configuración guardada en: {config_path}")

        # Guardar README
        readme_path = self.output_dir / 'MODEL_README.txt'

        with open(readme_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODELO - CORN DISEASE CLASSIFIER\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Fecha creación: {self.model_config['metadata']['fecha_creacion']}\n")
            f.write(f"Random seed: {self.random_seed}\n\n")

            f.write("ARQUITECTURA:\n")
            f.write(f"  Base model: MobileNetV3Large (ImageNet)\n")
            f.write(f"  Include top: False\n")
            f.write(f"  Pooling: Global Average Pooling\n")
            f.write(f"  Input shape: {self.input_shape}\n\n")

            f.write("CAPAS AÑADIDAS:\n")
            f.write(f"  1. BatchNormalization\n")
            f.write(f"  2. Dropout (rate=0.5)\n")
            f.write(f"  3. Dense({self.num_classes}, activation='softmax')\n\n")

            f.write("COMPILACIÓN:\n")
            f.write(f"  Optimizer: {self.optimizer_name}\n")
            f.write(f"  Learning rate: {self.learning_rate}\n")
            f.write(f"  Loss: {self.loss}\n")
            f.write(f"  Metrics: {', '.join(self.metrics)}\n\n")

            f.write("PARÁMETROS:\n")
            f.write(f"  Total: {self.model_config['parameters']['total']:,}\n")
            f.write(f"  Entrenables: {self.model_config['parameters']['trainable']:,}\n")
            f.write(f"  No entrenables: {self.model_config['parameters']['non_trainable']:,}\n\n")

            f.write("CLASES:\n")
            for i, class_name in enumerate(self.classes):
                f.write(f"  {i}: {class_name}\n")

        print(f"✓ README guardado en: {readme_path}")

    def save_model(self):
        """Guarda el modelo inicial (sin entrenar)"""
        print("\n" + "=" * 80)
        print("PASO 5: GUARDADO DEL MODELO INICIAL")
        print("=" * 80)

        model_path = self.output_dir / 'model_initial.keras'

        self.model.save(model_path)

        print(f"\n✓ Modelo inicial guardado en: {model_path}")
        print(f"  Formato: Keras (.keras)")
        print(f"  Tamaño: {model_path.stat().st_size / (1024**2):.2f} MB")

    def print_final_summary(self):
        """Imprime resumen final"""
        print("\n" + "=" * 80)
        print("RESUMEN FINAL")
        print("=" * 80)

        print(f"\n🎯 ARQUITECTURA:")
        print(f"  Base: MobileNetV3Large (ImageNet)")
        print(f"  Custom layers: BatchNorm → Dropout(0.5) → Dense(4, softmax)")
        print(f"  Total parámetros: {self.model_config['parameters']['total']:,}")

        print(f"\n⚙️  COMPILACIÓN:")
        print(f"  Optimizer: {self.optimizer_name} (lr={self.learning_rate})")
        print(f"  Loss: {self.loss}")

        print(f"\n📊 DATOS (del Paso 4):")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Clases: {', '.join(self.classes)}")

        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"  Modelo inicial: {self.output_dir / 'model_initial.keras'}")
        print(f"  Configuración: {self.output_dir / 'model_config.json'}")
        print(f"  Resumen: {self.output_dir / 'model_summary.txt'}")
        print(f"  README: {self.output_dir / 'MODEL_README.txt'}")

    def run(self):
        """Ejecuta la creación completa del modelo"""
        print("\n" + "🤖" * 40)
        print("CREACIÓN DEL MODELO - MOBILENETV3LARGE")
        print("🤖" * 40)

        # Paso 1: Crear arquitectura
        self.create_model()

        # Paso 2: Compilar
        self.compile_model()

        # Paso 3: Resumen
        self.print_model_summary()

        # Paso 4: Guardar configuración
        self.save_model_config()

        # Paso 5: Guardar modelo inicial
        self.save_model()

        # Resumen final
        self.print_final_summary()

        print("\n✅ MODELO CREADO EXITOSAMENTE\n")
        print("=" * 80)
        print("El modelo está listo para entrenamiento.")
        print("Usa training_config.py para obtener generadores y callbacks.")
        print("=" * 80 + "\n")

        return self.model


def main():
    """Función principal"""
    creator = ModelCreator(
        config_path='./training_output/training_config.json',
        output_dir='./model_output',
        random_seed=42
    )

    model = creator.run()

    print("\n📝 Siguiente paso:")
    print("   Entrenar el modelo usando:")
    print("   - Generadores del Paso 4 (training_config.py)")
    print("   - Este modelo (model_output/model_initial.keras)")
    print("   - Callbacks del Paso 4")


if __name__ == "__main__":
    main()
