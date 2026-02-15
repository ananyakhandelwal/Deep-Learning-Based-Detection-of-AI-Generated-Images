import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ---------------- CLEAN LOGS ----------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# Set global seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class RealVsAIClassifier:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.base_model = None # Keep reference to base for fine-tuning

    
    # ---------- MODEL ----------
    def create_model(self):
        # Load MobileNetV2 Base
        self.base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False  # Start frozen

        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # MobileNetV2 specific preprocessing (expects -1 to 1 range)
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        x = self.base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)

        self.model = keras.Model(inputs, outputs)
        self._compile_model(lr=5e-5)
        
        print("✓ Model created with MobileNetV2 (Base Frozen)")

    def _compile_model(self, lr):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

    # ---------- DATA PIPELINE ----------
    def prepare_data_generators(self, train_dir):
        # 1. Training Generator (With Augmentation)
        train_datagen = ImageDataGenerator(
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            channel_shift_range=30,
            # Note: No rescale needed for MobileNetV2 preprocess_input
        )

        # 2. Validation Generator (No Augmentation - Clean Data)
        # We must use the same validation_split but without distortion parameters
        val_datagen = ImageDataGenerator(validation_split=0.2)

        # SEED is critical here to ensure the split is the same for both generators
        seed = 42 

        train_gen = train_datagen.flow_from_directory(
            train_dir,
            subset='training',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            seed=seed,
            shuffle=True
        )

        val_gen = val_datagen.flow_from_directory(
            train_dir,
            subset='validation',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            seed=seed,
            shuffle=False 
        )

        return train_gen, val_gen

    # ---------- FINE TUNING (OPTIONAL) ----------
    def unfreeze_and_finetune(self, train_gen, val_gen):
        print("\n--- Starting Fine-Tuning Phase ---")
        self.base_model.trainable = True
        
        # MobileNetV2 has 154 layers. Fine-tune the last 30-40 layers
        # Freeze the earlier layers to preserve learned low-level features
        for layer in self.base_model.layers[:-30]:
            layer.trainable = False
        
        # Make the last 30 layers trainable
        for layer in self.base_model.layers[-30:]:
            layer.trainable = True

        print(f"Total layers: {len(self.base_model.layers)}")
        print(f"Trainable layers: {sum([layer.trainable for layer in self.base_model.layers])}")

        # Recompile with a much lower learning rate
        self._compile_model(lr=1e-5)
        
        # Train for a few more epochs
        self.train(train_gen, val_gen, epochs=15, is_finetuning=True)

    # ---------- TRAINING ----------
    def train(self, train_gen, val_gen, epochs=15, is_finetuning=False):
        
        checkpoint_name = 'MobileNetV2_best_model_finetuned.keras' if is_finetuning else 'MobileNetV2_best_model.keras'
        
        callbacks = [
            ModelCheckpoint(f'/kaggle/working/{checkpoint_name}', save_best_only=True, monitor='val_auc', mode='max'),
            EarlyStopping(patience=4, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-7)
        ]

        print(f"Training for {epochs} epochs...")
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Append history if fine-tuning, else overwrite
        if is_finetuning and self.history:
             for k, v in history.history.items():
                 self.history[k].extend(v)
        else:
            self.history = history.history

    # ---------- PLOTTING ----------
    def plot_results(self, test_gen):
        if not self.history:
            print("No training history to plot.")
            return

        # 1. Loss & Accuracy
        epochs_range = range(1, len(self.history['loss']) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        ax[0].plot(epochs_range, self.history['loss'], label='Train Loss')
        ax[0].plot(epochs_range, self.history['val_loss'], label='Val Loss')
        ax[0].set_title('Loss')
        ax[0].legend()
        
        ax[1].plot(epochs_range, self.history['accuracy'], label='Train Acc')
        ax[1].plot(epochs_range, self.history['val_accuracy'], label='Val Acc')
        ax[1].set_title('Accuracy')
        ax[1].legend()
        plt.savefig("/kaggle/working/MobileNetV2_accuracy_loss.png", dpi=300, bbox_inches='tight')
        plt.show()

        # 2. ROC & PR Curves
        # Ensure test_gen is reset
        test_gen.reset()
        y_true = test_gen.classes
        print("Generating predictions for ROC/PR curves...")
        y_pred = self.model.predict(test_gen, verbose=1).ravel()

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC
        ax[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        ax[0].plot([0, 1], [0, 1], 'k--')
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_title('ROC Curve')
        ax[0].legend()

        # Precision-Recall
        ax[1].plot(recall, precision, label=f'AP = {ap:.4f}')
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].set_title('Precision-Recall Curve')
        ax[1].legend()
        plt.savefig("/kaggle/working/MobileNetV2_roc_pr.png", dpi=300, bbox_inches='tight')
        plt.show()

    def save_final(self):
        self.model.save('/kaggle/working/MobileNetV2_cifake_final.keras')
        print("✓ Model saved as MobileNetV2_cifake_final.keras")
        self.model.save('/kaggle/working/MobileNetV2_cifake_final.h5')
        print("✓ Model saved as MobileNetV2_cifake_final.h5")

# ---------------- MAIN ----------------
def main():
    # Adjust paths as necessary
    BASE_PATH = '/kaggle/input/cifake-real-and-ai-generated-synthetic-images'
    TRAIN_DIR = os.path.join(BASE_PATH, 'train')
    TEST_DIR = os.path.join(BASE_PATH, 'test')

    # Initialize
    classifier = RealVsAIClassifier(img_size=(224, 224), batch_size=64)
    classifier.create_model()

    # Prepare Data
    train_gen, val_gen = classifier.prepare_data_generators(TRAIN_DIR)

    # Phase 1: Train Head (Base Frozen)
    classifier.train(train_gen, val_gen, epochs=15)

    # Phase 2: Fine-Tune (Optional but recommended)
    # Uncomment to enable fine-tuning
    # classifier.unfreeze_and_finetune(train_gen, val_gen)

    # Test Data Generator (Clean, No Shuffle for metrics)
    test_datagen = ImageDataGenerator() # No rescale needed due to internal preprocessing
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=64,
        class_mode='binary',
        shuffle=False 
    )

    # Visualize and Save
    classifier.plot_results(test_gen)
    classifier.save_final()

if __name__ == "__main__":
    main()