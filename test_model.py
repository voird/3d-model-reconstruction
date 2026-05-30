import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mpl_toolkits.mplot3d import Axes3D

IMG_SIZE = 64
VOXEL_SIZE = 32
BATCH_SIZE = 16

DATA_DIR = r'D:\kurs2026\renders'
MODEL_SAVE_PATH = r'D:\kurs2026\model.keras'
PREDICTIONS_SAVE_DIR = r'D:\kurs2026\prediction'

class IoU(keras.metrics.Metric):
    def __init__(self, name='iou_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        numerator = self.true_positives
        denominator = self.true_positives + self.false_positives + self.false_negatives
        return tf.math.divide_no_nan(numerator, denominator)

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

def voxel_accuracy(y_true, y_pred):
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_binary), tf.float32))

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce_loss = keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
    return bce_loss + dice_loss(y_true, y_pred)

def load_image(image_path):
    image_path = tf.cast(image_path, tf.string)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

def load_voxel_shape(shape_path):
    shape_path_str = shape_path.numpy().decode('utf-8')
    shape = np.load(shape_path_str)
    shape = shape.astype(np.float32)
    shape = shape[..., np.newaxis]
    return tf.convert_to_tensor(shape)

def process_path(front_path, top_path, back_path, shape_path):
    front_img = load_image(front_path)
    top_img = load_image(top_path)
    back_img = load_image(back_path)
    
    shape = tf.py_function(load_voxel_shape, [shape_path], tf.float32)
    shape.set_shape([VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1])
    
    combined_images = tf.concat([front_img, top_img, back_img], axis=-1)
    return combined_images, shape

def load_test_data(data_dir, num_samples=10):
    image_dir = os.path.join(data_dir, 'images')
    shape_dir = os.path.join(data_dir, 'shapes')
    
    all_front_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if 'front' in f and f.endswith('.png')])
    all_top_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if 'top' in f and f.endswith('.png')])
    all_back_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if 'back' in f and f.endswith('.png')])
    all_shape_paths = sorted([os.path.join(shape_dir, f) for f in os.listdir(shape_dir) if f.endswith('.npy')])
    
    test_front_paths = tf.constant(all_front_paths[-num_samples:], dtype=tf.string)
    test_top_paths = tf.constant(all_top_paths[-num_samples:], dtype=tf.string)
    test_back_paths = tf.constant(all_back_paths[-num_samples:], dtype=tf.string)
    test_shape_paths = tf.constant(all_shape_paths[-num_samples:], dtype=tf.string)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_front_paths, test_top_paths, test_back_paths, test_shape_paths))
    test_dataset = test_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(1).prefetch(tf.data.AUTOTUNE)
    
    return test_dataset

def visualize_voxel(voxel_data, title="Voxel Reconstruction", save_path=None):
    if voxel_data.ndim == 4:
        voxel_data = voxel_data[..., 0]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = np.where(voxel_data > 0.5)
    
    if len(x) > 0:
        ax.scatter(x, y, z, c='red', marker='s', s=5, alpha=0.7)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_xlim(0, VOXEL_SIZE)
    ax.set_ylim(0, VOXEL_SIZE)
    ax.set_zlim(0, VOXEL_SIZE)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def test_model():
    print("="*50)
    print("ЗАПУСК ТЕСТИРОВАНИЯ МОДЕЛИ")
    print("="*50)
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Ошибка: Модель не найдена по пути {MODEL_SAVE_PATH}")
        print("Сначала обучите модель с помощью main.py")
        return
    
    print("Загрузка модели...")
    try:
        model = keras.models.load_model(
            MODEL_SAVE_PATH,
            custom_objects={
                'IoU': IoU,
                'combined_loss': combined_loss,
                'voxel_accuracy': voxel_accuracy
            }
        )
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return
    
    print("Загрузка тестовых данных...")
    try:
        test_dataset = load_test_data(DATA_DIR, num_samples=10)
        print("Тестовые данные загружены!")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return
    
    os.makedirs(PREDICTIONS_SAVE_DIR, exist_ok=True)
    
    print("\nНачало тестирования...\n")
    test_ious = []
    test_accuracies = []
    
    for i, (images, true_voxels) in enumerate(test_dataset):
        print(f"Обработка образца {i+1}/10...")
        
        predictions = model.predict(images, verbose=0)
        
        input_img_combined = images[0].numpy()
        true_voxel = true_voxels[0].numpy()
        predicted_voxel = predictions[0]
        
        pred_binary = (predicted_voxel > 0.5).astype(np.float32)
        true_binary = (true_voxel > 0.5).astype(np.float32)
        
        intersection = np.sum(true_binary * pred_binary)
        union = np.sum(true_binary) + np.sum(pred_binary) - intersection
        iou = intersection / (union + 1e-6)
        accuracy = np.mean(true_binary == pred_binary)
        
        test_ious.append(iou)
        test_accuracies.append(accuracy)
        
        np.save(os.path.join(PREDICTIONS_SAVE_DIR, f'predicted_model_{i:02d}.npy'), predicted_voxel)
        np.save(os.path.join(PREDICTIONS_SAVE_DIR, f'true_model_{i:02d}.npy'), true_voxel)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(input_img_combined[..., 0], cmap='gray')
        plt.title("Front View", fontsize=12)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(input_img_combined[..., 1], cmap='gray')
        plt.title("Top View", fontsize=12)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(input_img_combined[..., 2], cmap='gray')
        plt.title("Back View", fontsize=12)
        plt.axis('off')
        
        plt.suptitle(f"Sample {i} - Input Images", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PREDICTIONS_SAVE_DIR, f'sample_{i:02d}_inputs.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        visualize_voxel(
            true_voxel,
            title=f"Sample {i} - Ground Truth",
            save_path=os.path.join(PREDICTIONS_SAVE_DIR, f'sample_{i:02d}_ground_truth.png')
        )
        
        visualize_voxel(
            predicted_voxel,
            title=f"Sample {i} - Predicted (IoU: {iou:.4f}, Acc: {accuracy:.4f})",
            save_path=os.path.join(PREDICTIONS_SAVE_DIR, f'sample_{i:02d}_prediction.png')
        )
        
        print(f"  IoU = {iou:.4f}, Accuracy = {accuracy:.4f}")
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*50)
    print(f"Средний IoU: {np.mean(test_ious):.4f} ± {np.std(test_ious):.4f}")
    print(f"Средняя Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
    print(f"Min IoU: {np.min(test_ious):.4f}, Max IoU: {np.max(test_ious):.4f}")
    print("="*50)
    
    with open(os.path.join(PREDICTIONS_SAVE_DIR, 'test_results.txt'), 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ\n")
        f.write("="*50 + "\n")
        f.write(f"Средний IoU: {np.mean(test_ious):.4f} ± {np.std(test_ious):.4f}\n")
        f.write(f"Средняя Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}\n")
        f.write(f"Min IoU: {np.min(test_ious):.4f}, Max IoU: {np.max(test_ious):.4f}\n")
        f.write("="*50 + "\n\n")
        f.write("Детальные результаты по образцам:\n")
        for i, (iou, acc) in enumerate(zip(test_ious, test_accuracies)):
            f.write(f"Sample {i}: IoU = {iou:.4f}, Accuracy = {acc:.4f}\n")
    
    print(f"\nРезультаты сохранены в: {PREDICTIONS_SAVE_DIR}")
    print("Тестирование завершено!")

if __name__ == "__main__":
    test_model()
