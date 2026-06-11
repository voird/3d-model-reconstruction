import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skimage.draw import line, rectangle, ellipse, circle_perimeter, polygon
from mpl_toolkits.mplot3d import Axes3D
import random

IMG_SIZE = 64
VOXEL_SIZE = 32
BATCH_SIZE = 16
NUM_SAMPLES = 3000
EPOCHS = 40
LEARNING_RATE = 1e-4

DATA_DIR = r'D:\kurs2026\renders'
MODEL_SAVE_PATH = r'D:\kurs2026\model.keras'
HISTORY_SAVE_PATH = r'D:\kurs2026\history.png'
PREDICTIONS_SAVE_DIR = r'D:\kurs2026\prediction'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Найдено GPU: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("GPU не найден. Проверьте установку драйверов/CUDA.")

def add_cube(shape, voxel_size):
    x_start, y_start, z_start = np.random.randint(0, voxel_size // 2, size=3)
    x_end, y_end, z_end = np.random.randint(voxel_size // 2, voxel_size, size=3)
    
    x_start, x_end = sorted([x_start, x_end])
    y_start, y_end = sorted([y_start, y_end])
    z_start, z_end = sorted([z_start, z_end])
    
    shape[x_start:x_end, y_start:y_end, z_start:z_end] = 1.0
    return shape

def add_sphere(shape, voxel_size):
    center_x = np.random.randint(5, voxel_size-5)
    center_y = np.random.randint(5, voxel_size-5)
    center_z = np.random.randint(5, voxel_size-5)
    radius = np.random.randint(3, 8)
    
    for i in range(voxel_size):
        for j in range(voxel_size):
            for k in range(voxel_size):
                if (i-center_x)**2 + (j-center_y)**2 + (k-center_z)**2 < radius**2:
                    shape[i, j, k] = 1.0
    return shape

def add_cylinder(shape, voxel_size):
    center_x = np.random.randint(5, voxel_size-5)
    center_z = np.random.randint(5, voxel_size-5)
    radius = np.random.randint(3, 7)
    height_start = np.random.randint(5, voxel_size-10)
    height_end = np.random.randint(height_start+3, voxel_size-5)
    
    for i in range(voxel_size):
        for j in range(height_start, height_end):
            for k in range(voxel_size):
                if (i-center_x)**2 + (k-center_z)**2 < radius**2:
                    shape[i, j, k] = 1.0
    return shape

def add_cone(shape, voxel_size):
    center_x = np.random.randint(5, voxel_size-5)
    center_z = np.random.randint(5, voxel_size-5)
    base_radius = np.random.randint(4, 8)
    height_start = np.random.randint(5, voxel_size-10)
    height_end = np.random.randint(height_start+3, voxel_size-5)
    
    for j in range(height_start, height_end):
        t = (j - height_start) / (height_end - height_start)
        radius = int(base_radius * (1 - t))
        for i in range(center_x - radius, center_x + radius):
            for k in range(center_z - radius, center_z + radius):
                if (i-center_x)**2 + (k-center_z)**2 < radius**2:
                    if 0 <= i < voxel_size and 0 <= k < voxel_size:
                        shape[i, j, k] = 1.0
    return shape

def add_torus(shape, voxel_size):
    center_x = voxel_size // 2
    center_y = voxel_size // 2
    center_z = voxel_size // 2
    R = np.random.randint(8, 12)
    r = np.random.randint(3, 5)
    
    for i in range(voxel_size):
        for j in range(voxel_size):
            for k in range(voxel_size):
                x = i - center_x
                y = j - center_y
                z = k - center_z
                distance_to_center = np.sqrt(x**2 + z**2)
                if abs((R - distance_to_center)**2 + y**2 - r**2) < 3:
                    shape[i, j, k] = 1.0
    return shape

def add_pyramid(shape, voxel_size):
    center_x = voxel_size // 2
    center_z = voxel_size // 2
    base_size = np.random.randint(8, 14)
    height = np.random.randint(8, 16)
    base_y = np.random.randint(5, voxel_size - height - 5)
    
    for j in range(base_y, base_y + height):
        t = (j - base_y) / height
        current_size = int(base_size * (1 - t))
        for i in range(center_x - current_size//2, center_x + current_size//2):
            for k in range(center_z - current_size//2, center_z + current_size//2):
                if 0 <= i < voxel_size and 0 <= k < voxel_size:
                    shape[i, j, k] = 1.0
    return shape

def add_rectangular_prism(shape, voxel_size):
    x_start = np.random.randint(3, voxel_size-10)
    y_start = np.random.randint(3, voxel_size-10)
    z_start = np.random.randint(3, voxel_size-10)
    x_end = x_start + np.random.randint(5, 12)
    y_end = y_start + np.random.randint(5, 12)
    z_end = z_start + np.random.randint(5, 12)
    
    x_start, x_end = min(x_start, x_end), max(x_start, x_end)
    y_start, y_end = min(y_start, y_end), max(y_start, y_end)
    z_start, z_end = min(z_start, z_end), max(z_start, z_end)
    
    shape[x_start:x_end, y_start:y_end, z_start:z_end] = 1.0
    return shape

def add_cross(shape, voxel_size):
    center_x = np.random.randint(8, voxel_size-8)
    center_y = np.random.randint(8, voxel_size-8)
    center_z = np.random.randint(8, voxel_size-8)
    arm_length = np.random.randint(3, 6)
    arm_thickness = np.random.randint(2, 4)
    
    for i in range(center_x - arm_length, center_x + arm_length):
        for dy in range(-arm_thickness//2, arm_thickness//2):
            for dz in range(-arm_thickness//2, arm_thickness//2):
                if 0 <= i < voxel_size and 0 <= center_y+dy < voxel_size and 0 <= center_z+dz < voxel_size:
                    shape[i, center_y+dy, center_z+dz] = 1.0
    
    for j in range(center_y - arm_length, center_y + arm_length):
        for dx in range(-arm_thickness//2, arm_thickness//2):
            for dz in range(-arm_thickness//2, arm_thickness//2):
                if 0 <= center_x+dx < voxel_size and 0 <= j < voxel_size and 0 <= center_z+dz < voxel_size:
                    shape[center_x+dx, j, center_z+dz] = 1.0
    
    for k in range(center_z - arm_length, center_z + arm_length):
        for dx in range(-arm_thickness//2, arm_thickness//2):
            for dy in range(-arm_thickness//2, arm_thickness//2):
                if 0 <= center_x+dx < voxel_size and 0 <= center_y+dy < voxel_size and 0 <= k < voxel_size:
                    shape[center_x+dx, center_y+dy, k] = 1.0
    return shape

def add_letter_l(shape, voxel_size):
    corner_x = np.random.randint(5, voxel_size-10)
    corner_y = np.random.randint(5, voxel_size-10)
    corner_z = np.random.randint(5, voxel_size-10)
    length = np.random.randint(5, 9)
    thickness = np.random.randint(2, 4)
    
    for i in range(corner_x, corner_x + thickness):
        for j in range(corner_y, corner_y + length):
            for k in range(corner_z, corner_z + thickness):
                if 0 <= i < voxel_size and 0 <= j < voxel_size and 0 <= k < voxel_size:
                    shape[i, j, k] = 1.0
    
    for i in range(corner_x, corner_x + length):
        for j in range(corner_y, corner_y + thickness):
            for k in range(corner_z, corner_z + thickness):
                if 0 <= i < voxel_size and 0 <= j < voxel_size and 0 <= k < voxel_size:
                    shape[i, j, k] = 1.0
    return shape

def add_ring(shape, voxel_size):
    center_x = np.random.randint(8, voxel_size-8)
    center_z = np.random.randint(8, voxel_size-8)
    outer_radius = np.random.randint(6, 10)
    inner_radius = np.random.randint(3, outer_radius-1)
    height_start = np.random.randint(5, voxel_size-10)
    height_end = np.random.randint(height_start+3, voxel_size-5)
    
    for i in range(voxel_size):
        for j in range(height_start, height_end):
            for k in range(voxel_size):
                dist = np.sqrt((i-center_x)**2 + (k-center_z)**2)
                if inner_radius <= dist <= outer_radius:
                    shape[i, j, k] = 1.0
    return shape

def add_random_shape(shape, voxel_size):
    num_shapes = np.random.randint(1, 4)
    
    for _ in range(num_shapes):
        shape_type = np.random.choice([
            'cube', 'sphere', 'cylinder', 'cone', 'pyramid', 
            'prism', 'cross', 'letter_l', 'ring'
        ])
        
        if shape_type == 'cube':
            shape = add_cube(shape, voxel_size)
        elif shape_type == 'sphere':
            shape = add_sphere(shape, voxel_size)
        elif shape_type == 'cylinder':
            shape = add_cylinder(shape, voxel_size)
        elif shape_type == 'cone':
            shape = add_cone(shape, voxel_size)
        elif shape_type == 'pyramid':
            shape = add_pyramid(shape, voxel_size)
        elif shape_type == 'prism':
            shape = add_rectangular_prism(shape, voxel_size)
        elif shape_type == 'cross':
            shape = add_cross(shape, voxel_size)
        elif shape_type == 'letter_l':
            shape = add_letter_l(shape, voxel_size)
        elif shape_type == 'ring':
            shape = add_ring(shape, voxel_size)
    
    if np.random.rand() > 0.7:
        num_protrusions = np.random.randint(1, 4)
        for _ in range(num_protrusions):
            protrusion_type = np.random.choice(['sphere', 'cube'])
            if protrusion_type == 'sphere':
                shape = add_sphere(shape, voxel_size)
            else:
                shape = add_cube(shape, voxel_size)
    
    return shape

def generate_voxel_shape(voxel_size):
    shape = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    
    shape = add_random_shape(shape, voxel_size)
    
    if np.random.rand() > 0.8:
        num_holes = np.random.randint(1, 3)
        for _ in range(num_holes):
            hole_center_x = np.random.randint(5, voxel_size-5)
            hole_center_y = np.random.randint(5, voxel_size-5)
            hole_center_z = np.random.randint(5, voxel_size-5)
            hole_radius = np.random.randint(2, 4)
            
            for i in range(voxel_size):
                for j in range(voxel_size):
                    for k in range(voxel_size):
                        if (i-hole_center_x)**2 + (j-hole_center_y)**2 + (k-hole_center_z)**2 < hole_radius**2:
                            shape[i, j, k] = 0.0
    
    if np.sum(shape) == 0:
        shape = add_cube(shape, voxel_size)
    
    return shape

def project_voxel_to_2d(voxel_shape, view_axis):
    if view_axis == 0:
        img = np.max(voxel_shape, axis=2)
    elif view_axis == 1:
        img = np.max(voxel_shape, axis=1)
    elif view_axis == 2:
        img = np.max(voxel_shape, axis=2)
        img = np.fliplr(img)
    elif view_axis == 3:
        img = np.max(voxel_shape, axis=0)
    elif view_axis == 4:
        img = np.max(voxel_shape, axis=0)
        img = np.fliplr(img)
        
    img = (img > 0).astype(np.float32)
    
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img_tf = tf.convert_to_tensor(img[None, ..., None], dtype=tf.float32)
        img = tf.image.resize(img_tf, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = img[0, ..., 0].numpy()
        
    if np.random.rand() > 0.9:
        noise = np.random.randn(*img.shape) * 0.05
        img = np.clip(img + noise, 0, 1)
        
    return img

def create_synthetic_dataset(num_samples, img_size, voxel_size, data_dir):
    print(f"Generating {num_samples} synthetic samples with complex shapes...")
    img_path = os.path.join(data_dir, 'images')
    shape_path = os.path.join(data_dir, 'shapes')
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(shape_path, exist_ok=True)
    
    existing_images = [f for f in os.listdir(img_path) if f.endswith('.png')]
    if len(existing_images) >= num_samples * 3:
        print(f"Dataset already exists with {len(existing_images)//3} samples. Skipping generation.")
        return
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Generating sample {i}/{num_samples}")
        
        model_id = f"{i:04d}"
        voxel_shape = generate_voxel_shape(voxel_size)
        
        np.save(os.path.join(shape_path, f'model_{model_id}.npy'), voxel_shape)
        
        front_img = project_voxel_to_2d(voxel_shape, 0)
        top_img = project_voxel_to_2d(voxel_shape, 1)
        back_img = project_voxel_to_2d(voxel_shape, 2)
        right_img = project_voxel_to_2d(voxel_shape, 3)
        left_img = project_voxel_to_2d(voxel_shape, 4)
        
        plt.imsave(os.path.join(img_path, f'model_{model_id}_front.png'), front_img, cmap='gray')
        plt.imsave(os.path.join(img_path, f'model_{model_id}_top.png'), top_img, cmap='gray')
        plt.imsave(os.path.join(img_path, f'model_{model_id}_back.png'), back_img, cmap='gray')
        plt.imsave(os.path.join(img_path, f'model_{model_id}_right.png'), right_img, cmap='gray')
        plt.imsave(os.path.join(img_path, f'model_{model_id}_left.png'), left_img, cmap='gray')
        
        metadata = {
            'id': model_id,
            'voxel_count': int(np.sum(voxel_shape)),
            'voxel_density': float(np.mean(voxel_shape))
        }
        
        with open(os.path.join(shape_path, f'model_{model_id}_metadata.txt'), 'w') as f:
            f.write(str(metadata))
        
    print(f"Synthetic dataset generated with {num_samples} samples.")
    print(f"Each sample includes: front, top, back, right, left views")
    print(f"Complex shapes include cubes, spheres, cylinders, cones, pyramids, crosses, rings, and more!")

if not os.path.exists(DATA_DIR):
    print(f"Creating dataset at {DATA_DIR}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    create_synthetic_dataset(NUM_SAMPLES, IMG_SIZE, VOXEL_SIZE, DATA_DIR)
else:
    print(f"Dataset directory {DATA_DIR} exists.")
    img_dir = os.path.join(DATA_DIR, 'images')
    if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
        print("Dataset is empty. Generating...")
        create_synthetic_dataset(NUM_SAMPLES, IMG_SIZE, VOXEL_SIZE, DATA_DIR)
    else:
        print(f"Found {len(os.listdir(img_dir))} images.")

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
    
    shape = tf.py_function(
        load_voxel_shape, 
        [shape_path], 
        tf.float32
    )
    shape.set_shape([VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1])
    
    combined_images = tf.concat([front_img, top_img, back_img], axis=-1)
    return combined_images, shape

image_dir = os.path.join(DATA_DIR, 'images')
shape_dir = os.path.join(DATA_DIR, 'shapes')

all_front_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if 'front' in f and f.endswith('.png')])
all_top_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if 'top' in f and f.endswith('.png')])
all_back_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if 'back' in f and f.endswith('.png')])
all_shape_paths = sorted([os.path.join(shape_dir, f) for f in os.listdir(shape_dir) if f.endswith('.npy')])

print(f"Found images: {len(all_front_paths)} front, {len(all_top_paths)} top, {len(all_back_paths)} back")
print(f"Found shapes: {len(all_shape_paths)}")

if len(all_front_paths) == 0:
    raise RuntimeError(f"No images found in {image_dir}. Please check the dataset generation.")

min_count = min(len(all_front_paths), len(all_top_paths), len(all_back_paths), len(all_shape_paths))
if min_count == 0:
    raise RuntimeError("Some file categories are empty!")

all_front_paths = all_front_paths[:min_count]
all_top_paths = all_top_paths[:min_count]
all_back_paths = all_back_paths[:min_count]
all_shape_paths = all_shape_paths[:min_count]

train_size = int(min_count * 0.8)
train_front_paths, val_front_paths = all_front_paths[:train_size], all_front_paths[train_size:]
train_top_paths, val_top_paths = all_top_paths[:train_size], all_top_paths[train_size:]
train_back_paths, val_back_paths = all_back_paths[:train_size], all_back_paths[train_size:]
train_shape_paths, val_shape_paths = all_shape_paths[:train_size], all_shape_paths[train_size:]

train_front_paths = tf.constant(train_front_paths, dtype=tf.string)
train_top_paths = tf.constant(train_top_paths, dtype=tf.string)
train_back_paths = tf.constant(train_back_paths, dtype=tf.string)
train_shape_paths = tf.constant(train_shape_paths, dtype=tf.string)

val_front_paths = tf.constant(val_front_paths, dtype=tf.string)
val_top_paths = tf.constant(val_top_paths, dtype=tf.string)
val_back_paths = tf.constant(val_back_paths, dtype=tf.string)
val_shape_paths = tf.constant(val_shape_paths, dtype=tf.string)

train_dataset = tf.data.Dataset.from_tensor_slices((train_front_paths, train_top_paths, train_back_paths, train_shape_paths))
train_dataset = train_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_front_paths, val_top_paths, val_back_paths, val_shape_paths))
val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Train dataset size: {train_size}")
print(f"Validation dataset size: {min_count - train_size}")

def build_model(img_size, voxel_size):
    input_images = keras.Input(shape=(img_size, img_size, 3))

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_images)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Dropout(0.3)(x) 
    x = layers.Flatten()(x)
    x = layers.Flatten()(x)
    
    initial_3d_dim = voxel_size // 8
    
    x = layers.Dense(initial_3d_dim * initial_3d_dim * initial_3d_dim * 256, activation='relu')(x)
    x = layers.Reshape((initial_3d_dim, initial_3d_dim, initial_3d_dim, 256))(x)

    x = layers.Conv3DTranspose(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling3D(2)(x)

    x = layers.Conv3DTranspose(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling3D(2)(x)
    
    x = layers.Conv3DTranspose(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling3D(2)(x)
    
    output_voxels = layers.Conv3D(1, 1, activation='sigmoid', padding='same')(x)

    model = keras.Model(inputs=input_images, outputs=output_voxels)
    return model

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

def add_helix(shape, voxel_size):
    """Добавляет трехмерную спираль."""
    center_x, center_z = voxel_size // 2, voxel_size // 2
    turns = np.random.randint(2, 4)
    for t_val in np.linspace(0, turns * 2 * np.pi, 200):
        r = 6
        x = int(center_x + r * np.cos(t_val))
        z = int(center_z + r * np.sin(t_val))
        y = int(5 + (t_val / (turns * 2 * np.pi)) * (voxel_size - 10))
        if 0 <= x < voxel_size and 0 <= y < voxel_size and 0 <= z < voxel_size:
            shape[x, y, z] = 1.0
    return shape


model = build_model(IMG_SIZE, VOXEL_SIZE)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=combined_loss,
    metrics=[IoU(), voxel_accuracy, keras.metrics.Precision(thresholds=0.5), keras.metrics.Recall(thresholds=0.5)]
)

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_iou_score', mode='max', verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

print("\nStarting model training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)
print("Training finished.")

def plot_training_history(history, save_path):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['iou_score'], label='Train IoU', linewidth=2)
    plt.plot(history.history['val_iou_score'], label='Validation IoU', linewidth=2)
    plt.title('IoU Score over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('IoU Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(history.history['voxel_accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_voxel_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to {save_path}")

plot_training_history(history, HISTORY_SAVE_PATH)

try:
    best_model = keras.models.load_model(
        MODEL_SAVE_PATH, 
        custom_objects={
            'IoU': IoU, 
            'combined_loss': combined_loss,
            'voxel_accuracy': voxel_accuracy
        }
    )
    print(f"Loaded best model from {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Error loading model, using the one from training: {e}")
    best_model = model

os.makedirs(PREDICTIONS_SAVE_DIR, exist_ok=True)

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

print("\nGenerating predictions for validation samples...")

test_ious = []
test_accuracies = []

num_test_samples = min(10, len(list(val_dataset)))
val_dataset_iter = iter(val_dataset)

for i in range(num_test_samples):
    try:
        images, true_voxels = next(val_dataset_iter)
        predictions = best_model.predict(images, verbose=0)
        
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
        
        print(f"Sample {i}: IoU = {iou:.4f}, Accuracy = {accuracy:.4f}")
        
    except StopIteration:
        break
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        continue

if len(test_ious) > 0:
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Average IoU: {np.mean(test_ious):.4f} ± {np.std(test_ious):.4f}")
    print(f"Average Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
    print(f"Min IoU: {np.min(test_ious):.4f}, Max IoU: {np.max(test_ious):.4f}")
    print("="*50)
    
    with open(os.path.join(PREDICTIONS_SAVE_DIR, 'test_results.txt'), 'w') as f:
        f.write("TEST RESULTS SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Average IoU: {np.mean(test_ious):.4f
