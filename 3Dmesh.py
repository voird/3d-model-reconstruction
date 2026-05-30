import numpy as np
import trimesh
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import glob

def voxel_to_mesh(voxel_data, threshold=0.5):
    if voxel_data.ndim == 4:
        voxel_data = voxel_data[..., 0]
    
    binary_voxel = (voxel_data > threshold).astype(np.uint8)
    
    if np.sum(binary_voxel) == 0:
        print("Предупреждение: воксельная модель пуста")
        return None
    
    try:
        verts, faces, normals, values = marching_cubes(binary_voxel, level=0.5)
    except Exception as e:
        print(f"Ошибка marching_cubes: {e}")
        return None
    
    if len(verts) == 0 or len(faces) == 0:
        print("Предупреждение: не удалось извлечь поверхность")
        return None
    
    if verts.max() > verts.min():
        verts = (verts - verts.min()) / (verts.max() - verts.min())
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    if len(mesh.vertices) > 0:
        try:
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()
        except Exception as e:
            print(f"  Ошибка очистки mesh: {e}")
    
    return mesh

def visualize_mesh(mesh, title="3D Mesh", save_path=None):
    if mesh is None or len(mesh.vertices) == 0:
        print("Невозможно визуализировать: mesh пуст")
        return
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    polygons = []
    for face in faces:
        try:
            polygons.append([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
        except IndexError:
            continue
    
    if len(polygons) == 0:
        print("Нет валидных полигонов для визуализации")
        return
    
    mesh_collection = Poly3DCollection(polygons, alpha=0.7, facecolor='cyan', edgecolor='navy', linewidths=0.1)
    ax.add_collection3d(mesh_collection)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def convert_npy_to_mesh_files(npy_path, output_dir, formats=['obj', 'ply', 'stl']):
    try:
        voxel_data = np.load(npy_path)
        print(f"Загружен файл: {npy_path}")
        print(f"Форма данных: {voxel_data.shape}")
        print(f"Диапазон значений: [{voxel_data.min():.3f}, {voxel_data.max():.3f}]")
        
        if voxel_data.max() == 0:
            print("Предупреждение: все воксели равны нулю")
            return None, []
            
    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        return None, []
    
    print("Конвертация вокселей в mesh...")
    mesh = voxel_to_mesh(voxel_data, threshold=0.5)
    
    if mesh is None or len(mesh.vertices) == 0:
        print("Ошибка: не удалось создать mesh")
        return None, []
    
    print(f"Mesh создан: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    
    saved_files = []
    for fmt in formats:
        output_path = os.path.join(output_dir, f"{base_name}.{fmt}")
        
        try:
            if fmt == 'obj':
                mesh.export(output_path, file_type='obj')
            elif fmt == 'ply':
                mesh.export(output_path, file_type='ply')
            elif fmt == 'stl':
                mesh.export(output_path, file_type='stl')
            else:
                print(f"Неподдерживаемый формат: {fmt}")
                continue
            
            saved_files.append(output_path)
            print(f"Сохранен: {output_path}")
        except Exception as e:
            print(f"Ошибка сохранения в формате {fmt}: {e}")
    
    if len(saved_files) > 0:
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        visualize_mesh(mesh, title=base_name, save_path=vis_path)
        print(f"Визуализация сохранена: {vis_path}")
    
    return mesh, saved_files

def batch_convert_npy_to_mesh(input_dir, output_dir, patterns=['predicted_model_*.npy', 'true_model_*.npy']):
    os.makedirs(output_dir, exist_ok=True)
    
    all_meshes = {}
    
    for pattern in patterns:
        npy_files = glob.glob(os.path.join(input_dir, pattern))
        print(f"\nНайдено {len(npy_files)} файлов по шаблону: {pattern}")
        
        for idx, npy_path in enumerate(sorted(npy_files)):
            print(f"\n--- Обработка: {os.path.basename(npy_path)} ({idx+1}/{len(npy_files)}) ---")
            
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            file_output_dir = os.path.join(output_dir, base_name)
            
            mesh, saved_files = convert_npy_to_mesh_files(
                npy_path, 
                file_output_dir,
                formats=['obj', 'ply', 'stl']
            )
            
            if mesh and len(saved_files) > 0:
                all_meshes[base_name] = {
                    'mesh': mesh,
                    'files': saved_files,
                    'vertices': len(mesh.vertices),
                    'faces': len(mesh.faces)
                }
                print(f"  ✓ Успешно обработан: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
            else:
                print(f"  ✗ Не удалось обработать файл")
    
    report_path = os.path.join(output_dir, 'conversion_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ О КОНВЕРТАЦИИ ВОКСЕЛЕЙ В 3D MESH\n")
        f.write("="*60 + "\n\n")
        
        if all_meshes:
            total_vertices = 0
            total_faces = 0
            for name, info in all_meshes.items():
                f.write(f"Файл: {name}\n")
                f.write(f"  Вершин: {info['vertices']}\n")
                f.write(f"  Граней: {info['faces']}\n")
                f.write(f"  Сохраненные файлы:\n")
                for file_path in info['files']:
                    f.write(f"    - {file_path}\n")
                f.write("\n")
                total_vertices += info['vertices']
                total_faces += info['faces']
            
            f.write("="*60 + "\n")
            f.write(f"ИТОГО обработано файлов: {len(all_meshes)}\n")
            f.write(f"Всего вершин: {total_vertices}\n")
            f.write(f"Всего граней: {total_faces}\n")
        else:
            f.write("Не удалось обработать ни одного файла!\n")
            f.write("Возможные причины:\n")
            f.write("  - Воксельные данные пусты\n")
            f.write("  - Ошибки при загрузке файлов\n")
    
    print(f"\n" + "="*60)
    print(f"КОНВЕРТАЦИЯ ЗАВЕРШЕНА!")
    print(f"Обработано файлов: {len(all_meshes)}")
    print(f"Результаты сохранены в: {output_dir}")
    print(f"Отчет сохранен: {report_path}")
    
    return all_meshes

def create_comparison_mesh(predicted_npy, true_npy, output_dir):
    pred_voxel = np.load(predicted_npy)
    true_voxel = np.load(true_npy)
    
    pred_mesh = voxel_to_mesh(pred_voxel, threshold=0.5)
    true_mesh = voxel_to_mesh(true_voxel, threshold=0.5)
    
    if pred_mesh is None or true_mesh is None:
        print("Ошибка создания mesh")
        return
    
    if len(pred_mesh.vertices) == 0 or len(true_mesh.vertices) == 0:
        print("Один из mesh пуст")
        return
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    verts_true = true_mesh.vertices
    faces_true = true_mesh.faces
    
    polygons_true = []
    for face in faces_true:
        try:
            polygons_true.append([verts_true[face[0]], verts_true[face[1]], verts_true[face[2]]])
        except IndexError:
            continue
    
    if len(polygons_true) > 0:
        mesh_true = Poly3DCollection(polygons_true, alpha=0.7, facecolor='green', edgecolor='darkgreen', linewidths=0.1)
        ax1.add_collection3d(mesh_true)
    ax1.set_title(f'Ground Truth\n({len(verts_true)} vertices, {len(faces_true)} faces)')
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1]); ax1.set_zlim([0, 1])
    
    ax2 = fig.add_subplot(132, projection='3d')
    verts_pred = pred_mesh.vertices
    faces_pred = pred_mesh.faces
    
    polygons_pred = []
    for face in faces_pred:
        try:
            polygons_pred.append([verts_pred[face[0]], verts_pred[face[1]], verts_pred[face[2]]])
        except IndexError:
            continue
    
    if len(polygons_pred) > 0:
        mesh_pred = Poly3DCollection(polygons_pred, alpha=0.7, facecolor='blue', edgecolor='darkblue', linewidths=0.1)
        ax2.add_collection3d(mesh_pred)
    ax2.set_title(f'Predicted\n({len(verts_pred)} vertices, {len(faces_pred)} faces)')
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1]); ax2.set_zlim([0, 1])
    
    ax3 = fig.add_subplot(133, projection='3d')
    if len(polygons_true) > 0:
        mesh_true_overlay = Poly3DCollection(polygons_true, alpha=0.5, facecolor='red', edgecolor='darkred', linewidths=0.1)
        ax3.add_collection3d(mesh_true_overlay)
    if len(polygons_pred) > 0:
        mesh_pred_overlay = Poly3DCollection(polygons_pred, alpha=0.5, facecolor='blue', edgecolor='darkblue', linewidths=0.1)
        ax3.add_collection3d(mesh_pred_overlay)
    ax3.set_title('Overlay (Red=True, Blue=Pred)')
    ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1]); ax3.set_zlim([0, 1])
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, 'comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сравнительная визуализация сохранена: {comparison_path}")

def example_single_file():
    npy_file = input("Введите путь к npy файлу: ").strip()
    if not npy_file:
        npy_file = r'D:\kurs2026\prediction\predicted_model_00.npy'
    
    output_dir = os.path.join(os.path.dirname(npy_file), 'mesh_output')
    
    if os.path.exists(npy_file):
        mesh, files = convert_npy_to_mesh_files(
            npy_file, 
            output_dir,
            formats=['obj', 'ply', 'stl']
        )
        if mesh:
            print(f"\nMesh успешно создан и сохранен в {output_dir}")
        return mesh
    else:
        print(f"Файл не найден: {npy_file}")
        return None

def example_batch_conversion():
    input_dir = r'D:\kurs2026\prediction'
    output_dir = r'D:\kurs2026\meshes\batch'
    
    if not os.path.exists(input_dir):
        print(f"Директория {input_dir} не найдена!")
        return None
    
    all_meshes = batch_convert_npy_to_mesh(
        input_dir,
        output_dir,
        patterns=['predicted_model_*.npy', 'true_model_*.npy']
    )
    
    return all_meshes

if __name__ == "__main__":
    print("ВЫБЕРИТЕ РЕЖИМ РАБОТЫ:")
    print("1. Конвертация одного файла")
    print("2. Пакетная конвертация всех файлов из prediction")
    print("3. Создание сравнения предсказанной и истинной модели")
    
    choice = input("Ваш выбор (1/2/3): ").strip()
    
    if choice == '1':
        example_single_file()
        
    elif choice == '2':
        example_batch_conversion()
        
    elif choice == '3':
        pred_path = input("Путь к предсказанной модели: ").strip()
        true_path = input("Путь к истинной модели: ").strip()
        output_dir = input("Директория для сохранения: ").strip()
        
        if not pred_path:
            pred_path = r'D:\kurs2026\prediction\predicted_model_00.npy'
        if not true_path:
            true_path = r'D:\kurs2026\prediction\true_model_00.npy'
        if not output_dir:
            output_dir = r'D:\kurs2026\meshes\comparison'
        
        if os.path.exists(pred_path) and os.path.exists(true_path):
            create_comparison_mesh(pred_path, true_path, output_dir)
        else:
            print("Один из файлов не найден!")
    
    else:
        print("Неверный выбор. Запуск пакетной конвертации...")
        example_batch_conversion()
