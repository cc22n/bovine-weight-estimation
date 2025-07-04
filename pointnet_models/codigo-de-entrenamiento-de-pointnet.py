import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import joblib
import open3d as o3d
import random
from datetime import datetime
from tqdm import tqdm
from google.colab import drive

print("""
===============================================================
AUMENTO DE DATOS Y POINTNET PARA PREDICCIÓN DE
MÉTRICAS BOVINAS A PARTIR DE NUBES DE PUNTOS 3D
===============================================================
""")

# Montar Google Drive
drive.mount('/content/drive')

# Configurar directorios
DRIVE_PATH = '/content/drive/MyDrive'
PLY_FOLDER = f"{DRIVE_PATH}/nubes-de-puntos-filtradas"  # Carpeta con nubes originales
AUGMENTED_FOLDER = f"{DRIVE_PATH}/nubes_aumentadas_{datetime.now().strftime('%Y%m%d_%H%M')}"  # Carpeta para nubes aumentadas
MODELS_FOLDER = f"{DRIVE_PATH}/pointnet_models_{datetime.now().strftime('%Y%m%d_%H%M')}"  # Carpeta para modelos de DL
EXCEL_PATH = f"{DRIVE_PATH}/Measurements.xlsx"  # Ruta al Excel con mediciones
RF_MODELS_FOLDER = f"{DRIVE_PATH}/Random-Forest-optimizado(peso)"  # Carpeta con modelos Random Forest
MULTITRAIT_MODELS_FOLDER = f"{DRIVE_PATH}/Random-Forest-las-de-mas-métricas"  # Carpeta con modelos multi-atributos

# Crear carpetas necesarias
for folder in [AUGMENTED_FOLDER, MODELS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Carpeta creada: {folder}")
# Verificar si existen las rutas importantes
print(f"PLY_FOLDER existe: {os.path.exists(PLY_FOLDER)}")
print(f"EXCEL_PATH existe: {os.path.exists(EXCEL_PATH)}")
print(f"RF_MODELS_FOLDER existe: {os.path.exists(RF_MODELS_FOLDER)}")
print(f"MULTITRAIT_MODELS_FOLDER existe: {os.path.exists(MULTITRAIT_MODELS_FOLDER)}")


# Función para extraer características volumétricas (misma que usaste en tus modelos RF)
def extract_volumetric_features(ply_file):
    """Extrae características volumétricas básicas de nubes de puntos"""
    try:
        # Cargar nube de puntos
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)

        if len(points) == 0:
            print(f"Advertencia: {ply_file} no contiene puntos.")
            return None

        # Calcular bounding box y dimensiones
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords

        # Extraer medidas fundamentales
        length = dimensions[2]  # Longitud (eje Z)
        width = dimensions[0]   # Ancho (eje X)
        height = dimensions[1]  # Altura (eje Y)

        # Aproximar volumen y área superficial
        volume = length * width * height
        surface_area = 2 * (length*width + length*height + width*height)

        # Calcular estadísticas de distribución de puntos
        centroid = np.mean(points, axis=0)
        std_dev = np.std(points, axis=0)

        # Calcular densidad de puntos
        density = len(points) / volume if volume > 0 else 0

        # Calcular distancia promedio al centroide
        distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)

        # Calcular otras proporciones
        aspect_ratio_1 = length / width if width > 0 else 0
        aspect_ratio_2 = length / height if height > 0 else 0
        aspect_ratio_3 = width / height if height > 0 else 0

        # Calcular cuartiles de distancias
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)

        # Crear vector de características
        features = {
            'length': length,
            'width': width,
            'height': height,
            'volume': volume,
            'surface_area': surface_area,
            'point_count': len(points),
            'density': density,
            'std_x': std_dev[0],
            'std_y': std_dev[1],
            'std_z': std_dev[2],
            'avg_distance': avg_distance,
            'max_distance': max_distance,
            'aspect_ratio_1': aspect_ratio_1,
            'aspect_ratio_2': aspect_ratio_2,
            'aspect_ratio_3': aspect_ratio_3,
            'distance_q1': q1,
            'distance_q3': q3
        }

        # Añadir características avanzadas de PCA (consistente con tu código multi-atributos)
        try:
            # Calcular momentos de segundo orden (momento de inercia)
            centered_points = points - centroid
            covariance = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)

            # Calcular principales ejes de inercia
            principal_axis_1 = eigenvectors[:, 2]  # Eje mayor
            principal_axis_2 = eigenvectors[:, 1]  # Eje intermedio
            principal_axis_3 = eigenvectors[:, 0]  # Eje menor

            # Calcular proyecciones en los ejes principales
            proj_1 = np.dot(centered_points, principal_axis_1)
            proj_2 = np.dot(centered_points, principal_axis_2)
            proj_3 = np.dot(centered_points, principal_axis_3)

            # Calcular longitudes a lo largo de los ejes principales
            principal_length_1 = np.max(proj_1) - np.min(proj_1)
            principal_length_2 = np.max(proj_2) - np.min(proj_2)
            principal_length_3 = np.max(proj_3) - np.min(proj_3)

            # Añadir a características
            features.update({
                'eigenvalue_1': eigenvalues[2],
                'eigenvalue_2': eigenvalues[1],
                'eigenvalue_3': eigenvalues[0],
                'principal_length_1': principal_length_1,
                'principal_length_2': principal_length_2,
                'principal_length_3': principal_length_3,
                'sphericity': eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 0 else 0,
                'elongation': 1 - (eigenvalues[1] / eigenvalues[2]) if eigenvalues[2] > 0 else 0,
                'flatness': 1 - (eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else 0
            })
        except Exception as e:
            print(f"Error calculando características PCA: {e}")

        return features, points
    except Exception as e:
        print(f"Error procesando {ply_file}: {e}")
        return None, None

# Función para extraer número de vaca
def extract_number(filename):
    """Extrae el número de identificación de la vaca del nombre del archivo"""
    match = re.search(r'procesada_(\d+)_AutoAligned', filename)
    if match:
        return int(match.group(1))
    return None

# Función para generar variaciones sutiles de una nube de puntos
def generate_subtle_variation(ply_path, output_path, variation_id):
    """Genera una variación sutil de una nube de puntos y la guarda"""
    try:
        # Extraer características y puntos de la nube original
        features, points = extract_volumetric_features(ply_path)

        if features is None or points is None:
            print(f"Error: No se pudieron extraer características de {ply_path}")
            return None

        # Obtener el centroide original
        centroid = np.mean(points, axis=0)

        # Determinar tipo de modificación basada en el ID de variación
        mod_type = variation_id % 5

        # Crear información sobre la modificación
        mod_info = {
            'tipo': None,
            'parametros': {}
        }

        # Aplicar diferentes tipos de modificaciones sutiles
        if mod_type == 0:
            # Escalado sutil casi uniforme
            base_factor = 0.97 + random.random() * 0.06  # Entre 0.97 y 1.03
            # Pequeñas variaciones por eje
            factor_x = base_factor * (0.99 + random.random() * 0.02)  # ±1%
            factor_y = base_factor * (0.99 + random.random() * 0.02)
            factor_z = base_factor * (0.99 + random.random() * 0.02)

            # Aplicar transformación
            points_mod = np.copy(points)
            points_mod[:, 0] = centroid[0] + (points[:, 0] - centroid[0]) * factor_x
            points_mod[:, 1] = centroid[1] + (points[:, 1] - centroid[1]) * factor_y
            points_mod[:, 2] = centroid[2] + (points[:, 2] - centroid[2]) * factor_z

            mod_info['tipo'] = 'escalado_sutil'
            mod_info['parametros'] = {
                'factor_x': factor_x,
                'factor_y': factor_y,
                'factor_z': factor_z
            }

        elif mod_type == 1:
            # Compresión leve en una dimensión
            dim_to_compress = random.choice([0, 1, 2])
            compression_factor = 0.96 + random.random() * 0.08  # Entre 0.96 y 1.04

            points_mod = np.copy(points)

            if dim_to_compress == 0:
                points_mod[:, 0] = centroid[0] + (points[:, 0] - centroid[0]) * compression_factor
                dim_name = 'X'
            elif dim_to_compress == 1:
                points_mod[:, 1] = centroid[1] + (points[:, 1] - centroid[1]) * compression_factor
                dim_name = 'Y'
            else:
                points_mod[:, 2] = centroid[2] + (points[:, 2] - centroid[2]) * compression_factor
                dim_name = 'Z'

            mod_info['tipo'] = 'compresion_leve'
            mod_info['parametros'] = {
                'dimension': dim_name,
                'factor': compression_factor
            }

        elif mod_type == 2:
            # Rotación muy sutil
            angle = np.radians(random.uniform(-3, 3))  # -3 a 3 grados
            axis = random.choice(['x', 'y', 'z'])

            # Crear matriz de rotación para el eje seleccionado
            if axis == 'x':
                rot_matrix = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]
                ])
            elif axis == 'y':
                rot_matrix = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
            else:
                rot_matrix = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])

            # Aplicar rotación: puntos centrados, rotados y descentrados
            centered_points = points - centroid
            rotated_points = np.dot(centered_points, rot_matrix.T)
            points_mod = rotated_points + centroid

            mod_info['tipo'] = 'rotacion_sutil'
            mod_info['parametros'] = {
                'eje': axis,
                'angulo_grados': np.degrees(angle)
            }

        elif mod_type == 3:
            # Ruido muy sutil
            noise_level = 0.003 + random.random() * 0.007  # Entre 0.3% y 1% de ruido

            # Calcular magnitudes máximas por dimensión
            range_x = np.max(points[:, 0]) - np.min(points[:, 0])
            range_y = np.max(points[:, 1]) - np.min(points[:, 1])
            range_z = np.max(points[:, 2]) - np.min(points[:, 2])

            # Generar ruido proporcional al rango de cada dimensión
            noise_x = noise_level * range_x * (np.random.random(len(points)) - 0.5)
            noise_y = noise_level * range_y * (np.random.random(len(points)) - 0.5)
            noise_z = noise_level * range_z * (np.random.random(len(points)) - 0.5)

            # Aplicar ruido
            points_mod = np.copy(points)
            points_mod[:, 0] += noise_x
            points_mod[:, 1] += noise_y
            points_mod[:, 2] += noise_z

            mod_info['tipo'] = 'ruido_sutil'
            mod_info['parametros'] = {
                'nivel_ruido': noise_level
            }

        else:
            # Deformación no lineal sutil
            # Afecta más a los puntos más alejados del centro
            dist_factor = 0.95 + random.random() * 0.1  # Entre 0.95 y 1.05

            # Calcular distancias al centroide
            distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
            max_dist = np.max(distances)

            # Normalizar distancias y crear factores de escala no lineales
            norm_distances = distances / max_dist
            scale_factors = 1.0 + (norm_distances * 0.5) * (dist_factor - 1.0)

            # Aplicar factores de escala
            points_mod = np.copy(points)
            for i in range(len(points)):
                points_mod[i] = centroid + (points[i] - centroid) * scale_factors[i]

            mod_info['tipo'] = 'deformacion_no_lineal'
            mod_info['parametros'] = {
                'factor_distorsion': dist_factor
            }

        # Crear nube de puntos modificada
        pcd_mod = o3d.geometry.PointCloud()
        pcd_mod.points = o3d.utility.Vector3dVector(points_mod)

        # Guardar en nuevo archivo PLY
        o3d.io.write_point_cloud(output_path, pcd_mod)

        return mod_info

    except Exception as e:
        print(f"Error generando variación: {str(e)}")
        return None

# Función load_ml_models mejorada con diagnóstico detallado
def load_ml_models():
    """Carga modelos de Random Forest y multi-atributos para predicción con diagnóstico detallado"""
    print("\nCargando modelos de ML existentes...")

    models = {}

    # Intentar cargar modelo de peso optimizado
    try:
        rf_path = os.path.join(RF_MODELS_FOLDER, 'random_forest_optimizado.pkl')
        if not os.path.exists(rf_path):
            rf_path = os.path.join(RF_MODELS_FOLDER, 'random_forest_final.pkl')
        if not os.path.exists(rf_path):
            rf_path = os.path.join(RF_MODELS_FOLDER, 'random_forest_model.pkl')

        if os.path.exists(rf_path):
            print(f"Cargando modelo de peso desde: {rf_path}")
            models['peso'] = joblib.load(rf_path)
            print("✓ Modelo de peso cargado correctamente")

            # Cargar también el scaler si existe
            scaler_path = os.path.join(RF_MODELS_FOLDER, 'feature_scaler.pkl')
            if os.path.exists(scaler_path):
                print(f"Cargando scaler de peso desde: {scaler_path}")
                models['peso_scaler'] = joblib.load(scaler_path)
                print("✓ Scaler para modelo de peso cargado")

                # Intentar extraer nombres de características del scaler
                if hasattr(models['peso_scaler'], 'feature_names_in_'):
                    models['peso_features'] = models['peso_scaler'].feature_names_in_.tolist()
                    print(f"✓ Nombres de características extraídos del scaler: {len(models['peso_features'])}")

            # Cargar nombres de características para modelo de peso
            try:
                feature_names_path = os.path.join(RF_MODELS_FOLDER, 'feature_names.csv')
                if os.path.exists(feature_names_path):
                    print(f"Cargando nombres de características desde: {feature_names_path}")
                    feature_names_df = pd.read_csv(feature_names_path)
                    models['peso_features'] = feature_names_df['feature'].tolist()
                    print(f"✓ Nombres de características cargados de CSV: {len(models['peso_features'])}")
                else:
                    feature_names_path = os.path.join(RF_MODELS_FOLDER, 'caracteristicas_seleccionadas.csv')
                    if os.path.exists(feature_names_path):
                        print(f"Cargando nombres de características desde: {feature_names_path}")
                        feature_names_df = pd.read_csv(feature_names_path)
                        models['peso_features'] = feature_names_df['feature'].tolist()
                        print(f"✓ Nombres de características cargados de CSV alternativo: {len(models['peso_features'])}")

                # Mostrar las primeras 5 características para diagnóstico
                if 'peso_features' in models:
                    print("\nPrimeras 5 características esperadas por modelo de peso:")
                    for i, feature in enumerate(models['peso_features'][:5]):
                        print(f"  {i+1}. '{feature}'")
                else:
                    print("⚠️ No se pudieron cargar nombres de características para el modelo de peso")

            except Exception as e:
                print(f"Advertencia: No se pudieron cargar nombres de características para peso: {e}")

                # Si no hay nombres explícitos, tratar de inferirlos del modelo
                if 'peso' in models and hasattr(models['peso'], 'feature_importances_'):
                    n_features = len(models['peso'].feature_importances_)
                    print(f"⚠️ Inferido del modelo: {n_features} características (sin nombres)")
                    # Crear nombres genéricos
                    models['peso_features'] = [f'feature_{i}' for i in range(n_features)]

    except Exception as e:
        print(f"Error cargando modelo de peso: {e}")

    # Cargar modelos multi-atributos
    try:
        multitrait_path = MULTITRAIT_MODELS_FOLDER
        if os.path.exists(multitrait_path):
            # Intentar cargar scaler general
            scaler_path = os.path.join(multitrait_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                print(f"\nCargando scaler multi-atributos desde: {scaler_path}")
                models['multitrait_scaler'] = joblib.load(scaler_path)
                print("✓ Scaler para modelos multi-atributos cargado")

                # Mostrar información sobre el scaler
                if hasattr(models['multitrait_scaler'], 'feature_names_in_'):
                    feature_names = models['multitrait_scaler'].feature_names_in_
                    print(f"✓ Scaler espera {len(feature_names)} características")
                    print("\nPrimeras 5 características esperadas por scaler multi-atributos:")
                    for i, feature in enumerate(feature_names[:5]):
                        print(f"  {i+1}. '{feature}'")

            # Buscar modelos individuales
            print("\nBuscando modelos individuales en: {multitrait_path}")
            model_count = 0
            for file in os.listdir(multitrait_path):
                if file.startswith('modelo_') and file.endswith('.pkl'):
                    trait_name = file.replace('modelo_', '').replace('.pkl', '')
                    model_path = os.path.join(multitrait_path, file)

                    # Cargar el modelo
                    models[trait_name] = joblib.load(model_path)
                    model_count += 1

                    # Buscar selector de características
                    selector_path = os.path.join(multitrait_path, f'selector_{trait_name}.pkl')
                    if os.path.exists(selector_path):
                        models[f'{trait_name}_selector'] = joblib.load(selector_path)

            print(f"✓ Cargados {model_count} modelos individuales")
    except Exception as e:
        print(f"Error cargando modelos multi-atributos: {e}")

    print(f"Total de modelos cargados: {len(models)} componentes")
    return models

# Diagnóstico específico para el modelo de peso
def diagnosticar_modelo_peso(rf_models_folder):
    """Analiza el modelo de peso para entender qué características espera"""
    print("\n=== DIAGNÓSTICO DEL MODELO DE PESO ===")

    import os
    import joblib
    import numpy as np
    import pandas as pd

    # Buscar todos los archivos en la carpeta del modelo de peso
    try:
        print(f"Analizando carpeta: {rf_models_folder}")
        files = os.listdir(rf_models_folder)
        print(f"Archivos encontrados: {len(files)}")
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"Error listando archivos: {e}")
        return

    # Buscar el modelo principal
    model_path = None
    for name in ['random_forest_optimizado.pkl', 'random_forest_final.pkl', 'random_forest_model.pkl']:
        if name in files:
            model_path = os.path.join(rf_models_folder, name)
            break

    if not model_path:
        print("No se encontró ningún modelo de Random Forest")
        return

    # Cargar el modelo
    try:
        print(f"Cargando modelo: {model_path}")
        model = joblib.load(model_path)
        print(f"Tipo de modelo: {type(model).__name__}")

        # Intentar obtener información sobre características
        if hasattr(model, 'feature_importances_'):
            print(f"Número de características: {len(model.feature_importances_)}")
    except Exception as e:
        print(f"Error cargando modelo: {e}")

    # Buscar scaler
    scaler_path = os.path.join(rf_models_folder, 'feature_scaler.pkl')
    if os.path.exists(scaler_path):
        try:
            print(f"\nCargando scaler: {scaler_path}")
            scaler = joblib.load(scaler_path)
            print(f"Tipo de scaler: {type(scaler).__name__}")

            if hasattr(scaler, 'feature_names_in_'):
                print(f"Características esperadas por el scaler ({len(scaler.feature_names_in_)}): ")
                for i, name in enumerate(scaler.feature_names_in_):
                    print(f"  {i+1}. '{name}'")
            else:
                print("El scaler no tiene nombres de características guardados")
        except Exception as e:
            print(f"Error cargando scaler: {e}")
    else:
        print("No se encontró un scaler")

    # Buscar nombres de características
    for name in ['feature_names.csv', 'caracteristicas_seleccionadas.csv']:
        features_path = os.path.join(rf_models_folder, name)
        if os.path.exists(features_path):
            try:
                print(f"\nCargando nombres de características: {features_path}")
                features_df = pd.read_csv(features_path)
                print(f"Columnas del CSV: {features_df.columns.tolist()}")

                if 'feature' in features_df.columns:
                    features = features_df['feature'].tolist()
                    print(f"Características utilizadas ({len(features)}):")
                    for i, feature in enumerate(features):
                        print(f"  {i+1}. '{feature}'")
            except Exception as e:
                print(f"Error cargando nombres de características: {e}")
            break


# Corrección de la función predict_metrics_with_ml_models
# Elimina el código duplicado/mal insertado
def predict_metrics_with_ml_models(features, ml_models):
    """Predice métricas usando modelos de ML con énfasis en la predicción correcta del peso"""
    predictions = {}

    # PRIMERO: Predecir el peso usando el modelo específico de peso
    peso_pred = None
    if 'peso' in ml_models:
        try:
            print("\n--- DIAGNÓSTICO DE PREDICCIÓN DE PESO ---")

            # Las 3 características que necesita el modelo de peso según el código original
            REQUIRED_FEATURES = ['width', 'point_count', 'max_distance']

            # Verificar si estas características están disponibles
            missing = [f for f in REQUIRED_FEATURES if f not in features]
            if missing:
                print(f"Advertencia: Faltan características requeridas: {', '.join(missing)}")

                # Verificar si podemos extraer o calcular las que faltan
                if 'width' in missing and 'dimensions' in features:
                    print("Intentando extraer 'width' de 'dimensions'...")
                    features['width'] = features['dimensions'][0]

                # Verificar si todavía faltan características
                missing = [f for f in REQUIRED_FEATURES if f not in features]

            print(f"Características disponibles: {list(features.keys())}")
            print(f"Características requeridas: {REQUIRED_FEATURES}")

            if not missing:
                # Crear vector con las características exactas que necesita el modelo
                feature_values = [features[f] for f in REQUIRED_FEATURES]
                print(f"Valores de características: {feature_values}")

                X = np.array([feature_values])

                # Verificar el scaler
                if 'peso_scaler' in ml_models:
                    print("Aplicando scaler para modelo de peso...")
                    print(f"Forma del vector antes de escalar: {X.shape}")
                    X_scaled = ml_models['peso_scaler'].transform(X)
                    print(f"Forma del vector después de escalar: {X_scaled.shape}")
                else:
                    print("No se encontró scaler para el modelo de peso, usando valores sin escalar")
                    X_scaled = X

                # Verificar el modelo
                print(f"Tipo de modelo de peso: {type(ml_models['peso']).__name__}")

                # Realizar predicción
                print("Prediciendo peso...")

                # En caso de que el modelo espere un dataframe
                try:
                    peso_pred = ml_models['peso'].predict(X_scaled)[0]
                    print(f"Predicción exitosa: {peso_pred:.2f} kg")

                    # Añadir pequeña variación aleatoria (±3%)
                    peso_pred *= (1 + random.uniform(-0.03, 0.03))

                    # Guardar predicción con ambos nombres para compatibilidad
                    predictions['live weithg'] = peso_pred
                    predictions['peso_real_kg'] = peso_pred

                    # También guardar valor en feature para que esté disponible para otras predicciones
                    features['live weithg'] = peso_pred

                except Exception as e:
                    print(f"Error en predict del modelo: {e}")
                    import traceback
                    traceback.print_exc()

                    # Intento alternativo con DataFrame
                    try:
                        print("Intentando con DataFrame...")
                        import pandas as pd
                        X_df = pd.DataFrame([dict(zip(REQUIRED_FEATURES, feature_values))])
                        peso_pred = ml_models['peso'].predict(X_df)[0]
                        print(f"Predicción exitosa con DataFrame: {peso_pred:.2f} kg")

                        # Añadir pequeña variación aleatoria (±3%)
                        peso_pred *= (1 + random.uniform(-0.03, 0.03))

                        # Guardar predicción
                        predictions['live weithg'] = peso_pred
                        predictions['peso_real_kg'] = peso_pred

                        # También guardar en features
                        features['live weithg'] = peso_pred
                    except Exception as e2:
                        print(f"Error en segundo intento: {e2}")
            else:
                print(f"No se puede predecir peso porque faltan características: {missing}")
        except Exception as e:
            print(f"Error general prediciendo peso: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No se encontró modelo de peso ('peso') en ml_models")
        print(f"Claves disponibles en ml_models: {list(ml_models.keys())}")

    # Si no se pudo predecir peso, usar peso por defecto basado en estadísticas
    if 'live weithg' not in predictions:
        default_weight = 415.0  # Valor del primer animal en el Excel (más realista que 500)
        print(f"Usando peso por defecto: {default_weight} kg")
        predictions['live weithg'] = default_weight
        features['live weithg'] = default_weight  # También añadimos a features

    # SEGUNDO: Predecir otros atributos usando el peso predicho
    if 'multitrait_scaler' in ml_models:
        try:
            print("\n--- PREDICCIÓN DE ATRIBUTOS MÚLTIPLES ---")

            # Obtener características esperadas por el scaler
            expected_features = None
            if hasattr(ml_models['multitrait_scaler'], 'feature_names_in_'):
                expected_features = ml_models['multitrait_scaler'].feature_names_in_.tolist()
                print(f"Scaler espera {len(expected_features)} características")
                print(f"Primeras 5: {expected_features[:5]}")
            else:
                print("No se pudieron obtener nombres de características del scaler")

            # Crear vector con todas las características necesarias
            if expected_features:
                # Asegurar que 'live weithg' está presente
                if 'live weithg' in expected_features and 'live weithg' in predictions:
                    print(f"Usando predicción de peso ({predictions['live weithg']:.2f} kg) para 'live weithg'")

                # Crear vector en orden correcto
                X = np.zeros((1, len(expected_features)))
                for i, feat_name in enumerate(expected_features):
                    if feat_name in features:
                        X[0, i] = features[feat_name]
                    elif feat_name == 'live weithg' and 'live weithg' in predictions:
                        X[0, i] = predictions['live weithg']
                    else:
                        X[0, i] = 0.0
                        print(f"Usando 0 para característica '{feat_name}'")
            else:
                # Si no tenemos nombres, construir vector menos optimizado
                print("Construyendo vector genérico...")
                # Asegurarnos que 'live weithg' esté primero
                feature_keys = ['live weithg'] + [k for k in features.keys() if k != 'live weithg']
                X = np.array([[features.get(k, 0.0) if k != 'live weithg' else predictions.get('live weithg', 0.0)
                              for k in feature_keys]])

            # Escalar características
            print(f"Forma del vector antes de escalar: {X.shape}")
            X_scaled = ml_models['multitrait_scaler'].transform(X)
            print(f"Forma del vector después de escalar: {X_scaled.shape}")

            # Mapeo entre nombres de modelos y nombres originales del Excel
            trait_mapping = {
                'altura_cruz': 'withers height',
                'altura_cadera': 'hip height',
                'circunferencia_torax': 'heart girth',
                'profundidad_pecho': 'chest depth',
                'ancho_pecho': 'chest width',
                'ancho_ilion': 'ilium width',
                'ancho_articulacion_cadera': 'hip joint width',
                'longitud_oblicua': 'oblique body length',
                'longitud_cadera': 'hip length'
            }

            # Predecir cada atributo
            for trait, output_name in trait_mapping.items():
                if trait in ml_models and f'{trait}_selector' in ml_models:
                    try:
                        # Aplicar selector de características
                        X_selected = ml_models[f'{trait}_selector'].transform(X_scaled)

                        # Predecir
                        pred = ml_models[trait].predict(X_selected)[0]

                        # Añadir pequeña variación aleatoria (±3%)
                        pred *= (1 + random.uniform(-0.03, 0.03))

                        # Guardar con nombre original
                        predictions[output_name] = pred
                        print(f"Predicción para {output_name}: {pred:.2f}")
                    except Exception as e:
                        print(f"Error prediciendo {output_name}: {e}")
        except Exception as e:
            print(f"Error general en predicciones multiatributo: {e}")
            import traceback
            traceback.print_exc()

    # Verificar predicciones finales
    print("\n--- RESUMEN DE PREDICCIONES ---")
    for k, v in predictions.items():
        print(f"{k}: {v:.2f}")

    return predictions

# Función main corregida para verificar correctamente los resultados de prepare_datasets
# Corrección del main para que llame correctamente a las funciones
def main():
    """Función principal que ejecuta todo el proceso con verificación de resultados mejorada"""
    print("Iniciando procesamiento...")

    # 1. Cargar datos y archivos
    metrics_df, metric_columns, column_mapping, ply_files = load_data_and_files()

    # CAMBIO CLAVE: Usar directamente los nombres originales del Excel como métricas objetivo
    # En lugar de los nombres traducidos al español
    target_metrics = metric_columns

    print(f"Métricas objetivo (nombres originales del Excel): {target_metrics}")

    # 2. Cargar modelos ML existentes
    ml_models = load_ml_models()

    # 3. Procesar nubes de puntos y generar variaciones
    all_samples = process_and_augment_point_clouds(metrics_df, ply_files, ml_models)

    # Verificar si tenemos suficientes muestras
    if not all_samples or len(all_samples) == 0:
        print("\nERROR: No se pudieron procesar muestras. Abortando.")
        return

    # 4. Preparar conjuntos de datos para entrenamiento
    result = prepare_datasets(all_samples, target_metrics)

    # CORREGIDO: Verificación adecuada de valores None en el resultado
    if result is None:
        print("\nERROR: prepare_datasets devolvió None. Abortando el entrenamiento.")
        return

    # Desempaquetar resultados
    try:
        X_train, y_train, X_val, y_val, feature_names, returned_metrics, _, scaler_y = result

        # Verificar componentes individuales
        if X_train is None or y_train is None or X_val is None or y_val is None:
            print("\nERROR: Alguno de los datasets de entrenamiento/validación es None. Abortando.")
            return

        # Verificar dimensiones
        if len(X_train) == 0 or len(y_train) == 0:
            print("\nERROR: Los conjuntos de entrenamiento están vacíos. Abortando.")
            return
    except Exception as e:
        print(f"\nERROR al desempaquetar los resultados de prepare_datasets: {e}")
        return

    # 5. Entrenar modelo PointNet
    print("\nComenzando entrenamiento del modelo PointNet...")
    model, train_losses, val_losses = train_pointnet_model(X_train, y_train, X_val, y_val, returned_metrics)

    # 6. Evaluar modelo
    print("\nEvaluando modelo entrenado...")
    evaluation_results, results_df = evaluate_model(model, X_val, y_val, scaler_y, returned_metrics)

    # 7. Crear función de predicción
    print("\nCreando función de predicción...")
    prediction_script_path = create_prediction_function()

    # 8. Comparar con modelos anteriores
    print("\nComparando con modelos anteriores...")
    comparison_df = compare_with_ml_models(evaluation_results, returned_metrics)

    # 9. Crear README con instrucciones
    # Aquí podemos traducir los nombres para que el README sea más legible
    readable_names = {
        'live weithg': 'Peso vivo (kg)',
        'withers height': 'Altura a la cruz (cm)',
        'hip height': 'Altura a la cadera (cm)',
        'chest depth': 'Profundidad de pecho (cm)',
        'chest width': 'Ancho de pecho (cm)',
        'ilium width': 'Ancho de ilion (cm)',
        'hip joint width': 'Ancho de articulación de cadera (cm)',
        'oblique body length': 'Longitud oblicua del cuerpo (cm)',
        'hip length': 'Longitud de cadera (cm)',
        'heart girth': 'Perímetro torácico (cm)'
    }

    # Convertir a nombres legibles para el README
    readable_metrics = [readable_names.get(metric, metric) for metric in returned_metrics]

    readme_content = f"""# Modelo PointNet para Predicción de Métricas Bovinas

Este directorio contiene modelos de PointNet entrenados para predecir diversas métricas bovinas a partir de nubes de puntos 3D.

## Resumen del Proyecto

- Se utilizaron **{{len(ply_files)}}** nubes de puntos originales
- Se generaron variaciones sutiles para aumentar el conjunto de datos
- Se entrenó una red neuronal PointNet para predecir múltiples métricas utilizando directamente las nubes de puntos 3D

## Métricas Predichas

{{', '.join(readable_metrics)}}

## Ventajas de PointNet

- Trabaja directamente con nubes de puntos sin necesidad de voxelización
- Es invariante a permutaciones (el orden de los puntos no importa)
- Requiere menos preprocesamiento y es más eficiente en memoria
- Captura mejor la estructura geométrica de las nubes de puntos 3D

## Uso del Modelo

Para utilizar el modelo con nuevas nubes de puntos:

```python
# Ejemplo de uso
from predict_metrics_pointnet import predict_metrics

# Predecir métricas
predictions = predict_metrics('ruta/a/nube_puntos.ply')

# Mostrar predicciones
for metric, value in predictions.items():
    print(f"{{metric}}: {{value:.2f}}")
```"""

    # Guardar README
    with open(os.path.join(MODELS_FOLDER, 'README.md'), 'w') as f:
        f.write(readme_content)

    print("\n¡Proceso completado con éxito!")
    print(f"Modelos guardados en: {MODELS_FOLDER}")
    print(f"Script de predicción: {prediction_script_path}")


# Versión corregida de la función load_data_and_files()
def load_data_and_files():
    """Carga datos del Excel y lista las nubes de puntos disponibles manteniendo los nombres originales"""
    print("\nCargando datos y archivos...")

    # Cargar Excel con métricas reales
    metrics_df = pd.read_excel(EXCEL_PATH)
    print(f"Excel cargado: {len(metrics_df)} registros")

    # IMPORTANTE: Usar los nombres exactos del Excel
    metric_columns = [
        'live weithg',        # Mantener el error ortográfico original
        'withers height',
        'hip height',
        'chest depth',
        'chest width',
        'ilium width',
        'hip joint width',
        'oblique body length',
        'hip length',
        'heart girth'
    ]

    # Verificar qué columnas existen en el Excel
    existing_columns = [col for col in metric_columns if col in metrics_df.columns]
    missing_columns = [col for col in metric_columns if col not in metrics_df.columns]

    if missing_columns:
        print("ADVERTENCIA: Las siguientes columnas no se encontraron en el Excel:")
        for col in missing_columns:
            print(f"  - '{col}'")

    print(f"Columnas de métricas encontradas: {len(existing_columns)}")

    # Para referencia interna (no para renombrar el DataFrame)
    column_mapping = {
                'altura_cruz': 'withers height',       # IMPORTANTE: Guardar con nombre original
                'altura_cadera': 'hip height',
                'profundidad_pecho': 'chest depth',
                'ancho_pecho': 'chest width',
                'ancho_ilion': 'ilium width',
                'ancho_articulacion_cadera': 'hip joint width',
                'longitud_oblicua': 'oblique body length',
                'longitud_cadera': 'hip length',
                'circunferencia_torax': 'heart girth'
    }

    # Listar archivos PLY disponibles
    ply_files = [f for f in os.listdir(PLY_FOLDER) if f.endswith('.ply')]
    print(f"Archivos PLY disponibles: {len(ply_files)}")

    # Devolver el DataFrame sin renombrar las columnas
    return metrics_df, existing_columns, column_mapping, ply_files

# Procesar nubes de puntos y generar variaciones
def process_and_augment_point_clouds(metrics_df, ply_files, ml_models):
    """Procesa nubes de puntos, extrae características y genera variaciones"""
    print("\n=== PROCESANDO NUBES DE PUNTOS Y GENERANDO VARIACIONES ===")

    # Dicionario para almacenar todas las muestras (originales y aumentadas)
    all_samples = []

    # Procesar archivos PLY originales
    for filename in tqdm(ply_files, desc="Procesando nubes originales"):
        file_path = os.path.join(PLY_FOLDER, filename)

        # Extraer ID de vaca
        cow_id = extract_number(filename)

        # Extraer características
        features, points = extract_volumetric_features(file_path)

        if features is None or points is None:
            print(f"  ❌ Error procesando {filename}")
            continue

        # Obtener métricas reales si existen
        real_metrics = None
        if cow_id is not None and cow_id - 1 < len(metrics_df):
            real_metrics = {}
            for col in metrics_df.columns:
                real_metrics[col] = metrics_df.iloc[cow_id - 1][col]

        # Registrar muestra original
        sample = {
            'tipo': 'original',
            'archivo': filename,
            'ruta': file_path,
            'id_vaca': cow_id,
            'caracteristicas': features,
            'puntos': points,  # Guardamos los puntos para PointNet
            'metricas_reales': real_metrics
        }

        all_samples.append(sample)

        # Generar variaciones (entre 4 y 8 por cada original)
        n_variations = random.randint(4, 8)

        for i in range(n_variations):
            # Nombre para archivo de variación
            var_name = f"var_{cow_id}_{i+1}.ply" if cow_id else f"var_{filename.split('.')[0]}_{i+1}.ply"
            var_path = os.path.join(AUGMENTED_FOLDER, var_name)

            # Generar variación
            mod_info = generate_subtle_variation(file_path, var_path, i)

            if mod_info:
                # Extraer características de la variación
                var_features, var_points = extract_volumetric_features(var_path)

                if var_features and var_points is not None:
                    # Predecir métricas para la variación usando modelos ML
                    predicted_metrics = predict_metrics_with_ml_models(var_features, ml_models)

                    # Registrar variación
                    var_sample = {
                        'tipo': f"variacion_{mod_info['tipo']}",
                        'archivo': var_name,
                        'ruta': var_path,
                        'id_vaca': cow_id,
                        'caracteristicas': var_features,
                        'puntos': var_points,  # Guardamos los puntos para PointNet
                        'metricas_reales': None,
                        'metricas_predichas': predicted_metrics,
                        'parametros_modificacion': mod_info['parametros']
                    }

                    all_samples.append(var_sample)

    print(f"Total de muestras procesadas: {len(all_samples)}")
    print(f"  - Originales: {sum(1 for s in all_samples if s['tipo'] == 'original')}")
    print(f"  - Variaciones: {sum(1 for s in all_samples if s['tipo'] != 'original')}")

    return all_samples

# Función prepare_datasets corregida para manejar casos con datos insuficientes
def prepare_datasets(all_samples, target_metrics):
    """Prepara conjuntos de datos para entrenamiento de PointNet con manejo de errores mejorado"""
    print("\n=== PREPARANDO CONJUNTOS DE DATOS PARA ENTRENAMIENTO ===")

    # Extraer puntos, características y métricas para todas las muestras
    point_clouds = []  # Nubes de puntos para PointNet
    X_features = []    # Características volumétricas para referencia
    y_metrics = []     # Métricas objetivo
    sample_types = []  # Para identificar originales vs. variaciones

    feature_names = None

    # Contar las muestras procesadas para diagnóstico
    originals_processed = 0
    originals_complete = 0
    augmented_processed = 0
    augmented_complete = 0

    # Definir número de puntos fijo para todas las nubes (necesario para PointNet)
    NUM_POINTS = 1024

    for sample in all_samples:
        # Verificar si tiene puntos
        if 'puntos' not in sample or sample['puntos'] is None or len(sample['puntos']) == 0:
            continue

        # Para muestras originales, usar métricas reales si están completas
        if sample['tipo'] == 'original' and sample['metricas_reales']:
            originals_processed += 1

            # Verificar que tengamos todas las métricas necesarias
            has_all_metrics = True
            metrics_dict = {}

            for metric in target_metrics:
                if metric not in sample['metricas_reales'] or pd.isna(sample['metricas_reales'][metric]):
                    has_all_metrics = False
                    print(f"Advertencia: Muestra original sin métrica '{metric}'")
                    break
                metrics_dict[metric] = sample['metricas_reales'][metric]

            if has_all_metrics:
                originals_complete += 1

                # Guardar nombres de características la primera vez
                if feature_names is None:
                    feature_names = list(sample['caracteristicas'].keys())

                # Preprocesar nube de puntos para PointNet
                points = sample['puntos']

                # Normalizar centroide a origen
                centroid = np.mean(points, axis=0)
                points = points - centroid

                # Normalizar a esfera unitaria
                furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
                points = points / furthest_distance

                # Muestrear un número fijo de puntos (necesario para PointNet)
                if len(points) >= NUM_POINTS:
                    # Muestreo aleatorio sin reemplazo
                    indices = np.random.choice(len(points), NUM_POINTS, replace=False)
                    sampled_points = points[indices]
                else:
                    # Si hay menos puntos que NUM_POINTS, repetir puntos
                    indices = np.random.choice(len(points), NUM_POINTS, replace=True)
                    sampled_points = points[indices]

                # Guardar nube de puntos preprocesada
                point_clouds.append(sampled_points)

                # Extraer características como vector (para referencia)
                features = [sample['caracteristicas'][name] for name in feature_names]
                X_features.append(features)

                # Extraer métricas como vector
                metrics = [metrics_dict[name] for name in target_metrics]
                y_metrics.append(metrics)

                sample_types.append('original')

        # Para variaciones, usar métricas predichas
        elif sample['tipo'] != 'original' and sample['metricas_predichas']:
            augmented_processed += 1

            # Verificar que tengamos todas las métricas necesarias
            has_all_metrics = True
            metrics_dict = {}

            for metric in target_metrics:
                if metric not in sample['metricas_predichas'] or pd.isna(sample['metricas_predichas'][metric]):
                    has_all_metrics = False
                    print(f"Advertencia: Variación sin métrica '{metric}'")
                    break
                metrics_dict[metric] = sample['metricas_predichas'][metric]

            if has_all_metrics:
                augmented_complete += 1

                # Guardar nombres de características la primera vez
                if feature_names is None:
                    feature_names = list(sample['caracteristicas'].keys())

                # Preprocesar nube de puntos para PointNet (igual que arriba)
                points = sample['puntos']

                # Normalizar centroide a origen
                centroid = np.mean(points, axis=0)
                points = points - centroid

                # Normalizar a esfera unitaria
                furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
                points = points / furthest_distance

                # Muestrear un número fijo de puntos
                if len(points) >= NUM_POINTS:
                    indices = np.random.choice(len(points), NUM_POINTS, replace=False)
                    sampled_points = points[indices]
                else:
                    indices = np.random.choice(len(points), NUM_POINTS, replace=True)
                    sampled_points = points[indices]

                # Guardar nube de puntos preprocesada
                point_clouds.append(sampled_points)

                # Extraer características como vector (para referencia)
                features = [sample['caracteristicas'][name] for name in feature_names]
                X_features.append(features)

                # Extraer métricas como vector
                metrics = [metrics_dict[name] for name in target_metrics]
                y_metrics.append(metrics)

                sample_types.append('augmented')

    # Mostrar estadísticas de procesamiento
    print(f"\nEstadísticas de procesamiento de muestras:")
    print(f"  - Originales procesadas: {originals_processed}")
    print(f"  - Originales completas: {originals_complete}")
    print(f"  - Variaciones procesadas: {augmented_processed}")
    print(f"  - Variaciones completas: {augmented_complete}")

    # Verificar si hay suficientes datos
    if len(point_clouds) == 0 or len(y_metrics) == 0:
        print("\n¡ERROR CRÍTICO! No hay suficientes datos para entrenar el modelo.")
        print("Posibles causas:")
        print("  - Ninguna muestra tiene todas las métricas requeridas")
        print("  - La generación de variaciones falló para todas las muestras")
        print("  - Las métricas objetivo no coinciden con las disponibles en los datos")

        print("\nMétricas objetivo que estamos buscando:")
        for metric in target_metrics:
            print(f"  - '{metric}'")

        # Muestrar algunas métricas disponibles en las muestras originales para diagnóstico
        if all_samples and len(all_samples) > 0:
            original_samples = [s for s in all_samples if s['tipo'] == 'original' and s['metricas_reales']]
            if original_samples:
                print("\nMétricas disponibles en la primera muestra original:")
                for key, value in original_samples[0]['metricas_reales'].items():
                    print(f"  - '{key}': {value}")

        # Devolver valores nulos para evitar errores en el resto del código
        return None, None, None, None, None, None, None, None

    # Convertir a arrays de NumPy
    point_clouds = np.array(point_clouds)
    X_features = np.array(X_features)
    y = np.array(y_metrics)
    sample_types = np.array(sample_types)

    # Ahora podemos imprimir dimensiones con seguridad
    print(f"\nDatos preparados: {point_clouds.shape[0]} muestras, cada una con {point_clouds.shape[1]} puntos y {point_clouds.shape[2]} coordenadas")
    print(f"Métricas objetivo: {y.shape[1]} métricas")
    print(f"  - Datos originales: {np.sum(sample_types == 'original')}")
    print(f"  - Datos aumentados: {np.sum(sample_types == 'augmented')}")

    # Separar muestras originales para validación
    is_original = (sample_types == 'original')

    # Verificar si hay suficientes muestras originales para dividir
    if np.sum(is_original) < 2:
        print("\nAdvertencia: No hay suficientes muestras originales para crear conjuntos de validación.")
        print("Se utilizará una división simple sin validación.")

        # División simple
        train_idx = np.arange(len(point_clouds))
        val_idx = np.array([0])  # Usar la primera muestra como validación (solo para forma)
    else:
        # Dividir datos originales en entrenamiento y validación
        orig_indices = np.where(is_original)[0]
        orig_train_idx, orig_val_idx = train_test_split(orig_indices, test_size=0.3, random_state=42)

        # Índices finales para entrenamiento (originales seleccionados + todas las variaciones)
        augmented_indices = np.where(~is_original)[0]
        train_idx = np.concatenate([orig_train_idx, augmented_indices])
        val_idx = orig_val_idx  # Solo originales para validación

    # Dividir datos
    X_train_points = point_clouds[train_idx]
    X_val_points = point_clouds[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    print(f"Conjunto de entrenamiento: {len(X_train_points)} muestras")
    print(f"Conjunto de validación: {len(X_val_points)} muestras")

    # Normalizar etiquetas (métricas objetivo)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # Guardar scaler de métricas
    joblib.dump(scaler_y, os.path.join(MODELS_FOLDER, 'metric_scaler_dl.pkl'))

    # Guardar nombres de métricas
    pd.DataFrame({'metric': target_metrics}).to_csv(os.path.join(MODELS_FOLDER, 'metric_names_dl.csv'), index=False)

    return X_train_points, y_train_scaled, X_val_points, y_val_scaled, feature_names, target_metrics, None, scaler_y

# AQUÍ COMIENZA LA IMPLEMENTACIÓN DE POINTNET
# Definir las capas de transformación T-Net para PointNet
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        # Red de convolución 1D para proceso de características
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # MLP para transformación de características
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        # Activaciones
        self.relu = nn.ReLU()

        # Normalización por lotes
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Inicialización de la última capa como matriz identidad
        self.k = k
        self.register_buffer('identity', torch.eye(k).flatten())

    def forward(self, x):
        batch_size = x.size(0)

        # Aplicar convoluciones 1D
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Max pooling a lo largo de la dimensión de puntos
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # MLP para generar matriz de transformación
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Agregar identidad para inicialización estable
        x = x + self.identity

        # Reshapear a matriz de transformación
        x = x.view(-1, self.k, self.k)

        return x

# Definir arquitectura PointNet principal
class PointNet(nn.Module):
    def __init__(self, output_size, feature_transform=True):
        super(PointNet, self).__init__()
        self.feature_transform = feature_transform

        # Transformación de entrada (3x3 matriz)
        self.input_transform = TNet(k=3)

        # Primera capa de extracción de características
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        # Transformación de características (64x64 matriz) - opcional
        self.feature_transform_net = TNet(k=64)

        # Capas de extracción de características profundas
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # Capas regresoras para métricas bovinas
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, output_size)

        # Activación ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        n_points = x.size(2)

        # Aplicar transformación de entrada
        trans_input = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_input)
        x = x.transpose(2, 1)

        # Primera capa de características
        x = self.relu(self.bn1(self.conv1(x)))

        # Aplicar transformación de características si está habilitada
        if self.feature_transform:
            trans_feature = self.feature_transform_net(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feature)
            x = x.transpose(2, 1)
        else:
            trans_feature = None

        # Capas de extracción de características
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Max pooling (agregación global)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Capas regresoras para métricas bovinas
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x, trans_feature

# Función de pérdida para regularizar matriz de transformación de características
def feature_transform_regularizer(trans):
    """Regulariza la matriz de transformación para que sea lo más cercana a una matriz ortogonal"""
    batch_size = trans.size(0)
    k = trans.size(1)
    identity = torch.eye(k, dtype=trans.dtype, device=trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - identity, dim=(1, 2)))
    return loss

# Clase para la pérdida ponderada por métrica
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, pred, target):
        # Pérdida MSE para cada métrica
        loss = ((pred - target) ** 2).mean(dim=0)
        # Aplicar pesos a cada métrica
        weighted_loss = (loss * self.weights).sum()
        return weighted_loss

# Entrenar modelo de PointNet
def train_pointnet_model(X_train, y_train, X_val, y_val, target_metrics):
    """Entrena el modelo de PointNet con los datos preparados"""
    print("\n=== ENTRENANDO MODELO POINTNET ===")

    # Configurar dispositivo (GPU si está disponible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Convertir a tensores de PyTorch
    X_train_tensor = torch.FloatTensor(X_train).transpose(2, 1).to(device)  # [B, N, 3] -> [B, 3, N]
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).transpose(2, 1).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    # Crear datasets y dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Inicializar modelo
    output_size = y_train.shape[1]

    # Definir pesos para cada métrica
    # Dar más peso a métricas importantes como peso y altura
    metric_weights = torch.ones(output_size)

    for i, metric in enumerate(target_metrics):
        if 'weithg' in metric or 'peso' in metric:
            metric_weights[i] = 2.0  # Mayor peso para el peso
        elif 'height' in metric or 'altura' in metric:
            metric_weights[i] = 1.5  # Peso intermedio para alturas
        # Resto de métricas con peso 1.0

    metric_weights = metric_weights.to(device)

    # Crear modelo PointNet
    model = PointNet(output_size, feature_transform=True).to(device)

    # Definir función de pérdida y optimizador
    criterion = WeightedMSELoss(metric_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Parámetros de entrenamiento
    num_epochs = 250
    patience = 25  # Early stopping
    best_val_loss = float('inf')
    counter = 0

    # Para guardar historial de entrenamiento
    train_losses = []
    val_losses = []
    best_model_state = None

    # Entrenamiento
    print(f"Iniciando entrenamiento por {num_epochs} épocas (con early stopping)...")

    for epoch in range(num_epochs):
        # Modo entrenamiento
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # Forward pass: obtiene predicciones y matriz de transformación
            predictions, trans_feature = model(inputs)

            # Calcular pérdida principal (MSE ponderado)
            loss = criterion(predictions, targets)

            # Añadir pérdida de regularización si hay transformación de características
            if trans_feature is not None:
                loss += 0.001 * feature_transform_regularizer(trans_feature)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # Calcular pérdida promedio de entrenamiento
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluación
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                predictions, _ = model(inputs)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

        # Actualizar learning rate
        scheduler.step()

        # Mostrar progreso
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Época {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping en época {epoch+1}")
                break

    # Cargar mejor modelo
    model.load_state_dict(best_model_state)

    # Guardar modelo
    torch.save({
        'model_state_dict': best_model_state,
        'output_size': output_size,
        'target_metrics': target_metrics
    }, os.path.join(MODELS_FOLDER, 'pointnet_model.pth'))

    # Graficar curva de aprendizaje
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Entrenamiento')
    plt.plot(val_losses, label='Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Curva de Aprendizaje - Modelo PointNet')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(MODELS_FOLDER, 'learning_curve.png'))
    plt.close()

    return model, train_losses, val_losses

# Evaluar el modelo
def evaluate_model(model, X_val, y_val, scaler_y, target_metrics):
    """Evalúa el modelo PointNet en conjunto de validación y guarda resultados"""
    print("\n=== EVALUANDO MODELO POINTNET ===")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_val_tensor = torch.FloatTensor(X_val).transpose(2, 1).to(device)  # [B, N, 3] -> [B, 3, N]

    # Obtener predicciones del modelo
    model.eval()
    with torch.no_grad():
        y_pred_scaled, _ = model(X_val_tensor)
        y_pred_scaled = y_pred_scaled.cpu().numpy()

    # Desnormalizar predicciones y valores reales
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_val_denorm = scaler_y.inverse_transform(y_val)

    # Calcular métricas de evaluación para cada objetivo
    results = {}

    for i, metric_name in enumerate(target_metrics):
        # Valores reales y predicciones para esta métrica
        y_true = y_val_denorm[:, i]
        y_pred_i = y_pred[:, i]

        # Calcular métricas
        mse = mean_squared_error(y_true, y_pred_i)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred_i)
        r2 = r2_score(y_true, y_pred_i)

        # Calcular error porcentual
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_errors = np.abs(y_pred_i - y_true) / y_true * 100
            rel_errors = rel_errors[~np.isnan(rel_errors)]  # Ignorar NaN
            rel_errors = rel_errors[~np.isinf(rel_errors)]  # Ignorar infinitos

        mean_rel_error = np.mean(rel_errors) if len(rel_errors) > 0 else float('nan')
        within_5pct = np.sum(rel_errors <= 5) / len(rel_errors) * 100 if len(rel_errors) > 0 else 0
        within_10pct = np.sum(rel_errors <= 10) / len(rel_errors) * 100 if len(rel_errors) > 0 else 0

        # Guardar resultados
        results[metric_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_rel_error': mean_rel_error,
            'within_5pct': within_5pct,
            'within_10pct': within_10pct
        }

        # Mostrar resultados
        print(f"\nResultados para {metric_name}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Error relativo medio: {mean_rel_error:.2f}%")
        print(f"  Dentro del 5%: {within_5pct:.1f}%")
        print(f"  Dentro del 10%: {within_10pct:.1f}%")

        # Visualizar gráfico de dispersión
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred_i, alpha=0.7)

        # Línea 1:1 (predicción perfecta)
        min_val = min(min(y_true), min(y_pred_i))
        max_val = max(max(y_true), max(y_pred_i))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Regresión lineal para visualizar tendencia
        z = np.polyfit(y_true, y_pred_i, 1)
        p = np.poly1d(z)
        plt.plot(y_true, p(y_true), "g-", alpha=0.7)

        plt.xlabel(f"{metric_name} (Real)")
        plt.ylabel(f"{metric_name} (Predicción)")
        plt.title(f"Predicción vs. Real - {metric_name}\nR² = {r2:.4f}")
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(MODELS_FOLDER, f'predictions_{metric_name}.png'))
        plt.close()

    # Guardar resultados en CSV
    results_df = pd.DataFrame({
        'Métrica': [metric for metric in results.keys()],
        'MAE': [results[metric]['mae'] for metric in results],
        'RMSE': [results[metric]['rmse'] for metric in results],
        'R²': [results[metric]['r2'] for metric in results],
        'Error Rel. (%)': [results[metric]['mean_rel_error'] for metric in results],
        'Dentro 5% (%)': [results[metric]['within_5pct'] for metric in results],
        'Dentro 10% (%)': [results[metric]['within_10pct'] for metric in results]
    })

    results_df.to_csv(os.path.join(MODELS_FOLDER, 'resultados_evaluacion.csv'), index=False)

    # Generar gráfico resumido de R²
    plt.figure(figsize=(12, 6))
    metrics = list(results.keys())
    r2_values = [results[metric]['r2'] for metric in metrics]

    # Ordenar por R²
    sorted_indices = np.argsort(r2_values)[::-1]
    sorted_metrics = [metrics[i] for i in sorted_indices]
    sorted_r2 = [r2_values[i] for i in sorted_indices]

    # Crear gráfico de barras
    bars = plt.bar(sorted_metrics, sorted_r2)

    # Colorear barras según R²
    for i, bar in enumerate(bars):
        if sorted_r2[i] > 0.5:
            bar.set_color('green')
        elif sorted_r2[i] > 0.25:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Métrica')
    plt.ylabel('Coeficiente de Determinación (R²)')
    plt.title('Precisión de Predicción por Métrica - Modelo PointNet')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FOLDER, 'r2_por_metrica.png'))
    plt.close()

    return results, results_df

# Crear función de predicción para nuevas nubes
def create_prediction_function():
    #Crea y guarda función para usar el modelo PointNet con nuevas nubes de puntos"""
    prediction_script = f"""
import torch
import numpy as np
import joblib
import os
import open3d as o3d

# Rutas a modelos y archivos auxiliares
MODEL_FOLDER = "{{MODELS_FOLDER}}"
MODEL_PATH = os.path.join(MODEL_FOLDER, "pointnet_model.pth")
METRIC_SCALER_PATH = os.path.join(MODEL_FOLDER, "metric_scaler_dl.pkl")
METRIC_NAMES_PATH = os.path.join(MODEL_FOLDER, "metric_names_dl.csv")

# Cargar nombres de métricas
import pandas as pd
metric_names = pd.read_csv(METRIC_NAMES_PATH)['metric'].tolist()

# Definir capas de transformación T-Net para PointNet
class TNet(torch.nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        # Red de convolución 1D para proceso de características
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        # MLP para transformación de características
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k*k)

        # Activaciones
        self.relu = torch.nn.ReLU()

        # Normalización por lotes
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

        # Inicialización de la última capa como matriz identidad
        self.k = k
        self.register_buffer('identity', torch.eye(k).flatten())

    def forward(self, x):
        batch_size = x.size(0)

        # Aplicar convoluciones 1D
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Max pooling a lo largo de la dimensión de puntos
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # MLP para generar matriz de transformación
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Agregar identidad para inicialización estable
        x = x + self.identity

        # Reshapear a matriz de transformación
        x = x.view(-1, self.k, self.k)

        return x

# Definir arquitectura PointNet principal
class PointNet(torch.nn.Module):
    def __init__(self, output_size, feature_transform=True):
        super(PointNet, self).__init__()
        self.feature_transform = feature_transform

        # Transformación de entrada (3x3 matriz)
        self.input_transform = TNet(k=3)

        # Primera capa de extracción de características
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)

        # Transformación de características (64x64 matriz) - opcional
        self.feature_transform_net = TNet(k=64)

        # Capas de extracción de características profundas
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        # Capas regresoras para métricas bovinas
        self.fc1 = torch.nn.Linear(1024, 512)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.dropout1 = torch.nn.Dropout(p=0.3)
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.dropout2 = torch.nn.Dropout(p=0.3)
        self.fc3 = torch.nn.Linear(256, output_size)

        # Activación ReLU
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        n_points = x.size(2)

        # Aplicar transformación de entrada
        trans_input = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_input)
        x = x.transpose(2, 1)

        # Primera capa de características
        x = self.relu(self.bn1(self.conv1(x)))

        # Aplicar transformación de características si está habilitada
        if self.feature_transform:
            trans_feature = self.feature_transform_net(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feature)
            x = x.transpose(2, 1)
        else:
            trans_feature = None

        # Capas de extracción de características
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Max pooling (agregación global)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Capas regresoras para métricas bovinas
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x, trans_feature

# Función para extraer nube de puntos de un archivo PLY
def extract_point_cloud(ply_file, num_points=1024):
    #Extrae y preprocesa nube de puntos para PointNet
    try:
        # Cargar nube de puntos
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)

        if len(points) == 0:
            print(f"Advertencia: {{ply_file}} no contiene puntos.")
            return None

        # Normalizar centroide a origen
        centroid = np.mean(points, axis=0)
        points = points - centroid

        # Normalizar a esfera unitaria
        furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / furthest_distance

        # Muestrear un número fijo de puntos
        if len(points) >= num_points:
            # Muestreo aleatorio sin reemplazo
            indices = np.random.choice(len(points), num_points, replace=False)
            sampled_points = points[indices]
        else:
            # Si hay menos puntos, repetir puntos
            indices = np.random.choice(len(points), num_points, replace=True)
            sampled_points = points[indices]

        return sampled_points
    except Exception as e:
        print(f"Error procesando {{ply_file}}: {{e}}")
        return None

# Función principal para predecir métricas
def predict_metrics(ply_file):
    \"\"\"
    Predice métricas para una vaca a partir de su nube de puntos usando PointNet.

    Args:
        ply_file: Ruta al archivo PLY con la nube de puntos

    Returns:
        Un diccionario con las métricas predichas
    \"\"\"
    try:
        # Verificar archivo
        if not os.path.exists(ply_file):
            print(f"Error: El archivo {{ply_file}} no existe")
            return None

        # Extraer nube de puntos
        points = extract_point_cloud(ply_file)

        if points is None:
            print("Error extrayendo puntos")
            return None

        # Cargar modelo y scaler
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        output_size = checkpoint['output_size']

        # Inicializar modelo
        model = PointNet(output_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Cargar scaler para desnormalizar predicciones
        metric_scaler = joblib.load(METRIC_SCALER_PATH)

        # Preparar input
        points_tensor = torch.FloatTensor(points).unsqueeze(0)  # [N, 3] -> [1, N, 3]
        points_tensor = points_tensor.transpose(2, 1)  # [1, N, 3] -> [1, 3, N]

        # Realizar predicción
        with torch.no_grad():
            y_pred_scaled, _ = model(points_tensor)
            y_pred_scaled = y_pred_scaled.cpu().numpy()

        # Desnormalizar predicciones
        y_pred = metric_scaler.inverse_transform(y_pred_scaled)

        # Crear diccionario de resultados
        predictions = {{}}
        for i, metric in enumerate(metric_names):
            predictions[metric] = float(y_pred[0, i])

        return predictions

    except Exception as e:
        print(f"Error en la predicción: {{e}}")
        return None

# Ejemplo de uso
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predecir métricas bovinas a partir de nube de puntos usando PointNet')
    parser.add_argument('ply_file', help='Ruta al archivo PLY')

    args = parser.parse_args()

    print(f"Procesando {{args.ply_file}}...")
    predictions = predict_metrics(args.ply_file)

    if predictions:
        print("\\nResultados de la predicción:")
        print("-" * 50)
        for metric, value in sorted(predictions.items()):
            print(f"{{metric:<30}} {{value:.2f}}")
    else:
        print("No se pudieron obtener predicciones")
"""

    # Guardar script de predicción
    script_path = os.path.join(MODELS_FOLDER, 'predict_metrics_pointnet.py')
    with open(script_path, 'w') as f:
        f.write(prediction_script)

    print(f"Función de predicción PointNet guardada en: {script_path}")
    return script_path

def compare_with_ml_models(pointnet_results, target_metrics):
    """Compara resultados del modelo PointNet con los modelos anteriores"""
    print("\n=== COMPARACIÓN CON MODELOS ANTERIORES ===")

    # Intentar cargar resultados de modelos anteriores
    rf_results_path = os.path.join(RF_MODELS_FOLDER, 'estadisticas_validacion_cruzada.csv')
    multi_results_path = os.path.join(MULTITRAIT_MODELS_FOLDER, 'resultados_modelos.csv')

    # Diccionario para almacenar resultados por modelo y métrica
    comparison_data = []

    # Añadir resultados de PointNet
    for metric in target_metrics:
        if metric in pointnet_results:
            result = pointnet_results[metric]
            comparison_data.append({
                'Métrica': metric,
                'Modelo': 'PointNet',
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'R²': result['r2'],
                'Error (%)': result['mean_rel_error'],
                'Dentro 10%': result['within_10pct']
            })

    # Añadir resultados del modelo de Random Forest para peso
    try:
        if os.path.exists(rf_results_path):
            rf_stats = pd.read_csv(rf_results_path)
            if 'live weithg' in target_metrics:
                comparison_data.append({
                    'Métrica': 'live weithg',
                    'Modelo': 'Random Forest',
                    'MAE': rf_stats['mae_mean'].iloc[0] if 'mae_mean' in rf_stats.columns else float('nan'),
                    'RMSE': np.sqrt(rf_stats['mse_mean'].iloc[0]) if 'mse_mean' in rf_stats.columns else float('nan'),
                    'R²': rf_stats['r2_mean'].iloc[0] if 'r2_mean' in rf_stats.columns else float('nan'),
                    'Error (%)': float('nan'),  # No disponible en estadísticas
                    'Dentro 10%': rf_stats['within_10pct_mean'].iloc[0] if 'within_10pct_mean' in rf_stats.columns else float('nan')
                })
        else:
            # Buscar en resultados de modelo optimizado
            rf_results_opt = os.path.join(RF_MODELS_FOLDER, 'resultados_modelo_optimizado.csv')
            if os.path.exists(rf_results_opt):
                rf_stats = pd.read_csv(rf_results_opt)

                # Extraer métricas
                mae_idx = rf_stats[rf_stats['Métrica'] == 'MAE'].index[0] if 'MAE' in rf_stats['Métrica'].values else -1
                rmse_idx = rf_stats[rf_stats['Métrica'] == 'RMSE'].index[0] if 'RMSE' in rf_stats['Métrica'].values else -1
                r2_idx = rf_stats[rf_stats['Métrica'] == 'R²'].index[0] if 'R²' in rf_stats['Métrica'].values else -1
                within_10pct_idx = rf_stats[rf_stats['Métrica'] == 'Dentro 10%'].index[0] if 'Dentro 10%' in rf_stats['Métrica'].values else -1

                if 'live weithg' in target_metrics:
                    comparison_data.append({
                        'Métrica': 'live weithg',
                        'Modelo': 'Random Forest',
                        'MAE': rf_stats['Valor'].iloc[mae_idx] if mae_idx >= 0 else float('nan'),
                        'RMSE': rf_stats['Valor'].iloc[rmse_idx] if rmse_idx >= 0 else float('nan'),
                        'R²': rf_stats['Valor'].iloc[r2_idx] if r2_idx >= 0 else float('nan'),
                        'Error (%)': float('nan'),
                        'Dentro 10%': rf_stats['Valor'].iloc[within_10pct_idx] if within_10pct_idx >= 0 else float('nan')
                    })
    except Exception as e:
        print(f"Error cargando resultados de Random Forest: {e}")

    # Añadir resultados de modelos multi-atributos
    try:
        if os.path.exists(multi_results_path):
            multi_stats = pd.read_csv(multi_results_path)

            # Mapeo entre nombres de columnas y métricas
            metric_mapping = {
                'Altura Cruz': 'withers height',
                'Altura Cadera': 'hip height',
                'Perimetro Torácico': 'heart girth',
                'altura_cruz': 'withers height',
                'altura_cadera': 'hip height',
                'circunferencia_torax': 'heart girth',
                'profundidad_pecho': 'chest depth',
                'ancho_pecho': 'chest width',
                'ancho_ilion': 'ilium width'
            }

            # Iterar por filas para encontrar métricas coincidentes
            for _, row in multi_stats.iterrows():
                metric_name = row['Característica'] if 'Característica' in multi_stats.columns else row['Métrica']

                # Convertir al nombre estándar si existe mapeo
                if metric_name in metric_mapping and metric_mapping[metric_name] in target_metrics:
                    std_name = metric_mapping[metric_name]
                elif metric_name in target_metrics:
                    std_name = metric_name
                else:
                    continue

                comparison_data.append({
                    'Métrica': std_name,
                    'Modelo': 'ML Tradicional',
                    'MAE': row['MAE'] if 'MAE' in multi_stats.columns else float('nan'),
                    'RMSE': row['RMSE'] if 'RMSE' in multi_stats.columns else float('nan'),
                    'R²': row['R²'] if 'R²' in multi_stats.columns else float('nan'),
                    'Error (%)': row['Error Rel. Medio (%)'] if 'Error Rel. Medio (%)' in multi_stats.columns else float('nan'),
                    'Dentro 10%': row['Dentro 10% (%)'] if 'Dentro 10% (%)' in multi_stats.columns else float('nan')
                })
    except Exception as e:
        print(f"Error cargando resultados de modelos multi-atributos: {e}")

    # Crear DataFrame de comparación
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(MODELS_FOLDER, 'comparacion_modelos.csv'), index=False)

    print("\nComparación de modelos:")
    print(comparison_df.to_string())

    # Visualizar comparación de R²
    plt.figure(figsize=(14, 7))

    # Obtener métricas únicas
    metrics = comparison_df['Métrica'].unique()

    # Para cada métrica, graficar barras de R² por modelo
    x = np.arange(len(metrics))
    width = 0.3  # Ancho de las barras

    models = comparison_df['Modelo'].unique()

    for i, model in enumerate(models):
        r2_values = []

        for metric in metrics:
            model_metric_data = comparison_df[(comparison_df['Modelo'] == model) &
                                             (comparison_df['Métrica'] == metric)]

            if not model_metric_data.empty and not pd.isna(model_metric_data['R²'].iloc[0]):
                r2_values.append(model_metric_data['R²'].iloc[0])
            else:
                r2_values.append(0)  # Valor por defecto si no hay datos

        plt.bar(x + i*width, r2_values, width, label=model)

    plt.xlabel('Métrica')
    plt.ylabel('R²')
    plt.title('Comparación de R² por Modelo y Métrica')
    plt.xticks(x + width, metrics, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(MODELS_FOLDER, 'comparacion_r2_modelos.png'))
    plt.close()

    return comparison_df
    # Añade esta línea al final de tu notebook
main()