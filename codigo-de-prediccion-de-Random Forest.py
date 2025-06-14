import numpy as np
import pandas as pd
import os
import re
import joblib
import open3d as o3d
import argparse
from datetime import datetime
from tqdm import tqdm
from google.colab import drive

print("""
===============================================================
PREDICCIÓN DE MÉTRICAS BOVINAS CON MODELOS DE ÁRBOLES DE DECISIÓN
A PARTIR DE NUBES DE PUNTOS 3D
===============================================================
""")

# Montar Google Drive
drive.mount('/content/drive')

# Configurar directorios
DRIVE_PATH = '/content/drive/MyDrive'
PLY_FOLDER = f"{DRIVE_PATH}/nubes-de-puntos-filtradas"  # Carpeta con nubes para predecir
RESULTS_FOLDER = f"{DRIVE_PATH}/resultados_predicciones_{datetime.now().strftime('%Y%m%d_%H%M')}"  # Carpeta para resultados
RF_MODELS_FOLDER = f"{DRIVE_PATH}/Random-Forest-optimizado(peso)"  # Carpeta con modelos Random Forest
MULTITRAIT_MODELS_FOLDER = f"{DRIVE_PATH}/Random-Forest-las-de-mas-métricas"  # Carpeta con modelos multi-atributos

# Crear carpeta de resultados si no existe
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
    print(f"Carpeta creada: {RESULTS_FOLDER}")

# Verificar si existen las rutas importantes
print(f"PLY_FOLDER existe: {os.path.exists(PLY_FOLDER)}")
print(f"RF_MODELS_FOLDER existe: {os.path.exists(RF_MODELS_FOLDER)}")
print(f"MULTITRAIT_MODELS_FOLDER existe: {os.path.exists(MULTITRAIT_MODELS_FOLDER)}")

# Función para extraer características volumétricas
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

        return features
    except Exception as e:
        print(f"Error procesando {ply_file}: {e}")
        return None

# Función para extraer número de vaca
def extract_number(filename):
    """Extrae el número de identificación de la vaca del nombre del archivo"""
    match = re.search(r'procesada_(\d+)_AutoAligned', filename)
    if match:
        return int(match.group(1))
    return None

# Función para cargar modelos de árboles de decisión y multi-atributos
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

# Función optimizada para garantizar que el modelo de peso funcione correctamente
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

            if not missing:
                # Crear vector con las características exactas que necesita el modelo
                feature_values = [features[f] for f in REQUIRED_FEATURES]

                X = np.array([feature_values])

                # Verificar el scaler
                if 'peso_scaler' in ml_models:
                    print("Aplicando scaler para modelo de peso...")
                    X_scaled = ml_models['peso_scaler'].transform(X)
                else:
                    print("No se encontró scaler para el modelo de peso, usando valores sin escalar")
                    X_scaled = X

                # Realizar predicción
                try:
                    peso_pred = ml_models['peso'].predict(X_scaled)[0]
                    print(f"Predicción exitosa: {peso_pred:.2f} kg")

                    # Guardar predicción con ambos nombres para compatibilidad
                    predictions['live weithg'] = peso_pred
                    predictions['peso_real_kg'] = peso_pred

                    # También guardar valor en feature para que esté disponible para otras predicciones
                    features['live weithg'] = peso_pred

                except Exception as e:
                    print(f"Error en predict del modelo: {e}")

                    # Intento alternativo con DataFrame
                    try:
                        print("Intentando con DataFrame...")
                        X_df = pd.DataFrame([dict(zip(REQUIRED_FEATURES, feature_values))])
                        peso_pred = ml_models['peso'].predict(X_df)[0]
                        print(f"Predicción exitosa con DataFrame: {peso_pred:.2f} kg")

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
    else:
        print("No se encontró modelo de peso ('peso') en ml_models")

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
            X_scaled = ml_models['multitrait_scaler'].transform(X)

            # Mapeo entre nombres de modelos y nombres originales del Excel
            trait_mapping = {
                'altura_cruz': 'withers height',
                'altura_cadera': 'hip height',
                'profundidad_pecho': 'chest depth',
                'ancho_pecho': 'chest width',
                'ancho_ilion': 'ilium width',
                'ancho_articulacion_cadera': 'hip joint width',
                'longitud_oblicua': 'oblique body length',
                'longitud_cadera': 'hip length',
                'circunferencia_torax': 'heart girth'
            }

            # Predecir cada atributo
            for trait, output_name in trait_mapping.items():
                if trait in ml_models and f'{trait}_selector' in ml_models:
                    try:
                        # Aplicar selector de características
                        X_selected = ml_models[f'{trait}_selector'].transform(X_scaled)

                        # Predecir
                        pred = ml_models[trait].predict(X_selected)[0]

                        # Guardar con nombre original
                        predictions[output_name] = pred
                        print(f"Predicción para {output_name}: {pred:.2f}")
                    except Exception as e:
                        print(f"Error prediciendo {output_name}: {e}")
        except Exception as e:
            print(f"Error general en predicciones multiatributo: {e}")

    # Verificar predicciones finales
    print("\n--- RESUMEN DE PREDICCIONES ---")
    for k, v in predictions.items():
        print(f"{k}: {v:.2f}")

    return predictions

# Función para procesar un archivo PLY y obtener predicciones
def process_ply_file(ply_path, ml_models):
    """Procesa un archivo PLY y devuelve predicciones usando modelos ML"""
    print(f"\nProcesando: {os.path.basename(ply_path)}")

    # Extraer características del archivo PLY
    features = extract_volumetric_features(ply_path)

    if features is None:
        print(f"Error: No se pudieron extraer características de {ply_path}")
        return None

    # Predecir métricas usando modelos ML
    predictions = predict_metrics_with_ml_models(features, ml_models)

    # Añadir información de archivo
    predictions['archivo'] = os.path.basename(ply_path)
    predictions['id_vaca'] = extract_number(os.path.basename(ply_path))

    # Añadir características principales como referencia
    for key in ['length', 'width', 'height', 'volume', 'point_count']:
        if key in features:
            predictions[f'feature_{key}'] = features[key]

    return predictions

# Función principal
"""
def main():
    # Parsear argumentos
    parser = argparse.ArgumentParser(description='Predecir métricas bovinas a partir de nubes de puntos PLY')
    parser.add_argument('--input', help='Carpeta de entrada con archivos PLY (por defecto en Drive)', default=PLY_FOLDER)
    parser.add_argument('--output', help='Carpeta de salida para resultados (por defecto en Drive)', default=RESULTS_FOLDER)

    args = parser.parse_args()

    # Actualizar rutas si se proporcionaron argumentos
    ply_folder = args.input
    results_folder = args.output
    """
def main():
    # Rutas de entrada y salida directamente como variables (no usar argparse en Colab)
    ply_folder = PLY_FOLDER
    results_folder = RESULTS_FOLDER

    print(f"Carpeta de entrada: {ply_folder}")
    print(f"Carpeta de salida: {results_folder}")



    # Cargar modelos ML
    ml_models = load_ml_models()

    # Listar archivos PLY para procesar
    ply_files = [f for f in os.listdir(ply_folder) if f.endswith('.ply')]
    print(f"\nArchivos PLY encontrados: {len(ply_files)}")

    if len(ply_files) == 0:
        print("¡Error! No se encontraron archivos PLY en la carpeta especificada.")
        return

    # Procesar cada archivo y recopilar resultados
    all_predictions = []

    for filename in tqdm(ply_files, desc="Procesando archivos PLY"):
        ply_path = os.path.join(ply_folder, filename)
        predictions = process_ply_file(ply_path, ml_models)

        if predictions:
            all_predictions.append(predictions)

    # Crear DataFrame con todas las predicciones
    if all_predictions:
        results_df = pd.DataFrame(all_predictions)

        # Ordenar columnas para mejor legibilidad
        col_order = ['archivo', 'id_vaca']

        # Añadir primero métricas importantes
        important_metrics = [
            'live weithg', 'peso_real_kg',
            'withers height', 'hip height', 'heart girth',
            'chest depth', 'chest width', 'ilium width',
            'hip joint width', 'oblique body length', 'hip length'
        ]

        for metric in important_metrics:
            if metric in results_df.columns:
                col_order.append(metric)

        # Añadir resto de columnas
        remaining_cols = [c for c in results_df.columns if c not in col_order]
        col_order.extend(remaining_cols)

        # Reordenar y guardar
        results_df = results_df[col_order]

        # Guardar resultados
        csv_path = os.path.join(results_folder, f'predicciones_metricas_{datetime.now().strftime("%Y%m%d_%H%M")}.csv')
        results_df.to_csv(csv_path, index=False)

        print(f"\n¡Procesamiento completado!")
        print(f"Se procesaron {len(results_df)} archivos PLY")
        print(f"Resultados guardados en: {csv_path}")

        # Mostrar estadísticas básicas
        print("\nEstadísticas de predicciones:")
        for metric in important_metrics:
            if metric in results_df.columns:
                print(f"{metric}:")
                print(f"  Media: {results_df[metric].mean():.2f}")
                print(f"  Mín: {results_df[metric].min():.2f}")
                print(f"  Máx: {results_df[metric].max():.2f}")
                print()
    else:
        print("No se pudieron generar predicciones para ningún archivo.")

# Ejecutar la función principal si el script se ejecuta directamente
if __name__ == "__main__":
    main()