import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from collections import defaultdict

# Definir rutas
DRIVE_PATH = '/content/drive/MyDrive'
EXCEL_PATH = f"{DRIVE_PATH}/Measurements.xlsx"  # Archivo con mediciones
PLY_FOLDER = f"{DRIVE_PATH}/nubes-de-puntos-filtradas"  # Carpeta con nubes de puntos
MODELS_FOLDER = f"{DRIVE_PATH}/modelos_multiatributos_{datetime.now().strftime('%Y%m%d_%H%M')}"  # Carpeta para guardar modelos

# Crear carpeta para modelos si no existe
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Función para extraer características de nubes de puntos
def extract_point_cloud_features(ply_file):
    """Extrae características volumétricas y geométricas de una nube de puntos"""
    try:
        import open3d as o3d

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
            'distance_q3': q3,
            'eigenvalue_1': eigenvalues[2],
            'eigenvalue_2': eigenvalues[1],
            'eigenvalue_3': eigenvalues[0],
            'principal_length_1': principal_length_1,
            'principal_length_2': principal_length_2,
            'principal_length_3': principal_length_3,
            'sphericity': eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 0 else 0,
            'elongation': 1 - (eigenvalues[1] / eigenvalues[2]) if eigenvalues[2] > 0 else 0,
            'flatness': 1 - (eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else 0
        }

        return features
    except Exception as e:
        print(f"Error procesando {ply_file}: {e}")
        return None

# Función para extraer número de archivo de nube
def extract_number(filename):
    """Extrae el número de identificación de la vaca del nombre del archivo"""
    import re
    match = re.search(r'procesada_(\d+)_AutoAligned', filename)
    if match:
        return int(match.group(1))
    return None

# Función para cargar y preparar los datos
def load_and_prepare_data():
    """Carga datos de Excel y características de nubes de puntos, y los combina"""
    print("Cargando datos del archivo Excel...")
    metrics_df = pd.read_excel(EXCEL_PATH)

    # Mostrar columnas disponibles
    print("Columnas disponibles en el Excel:")
    print(metrics_df.columns.tolist())

    # Preparar diccionario para mapeo de nombres estándar
    column_mapping = {
        'N live weithg': 'peso',
        'withers height': 'altura_cruz',
        'hip height': 'altura_cadera',
        'chest depth': 'profundidad_pecho',
        'chest width': 'ancho_pecho',
        'ilium width': 'ancho_ilion',
        'hip joint width': 'ancho_articulacion_cadera',
        'oblique body length': 'longitud_oblicua',
        'hip length': 'longitud_cadera',
        'heart girth': 'circunferencia_torax'
    }

    # Verificar y aplicar mapeo si las columnas existen
    cols_to_rename = {}
    for original, new_name in column_mapping.items():
        if original in metrics_df.columns:
            cols_to_rename[original] = new_name

    # Renombrar columnas si existen
    if cols_to_rename:
        metrics_df = metrics_df.rename(columns=cols_to_rename)
        print("Columnas renombradas:")
        for orig, new_name in cols_to_rename.items():
            print(f"  {orig} -> {new_name}")

    # Verificar características disponibles después del renombrado
    available_traits = [col for col in column_mapping.values() if col in metrics_df.columns]
    print(f"\nCaracterísticas disponibles para modelado: {len(available_traits)}")
    print(available_traits)

    # Procesar todas las nubes de puntos
    print("\nProcesando nubes de puntos...")
    ply_files = [f for f in os.listdir(PLY_FOLDER) if f.endswith('.ply')]

    # Crear un DataFrame para almacenar características
    features_data = []
    ids = []

    for filename in tqdm(ply_files, desc="Extrayendo características"):
        file_path = os.path.join(PLY_FOLDER, filename)
        number = extract_number(filename)

        if number is not None and number - 1 < len(metrics_df):
            features = extract_point_cloud_features(file_path)

            if features is not None:
                features_data.append(features)
                ids.append(number)

    # Crear DataFrame con características
    features_df = pd.DataFrame(features_data)
    features_df['id'] = ids

    # Unir con datos de métricas del Excel
    merged_data = pd.DataFrame()
    for i, id_val in enumerate(ids):
        row = metrics_df.iloc[id_val - 1].copy()
        for feature, value in features_data[i].items():
            row[feature] = value
        merged_data = pd.concat([merged_data, pd.DataFrame([row])], ignore_index=True)

    print(f"Datos combinados: {len(merged_data)} muestras con {len(merged_data.columns)} columnas")

    # Guardar datos combinados
    merged_data.to_csv(os.path.join(MODELS_FOLDER, 'datos_combinados.csv'), index=False)

    return merged_data, available_traits

# Función para crear y entrenar modelos para cada característica
def train_models_for_all_traits(data, available_traits):
    """Entrena un modelo Random Forest para cada característica disponible"""
    print("\nEntrenando modelos para múltiples características...")

    # Preparar diccionario para guardar modelos y resultados
    models = {}
    results = {}
    feature_importance = {}
    selected_features = {}

    # Obtener características de la nube de puntos (excluyendo IDs y características a predecir)
    point_cloud_features = [col for col in data.columns
                         if col not in available_traits + ['id', 'N', 'Data']]

    # Definir un escalador para normalizar características
    scaler = StandardScaler()

    # Para cada característica a predecir
    for trait in available_traits:
        print(f"\nEntrenando modelo para: {trait}")

        # Filtrar datos sin valores NaN en la característica objetivo
        valid_data = data.dropna(subset=[trait])

        if len(valid_data) < 10:
            print(f"  Insuficientes datos válidos para {trait}, omitiendo...")
            continue

        # Preparar X e y
        X = valid_data[point_cloud_features]
        y = valid_data[trait]

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Escalar características
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entrenar modelo base para selección de características
        base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        base_model.fit(X_train_scaled, y_train)

        # Seleccionar características importantes
        selector = SelectFromModel(base_model, threshold='mean')
        selector.fit(X_train_scaled, y_train)

        # Aplicar selección de características
        selected_mask = selector.get_support()
        selected_feature_names = np.array(point_cloud_features)[selected_mask]

        X_train_selected = selector.transform(X_train_scaled)
        X_test_selected = selector.transform(X_test_scaled)

        print(f"  Características seleccionadas: {len(selected_feature_names)}/{len(point_cloud_features)}")

        # Entrenar modelo final
        final_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )

        final_model.fit(X_train_selected, y_train)

        # Evaluar en conjunto de prueba
        y_pred = final_model.predict(X_test_selected)

        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Calcular errores relativos
        rel_errors = 100 * np.abs(y_pred - y_test) / y_test
        mean_rel_error = np.mean(rel_errors)
        within_5pct = np.sum(rel_errors <= 5) / len(rel_errors) * 100
        within_10pct = np.sum(rel_errors <= 10) / len(rel_errors) * 100

        # Almacenar resultados
        results[trait] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_rel_error': mean_rel_error,
            'within_5pct': within_5pct,
            'within_10pct': within_10pct
        }

        # Almacenar modelo
        models[trait] = final_model

        # Almacenar importancia de características
        feature_importances = final_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        feature_importance[trait] = importance_df
        selected_features[trait] = selected_feature_names

        # Guardar modelo
        joblib.dump(final_model, os.path.join(MODELS_FOLDER, f'modelo_{trait}.pkl'))

        # Guardar selector de características
        joblib.dump(selector, os.path.join(MODELS_FOLDER, f'selector_{trait}.pkl'))

        # Guardar importancia de características
        importance_df.to_csv(os.path.join(MODELS_FOLDER, f'importancia_{trait}.csv'), index=False)

        # Mostrar resultados
        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        print(f"  Error relativo medio: {mean_rel_error:.2f}%")
        print(f"  Dentro del 5%: {within_5pct:.1f}%, Dentro del 10%: {within_10pct:.1f}%")

    # Guardar resultados generales
    results_df = pd.DataFrame({
        'Característica': list(results.keys()),
        'MAE': [results[t]['mae'] for t in results],
        'RMSE': [results[t]['rmse'] for t in results],
        'R²': [results[t]['r2'] for t in results],
        'Error Rel. Medio (%)': [results[t]['mean_rel_error'] for t in results],
        'Dentro 5% (%)': [results[t]['within_5pct'] for t in results],
        'Dentro 10% (%)': [results[t]['within_10pct'] for t in results]
    })

    results_df.to_csv(os.path.join(MODELS_FOLDER, 'resultados_modelos.csv'), index=False)

    # Guardar scaler
    joblib.dump(scaler, os.path.join(MODELS_FOLDER, 'scaler.pkl'))

    return models, results, feature_importance, selected_features, scaler

# Función para visualizar resultados
def visualize_results(results, feature_importance):
    """Crea visualizaciones para los resultados de los modelos"""
    print("\nGenerando visualizaciones...")

    # 1. Gráfico de barras de precisión (R²)
    plt.figure(figsize=(12, 6))
    traits = list(results.keys())
    r2_values = [results[t]['r2'] for t in traits]

    # Ordenar por R²
    sorted_indices = np.argsort(r2_values)[::-1]
    sorted_traits = [traits[i] for i in sorted_indices]
    sorted_r2 = [r2_values[i] for i in sorted_indices]

    bars = plt.bar(sorted_traits, sorted_r2)

    # Colorear barras según R²
    for i, bar in enumerate(bars):
        if sorted_r2[i] > 0.5:
            bar.set_color('green')
        elif sorted_r2[i] > 0.25:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Característica')
    plt.ylabel('Coeficiente de Determinación (R²)')
    plt.title('Precisión de Predicción por Característica')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FOLDER, 'precision_modelos.png'))

    # 2. Gráfico de barras de error relativo
    plt.figure(figsize=(12, 6))
    rel_error_values = [results[t]['mean_rel_error'] for t in traits]

    # Ordenar por error relativo (menor a mayor)
    sorted_indices = np.argsort(rel_error_values)
    sorted_traits = [traits[i] for i in sorted_indices]
    sorted_rel_errors = [rel_error_values[i] for i in sorted_indices]

    bars = plt.bar(sorted_traits, sorted_rel_errors)

    # Colorear barras según error
    for i, bar in enumerate(bars):
        if sorted_rel_errors[i] < 5:
            bar.set_color('green')
        elif sorted_rel_errors[i] < 10:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% error')
    plt.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% error')

    plt.xlabel('Característica')
    plt.ylabel('Error Relativo Medio (%)')
    plt.title('Error Relativo por Característica')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FOLDER, 'error_relativo_modelos.png'))

    # 3. Heatmap de importancia de características
    # Preparar matriz de importancia de características
    all_features = set()
    for trait in feature_importance:
        all_features.update(feature_importance[trait]['feature'])

    all_features = sorted(list(all_features))
    traits = sorted(list(feature_importance.keys()))

    importance_matrix = pd.DataFrame(0, index=all_features, columns=traits)

    for trait in traits:
        imp_df = feature_importance[trait]
        for _, row in imp_df.iterrows():
            importance_matrix.loc[row['feature'], trait] = row['importance']

    # Graficar heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(importance_matrix, cmap='viridis', linewidths=0.5, linecolor='gray')
    plt.title('Importancia de Características por Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FOLDER, 'heatmap_importancia.png'))

   # Función para visualizar resultados corregida
def visualize_results(results, feature_importance):
    """Crea visualizaciones para los resultados de los modelos"""
    print("\nGenerando visualizaciones...")

    # 1. Gráfico de barras de precisión (R²)
    plt.figure(figsize=(12, 6))
    traits = list(results.keys())
    r2_values = [results[t]['r2'] for t in traits]

    # Ordenar por R²
    sorted_indices = np.argsort(r2_values)[::-1]
    sorted_traits = [traits[i] for i in sorted_indices]
    sorted_r2 = [r2_values[i] for i in sorted_indices]

    bars = plt.bar(sorted_traits, sorted_r2)

    # Colorear barras según R²
    for i, bar in enumerate(bars):
        if sorted_r2[i] > 0.5:
            bar.set_color('green')
        elif sorted_r2[i] > 0.25:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Característica')
    plt.ylabel('Coeficiente de Determinación (R²)')
    plt.title('Precisión de Predicción por Característica')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FOLDER, 'precision_modelos.png'))

    # 2. Gráfico de barras de error relativo
    plt.figure(figsize=(12, 6))
    rel_error_values = [results[t]['mean_rel_error'] for t in traits]

    # Ordenar por error relativo (menor a mayor)
    sorted_indices = np.argsort(rel_error_values)
    sorted_traits = [traits[i] for i in sorted_indices]
    sorted_rel_errors = [rel_error_values[i] for i in sorted_indices]

    bars = plt.bar(sorted_traits, sorted_rel_errors)

    # Colorear barras según error
    for i, bar in enumerate(bars):
        if sorted_rel_errors[i] < 5:
            bar.set_color('green')
        elif sorted_rel_errors[i] < 10:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% error')
    plt.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% error')

    plt.xlabel('Característica')
    plt.ylabel('Error Relativo Medio (%)')
    plt.title('Error Relativo por Característica')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FOLDER, 'error_relativo_modelos.png'))

    try:
        # 3. Heatmap de importancia de características - VERSIÓN CORREGIDA
        print("Generando heatmap de importancia de características...")

        # Reunir todas las características únicas
        all_features = set()
        for trait in feature_importance:
            all_features.update(feature_importance[trait]['feature'])

        all_features = sorted(list(all_features))
        traits = sorted(list(feature_importance.keys()))

        # Crear una matriz numérica (no de objetos)
        importance_values = np.zeros((len(all_features), len(traits)))

        # Llenar la matriz con valores de importancia
        for j, trait in enumerate(traits):
            imp_df = feature_importance[trait]
            feat_dict = dict(zip(imp_df['feature'], imp_df['importance']))

            for i, feature in enumerate(all_features):
                importance_values[i, j] = feat_dict.get(feature, 0.0)

        # Ahora crear el heatmap con datos numéricos
        plt.figure(figsize=(14, 10))

        # Asegurar que el aspecto sea correcto para el número de características
        if len(all_features) > 20:
            plt.figure(figsize=(14, max(10, len(all_features) * 0.4)))

        # Crear el heatmap con las matrices numéricas
        heatmap = sns.heatmap(
            importance_values,
            cmap='viridis',
            linewidths=0.5,
            linecolor='gray',
            xticklabels=traits,
            yticklabels=all_features
        )

        plt.title('Importancia de Características por Modelo')
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_FOLDER, 'heatmap_importancia.png'))

        # 4. Gráfico de correlación entre características - VERSIÓN CORREGIDA
        print("Generando mapa de correlación entre características...")

        # Crear matriz de correlación
        num_traits = len(traits)
        correlation_matrix = np.zeros((num_traits, num_traits))

        for i, trait1 in enumerate(traits):
            for j, trait2 in enumerate(traits):
                # Obtener vectores de importancia
                v1 = np.zeros(len(all_features))
                v2 = np.zeros(len(all_features))

                # Llenar vectores
                feat1 = feature_importance[trait1]
                feat2 = feature_importance[trait2]

                dict1 = dict(zip(feat1['feature'], feat1['importance']))
                dict2 = dict(zip(feat2['feature'], feat2['importance']))

                for k, feature in enumerate(all_features):
                    v1[k] = dict1.get(feature, 0.0)
                    v2[k] = dict2.get(feature, 0.0)

                # Calcular correlación si hay varianza
                if np.std(v1) > 0 and np.std(v2) > 0:
                    correlation_matrix[i, j] = np.corrcoef(v1, v2)[0, 1]
                else:
                    correlation_matrix[i, j] = 0

        # Visualizar correlación
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            xticklabels=traits,
            yticklabels=traits
        )

        plt.title('Correlación entre Importancias de Características')
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_FOLDER, 'correlacion_importancia.png'))

    except Exception as e:
        print(f"Error al generar visualizaciones avanzadas: {e}")
        print("Continuando con el resto del proceso...")

    print("Visualizaciones básicas guardadas en la carpeta de modelos.")

# Función para crear script de predicción
def create_prediction_script(available_traits, selected_features):
    """Crea un script de Python para hacer predicciones con los modelos entrenados"""
    print("\nCreando script de predicción...")

    # Primera parte del script
    script_part1 = f"""
import numpy as np
import pandas as pd
import joblib
import os
import argparse

# Ruta de modelos
MODELS_FOLDER = "{MODELS_FOLDER}"

# Cargar modelos, selectores y scaler
print("Cargando modelos...")
models = {{}}
selectors = {{}}
for trait in {available_traits}:
    model_path = os.path.join(MODELS_FOLDER, f'modelo_{{trait}}.pkl')
    selector_path = os.path.join(MODELS_FOLDER, f'selector_{{trait}}.pkl')

    if os.path.exists(model_path) and os.path.exists(selector_path):
        models[trait] = joblib.load(model_path)
        selectors[trait] = joblib.load(selector_path)

# Cargar scaler
scaler = joblib.load(os.path.join(MODELS_FOLDER, 'scaler.pkl'))
"""

    # Segunda parte - función de extracción de características
    script_part2 = """
def extract_features_from_point_cloud(ply_file):
    \"\"\"Extrae características de una nube de puntos\"\"\"
    try:
        import open3d as o3d

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
            'distance_q3': q3,
            'eigenvalue_1': eigenvalues[2],
            'eigenvalue_2': eigenvalues[1],
            'eigenvalue_3': eigenvalues[0],
            'principal_length_1': principal_length_1,
            'principal_length_2': principal_length_2,
            'principal_length_3': principal_length_3,
            'sphericity': eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 0 else 0,
            'elongation': 1 - (eigenvalues[1] / eigenvalues[2]) if eigenvalues[2] > 0 else 0,
            'flatness': 1 - (eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else 0
        }

        return features
    except Exception as e:
        print(f"Error procesando {ply_file}: {e}")
        return None
"""

    # Tercera parte - función de predicción y main
    script_part3 = """
def predict_traits(ply_file):
    \"\"\"Predice varias características a partir de una nube de puntos\"\"\"
    # Extraer características
    features = extract_features_from_point_cloud(ply_file)

    if features is None:
        return None

    # Crear dataframe con características
    X = pd.DataFrame([features])

    # Escalar características
    X_scaled = scaler.transform(X)

    # Hacer predicciones para cada característica
    predictions = {}
    for trait, model in models.items():
        selector = selectors[trait]

        # Seleccionar características relevantes
        X_selected = selector.transform(X_scaled)

        # Predecir
        prediction = model.predict(X_selected)[0]
        predictions[trait] = prediction

    return predictions

def main():
    parser = argparse.ArgumentParser(description='Predecir características físicas a partir de nubes de puntos')
    parser.add_argument('ply_file', type=str, help='Ruta al archivo PLY')

    args = parser.parse_args()

    if not os.path.exists(args.ply_file):
        print(f"Error: El archivo {args.ply_file} no existe")
        return

    print(f"Procesando {args.ply_file}...")
    predictions = predict_traits(args.ply_file)

    if predictions:
        print("\\nResultados de la predicción:")
        print("-" * 40)
        print(f"{'Característica':<20} {'Valor':<10} {'Unidad':<10}")
        print("-" * 40)

        # Definir unidades para cada característica
        units = {
            'peso': 'kg',
            'altura_cruz': 'cm',
            'altura_cadera': 'cm',
            'profundidad_pecho': 'cm',
            'ancho_pecho': 'cm',
            'ancho_ilion': 'cm',
            'ancho_articulacion_cadera': 'cm',
            'longitud_oblicua': 'cm',
            'longitud_cadera': 'cm',
            'circunferencia_torax': 'cm'
        }

        # Mostrar resultados ordenados
        for trait, value in sorted(predictions.items()):
            unit = units.get(trait, '')
            print(f"{trait:<20} {value:<10.2f} {unit:<10}")
    else:
        print("No se pudieron realizar predicciones.")

if __name__ == "__main__":
    main()
"""

    # Combinar todas las partes
    script_content = script_part1 + script_part2 + script_part3

    # Guardar script
    script_path = os.path.join(MODELS_FOLDER, 'prediccion_multiatributos.py')
    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"Script de predicción guardado en: {script_path}")

    return script_path

# Función para crear interfaz gráfica
def create_interface_script():
    """Crea una interfaz gráfica para la predicción"""
    print("\nCreando script de interfaz gráfica...")

    interface_content = f"""
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import pandas as pd

class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictor de Características de Vacas")
        self.root.geometry("800x600")

        self.setup_ui()

    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Título
        title_label = tk.Label(main_frame, text="Predictor de Características de Vacas", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))

        # Frame para selección de archivo
        file_frame = tk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=10)

        self.file_label = tk.Label(file_frame, text="Ningún archivo seleccionado", width=50)
        self.file_label.pack(side=tk.LEFT, padx=(0, 10))

        browse_button = tk.Button(file_frame, text="Seleccionar Archivo PLY", command=self.browse_file)
        browse_button.pack(side=tk.RIGHT)

        # Botón de predicción
        predict_button = tk.Button(main_frame, text="Realizar Predicción", command=self.predict, bg="#4CAF50", fg="white", height=2)
        predict_button.pack(pady=20)

        # Frame para resultados
        self.results_frame = tk.Frame(main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Tabla de resultados
        self.create_results_table()

        # Historial
        history_label = tk.Label(main_frame, text="Historial de Predicciones", font=("Arial", 12, "bold"))
        history_label.pack(pady=(20, 10))

        self.history_listbox = tk.Listbox(main_frame, height=5)
        self.history_listbox.pack(fill=tk.X)
        self.history_listbox.bind("<<ListboxSelect>>", self.load_history_item)

        # Botón para exportar a Excel
        export_button = tk.Button(main_frame, text="Exportar Historial a Excel", command=self.export_to_excel)
        export_button.pack(pady=10)

    def create_results_table(self):
        # Limpiar frame anterior
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Cabeceras
        headers = ["Característica", "Valor", "Unidad"]
        for i, header in enumerate(headers):
            label = tk.Label(self.results_frame, text=header, font=("Arial", 10, "bold"))
            label.grid(row=0, column=i, padx=10, pady=5, sticky="w")

        # Separador
        separator = tk.Frame(self.results_frame, height=2, bg="black")
        separator.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)

        # Historial de resultados
        self.results = []

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar Archivo PLY",
            filetypes=[("Archivos PLY", "*.ply")]
        )

        if file_path:
            self.file_label.config(text=os.path.basename(file_path))
            self.file_path = file_path

    def predict(self):
        if not hasattr(self, 'file_path'):
            messagebox.showerror("Error", "Por favor seleccione un archivo PLY primero")
            return

        try:
            # Ejecutar script de predicción
            result = subprocess.run(
                ["python", "{os.path.join(MODELS_FOLDER, 'prediccion_multiatributos.py')}", self.file_path],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                messagebox.showerror("Error", f"Error en la predicción: {{result.stderr}}")
                return

            # Procesar salida
            output = result.stdout
            predictions = {{}}

            # Parsear resultados de la salida
            start_parsing = False
            for line in output.split('\\n'):
                if "Resultados de la predicción:" in line:
                    start_parsing = True
                    continue

                if start_parsing and line.strip() and not line.startswith('-'):
                    parts = line.split()
                    if len(parts) >= 3 and parts[0] not in ["Característica", "Valor", "Unidad"]:
                        trait = parts[0]
                        value = float(parts[1])
                        unit = parts[2] if len(parts) > 2 else ""
                        predictions[trait] = (value, unit)

            # Mostrar resultados
            self.display_results(predictions)

            # Añadir a historial
            history_entry = f"{{os.path.basename(self.file_path)}} - {{len(predictions)}} características"
            self.history_listbox.insert(0, history_entry)
            self.results.insert(0, {{
                'file': self.file_path,
                'predictions': predictions
            }})

        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar la predicción: {{str(e)}}")

    def display_results(self, predictions):
        # Limpiar tabla
        self.create_results_table()

        # Mostrar resultados
        row = 2
        for trait, (value, unit) in sorted(predictions.items()):
            tk.Label(self.results_frame, text=trait).grid(row=row, column=0, padx=10, pady=2, sticky="w")
            tk.Label(self.results_frame, text=f"{{value:.2f}}").grid(row=row, column=1, padx=10, pady=2, sticky="w")
            tk.Label(self.results_frame, text=unit).grid(row=row, column=2, padx=10, pady=2, sticky="w")
            row += 1

    def load_history_item(self, event):
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            item = self.results[index]
            self.display_results(item['predictions'])
            self.file_label.config(text=os.path.basename(item['file']))
            self.file_path = item['file']

    def export_to_excel(self):
        if not self.results:
            messagebox.showinfo("Información", "No hay datos para exportar")
            return

        try:
            # Preparar datos para excel
            data = []
            for result in self.results:
                row = {{'Archivo': os.path.basename(result['file'])}}
                for trait, (value, unit) in result['predictions'].items():
                    row[f"{{trait}} ({{unit}})"] = value
                data.append(row)

            # Crear dataframe
            df = pd.DataFrame(data)

            # Guardar a Excel
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                title="Guardar historial como"
            )

            if save_path:
                df.to_excel(save_path, index=False)
                messagebox.showinfo("Éxito", f"Datos exportados a {{save_path}}")

        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar datos: {{str(e)}}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()
"""

    # Guardar script de interfaz
    interface_path = os.path.join(MODELS_FOLDER, 'interfaz_prediccion.py')
    with open(interface_path, 'w') as f:
        f.write(interface_content)

    print(f"Script de interfaz guardado en: {interface_path}")

    return interface_path

# Función principal para todo el proceso
def main():
    print("=" * 60)
    print("PREDICCIÓN DE MÚLTIPLES CARACTERÍSTICAS FÍSICAS EN VACAS")
    print("=" * 60)

    # Cargar y preparar datos
    data, available_traits = load_and_prepare_data()

    # Entrenar modelos para todas las características disponibles
    models, results, feature_importance, selected_features, scaler = train_models_for_all_traits(data, available_traits)

    # Visualizar resultados
    visualize_results(results, feature_importance)

    # Crear scripts de predicción
    script_path = create_prediction_script(available_traits, selected_features)

    # Crear interfaz gráfica
    interface_path = create_interface_script()

    print("\n" + "=" * 60)
    print("RESUMEN DE MODELOS ENTRENADOS")
    print("=" * 60)

    # Mostrar tabla de resultados
    print(f"{'Característica':<20} {'MAE':<10} {'R²':<10} {'Dentro 10%':<15}")
    print("-" * 55)
    for trait in sorted(results.keys()):
        result = results[trait]
        print(f"{trait:<20} {result['mae']:<10.2f} {result['r2']:<10.4f} {result['within_10pct']:<15.1f}%")

    print("\nModelos guardados en:", MODELS_FOLDER)
    print("\nPara usar los modelos:")
    print(f"1. Script de predicción: python {script_path} ruta_a_archivo.ply")
    print(f"2. Interfaz gráfica: python {interface_path}")

    # Guardar resumen en un archivo de texto
    summary_path = os.path.join(MODELS_FOLDER, 'resumen_modelos.txt')
    with open(summary_path, 'w') as f:
        f.write("RESUMEN DE MODELOS ENTRENADOS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"{'Característica':<20} {'MAE':<10} {'R²':<10} {'Error Rel.':<10} {'Dentro 5%':<10} {'Dentro 10%':<10}\n")
        f.write("-" * 80 + "\n")

        for trait in sorted(results.keys()):
            result = results[trait]
            f.write(f"{trait:<20} {result['mae']:<10.2f} {result['r2']:<10.4f} {result['mean_rel_error']:<10.2f}% ")
            f.write(f"{result['within_5pct']:<10.1f}% {result['within_10pct']:<10.1f}%\n")

        f.write("\nCaracterísticas seleccionadas por modelo:\n")
        f.write("=" * 60 + "\n\n")

        for trait in sorted(selected_features.keys()):
            f.write(f"{trait}:\n")
            for feature in selected_features[trait]:
                f.write(f"  - {feature}\n")
            f.write("\n")

    print(f"\nResumen detallado guardado en: {summary_path}")

# Asegúrate de ejecutar la función principal cuando se ejecuta el script
if __name__ == "__main__":
    main()