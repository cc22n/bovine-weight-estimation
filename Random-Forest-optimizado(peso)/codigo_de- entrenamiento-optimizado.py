import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel, RFECV, RFE
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import randint, uniform

# Definir rutas
DRIVE_PATH = '/content/drive/MyDrive'
MODELS_FOLDER = f"{DRIVE_PATH}/modelos_ml_optimizado_{datetime.now().strftime('%Y%m%d_%H%M')}"
PREV_MODEL_FOLDER = f"{DRIVE_PATH}/Random-Forest-peso-sin-optimizar"  # Ajustar según la carpeta del modelo anterior

# Crear carpeta para modelos si no existe
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Función para cargar datos procesados del experimento anterior
def cargar_datos_procesados():
    """Carga los datos procesados y el modelo anterior"""
    print("Cargando datos procesados y modelos previos...")

    # Cargar features extraídos
    features_path = os.path.join(PREV_MODEL_FOLDER, 'caracteristicas_extraidas.csv')
    features_df = pd.read_csv(features_path)
    print(f"Cargadas {len(features_df)} muestras con {len(features_df.columns)-2} características")

    # Cargar modelo anterior para comparación
    model_path = os.path.join(PREV_MODEL_FOLDER, 'random_forest_model.pkl')
    prev_model = joblib.load(model_path)
    print("Modelo anterior cargado correctamente")

    # Cargar scaler si existe
    scaler_path = os.path.join(PREV_MODEL_FOLDER, 'feature_scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Scaler cargado correctamente")
    else:
        scaler = None
        print("No se encontró scaler previo")

    return features_df, prev_model, scaler

# Función para preparar datos para modelado
def preparar_datos(features_df, test_size=0.2, random_state=42):
    """Prepara los datos para entrenamiento y validación"""

    # Separar características y objetivo
    X = features_df.drop(['id', 'peso'], axis=1)
    y = features_df['peso']

    # Realizar división estratificada (basada en intervalos de peso)
    from sklearn.model_selection import train_test_split

    # Crear bins para estratificación
    bins = pd.qcut(y, q=5, labels=False, duplicates='drop')

    # División train/test estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=bins
    )

    print(f"División de datos: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba")

    return X_train, X_test, y_train, y_test, X.columns

# 1. Análisis de características y selección
def analizar_caracteristicas(X_train, y_train, X_test, y_test, columnas):
    """Analiza y selecciona las características más importantes"""
    print("Analizando importancia de características...")

    # Crear modelo base para análisis
    base_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Ajustar modelo
    base_model.fit(X_train, y_train)

    # Obtener importancia de características
    importances = base_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Crear DataFrame para visualización
    feature_importance_df = pd.DataFrame({
        'Característica': columnas[indices],
        'Importancia': importances[indices]
    })

    # Visualizar importancia de características
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importancia', y='Característica', data=feature_importance_df)
    plt.title('Importancia de Características - Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FOLDER, 'importancia_caracteristicas.png'))

    # Guardar importancias en CSV
    feature_importance_df.to_csv(
        os.path.join(MODELS_FOLDER, 'importancia_caracteristicas.csv'),
        index=False
    )

    print("Importancia de características guardada en CSV y visualizada")

    # Implementar diferentes métodos de selección de características
    print("\nAplicando diferentes métodos de selección de características...")

    # 1. Método basado en umbral de importancia
    print("\n1. Selección por umbral de importancia")
    threshold_selector = SelectFromModel(base_model, threshold='mean')
    threshold_selector.fit(X_train, y_train)

    # Obtener máscara de características seleccionadas
    threshold_support = threshold_selector.get_support()
    threshold_features = columnas[threshold_support]

    print(f"Características seleccionadas (umbral): {len(threshold_features)}/{len(columnas)}")
    print(threshold_features.tolist())

    # Evaluar rendimiento con estas características
    X_train_threshold = threshold_selector.transform(X_train)
    X_test_threshold = threshold_selector.transform(X_test)

    threshold_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    threshold_model.fit(X_train_threshold, y_train)

    threshold_pred = threshold_model.predict(X_test_threshold)
    threshold_mae = mean_absolute_error(y_test, threshold_pred)
    threshold_r2 = r2_score(y_test, threshold_pred)

    rel_errors = 100 * np.abs(threshold_pred - y_test) / y_test
    threshold_within_10pct = np.sum(rel_errors <= 10) / len(rel_errors) * 100

    print(f"Rendimiento con selección por umbral:")
    print(f"  MAE: {threshold_mae:.2f} kg")
    print(f"  R²: {threshold_r2:.4f}")
    print(f"  Dentro del 10%: {threshold_within_10pct:.1f}%")

    # 2. Eliminación recursiva de características (RFE)
    print("\n2. Eliminación recursiva de características (RFE)")

    # Determinar número óptimo de características con validación cruzada
    min_features_to_select = max(3, len(columnas) // 3)  # Al menos 3 o un tercio

    # Usar RFE simple en lugar de RFECV para ahorrar tiempo
    rfe = RFE(
        estimator=RandomForestRegressor(n_estimators=50, random_state=42),
        n_features_to_select=min_features_to_select,
        step=1
    )

    rfe.fit(X_train, y_train)

    # Obtener características seleccionadas
    rfe_support = rfe.support_
    rfe_features = columnas[rfe_support]

    print(f"Número de características seleccionadas: {len(rfe_features)}")
    print(f"Características seleccionadas (RFE): {len(rfe_features)}/{len(columnas)}")
    print(rfe_features.tolist())

    # Evaluar rendimiento con estas características
    X_train_rfe = X_train.iloc[:, rfe_support]
    X_test_rfe = X_test.iloc[:, rfe_support]

    rfe_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rfe_model.fit(X_train_rfe, y_train)

    rfe_pred = rfe_model.predict(X_test_rfe)
    rfe_mae = mean_absolute_error(y_test, rfe_pred)
    rfe_r2 = r2_score(y_test, rfe_pred)

    rel_errors = 100 * np.abs(rfe_pred - y_test) / y_test
    rfe_within_10pct = np.sum(rel_errors <= 10) / len(rel_errors) * 100

    print(f"Rendimiento con selección RFE:")
    print(f"  MAE: {rfe_mae:.2f} kg")
    print(f"  R²: {rfe_r2:.4f}")
    print(f"  Dentro del 10%: {rfe_within_10pct:.1f}%")

    # Resumen de métodos de selección
    comparison_df = pd.DataFrame({
        'Método': ['Todas las características', 'Selección por umbral', 'RFE'],
        'Num. Características': [len(columnas), len(threshold_features), len(rfe_features)],
        'MAE': [
            mean_absolute_error(y_test, base_model.predict(X_test)),
            threshold_mae,
            rfe_mae
        ],
        'R²': [
            r2_score(y_test, base_model.predict(X_test)),
            threshold_r2,
            rfe_r2
        ],
        'Dentro 10%': [
            np.sum(100 * np.abs(base_model.predict(X_test) - y_test) / y_test <= 10) / len(y_test) * 100,
            threshold_within_10pct,
            rfe_within_10pct
        ]
    })

    comparison_df.to_csv(
        os.path.join(MODELS_FOLDER, 'comparacion_seleccion_caracteristicas.csv'),
        index=False
    )

    print("\nComparación de métodos de selección de características:")
    print(comparison_df)

    # Determinar mejor método basado en MAE
    best_method_idx = comparison_df['MAE'].idxmin()
    best_method = comparison_df.iloc[best_method_idx]['Método']

    print(f"\nMejor método de selección: {best_method}")

    # Retornar datasets con características seleccionadas según el mejor método
    if best_method == 'Selección por umbral':
        selected_features = threshold_features
        X_train_selected = X_train_threshold
        X_test_selected = X_test_threshold
    elif best_method == 'RFE':
        selected_features = rfe_features
        X_train_selected = X_train_rfe
        X_test_selected = X_test_rfe
    else:
        selected_features = columnas
        X_train_selected = X_train
        X_test_selected = X_test

    return X_train_selected, X_test_selected, selected_features, best_method

# 2. Búsqueda de hiperparámetros (versión simplificada)
def optimizar_hiperparametros(X_train, y_train, cv=5):
    """Optimiza hiperparámetros usando búsqueda de cuadrícula y aleatoria"""
    print("\nRealizando búsqueda de hiperparámetros (versión simplificada)...")

    # Definir parámetros para la búsqueda
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    # Modelo base
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Usar directamente GridSearchCV (más pequeño para evitar errores)
    print("Ejecutando GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        verbose=1,
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )

    try:
        grid_search.fit(X_train, y_train)

        print(f"Mejores parámetros (GridSearchCV):")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        print(f"Mejor puntuación (GridSearchCV): MAE = {-grid_search.best_score_:.2f}")

        # Guardar resultados de búsqueda
        results = {
            'grid_search_best_params': grid_search.best_params_,
            'grid_search_best_score': -grid_search.best_score_
        }

        # Guardar como JSON
        import json
        with open(os.path.join(MODELS_FOLDER, 'hiperparametros_optimizados.json'), 'w') as f:
            json.dump(results, f, indent=4)

        return grid_search.best_params_

    except Exception as e:
        print(f"Error en la optimización de hiperparámetros: {e}")
        print("Usando parámetros por defecto")

        # Retornar parámetros por defecto en caso de error
        default_params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }

        return default_params

# 3. Validación cruzada extendida
def validacion_cruzada_extendida(X, y, params, n_splits=5):
    """Realiza validación cruzada extendida con los parámetros optimizados"""
    print("\nRealizando validación cruzada extendida...")
    print("Parámetros usados:", params)

    # Verificar si params es None y usar valores por defecto si es necesario
    if params is None:
        print("ADVERTENCIA: Parámetros no proporcionados, usando valores por defecto")
        params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }

    # Configurar validación cruzada
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Modelo con parámetros optimizados
    rf_optimized = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

    # Variables para almacenar métricas por fold
    mae_scores = []
    r2_scores = []
    within_10pct_scores = []

    # Para cada fold
    fold = 1
    for train_idx, test_idx in kf.split(X):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Entrenar modelo
        rf_optimized.fit(X_train_fold, y_train_fold)

        # Evaluar
        y_pred = rf_optimized.predict(X_test_fold)

        # Calcular métricas
        mae = mean_absolute_error(y_test_fold, y_pred)
        r2 = r2_score(y_test_fold, y_pred)

        rel_errors = 100 * np.abs(y_pred - y_test_fold) / y_test_fold
        within_10pct = np.sum(rel_errors <= 10) / len(rel_errors) * 100

        # Almacenar métricas
        mae_scores.append(mae)
        r2_scores.append(r2)
        within_10pct_scores.append(within_10pct)

        print(f"Fold {fold}: MAE = {mae:.2f}, R² = {r2:.4f}, Dentro 10% = {within_10pct:.1f}%")
        fold += 1

    # Calcular estadísticas de validación cruzada
    cv_stats = {
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'within_10pct_mean': np.mean(within_10pct_scores),
        'within_10pct_std': np.std(within_10pct_scores)
    }

    print("\nResultados de validación cruzada:")
    print(f"  MAE: {cv_stats['mae_mean']:.2f} ± {cv_stats['mae_std']:.2f} kg")
    print(f"  R²: {cv_stats['r2_mean']:.4f} ± {cv_stats['r2_std']:.4f}")
    print(f"  Dentro del 10%: {cv_stats['within_10pct_mean']:.1f} ± {cv_stats['within_10pct_std']:.1f}%")

    # Guardar estadísticas
    pd.DataFrame([cv_stats]).to_csv(
        os.path.join(MODELS_FOLDER, 'estadisticas_validacion_cruzada.csv'),
        index=False
    )

    # Guardar modelo final con todas las características y parámetros optimizados
    # para uso en futuras predicciones
    final_rf = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    final_rf.fit(X, y)
    joblib.dump(final_rf, os.path.join(MODELS_FOLDER, 'random_forest_final.pkl'))

    return cv_stats

# 4. Entrenamiento y evaluación del modelo final
def entrenar_modelo_final(X_train, y_train, X_test, y_test, params, features_names):
    """Entrena modelo final con parámetros optimizados y evalúa"""
    print("\nEntrenando modelo final con parámetros optimizados...")
    print("Parámetros usados:", params)

    # Verificar si params es None y usar valores por defecto si es necesario
    if params is None:
        print("ADVERTENCIA: Parámetros no proporcionados, usando valores por defecto")
        params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }

    # Crear modelo final
    rf_final = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

    # Entrenar
    rf_final.fit(X_train, y_train)

    # Evaluar en conjunto de prueba
    y_pred = rf_final.predict(X_test)

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rel_errors = 100 * np.abs(y_pred - y_test) / y_test
    mean_rel_error = np.mean(rel_errors)
    within_5pct = np.sum(rel_errors <= 5) / len(rel_errors) * 100
    within_10pct = np.sum(rel_errors <= 10) / len(rel_errors) * 100
    within_15pct = np.sum(rel_errors <= 15) / len(rel_errors) * 100

    print("\nRendimiento del modelo final:")
    print(f"  MAE: {mae:.2f} kg")
    print(f"  RMSE: {rmse:.2f} kg")
    print(f"  R²: {r2:.4f}")
    print(f"  Error relativo medio: {mean_rel_error:.2f}%")
    print(f"  Predicciones dentro del 5%: {within_5pct:.1f}%")
    print(f"  Predicciones dentro del 10%: {within_10pct:.1f}%")
    print(f"  Predicciones dentro del 15%: {within_15pct:.1f}%")

    # Guardar modelo final
    joblib.dump(rf_final, os.path.join(MODELS_FOLDER, 'random_forest_optimizado.pkl'))

    # Guardar características utilizadas
    pd.DataFrame({'feature': features_names}).to_csv(
        os.path.join(MODELS_FOLDER, 'caracteristicas_seleccionadas.csv'),
        index=False
    )

    # Visualizar predicciones vs reales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')

    # Añadir líneas para errores del 10%
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'g--', alpha=0.5, label='-10%')
    plt.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'g--', alpha=0.5, label='+10%')

    plt.xlabel('Peso Real (kg)')
    plt.ylabel('Peso Predicho (kg)')
    plt.title('Predicciones vs Valores Reales - Random Forest Optimizado')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(MODELS_FOLDER, 'predicciones_vs_reales_optimizado.png'))

    # Histograma de errores relativos
    plt.figure(figsize=(10, 6))
    plt.hist(rel_errors, bins=20, alpha=0.7)
    plt.axvline(x=10, color='r', linestyle='--', label='10% Error')
    plt.xlabel('Error Relativo (%)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Errores Relativos - Modelo Optimizado')
    plt.legend()
    plt.savefig(os.path.join(MODELS_FOLDER, 'histograma_errores_optimizado.png'))

    # Crear DataFrame con resultados
    final_results = {
        'Métrica': ['MAE', 'RMSE', 'R²', 'Error Rel. Medio', 'Dentro 5%', 'Dentro 10%', 'Dentro 15%'],
        'Valor': [mae, rmse, r2, mean_rel_error, within_5pct, within_10pct, within_15pct]
    }

    pd.DataFrame(final_results).to_csv(
        os.path.join(MODELS_FOLDER, 'resultados_modelo_optimizado.csv'),
        index=False
    )

    return rf_final, {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_rel_error': mean_rel_error,
        'within_5pct': within_5pct,
        'within_10pct': within_10pct,
        'within_15pct': within_15pct
    }

# Función para crear función de predicción sencilla
def crear_funcion_prediccion(model, selected_features):
    """Crea y guarda una función de predicción simple para uso futuro"""

    # Guardar función como script Python separado
    prediction_script = f"""
import joblib
import numpy as np
import pandas as pd

# Cargar modelo y scaler
model = joblib.load('random_forest_final.pkl')

# Definir características requeridas
REQUIRED_FEATURES = {list(selected_features)}

def predecir_peso(caracteristicas):
    \"\"\"
    Predice el peso de una vaca basado en características extraídas de nube de puntos.

    Args:
        caracteristicas: dict con valores para al menos las características requeridas

    Returns:
        float: Peso predicho en kg
    \"\"\"
    # Verificar características requeridas
    missing = [f for f in REQUIRED_FEATURES if f not in caracteristicas]
    if missing:
        raise ValueError(f"Faltan características requeridas: {{', '.join(missing)}}")

    # Crear dataframe con características
    X = pd.DataFrame([caracteristicas])[REQUIRED_FEATURES]

    # Realizar predicción
    peso_predicho = model.predict(X)[0]

    return peso_predicho

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de características extraídas
    ejemplo_caracteristicas = {{
        'length': 1.95,           # Longitud
        'width': 0.65,            # Ancho
        'height': 1.20,           # Altura
        'volume': 1.52,           # Volumen aproximado
        'surface_area': 6.8,      # Área de superficie aproximada
        'point_count': 95000,     # Número de puntos en la nube
        'density': 62500,         # Densidad de puntos
        'std_x': 0.32,            # Desviación estándar en eje X
        'std_y': 0.28,            # Desviación estándar en eje Y
        'std_z': 0.45,            # Desviación estándar en eje Z
        'avg_distance': 0.62,     # Distancia promedio al centroide
        'max_distance': 1.25,     # Distancia máxima al centroide
        'aspect_ratio_1': 3.0,    # Relación de aspecto 1
        'aspect_ratio_2': 1.6,    # Relación de aspecto 2
        'aspect_ratio_3': 0.54,   # Relación de aspecto 3
        'distance_q1': 0.35,      # Primer cuartil de distancias
        'distance_q3': 0.85,      # Tercer cuartil de distancias
    }}

    # Predecir peso
    peso = predecir_peso(ejemplo_caracteristicas)
    print(f"Peso predicho: {{peso:.2f}} kg")
"""

    # Guardar script
    with open(os.path.join(MODELS_FOLDER, 'predictor_peso.py'), 'w') as f:
        f.write(prediction_script)

    print(f"Función de predicción guardada en {os.path.join(MODELS_FOLDER, 'predictor_peso.py')}")

# Función para inferencia en nueva nube de puntos
def procesar_nueva_nube(ply_file, modelo, selected_features):
    """Procesa una nueva nube de puntos y predice su peso"""
    try:
        # Importar open3d (mejor importarlo solo cuando se necesite)
        import open3d as o3d

        print(f"Procesando archivo: {ply_file}")

        # Extraer características volumétricas
        try:
            # Cargar nube de puntos
            pcd = o3d.io.read_point_cloud(ply_file)
            points = np.asarray(pcd.points)

            if len(points) == 0:
                print(f"Error: El archivo {ply_file} no contiene puntos.")
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

            # Crear diccionario de características
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

            # Crear DataFrame con características seleccionadas
            features_df = pd.DataFrame([{f: features[f] for f in selected_features}])

            # Realizar predicción
            peso_predicho = modelo.predict(features_df)[0]

            print(f"Peso predicho: {peso_predicho:.2f} kg")

            return {
                'features': features,
                'peso_predicho': peso_predicho
            }

        except Exception as e:
            print(f"Error procesando {ply_file}: {e}")
            return None

    except ImportError:
        print("Error: No se pudo importar open3d. Instálelo con 'pip install open3d'")
        return None

# Función principal
def optimizar_random_forest():
    """Función principal que ejecuta todo el proceso de optimización"""
    print("=" * 50)
    print("OPTIMIZACIÓN DE RANDOM FOREST PARA PREDICCIÓN DE PESO DE VACAS")
    print("=" * 50)

    # 1. Cargar datos procesados
    features_df, prev_model, _ = cargar_datos_procesados()

    # 2. Preparar datos
    X_train, X_test, y_train, y_test, columnas = preparar_datos(features_df)

    # 3. Análisis y selección de características
    X_train_selected, X_test_selected, selected_features, best_selection_method = \
        analizar_caracteristicas(X_train, y_train, X_test, y_test, columnas)

    # 4. Optimización de hiperparámetros
    best_params = optimizar_hiperparametros(X_train_selected, y_train, cv=5)

    # 5. Validación cruzada extendida
    X_selected = features_df[selected_features]
    y = features_df['peso']
    cv_stats = validacion_cruzada_extendida(X_selected, y, best_params, n_splits=5)

    # 6. Entrenamiento y evaluación del modelo final
    final_model, final_metrics = entrenar_modelo_final(
        X_train_selected, y_train, X_test_selected, y_test, best_params, selected_features
    )

    # 7. Comparar con modelo anterior
    print("\nComparando con modelo anterior...")

    # Predicciones del modelo anterior
    prev_pred = prev_model.predict(X_test)

    # Métricas del modelo anterior
    prev_mae = mean_absolute_error(y_test, prev_pred)
    prev_r2 = r2_score(y_test, prev_pred)

    prev_rel_errors = 100 * np.abs(prev_pred - y_test) / y_test
    prev_within_10pct = np.sum(prev_rel_errors <= 10) / len(prev_rel_errors) * 100

    # Mejora porcentual
    mae_improvement = ((prev_mae - final_metrics['mae']) / prev_mae) * 100
    r2_improvement = ((final_metrics['r2'] - prev_r2) / abs(prev_r2)) * 100 if prev_r2 != 0 else float('inf')
    within_10pct_improvement = ((final_metrics['within_10pct'] - prev_within_10pct) / prev_within_10pct) * 100

    # Guardar comparación en un DataFrame
    comparison_df = pd.DataFrame({
        'Métrica': ['MAE (kg)', 'R²', 'Dentro 10%'],
        'Modelo Original': [prev_mae, prev_r2, prev_within_10pct],
        'Modelo Optimizado': [final_metrics['mae'], final_metrics['r2'], final_metrics['within_10pct']],
        'Mejora (%)': [mae_improvement, r2_improvement, within_10pct_improvement]
    })

    comparison_df.to_csv(
        os.path.join(MODELS_FOLDER, 'comparacion_modelos.csv'),
        index=False
    )

    print(f"Modelo anterior: MAE = {prev_mae:.2f} kg, R² = {prev_r2:.4f}, Dentro 10% = {prev_within_10pct:.1f}%")
    print(f"Modelo optimizado: MAE = {final_metrics['mae']:.2f} kg, R² = {final_metrics['r2']:.4f}, Dentro 10% = {final_metrics['within_10pct']:.1f}%")
    print(f"Mejora: MAE = {mae_improvement:.1f}%, R² = {r2_improvement:.1f}%, Dentro 10% = {within_10pct_improvement:.1f}%")

    # Crear función de predicción
    crear_funcion_prediccion(final_model, selected_features)

    # 8. Resumen final
    print("\n" + "=" * 50)
    print("RESUMEN DE OPTIMIZACIÓN")
    print("=" * 50)
    print(f"Método de selección de características: {best_selection_method}")
    print(f"Número de características seleccionadas: {len(selected_features)}")
    print(f"Características seleccionadas: {', '.join(selected_features)}")
    print("\nParámetros optimizados:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Guardar resumen en un archivo de texto
    with open(os.path.join(MODELS_FOLDER, 'resumen_optimizacion.txt'), 'w') as f:
        f.write("RESUMEN DE OPTIMIZACIÓN\n")
        f.write("=" * 50 + "\n")
        f.write(f"Método de selección de características: {best_selection_method}\n")
        f.write(f"Número de características seleccionadas: {len(selected_features)}\n")
        f.write(f"Características seleccionadas: {', '.join(selected_features)}\n\n")
        f.write("Parámetros optimizados:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\nMejora respecto al modelo anterior:\n")
        f.write(f"  MAE: {mae_improvement:.1f}%\n")
        f.write(f"  R²: {r2_improvement:.1f}%\n")
        f.write(f"  Dentro 10%: {within_10pct_improvement:.1f}%\n")

    print("\nRendimiento en validación cruzada:")
    print(f"  MAE: {cv_stats['mae_mean']:.2f} ± {cv_stats['mae_std']:.2f} kg")
    print(f"  R²: {cv_stats['r2_mean']:.4f} ± {cv_stats['r2_std']:.4f}")
    print(f"  Dentro del 10%: {cv_stats['within_10pct_mean']:.1f} ± {cv_stats['within_10pct_std']:.1f}%")

    print(f"\nTodos los resultados y modelos guardados en {MODELS_FOLDER}")

    return final_model, selected_features, best_params

if __name__ == "__main__":
    # Ejecutar optimización
    modelo_final, caracteristicas_seleccionadas, parametros_optimos = optimizar_random_forest()

    # Mostrar información para uso manual posterior
    print("\nPara usar este modelo para predecir el peso de nuevas nubes de puntos:")
    print(f"1. Utilice el script 'predictor_peso.py' en {MODELS_FOLDER}")
    print("2. O importe las funciones de este script:")
    print("   from optimizacion_rf import procesar_nueva_nube")
    print("   resultado = procesar_nueva_nube('ruta_a_nube.ply', modelo_final, caracteristicas_seleccionadas)")