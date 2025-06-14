
import numpy as np
import pandas as pd
import joblib
import os
import argparse

# Ruta de modelos
MODELS_FOLDER = "/content/drive/MyDrive/modelos_multiatributos_20250508_0425"

# Cargar modelos, selectores y scaler
print("Cargando modelos...")
models = {}
selectors = {}
for trait in ['altura_cruz', 'altura_cadera', 'profundidad_pecho', 'ancho_pecho', 'ancho_ilion', 'ancho_articulacion_cadera', 'longitud_oblicua', 'longitud_cadera', 'circunferencia_torax']:
    model_path = os.path.join(MODELS_FOLDER, f'modelo_{trait}.pkl')
    selector_path = os.path.join(MODELS_FOLDER, f'selector_{trait}.pkl')

    if os.path.exists(model_path) and os.path.exists(selector_path):
        models[trait] = joblib.load(model_path)
        selectors[trait] = joblib.load(selector_path)

# Cargar scaler
scaler = joblib.load(os.path.join(MODELS_FOLDER, 'scaler.pkl'))

def extract_features_from_point_cloud(ply_file):
    """Extrae características de una nube de puntos"""
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

def predict_traits(ply_file):
    """Predice varias características a partir de una nube de puntos"""
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
        print("\nResultados de la predicción:")
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
