
import torch
import numpy as np
import joblib
import os
import open3d as o3d

# Rutas a modelos y archivos auxiliares
MODEL_FOLDER = "/content/drive/MyDrive/deep_learning_models_20250515_0007"
MODEL_PATH = os.path.join(MODEL_FOLDER, "deep_learning_model.pth")
FEATURE_SCALER_PATH = os.path.join(MODEL_FOLDER, "feature_scaler_dl.pkl")
METRIC_SCALER_PATH = os.path.join(MODEL_FOLDER, "metric_scaler_dl.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_FOLDER, "feature_names_dl.csv")
METRIC_NAMES_PATH = os.path.join(MODEL_FOLDER, "metric_names_dl.csv")

# Cargar nombres de características y métricas
import pandas as pd
feature_names = pd.read_csv(FEATURE_NAMES_PATH)['feature'].tolist()
metric_names = pd.read_csv(METRIC_NAMES_PATH)['metric'].tolist()

# Función para extraer características de una nube de puntos
def extract_features(ply_file):
    """Extrae características de una nube de puntos PLY"""
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

        # Añadir características avanzadas de PCA
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
            print(f"Advertencia: Error calculando características PCA: {e}")

        return features
    except Exception as e:
        print(f"Error procesando {ply_file}: {e}")
        return None

# Clase de red neuronal (igual a la original)
class DeepCowNet(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.3):
        super(DeepCowNet, self).__init__()
        
        # Primera capa - extracción de características de bajo nivel
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(dropout_rate)
        )
        
        # Segunda capa - procesamiento intermedio
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(dropout_rate)
        )
        
        # Tercera capa - representación latente
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(dropout_rate)
        )
        
        # Cuarta capa - refinamiento
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(dropout_rate/2)
        )
        
        # Capa de salida - predicción multitarea
        self.output_layer = torch.nn.Linear(32, output_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x

# Función principal para predecir métricas
def predict_metrics(ply_file):
    """
    Predice métricas para una vaca a partir de su nube de puntos.
    
    Args:
        ply_file: Ruta al archivo PLY con la nube de puntos
        
    Returns:
        Un diccionario con las métricas predichas
    """
    try:
        # Verificar archivo
        if not os.path.exists(ply_file):
            print(f"Error: El archivo {ply_file} no existe")
            return None
            
        # Extraer características
        features = extract_features(ply_file)
        
        if features is None:
            print("Error extrayendo características")
            return None
            
        # Cargar modelo y scalers
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        input_size = checkpoint['input_size']
        output_size = checkpoint['output_size']
        target_metrics = checkpoint['target_metrics']
        
        model = DeepCowNet(input_size, output_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        metric_scaler = joblib.load(METRIC_SCALER_PATH)
        
        # Preparar características en el orden correcto
        X = []
        for name in feature_names:
            if name in features:
                X.append(features[name])
            else:
                X.append(0)  # Valor default
                
        X = np.array(X).reshape(1, -1)
        
        # Escalar características
        X_scaled = feature_scaler.transform(X)
        
        # Convertir a tensor y predecir
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            y_pred_scaled = model(X_tensor).numpy()
            
        # Desnormalizar predicciones
        y_pred = metric_scaler.inverse_transform(y_pred_scaled)
        
        # Crear diccionario de resultados
        predictions = {}
        for i, metric in enumerate(metric_names):
            predictions[metric] = float(y_pred[0, i])
            
        return predictions
            
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return None
        
# Ejemplo de uso
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predecir métricas bovinas a partir de nube de puntos')
    parser.add_argument('ply_file', help='Ruta al archivo PLY')
    
    args = parser.parse_args()
    
    print(f"Procesando {args.ply_file}...")
    predictions = predict_metrics(args.ply_file)
    
    if predictions:
        print("\nResultados de la predicción:")
        print("-" * 50)
        for metric, value in sorted(predictions.items()):
            print(f"{metric:<30} {value:.2f}")
    else:
        print("No se pudieron obtener predicciones")
