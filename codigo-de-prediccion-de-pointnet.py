# Este notebook carga los modelos de PointNet y procesa nubes de puntos PLY para
# predecir métricas bovinas, guardando los resultados en un archivo CSV.

# Primero montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Importar librerías necesarias
import os
import sys
import numpy as np
import pandas as pd
import torch
import open3d as o3d
import joblib
import re
from tqdm import tqdm
from datetime import datetime

# Definir la ruta al directorio de modelos PointNet en Google Drive
POINTNET_MODELS_FOLDER = "/content/drive/MyDrive/pointnet_models"  # Carpeta con modelos PointNet

# Definir la carpeta donde están las nubes de puntos para predecir
PLY_FOLDER = "/content/drive/MyDrive/nubes-de-puntos-filtradas"  # Carpeta con nubes para predecir
RESULTS_FOLDER = f"/content/drive/MyDrive/resultados_predicciones-pointnet3_{datetime.now().strftime('%Y%m%d_%H%M')}"

# Crear carpetas necesarias
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
    print(f"Carpeta creada: {RESULTS_FOLDER}")

# Verificar si existen las rutas importantes
print(f"PLY_FOLDER existe: {os.path.exists(PLY_FOLDER)}")
print(f"POINTNET_MODELS_FOLDER existe: {os.path.exists(POINTNET_MODELS_FOLDER)}")

# Definir rutas a modelos y archivos auxiliares
MODEL_PATH = os.path.join(POINTNET_MODELS_FOLDER, "pointnet_model.pth")
METRIC_SCALER_PATH = os.path.join(POINTNET_MODELS_FOLDER, "metric_scaler_dl.pkl")
METRIC_NAMES_PATH = os.path.join(POINTNET_MODELS_FOLDER, "metric_names_dl.csv")

# Verificar existencia de archivos de modelo
print(f"MODEL_PATH existe: {os.path.exists(MODEL_PATH)}")
print(f"METRIC_SCALER_PATH existe: {os.path.exists(METRIC_SCALER_PATH)}")
print(f"METRIC_NAMES_PATH existe: {os.path.exists(METRIC_NAMES_PATH)}")

# Definir función para extraer número de vaca
def extract_number(filename):
    """Extrae el número de identificación de la vaca del nombre del archivo"""
    match = re.search(r'procesada_(\d+)_AutoAligned', filename)
    if match:
        return int(match.group(1))
    return None

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
    """Extrae y preprocesa nube de puntos para PointNet"""
    try:
        # Cargar nube de puntos
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)

        if len(points) == 0:
            print(f"Advertencia: {ply_file} no contiene puntos.")
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
        print(f"Error procesando {ply_file}: {e}")
        return None

# Función para predecir métricas con PointNet
def predict_with_pointnet(ply_file):
    """Predice métricas para una vaca a partir de su nube de puntos usando PointNet"""
    try:
        # Verificar archivo
        if not os.path.exists(ply_file):
            print(f"Error: El archivo {ply_file} no existe")
            return None

        # Extraer nube de puntos
        points = extract_point_cloud(ply_file)

        if points is None:
            print("Error extrayendo puntos")
            return None

        # Cargar nombres de métricas
        if os.path.exists(METRIC_NAMES_PATH):
            metric_names = pd.read_csv(METRIC_NAMES_PATH)['metric'].tolist()
        else:
            print(f"Error: No se encontró el archivo de nombres de métricas: {METRIC_NAMES_PATH}")
            return None

        # Cargar modelo y scaler
        if not os.path.exists(MODEL_PATH):
            print(f"Error: No se encontró el modelo: {MODEL_PATH}")
            return None

        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

        # Verificar que el checkpoint tenga la información necesaria
        if 'output_size' not in checkpoint:
            print("Error: El checkpoint no contiene 'output_size'")
            # Inferir del número de métricas
            output_size = len(metric_names)
            print(f"Usando número de métricas como output_size: {output_size}")
        else:
            output_size = checkpoint['output_size']

        # Inicializar modelo
        model = PointNet(output_size)

        # Verificar si existe model_state_dict en el checkpoint
        if 'model_state_dict' not in checkpoint:
            print("Error: El checkpoint no contiene 'model_state_dict'")
            return None

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Cargar scaler para desnormalizar predicciones
        if not os.path.exists(METRIC_SCALER_PATH):
            print(f"Error: No se encontró el scaler: {METRIC_SCALER_PATH}")
            return None

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
        predictions = {}
        for i, metric in enumerate(metric_names):
            if i < y_pred.shape[1]:  # Asegurarse de que no se exceda el índice
                predictions[metric] = float(y_pred[0, i])

        return predictions

    except Exception as e:
        print(f"Error en la predicción con PointNet: {e}")
        import traceback
        traceback.print_exc()
        return None

# Ejecutar el proceso de predicción para todas las nubes de puntos
print("\n==== INICIANDO PROCESO DE PREDICCIÓN CON POINTNET ====")

# Listar archivos PLY en la carpeta
ply_files = [f for f in os.listdir(PLY_FOLDER) if f.endswith('.ply')]
print(f"Se encontraron {len(ply_files)} archivos PLY para procesar")

if len(ply_files) == 0:
    print(f"No hay archivos PLY para procesar en {PLY_FOLDER}")
else:
    # Crear un dataframe para almacenar todos los resultados
    results_df = pd.DataFrame()

    # Procesar cada nube de puntos
    for i, file in enumerate(tqdm(ply_files, desc="Procesando PLYs")):
        ply_path = os.path.join(PLY_FOLDER, file)
        print(f"\nProcesando {i+1}/{len(ply_files)}: {file}")

        # Predecir métricas
        predictions = predict_with_pointnet(ply_path)

        if predictions:
            # Agregar información del archivo
            predictions['archivo'] = file
            predictions['id_vaca'] = extract_number(file)

            # Agregar al dataframe
            results_row = pd.DataFrame([predictions])
            results_df = pd.concat([results_df, results_row], ignore_index=True)
        else:
            print(f"No se pudieron obtener predicciones para {file}")

    # Si se procesaron archivos con éxito, guardar resultados en CSV
    if not results_df.empty:
        # Reorganizar columnas para mejor legibilidad
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

        # Usar solo columnas que existen en el DataFrame
        col_order = [c for c in col_order if c in results_df.columns]

        # Reordenar y guardar
        results_df = results_df[col_order]

        # Guardar resultados
        csv_path = os.path.join(RESULTS_FOLDER, f'predicciones_pointnet_{datetime.now().strftime("%Y%m%d_%H%M")}.csv')
        results_df.to_csv(csv_path, index=False)

        print(f"\n¡Procesamiento completado!")
        print(f"Se procesaron {len(results_df)} archivos PLY")
        print(f"Resultados guardados en: {csv_path}")

        # Mostrar estadísticas básicas
        print("\nEstadísticas de predicciones:")
        for metric in [m for m in important_metrics if m in results_df.columns]:
            print(f"{metric}:")
            print(f"  Media: {results_df[metric].mean():.2f}")
            print(f"  Mín: {results_df[metric].min():.2f}")
            print(f"  Máx: {results_df[metric].max():.2f}")
            print()

        # Mostrar el dataframe
        print("\nResumen de predicciones:")
        print(results_df.head())
    else:
        print("No se pudieron generar predicciones para ningún archivo.")