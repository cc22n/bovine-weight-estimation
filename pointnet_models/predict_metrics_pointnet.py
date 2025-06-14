
import torch
import numpy as np
import joblib
import os
import open3d as o3d

# Rutas a modelos y archivos auxiliares
MODEL_FOLDER = "{MODELS_FOLDER}"
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

# Función principal para predecir métricas
def predict_metrics(ply_file):
    """
    Predice métricas para una vaca a partir de su nube de puntos usando PointNet.
    
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
    
    parser = argparse.ArgumentParser(description='Predecir métricas bovinas a partir de nube de puntos usando PointNet')
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
