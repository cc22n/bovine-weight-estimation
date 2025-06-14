import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open3d as o3d
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
from tqdm import tqdm
from google.colab import drive

# Montar Google Drive
try:
    drive.mount('/content/drive')
except:
    print("Google Drive ya está montado o hay un problema con el montaje")

# Definir rutas
DRIVE_PATH = '/content/drive/MyDrive'
PLY_FOLDER = f"{DRIVE_PATH}/nubes-de-puntos-filtradas"
EXCEL_PATH = f"{DRIVE_PATH}/Measurements.xlsx"
MODELS_FOLDER = f"{DRIVE_PATH}/modelos_ml_{datetime.now().strftime('%Y%m%d_%H%M')}"

# Crear carpeta para modelos si no existe
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Función para extraer características volumétricas básicas de nubes de puntos
def extract_volumetric_features(ply_file):
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

        return features
    except Exception as e:
        print(f"Error procesando {ply_file}: {e}")
        return None

# Función para extraer número de vaca
def extract_number(filename):
    match = re.search(r'procesada_(\d+)_AutoAligned', filename)
    if match:
        return int(match.group(1))
    return None

# Función para procesar y alinear todos los datos
def prepare_datasets(ply_folder, metrics_df):
    print("Extrayendo características de las nubes de puntos...")

    # Crear diccionario para almacenar características por número de vaca
    features_dict = {}
    weights_dict = {}

    # Procesar todas las nubes de puntos
    ply_files = [f for f in os.listdir(ply_folder) if f.endswith('.ply')]
    print(f"Encontrados {len(ply_files)} archivos PLY")

    # Primero verificar formato de las columnas
    print("Columnas disponibles en el Excel:")
    print(metrics_df.columns.tolist())

    # Determinar qué columna contiene los pesos
    weight_column = None
    for col in metrics_df.columns:
        if 'weight' in col.lower() or 'peso' in col.lower() or 'weithg' in col.lower():
            weight_column = col
            print(f"Usando columna '{weight_column}' para los pesos")
            break

    if weight_column is None:
        print("¡ADVERTENCIA! No se encontró columna de peso en el Excel")
        return None, None, None

    # Extraer características de cada nube de puntos
    feature_data = []
    weight_data = []
    ids = []

    for filename in tqdm(ply_files, desc="Procesando archivos PLY"):
        file_path = os.path.join(ply_folder, filename)
        number = extract_number(filename)

        if number is not None and number-1 < len(metrics_df):
            # Extraer características de la nube
            features = extract_volumetric_features(file_path)

            if features is not None:
                try:
                    # Obtener peso de la columna correspondiente
                    weight = float(metrics_df.iloc[number-1][weight_column])

                    # Almacenar datos
                    feature_data.append(features)
                    weight_data.append(weight)
                    ids.append(number)

                    # Imprimir información de diagnóstico para los primeros ejemplos
                    if len(feature_data) <= 3:
                        print(f"ID: {number}, Peso: {weight}, Volumen: {features['volume']:.2f}")
                except Exception as e:
                    print(f"Error procesando datos para {filename}: {e}")

    # Mostrar estadísticas de los pesos
    weights_array = np.array(weight_data)
    print("\nEstadísticas de los pesos:")
    print(f"  Min: {np.min(weights_array):.2f}")
    print(f"  Max: {np.max(weights_array):.2f}")
    print(f"  Mean: {np.mean(weights_array):.2f}")
    print(f"  Std: {np.std(weights_array):.2f}")

    # Convertir listas de diccionarios a DataFrame
    features_df = pd.DataFrame(feature_data)

    # Mostrar correlaciones con el peso
    print("\nCorrelaciones con el peso:")
    for column in features_df.columns:
        correlation = np.corrcoef(features_df[column], weights_array)[0, 1]
        print(f"  {column}: {correlation:.4f}")

    # Crear gráfico de correlaciones
    plt.figure(figsize=(12, 8))
    correlations = []
    for column in features_df.columns:
        correlation = np.corrcoef(features_df[column], weights_array)[0, 1]
        correlations.append((column, correlation))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    feature_names = [x[0] for x in correlations]
    correlation_values = [x[1] for x in correlations]

    plt.barh(feature_names, correlation_values)
    plt.xlabel('Correlación con el peso')
    plt.title('Correlaciones de características con el peso')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FOLDER, 'correlaciones.png'))

    # Guardar características extraídas
    features_with_id = features_df.copy()
    features_with_id['id'] = ids
    features_with_id['peso'] = weights_array
    features_with_id.to_csv(os.path.join(MODELS_FOLDER, 'caracteristicas_extraidas.csv'), index=False)

    # Dividir en conjuntos de entrenamiento, validación y prueba
    X = features_df.values
    y = weights_array

    # Crear un histograma de pesos
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=15)
    plt.xlabel('Peso (kg)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Pesos')
    plt.savefig(os.path.join(MODELS_FOLDER, 'distribucion_pesos.png'))

    # Dividir datos
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_test, y_test, ids_test, test_size=0.5, random_state=42
    )

    print(f"\nDivisión de datos completada: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")

    # Normalizar características (importante!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Guardar scaler para uso futuro
    import joblib
    joblib.dump(scaler, os.path.join(MODELS_FOLDER, 'feature_scaler.pkl'))

    # Guardar nombres de características
    pd.DataFrame({'feature': features_df.columns}).to_csv(
        os.path.join(MODELS_FOLDER, 'feature_names.csv'), index=False
    )

    # También retornar versiones no escaladas para algunas visualizaciones
    return (
        (X_train_scaled, y_train, ids_train),
        (X_val_scaled, y_val, ids_val),
        (X_test_scaled, y_test, ids_test),
        (X_train, X_val, X_test),  # Versiones no escaladas
        features_df.columns,        # Nombres de características
        scaler                      # Scaler entrenado
    )

# Función para entrenar modelo de Random Forest
def train_random_forest(X_train, y_train, X_val, y_val, feature_names):
    print("Entrenando modelo Random Forest...")

    # Definir hiperparámetros
    n_estimators = 100
    max_depth = 10
    min_samples_split = 2
    min_samples_leaf = 1

    # Crear y entrenar modelo
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1  # Usar todos los núcleos disponibles
    )

    model.fit(X_train, y_train)

    # Evaluar en conjunto de validación
    y_pred_val = model.predict(X_val)
    val_mae = np.mean(np.abs(y_pred_val - y_val))
    val_rmse = np.sqrt(np.mean((y_pred_val - y_val) ** 2))

    # Calcular R²
    val_r2 = 1 - np.sum((y_val - y_pred_val) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

    # Calcular errores relativos
    val_rel_errors = 100 * np.abs(y_pred_val - y_val) / y_val
    val_within_5pct = np.sum(val_rel_errors <= 5) / len(val_rel_errors) * 100
    val_within_10pct = np.sum(val_rel_errors <= 10) / len(val_rel_errors) * 100
    val_within_15pct = np.sum(val_rel_errors <= 15) / len(val_rel_errors) * 100

    print("\nResultados en validación:")
    print(f"  MAE: {val_mae:.2f} kg")
    print(f"  RMSE: {val_rmse:.2f} kg")
    print(f"  R²: {val_r2:.4f}")
    print(f"  Error relativo medio: {np.mean(val_rel_errors):.2f}%")
    print(f"  Predicciones dentro del 5%: {val_within_5pct:.1f}%")
    print(f"  Predicciones dentro del 10%: {val_within_10pct:.1f}%")
    print(f"  Predicciones dentro del 15%: {val_within_15pct:.1f}%")

    # Importancia de características
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    importance_df.to_csv(os.path.join(MODELS_FOLDER, 'feature_importance.csv'), index=False)

    # Visualizar importancia de características
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importancia')
    plt.title('Importancia de Características')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FOLDER, 'importancia_caracteristicas.png'))

    # Guardar modelo
    import joblib
    model_path = os.path.join(MODELS_FOLDER, 'random_forest_model.pkl')
    joblib.dump(model, model_path)
    print(f"Modelo guardado en {model_path}")

    return model, val_mae, val_r2, val_within_10pct

# Función para entrenar una red neuronal simple
def train_simple_neural_network(X_train, y_train, X_val, y_val, learning_rate=0.001):
    print("Entrenando red neuronal sencilla...")

    # Convertir a tensores
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)

    # Definir modelo sencillo
    input_dim = X_train.shape[1]

    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    model = SimpleNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Definir pérdida y optimizador
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Normalizar objetivo (pesos)
    y_mean = float(torch.mean(y_train_tensor))
    y_std = float(torch.std(y_train_tensor))

    y_train_normalized = (y_train_tensor - y_mean) / y_std
    y_val_normalized = (y_val_tensor - y_mean) / y_std

    print(f"Normalizando pesos: Media: {y_mean:.2f}, Std: {y_std:.2f}")

    # Entrenar modelo
    num_epochs = 100
    batch_size = 16
    best_val_loss = float('inf')
    patience = 15
    counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # Entrenamiento por mini-lotes
        indices = torch.randperm(X_train_tensor.shape[0])
        for start_idx in range(0, X_train_tensor.shape[0], batch_size):
            idx = indices[start_idx:start_idx + batch_size]

            inputs = X_train_tensor[idx].to(device)
            targets = y_train_normalized[idx].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward y optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        epoch_loss /= X_train_tensor.shape[0]
        train_losses.append(epoch_loss)

        # Evaluación
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            val_loss = criterion(val_outputs, y_val_normalized.to(device)).item()
            val_losses.append(val_loss)

            # Desnormalizar para obtener métricas
            val_pred = val_outputs.cpu() * y_std + y_mean

            val_mae = torch.mean(torch.abs(val_pred - y_val_tensor)).item()
            rel_errors = 100 * torch.abs(val_pred - y_val_tensor) / y_val_tensor
            within_10pct = torch.sum(rel_errors <= 10).item() / len(rel_errors) * 100

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "\
                  f"MAE: {val_mae:.2f} kg, Within 10%: {within_10pct:.1f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_FOLDER, 'best_simple_nn.pth'))
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping en época {epoch+1}")
                break

    # Cargar mejor modelo
    model.load_state_dict(torch.load(os.path.join(MODELS_FOLDER, 'best_simple_nn.pth')))

    # Graficar curvas de pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(MODELS_FOLDER, 'nn_loss_curves.png'))

    # Evaluación final
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor.to(device))
        val_pred = (val_outputs.cpu() * y_std + y_mean).numpy().flatten()

        val_mae = np.mean(np.abs(val_pred - y_val))
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        val_r2 = 1 - np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

        rel_errors = 100 * np.abs(val_pred - y_val) / y_val
        within_5pct = np.sum(rel_errors <= 5) / len(rel_errors) * 100
        within_10pct = np.sum(rel_errors <= 10) / len(rel_errors) * 100
        within_15pct = np.sum(rel_errors <= 15) / len(rel_errors) * 100

    print("\nResultados finales en validación (NN):")
    print(f"  MAE: {val_mae:.2f} kg")
    print(f"  RMSE: {val_rmse:.2f} kg")
    print(f"  R²: {val_r2:.4f}")
    print(f"  Error relativo medio: {np.mean(rel_errors):.2f}%")
    print(f"  Predicciones dentro del 5%: {within_5pct:.1f}%")
    print(f"  Predicciones dentro del 10%: {within_10pct:.1f}%")
    print(f"  Predicciones dentro del 15%: {within_15pct:.1f}%")

    # Para usar el modelo más adelante
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'y_mean': y_mean,
        'y_std': y_std
    }
    torch.save(model_info, os.path.join(MODELS_FOLDER, 'simple_nn_model_info.pth'))

    return model, val_mae, val_r2, within_10pct

# Función para evaluar los modelos en conjunto de prueba
def evaluate_models(rf_model, nn_model, X_test, y_test, ids_test, y_mean, y_std, nn_input_dim, unscaled_X_test):
    print("\nEvaluando modelos en conjunto de prueba...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Predicciones de Random Forest
    rf_preds = rf_model.predict(X_test)

    # Carga y predicciones de la red neuronal
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    nn_model = SimpleNN(nn_input_dim)
    nn_model.load_state_dict(torch.load(os.path.join(MODELS_FOLDER, 'best_simple_nn.pth')))
    nn_model = nn_model.to(device)
    nn_model.eval()

    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        nn_preds_normalized = nn_model(X_test_tensor)
        nn_preds = (nn_preds_normalized.cpu() * y_std + y_mean).numpy().flatten()

    # Cálculo de métricas para Random Forest
    rf_mae = np.mean(np.abs(rf_preds - y_test))
    rf_rmse = np.sqrt(np.mean((rf_preds - y_test) ** 2))
    rf_r2 = 1 - np.sum((y_test - rf_preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    rf_rel_errors = 100 * np.abs(rf_preds - y_test) / y_test
    rf_within_5pct = np.sum(rf_rel_errors <= 5) / len(rf_rel_errors) * 100
    rf_within_10pct = np.sum(rf_rel_errors <= 10) / len(rf_rel_errors) * 100
    rf_within_15pct = np.sum(rf_rel_errors <= 15) / len(rf_rel_errors) * 100

    # Cálculo de métricas para red neuronal
    nn_mae = np.mean(np.abs(nn_preds - y_test))
    nn_rmse = np.sqrt(np.mean((nn_preds - y_test) ** 2))
    nn_r2 = 1 - np.sum((y_test - nn_preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    nn_rel_errors = 100 * np.abs(nn_preds - y_test) / y_test
    nn_within_5pct = np.sum(nn_rel_errors <= 5) / len(nn_rel_errors) * 100
    nn_within_10pct = np.sum(nn_rel_errors <= 10) / len(nn_rel_errors) * 100
    nn_within_15pct = np.sum(nn_rel_errors <= 15) / len(nn_rel_errors) * 100

    # Modelo ensemble (promedio de predicciones)
    ensemble_preds = (rf_preds + nn_preds) / 2

    ensemble_mae = np.mean(np.abs(ensemble_preds - y_test))
    ensemble_rmse = np.sqrt(np.mean((ensemble_preds - y_test) ** 2))
    ensemble_r2 = 1 - np.sum((y_test - ensemble_preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    ensemble_rel_errors = 100 * np.abs(ensemble_preds - y_test) / y_test
    ensemble_within_5pct = np.sum(ensemble_rel_errors <= 5) / len(ensemble_rel_errors) * 100
    ensemble_within_10pct = np.sum(ensemble_rel_errors <= 10) / len(ensemble_rel_errors) * 100
    ensemble_within_15pct = np.sum(ensemble_rel_errors <= 15) / len(ensemble_rel_errors) * 100

    # Imprimir resultados
    print("\nResultados en conjunto de prueba (Random Forest):")
    print(f"  MAE: {rf_mae:.2f} kg")
    print(f"  RMSE: {rf_rmse:.2f} kg")
    print(f"  R²: {rf_r2:.4f}")
    print(f"  Error relativo medio: {np.mean(rf_rel_errors):.2f}%")
    print(f"  Predicciones dentro del 5%: {rf_within_5pct:.1f}%")
    print(f"  Predicciones dentro del 10%: {rf_within_10pct:.1f}%")
    print(f"  Predicciones dentro del 15%: {rf_within_15pct:.1f}%")

    print("\nResultados en conjunto de prueba (Red Neuronal):")
    print(f"  MAE: {nn_mae:.2f} kg")
    print(f"  RMSE: {nn_rmse:.2f} kg")
    print(f"  R²: {nn_r2:.4f}")
    print(f"  Error relativo medio: {np.mean(nn_rel_errors):.2f}%")
    print(f"  Predicciones dentro del 5%: {nn_within_5pct:.1f}%")
    print(f"  Predicciones dentro del 10%: {nn_within_10pct:.1f}%")
    print(f"  Predicciones dentro del 15%: {nn_within_15pct:.1f}%")

    print("\nResultados en conjunto de prueba (Modelo Ensemble):")
    print(f"  MAE: {ensemble_mae:.2f} kg")
    print(f"  RMSE: {ensemble_rmse:.2f} kg")
    print(f"  R²: {ensemble_r2:.4f}")
    print(f"  Error relativo medio: {np.mean(ensemble_rel_errors):.2f}%")
    print(f"  Predicciones dentro del 5%: {ensemble_within_5pct:.1f}%")
    print(f"  Predicciones dentro del 10%: {ensemble_within_10pct:.1f}%")
    print(f"  Predicciones dentro del 15%: {ensemble_within_15pct:.1f}%")

    # Crear DataFrame con todas las predicciones
    results_df = pd.DataFrame({
        'id': ids_test,
        'peso_real': y_test,
        'pred_rf': rf_preds,
        'pred_nn': nn_preds,
        'pred_ensemble': ensemble_preds,
        'error_rf': rf_rel_errors,
        'error_nn': nn_rel_errors,
        'error_ensemble': ensemble_rel_errors
    })

    results_df.to_csv(os.path.join(MODELS_FOLDER, 'resultados_test.csv'), index=False)

    # Gráfico de dispersión
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(y_test, rf_preds, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Peso Real (kg)')
    plt.ylabel('Predicción RF (kg)')
    plt.title('Random Forest')

    plt.subplot(1, 3, 2)
    plt.scatter(y_test, nn_preds, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Peso Real (kg)')
    plt.ylabel('Predicción NN (kg)')
    plt.title('Red Neuronal')

    plt.subplot(1, 3, 3)
    plt.scatter(y_test, ensemble_preds, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Peso Real (kg)')
    plt.ylabel('Predicción Ensemble (kg)')
    plt.title('Modelo Ensemble')

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_FOLDER, 'predicciones_vs_reales.png'))

    # Boxplot de errores relativos
    plt.figure(figsize=(10, 6))
    plt.boxplot([rf_rel_errors, nn_rel_errors, ensemble_rel_errors],
                labels=['Random Forest', 'Red Neuronal', 'Ensemble'])
    plt.ylabel('Error Relativo (%)')
    plt.title('Comparación de Errores Relativos por Modelo')
    plt.axhline(y=10, color='r', linestyle='--', label='10% Error')
    plt.legend()
    plt.savefig(os.path.join(MODELS_FOLDER, 'boxplot_errores.png'))

    return {
        'rf': {
            'mae': rf_mae,
            'rmse': rf_rmse,
            'r2': rf_r2,
            'within_10pct': rf_within_10pct
        },
        'nn': {
            'mae': nn_mae,
            'rmse': nn_rmse,
            'r2': nn_r2,
            'within_10pct': nn_within_10pct
        },
        'ensemble': {
            'mae': ensemble_mae,
            'rmse': ensemble_rmse,
            'r2': ensemble_r2,
            'within_10pct': ensemble_within_10pct
        }
    }

# Función principal
def main():
    print("Iniciando procesamiento de datos de peso de vacas...")

    # Cargar datos de métricas
    print(f"Cargando métricas desde {EXCEL_PATH}")
    try:
        metrics_df = pd.read_excel(EXCEL_PATH)
        print(f"Cargadas {len(metrics_df)} filas de datos")
    except Exception as e:
        print(f"Error cargando el archivo Excel: {e}")
        return

    # Preparar datasets
    datasets = prepare_datasets(PLY_FOLDER, metrics_df)
    if datasets is None:
        print("Error preparando los datasets. Abortando.")
        return

    (X_train, y_train, ids_train), \
    (X_val, y_val, ids_val), \
    (X_test, y_test, ids_test), \
    (X_train_unscaled, X_val_unscaled, X_test_unscaled), \
    feature_names, scaler = datasets

    # Entrenar Random Forest
    rf_model, rf_val_mae, rf_val_r2, rf_val_within_10pct = train_random_forest(
        X_train, y_train, X_val, y_val, feature_names
    )

    # Calcular media y desviación de pesos para normalización
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))

    # Entrenar red neuronal simple
    nn_model, nn_val_mae, nn_val_r2, nn_val_within_10pct = train_simple_neural_network(
        X_train, y_train, X_val, y_val, learning_rate=0.001
    )

    # Evaluar modelos
    test_results = evaluate_models(
        rf_model, nn_model, X_test, y_test, ids_test,
        y_mean, y_std, X_train.shape[1], X_test_unscaled
    )

    # Generar informe final
    print("\n" + "="*50)
    print("INFORME FINAL - PREDICCIÓN DE PESO DE VACAS")
    print("="*50)

    print("\nMejor modelo: ", end="")
    best_model = max(
        ['rf', 'nn', 'ensemble'],
        key=lambda x: test_results[x]['within_10pct']
    )

    if best_model == 'rf':
        print("Random Forest")
    elif best_model == 'nn':
        print("Red Neuronal")
    else:
        print("Ensemble (Combinación)")

    print(f"\nEstadísticas del mejor modelo ({best_model}):")
    print(f"  MAE: {test_results[best_model]['mae']:.2f} kg")
    print(f"  R²: {test_results[best_model]['r2']:.4f}")
    print(f"  Predicciones dentro del 10%: {test_results[best_model]['within_10pct']:.1f}%")

    print("\nComparación de modelos (% dentro del 10% de error):")
    print(f"  Random Forest: {test_results['rf']['within_10pct']:.1f}%")
    print(f"  Red Neuronal: {test_results['nn']['within_10pct']:.1f}%")
    print(f"  Ensemble: {test_results['ensemble']['within_10pct']:.1f}%")

    print(f"\nTodos los resultados guardados en {MODELS_FOLDER}")

if __name__ == "__main__":
    main()