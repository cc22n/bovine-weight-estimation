
import joblib
import numpy as np
import pandas as pd

# Cargar modelo y scaler
model = joblib.load('random_forest_final.pkl')

# Definir características requeridas
REQUIRED_FEATURES = ['width', 'point_count', 'max_distance']

def predecir_peso(caracteristicas):
    """
    Predice el peso de una vaca basado en características extraídas de nube de puntos.
    
    Args:
        caracteristicas: dict con valores para al menos las características requeridas
        
    Returns:
        float: Peso predicho en kg
    """
    # Verificar características requeridas
    missing = [f for f in REQUIRED_FEATURES if f not in caracteristicas]
    if missing:
        raise ValueError(f"Faltan características requeridas: {', '.join(missing)}")
    
    # Crear dataframe con características
    X = pd.DataFrame([caracteristicas])[REQUIRED_FEATURES]
    
    # Realizar predicción
    peso_predicho = model.predict(X)[0]
    
    return peso_predicho

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de características extraídas
    ejemplo_caracteristicas = {
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
    }
    
    # Predecir peso
    peso = predecir_peso(ejemplo_caracteristicas)
    print(f"Peso predicho: {peso:.2f} kg")
