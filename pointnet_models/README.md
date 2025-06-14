# Modelo PointNet para Predicción de Métricas Bovinas

Este directorio contiene modelos de PointNet entrenados para predecir diversas métricas bovinas a partir de nubes de puntos 3D.

## Resumen del Proyecto

- Se utilizaron **{len(ply_files)}** nubes de puntos originales
- Se generaron variaciones sutiles para aumentar el conjunto de datos
- Se entrenó una red neuronal PointNet para predecir múltiples métricas utilizando directamente las nubes de puntos 3D

## Métricas Predichas

{', '.join(readable_metrics)}

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
    print(f"{metric}: {value:.2f}")
```