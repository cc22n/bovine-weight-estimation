# Modelo de Aprendizaje Profundo para Predicción de Métricas Bovinas

Este directorio contiene modelos de deep learning entrenados para predecir diversas métricas bovinas a partir de nubes de puntos 3D.

## Resumen del Proyecto

- Se utilizaron **86** nubes de puntos originales
- Se generaron variaciones sutiles para aumentar el conjunto de datos
- Se entrenó una red neuronal profunda para predecir múltiples métricas

## Métricas Predichas

Peso vivo (kg), Altura a la cruz (cm), Altura a la cadera (cm), Profundidad de pecho (cm), Ancho de pecho (cm), Ancho de ilion (cm), Ancho de articulación de cadera (cm), Longitud oblicua del cuerpo (cm), Longitud de cadera (cm), Perímetro torácico (cm)

## Uso del Modelo

Para utilizar el modelo con nuevas nubes de puntos:

```python
# Ejemplo de uso
from predict_metrics import predict_metrics

# Predecir métricas
predictions = predict_metrics('ruta/a/nube_puntos.ply')

# Mostrar predicciones
for metric, value in predictions.items():
    print(f"{metric}: {value:.2f}")
```