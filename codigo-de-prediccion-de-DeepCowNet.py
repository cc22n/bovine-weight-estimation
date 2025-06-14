# Primero montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Importar librerías necesarias
import os
import sys

# Definir la ruta al directorio de modelos en Google Drive
# Cambia esto a la ruta específica donde guardaste tus modelos
MODELS_FOLDER = "/content/drive/MyDrive/deep_learning_models_(DeepCowNet)"  # Actualizar con la fecha correcta

# Agregar la carpeta de modelos al path para poder importar el script de predicción
sys.path.append(MODELS_FOLDER)

# Ahora puedes importar la función de predicción
from predict_metrics import predict_metrics

# Definir la carpeta donde están tus nubes de puntos para predecir
NUBES_PREDECIR_FOLDER = "/content/drive/MyDrive/nubes-de-puntos-filtradas"  # Cambia esto a tu carpeta de nubes

# Verificar que la carpeta existe
if not os.path.exists(NUBES_PREDECIR_FOLDER):
    print(f"¡Error! La carpeta {NUBES_PREDECIR_FOLDER} no existe.")
    # Opcional: crear la carpeta si no existe
    os.makedirs(NUBES_PREDECIR_FOLDER)
    print(f"Se ha creado la carpeta {NUBES_PREDECIR_FOLDER}. Por favor, coloca tus archivos PLY allí.")
else:
    # Listar todos los archivos PLY en la carpeta
    ply_files = [f for f in os.listdir(NUBES_PREDECIR_FOLDER) if f.endswith('.ply')]

    if not ply_files:
        print(f"No se encontraron archivos PLY en {NUBES_PREDECIR_FOLDER}.")
        print("Asegúrate de colocar tus nubes de puntos 3D (archivos .ply) en esta carpeta.")
    else:
        print(f"Se encontraron {len(ply_files)} archivos PLY:")
        for i, file in enumerate(ply_files):
            print(f"{i+1}. {file}")

        # Crear un dataframe para almacenar todos los resultados
        import pandas as pd
        results_df = pd.DataFrame()

        # Procesar cada nube de puntos
        print("\nProcesando nubes de puntos...")
        for i, file in enumerate(ply_files):
            ply_path = os.path.join(NUBES_PREDECIR_FOLDER, file)
            print(f"\nProcesando {i+1}/{len(ply_files)}: {file}")

            # Predecir métricas
            predictions = predict_metrics(ply_path)

            if predictions:
                # Mostrar predicciones
                print("Predicciones:")
                for metric, value in predictions.items():
                    print(f"  {metric}: {value:.2f}")

                # Agregar al dataframe
                predictions['filename'] = file
                results_row = pd.DataFrame([predictions])
                results_df = pd.concat([results_df, results_row], ignore_index=True)
            else:
                print(f"No se pudieron obtener predicciones para {file}")

        # Si se procesaron archivos con éxito, guardar resultados en CSV
        if not results_df.empty:
            # Reorganizar columnas para que filename sea la primera
            cols = ['filename'] + [col for col in results_df.columns if col != 'filename']
            results_df = results_df[cols]

            # Guardar resultados
            results_path = os.path.join(NUBES_PREDECIR_FOLDER, 'predicciones_metricas_bovinas.csv')
            results_df.to_csv(results_path, index=False)
            print(f"\nResultados guardados en: {results_path}")

            # Mostrar el dataframe
            print("\nResumen de predicciones:")
            print(results_df)