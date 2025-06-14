# bovine-weight-estimation
Este proyecto implementa un sistema basado en aprendizaje automático y redes neuronales profundas para la predicción de métricas morfométricas de ganado bovino (peso, altura, perímetro torácico, etc.) utilizando nubes de puntos 3D generadas a partir de cámaras de profundidad.

📌 Descripción del Proyecto
A partir de un conjunto de datos tridimensionales públicos, este sistema:

Extrae más de 26 características geométricas de nubes de puntos.

Aplica técnicas de aumento de datos tridimensional.

Entrena modelos Random Forest (utilizados para generar etiquetas sintéticas a partir de la función de aumento de datos, con el fin de entrenar modelos más robustos de aprendizaje profundo), así como los modelos MLP (DeepCowNet) y PointNet para realizar predicciones precisas.

Evalúa los modelos con métricas como MAE, RMSE, R², error relativo promedio, y porcentaje de predicciones con errores menores al 5% y 10%.

Exporta modelos entrenados listos para su uso en nuevas muestras.

El entorno de desarrollo fue Google Colab, y todo el código está preparado para su ejecución con mínima configuración (solo es necesario ajustar rutas a los archivos).

🔍 Métricas Predichas
Los modelos desarrollados permiten predecir de forma automática las siguientes métricas corporales:

Peso (WT)

Altura a la cruz (WH)

Altura de cadera (HH)

Profundidad de pecho (CD)

Perímetro torácico (HG)

Ancho de ilion (IW)

Ancho de cadera (HJW)

Largo oblicuo del cuerpo (OBL)

Largo de la cadera (HL)

Ancho del pecho (CW)

🧠 Metodología General
El desarrollo del sistema se estructuró según el enfoque CRISP-DM, ampliamente adoptado en ciencia de datos:

Comprensión del negocio: Automatizar la predicción de métricas corporales a partir de datos 3D para apoyar la ganadería de precisión.

Comprensión de los datos: Se utilizaron 103 nubes de puntos del repositorio CowDatabase, filtradas manualmente para obtener 86 modelos utilizables.

Preparación de datos: Extracción de características geométricas y aumento de datos 3D (rotación, escalado, compresión, deformación, ruido).

Modelado: Se entrenaron:

Modelos Random Forest para generar etiquetas sintéticas.

Una red MLP (DeepCowNet) con datos numéricos.

Una red PointNet, que trabaja directamente con nubes de puntos sin conversión previa.

Evaluación: Comparativa entre modelos clásicos y redes profundas, observando mejoras claras en precisión.

Implementación: Modelos y scripts listos para ser utilizados en nuevas predicciones.

🗂️ Base de Datos: CowDatabase
Este sistema se construyó utilizando la base de datos abierta CowDatabase, que contiene datos RGB-D, mapas de profundidad y nubes de puntos 3D de 103 vacas Hereford, recolectados en Rusia. Además, incluye mediciones manuales expertas de 10 métricas corporales.

Las mediciones disponibles son:

Altura a la cruz (WH)

Altura de cadera (HH)

Profundidad de pecho (CD)

Perímetro torácico (HG)

Ancho de ilion (IW)

Ancho de cadera (HJW)

Largo oblicuo del cuerpo (OBL)

Largo de la cadera (HL)

Ancho del pecho (CW)

Peso estimado (WT)

📚 Referencia del artículo original
Alexey Ruchay, Vitaly Kober, Konstantin Dorofeev, Vladimir Kolpakov, Sergei Miroshnikov.
Accurate body measurement of live cattle using three depth cameras and non-rigid 3-D shape recovery
Computers and Electronics in Agriculture, Volume 179, 2020, 105821.
https://doi.org/10.1016/j.compag.2020.105821
Enlace al artículo

bibtex
Copiar
Editar
@article{RUCHAY2020105821,
  title = "Accurate body measurement of live cattle using three depth cameras and non-rigid 3-D shape recovery",
  journal = "Computers and Electronics in Agriculture",
  volume = "179",
  pages = "105821",
  year = "2020",
  issn = "0168-1699",
  doi = "https://doi.org/10.1016/j.compag.2020.105821",
  url = "http://www.sciencedirect.com/science/article/pii/S0168169920321256",
  author = "Alexey Ruchay and Vitaly Kober and Konstantin Dorofeev and Vladimir Kolpakov and Sergei Miroshnikov"
}
Repositorio del dataset:
https://github.com/ruchaya/CowDatabase
