# CONCLUSION

En este estudio exhaustivo sobre los factores que influyen en los precios de los alojamientos de Airbnb en Madrid, hemos desarrollado y comparado varios modelos predictivos, culminando en un modelo Random Forest optimizado que utiliza las 30 características más relevantes. Este análisis ha proporcionado insights valiosos.

Principales hallazgos:
1.	Rendimiento del modelo:
El modelo Random Forest final, utilizando 30 características cuidadosamente seleccionadas, logró un R² de 0.7126 en el conjunto de prueba, explicando aproximadamente el 71% de la variabilidad en los precios. Con un RMSE de 22.80 euros en el conjunto de prueba, el modelo demuestra una capacidad predictiva sólida para una variable tan compleja como el precio de alojamientos.
2.	Factores clave que influyen en el precio:
a) Características del alojamiento: El tipo de habitación, la capacidad de alojamiento, y el número de habitaciones, camas y baños son factores cruciales.
b) Ubicación: Tanto la latitud y longitud como la proximidad al centro de la ciudad son determinantes significativos del precio.
c) Costos adicionales: La tarifa de limpieza y el depósito de seguridad tienen un impacto considerable en el precio final.
d) Reseñas y puntuaciones: La calificación general y específica (especialmente la ubicación) influyen en el precio.
e) Disponibilidad: La disponibilidad a largo plazo (365 días) es más influyente que la disponibilidad a corto plazo.
3.	Ingeniería de características:
La creación de características derivadas como 'Bathrooms_per_person', 'Beds_per_person', y 'Bedrooms_squared' mejoró significativamente la capacidad predictiva del modelo, capturando relaciones no lineales entre las variables.
4.	Comparación de modelos:
A lo largo del análisis, se exploraron varios modelos, incluyendo Lasso, Bagging, y Random Forest. El Random Forest con características seleccionadas demostró el mejor equilibrio entre interpretabilidad y rendimiento.
5.	Importancia de la selección de características:
La reducción del número de características de 75 a 30 no solo simplificó el modelo, sino que también mejoró su capacidad de generalización, reduciendo el sobreajuste observado en modelos anteriores.

Se adjunta código en python, y gráficas con diferentes pruebas realizadas, así como las utilizadas en los análisis durante el proceso.
