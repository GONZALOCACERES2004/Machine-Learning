## Aplicación de Machine Learning para el Análisis Predictivo de Precios en Alojamientos de Airbnb

Este estudio exhaustivo sobre los factores que influyen en los precios de los alojamientos de Airbnb ha desarrollado y comparado varios modelos predictivos, culminando en un modelo Random Forest optimizado con las 30 características más relevantes. El análisis proporciona insights valiosos tanto para anfitriones como para la plataforma Airbnb.

## Principales hallazgos

### Rendimiento del modelo
El modelo Random Forest final, con 30 características seleccionadas, logró un R² de 0.7126 en el conjunto de prueba, explicando aproximadamente el 71% de la variabilidad en los precios. Con un RMSE de 22.80 euros en el conjunto de prueba, el modelo demuestra una sólida capacidad predictiva para una variable compleja como el precio de alojamientos.

### Factores clave que influyen en el precio
- **Características del alojamiento**: Tipo de habitación, capacidad, número de habitaciones, camas y baños.
- **Ubicación**: Latitud, longitud y proximidad al centro de la ciudad.
- **Costos adicionales**: Tarifa de limpieza y depósito de seguridad.
- **Reseñas y puntuaciones**: Calificación general y específica, especialmente la ubicación.
- **Disponibilidad**: La disponibilidad a largo plazo (365 días) es más influyente que la de corto plazo.

### Ingeniería de características
La creación de características derivadas como 'Bathrooms_per_person', 'Beds_per_person', y 'Bedrooms_squared' mejoró significativamente la capacidad predictiva del modelo.

### Comparación de modelos
Se exploraron varios modelos, incluyendo Lasso, Bagging, y Random Forest. El Random Forest con características seleccionadas demostró el mejor equilibrio entre rendimiento e interpretabilidad.

### Importancia de la selección de características
La reducción de 75 a 30 características simplificó el modelo y mejoró su capacidad de generalización, reduciendo el sobreajuste.

## Implicaciones prácticas

### Para anfitriones
- Enfocarse en mejorar aspectos clave como capacidad, limpieza y comodidades puede justificar precios más altos.
- La ubicación es crucial; propiedades cercanas al centro pueden comandar precios premium.
- Mantener altas puntuaciones en reseñas, especialmente en ubicación, puede influir positivamente en el precio.

### Para Airbnb
- El modelo puede ser la base para un sistema de recomendación de precios para anfitriones.
- Los insights sobre ubicación y características del alojamiento pueden informar estrategias de marketing y expansión.

### Para usuarios
- Comprender los factores que influyen en el precio puede ayudar a los huéspedes a tomar decisiones más informadas.

## Limitaciones y trabajo futuro
- El modelo actual muestra cierto grado de sobreajuste, sugiriendo que podría beneficiarse de más datos o técnicas de regularización adicionales.
- La dinámica del mercado de Airbnb puede cambiar con el tiempo, requiriendo actualizaciones periódicas del modelo.
- Futuros estudios podrían explorar la segmentación del mercado o incorporar datos externos como eventos locales o tendencias turísticas.

En resumen, este análisis proporciona una comprensión profunda de los factores que influyen en los precios de Airbnb, ofreciendo una herramienta valiosa para la toma de decisiones y sentando las bases para futuros análisis y mejoras en la predicción de precios en el mercado de alojamientos de corto plazo.
