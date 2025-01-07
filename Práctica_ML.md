# Práctica de ML con Python
### Estimación del precio del airbnb 

En este notebook haremos un análisis exploratorio básico de la base de datos de airbnb, para familiarizarnos con los datos y posteriormente aplicar técnicas de machine learning sobre ellos. 




Vamos a utilizar **DataFrames** de [Pandas](http://pandas.pydata.org/). Como es sabido, Pandas es un módulo de python de código abierto para el análisis de datos, que proporciona estructuras de datos fáciles de utilizar.


```python
import pandas as pd               # Para manipulación de datos
import numpy as np                # Para operaciones numéricas
import matplotlib.pyplot as plt    # Para visualización de datos
from sklearn.model_selection import train_test_split  # Para dividir los datos
from sklearn.preprocessing import StandardScaler     # Para escalar características
from sklearn.model_selection import GridSearchCV    #Para ajuste de hiperparámetros
from sklearn.linear_model import Lasso                 # Modelo de Lasso
from sklearn.ensemble import BaggingRegressor    #Modelo de Baggin Regressor
from sklearn.tree import DecisionTreeRegressor   #Estimador base para BaggingRegressor
from sklearn.ensemble import RandomForestRegressor    # Modelo de Random Forest
from sklearn.metrics import mean_squared_error, r2_score  # Para evaluación del modelo
from sklearn.model_selection import KFold         # Para validación cruzada kFold
%matplotlib inline
```

## 1. Carga de datos y división train/test

Hay que tener mucho cuidado a la hora de realizar la división, para no cometer data leakage. Vamos a  mirar el dataset, eliminar todas aquellas columnas que sabemos que se pueden quitar (ids, URLs, etc) y a continuación dividiremos en train/test para evitar riesgos.


```python
# Cargar el dataset
airbnb_data = pd.read_csv('airbnb-listings-extract.csv', delimiter=';')

# Mostrar las dimensiones del DataFrame
print(f"\nDimensiones del DataFrame: {airbnb_data.shape}")

# Mostrar las primeras filas y la información del DataFrame
airbnb_data.head(5).T

```

    
    Dimensiones del DataFrame: (14780, 89)
    


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ID</th>
      <td>11210388</td>
      <td>17471131</td>
      <td>17584891</td>
      <td>5398030</td>
      <td>18104606</td>
    </tr>
    <tr>
      <th>Listing Url</th>
      <td>https://www.airbnb.com/rooms/11210388</td>
      <td>https://www.airbnb.com/rooms/17471131</td>
      <td>https://www.airbnb.com/rooms/17584891</td>
      <td>https://www.airbnb.com/rooms/5398030</td>
      <td>https://www.airbnb.com/rooms/18104606</td>
    </tr>
    <tr>
      <th>Scrape ID</th>
      <td>20170306202425</td>
      <td>20170407214050</td>
      <td>20170407214050</td>
      <td>20170407214050</td>
      <td>20170407214050</td>
    </tr>
    <tr>
      <th>Last Scraped</th>
      <td>2017-03-07</td>
      <td>2017-04-08</td>
      <td>2017-04-08</td>
      <td>2017-04-08</td>
      <td>2017-04-08</td>
    </tr>
    <tr>
      <th>Name</th>
      <td>The Loft-Full Bath-Deck w/View</td>
      <td>Claris I, Friendly Rentals</td>
      <td>Style Terrace Red, Friendly Rentals</td>
      <td>Picasso Suite 1.4 Paseo de Gracia</td>
      <td>Smart City Centre Apartment II</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Cancellation Policy</th>
      <td>moderate</td>
      <td>super_strict_30</td>
      <td>super_strict_30</td>
      <td>strict</td>
      <td>flexible</td>
    </tr>
    <tr>
      <th>Calculated host listings count</th>
      <td>1.0</td>
      <td>106.0</td>
      <td>106.0</td>
      <td>24.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>Reviews per Month</th>
      <td>3.5</td>
      <td>0.86</td>
      <td>NaN</td>
      <td>1.09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Geolocation</th>
      <td>30.3373609355,-97.8632766782</td>
      <td>41.3896829422,2.17262543017</td>
      <td>41.3930345489,2.16217327868</td>
      <td>41.3969668101,2.1674178103</td>
      <td>41.3886851936,2.15514963616</td>
    </tr>
    <tr>
      <th>Features</th>
      <td>Host Is Superhost,Host Has Profile Pic,Host Id...</td>
      <td>Host Has Profile Pic,Requires License,Instant ...</td>
      <td>Host Has Profile Pic,Requires License,Instant ...</td>
      <td>Host Has Profile Pic,Host Identity Verified,Re...</td>
      <td>Host Has Profile Pic,Host Identity Verified,Is...</td>
    </tr>
  </tbody>
</table>
<p>89 rows × 5 columns</p>
</div>




```python
print("\nInformación del DataFrame:")
print(airbnb_data.info())
```

    
    Información del DataFrame:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14780 entries, 0 to 14779
    Data columns (total 89 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   ID                              14780 non-null  int64  
     1   Listing Url                     14780 non-null  object 
     2   Scrape ID                       14780 non-null  int64  
     3   Last Scraped                    14780 non-null  object 
     4   Name                            14779 non-null  object 
     5   Summary                         14189 non-null  object 
     6   Space                           10888 non-null  object 
     7   Description                     14774 non-null  object 
     8   Experiences Offered             14780 non-null  object 
     9   Neighborhood Overview           9134 non-null   object 
     10  Notes                           5644 non-null   object 
     11  Transit                         9066 non-null   object 
     12  Access                          8318 non-null   object 
     13  Interaction                     8226 non-null   object 
     14  House Rules                     9619 non-null   object 
     15  Thumbnail Url                   11960 non-null  object 
     16  Medium Url                      11960 non-null  object 
     17  Picture Url                     14761 non-null  object 
     18  XL Picture Url                  11960 non-null  object 
     19  Host ID                         14780 non-null  int64  
     20  Host URL                        14780 non-null  object 
     21  Host Name                       14777 non-null  object 
     22  Host Since                      14777 non-null  object 
     23  Host Location                   14737 non-null  object 
     24  Host About                      9539 non-null   object 
     25  Host Response Time              12881 non-null  object 
     26  Host Response Rate              12881 non-null  float64
     27  Host Acceptance Rate            39 non-null     object 
     28  Host Thumbnail Url              14777 non-null  object 
     29  Host Picture Url                14777 non-null  object 
     30  Host Neighbourhood              10904 non-null  object 
     31  Host Listings Count             14777 non-null  float64
     32  Host Total Listings Count       14777 non-null  float64
     33  Host Verifications              14771 non-null  object 
     34  Street                          14780 non-null  object 
     35  Neighbourhood                   9551 non-null   object 
     36  Neighbourhood Cleansed          14780 non-null  object 
     37  Neighbourhood Group Cleansed    13760 non-null  object 
     38  City                            14774 non-null  object 
     39  State                           14636 non-null  object 
     40  Zipcode                         14274 non-null  object 
     41  Market                          14723 non-null  object 
     42  Smart Location                  14780 non-null  object 
     43  Country Code                    14780 non-null  object 
     44  Country                         14779 non-null  object 
     45  Latitude                        14780 non-null  float64
     46  Longitude                       14780 non-null  float64
     47  Property Type                   14780 non-null  object 
     48  Room Type                       14780 non-null  object 
     49  Accommodates                    14780 non-null  int64  
     50  Bathrooms                       14725 non-null  float64
     51  Bedrooms                        14755 non-null  float64
     52  Beds                            14731 non-null  float64
     53  Bed Type                        14780 non-null  object 
     54  Amenities                       14610 non-null  object 
     55  Square Feet                     598 non-null    float64
     56  Price                           14763 non-null  float64
     57  Weekly Price                    3590 non-null   float64
     58  Monthly Price                   3561 non-null   float64
     59  Security Deposit                6256 non-null   float64
     60  Cleaning Fee                    8687 non-null   float64
     61  Guests Included                 14780 non-null  int64  
     62  Extra People                    14780 non-null  int64  
     63  Minimum Nights                  14780 non-null  int64  
     64  Maximum Nights                  14780 non-null  int64  
     65  Calendar Updated                14780 non-null  object 
     66  Has Availability                12 non-null     object 
     67  Availability 30                 14780 non-null  int64  
     68  Availability 60                 14780 non-null  int64  
     69  Availability 90                 14780 non-null  int64  
     70  Availability 365                14780 non-null  int64  
     71  Calendar last Scraped           14780 non-null  object 
     72  Number of Reviews               14780 non-null  int64  
     73  First Review                    11618 non-null  object 
     74  Last Review                     11617 non-null  object 
     75  Review Scores Rating            11476 non-null  float64
     76  Review Scores Accuracy          11454 non-null  float64
     77  Review Scores Cleanliness       11460 non-null  float64
     78  Review Scores Checkin           11443 non-null  float64
     79  Review Scores Communication     11460 non-null  float64
     80  Review Scores Location          11440 non-null  float64
     81  Review Scores Value             11439 non-null  float64
     82  License                         349 non-null    object 
     83  Jurisdiction Names              227 non-null    object 
     84  Cancellation Policy             14780 non-null  object 
     85  Calculated host listings count  14776 non-null  float64
     86  Reviews per Month               11618 non-null  float64
     87  Geolocation                     14780 non-null  object 
     88  Features                        14779 non-null  object 
    dtypes: float64(23), int64(13), object(53)
    memory usage: 10.0+ MB
    None
    

Primero, eliminaremos las columnas que no son relevantes para la predicción del precio. o que podrían causar data leakage.


```python
# Definir las columnas a eliminar con sus justificaciones
columns_to_drop = {
    "Identificadores y URLs: No aportan información predictiva para el precio.": [
        'ID', 'Listing Url', 'Scrape ID', 'Host ID', 'Host URL', 
        'Thumbnail Url', 'Medium Url', 'Picture Url', 'XL Picture Url', 
        'Host Thumbnail Url', 'Host Picture Url'
    ],
    "Información temporal no relevante para la predicción": [
        'Last Scraped', 'Host Since', 'Calendar Updated', 
        'Calendar last Scraped', 'First Review', 'Last Review'
    ],
    "Texto largo y descripciones: Requieren procesamiento de lenguaje natural avanzado": [
        'Name', 'Summary', 'Space', 'Description', 'Neighborhood Overview', 
        'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Host About'
    ],
    "Información del host no directamente relacionada con el precio": [
        'Host Name', 'Host Location', 'Host Response Time', 'Host Response Rate', 
        'Host Acceptance Rate', 'Host Neighbourhood', 'Host Listings Count', 
        'Host Total Listings Count', 'Host Verifications'
    ],
    "Información geográfica redundante o muy específica: Ya tenemos otras columnas que capturan la ubicación": [
        'Street', 'Smart Location', 'Geolocation'
    ],
    "Columnas con muy pocos datos o irrelevantes para el precio": [
        'Experiences Offered', 'Has Availability', 'License', 'Jurisdiction Names'
    ],
    "Métricas derivadas que podrían causar data leakage": [
        'Calculated host listings count', 'Reviews per Month'
    ],
    "Columna que requiere procesamiento adicional y podría contener información ya capturada": [
        'Features'
    ]
}

# Imprimir justificaciones y contar columnas
total_columns = 0
print("Justificación para la eliminación de columnas:")
for category, cols in columns_to_drop.items():
    print(f"\n{category} ({len(cols)} columnas):")
    for col in cols:
        print(f"  - {col}")
    total_columns += len(cols)

print(f"\nTotal de columnas a eliminar: {total_columns}")

# Crear una lista plana de todas las columnas a eliminar
columns_to_drop_flat = [col for cols in columns_to_drop.values() for col in cols]
```

    Justificación para la eliminación de columnas:
    
    Identificadores y URLs: No aportan información predictiva para el precio. (11 columnas):
      - ID
      - Listing Url
      - Scrape ID
      - Host ID
      - Host URL
      - Thumbnail Url
      - Medium Url
      - Picture Url
      - XL Picture Url
      - Host Thumbnail Url
      - Host Picture Url
    
    Información temporal no relevante para la predicción (6 columnas):
      - Last Scraped
      - Host Since
      - Calendar Updated
      - Calendar last Scraped
      - First Review
      - Last Review
    
    Texto largo y descripciones: Requieren procesamiento de lenguaje natural avanzado (11 columnas):
      - Name
      - Summary
      - Space
      - Description
      - Neighborhood Overview
      - Notes
      - Transit
      - Access
      - Interaction
      - House Rules
      - Host About
    
    Información del host no directamente relacionada con el precio (9 columnas):
      - Host Name
      - Host Location
      - Host Response Time
      - Host Response Rate
      - Host Acceptance Rate
      - Host Neighbourhood
      - Host Listings Count
      - Host Total Listings Count
      - Host Verifications
    
    Información geográfica redundante o muy específica: Ya tenemos otras columnas que capturan la ubicación (3 columnas):
      - Street
      - Smart Location
      - Geolocation
    
    Columnas con muy pocos datos o irrelevantes para el precio (4 columnas):
      - Experiences Offered
      - Has Availability
      - License
      - Jurisdiction Names
    
    Métricas derivadas que podrían causar data leakage (2 columnas):
      - Calculated host listings count
      - Reviews per Month
    
    Columna que requiere procesamiento adicional y podría contener información ya capturada (1 columnas):
      - Features
    
    Total de columnas a eliminar: 47
    


```python
# Eliminar las columnas irrelevantes
airbnb_data_cleaned = airbnb_data.drop(columns=columns_to_drop_flat)
```

Dividiremos los datos en entrenamiento y prueba utilizando el método `train_test_split`  para obtener dos subconjuntos: train y test.


```python
train, test = train_test_split(airbnb_data_cleaned, test_size=0.2, shuffle=True, random_state=0)

print(f'Dimensiones del dataset de training: {train.shape}')
print(f'Dimensiones del dataset de test: {test.shape}')

# Guardamos
train.to_csv('airbnb_data_cleaned_train.csv', sep=';', decimal='.', index=False)
test.to_csv('airbnb_data_cleaned_test.csv', sep=';', decimal='.', index=False)
```

    Dimensiones del dataset de training: (11824, 42)
    Dimensiones del dataset de test: (2956, 42)
    

# A partir de este momento cargamos el dataset de train y trabajamos ÚNICAMENTE con él.


```python
airbnb_data = pd.read_csv('airbnb_data_cleaned_train.csv', sep=';', decimal='.')
```

## 2. Análisis exploratorio

Podemos analizar la estructura básica del dataset con las funciones de Pandas que ya conocemos: `describe`, `dtypes`, `shape`, etc.


```python
print(f'Dimensiones del dataset airbnb_data: {airbnb_data.shape}')
airbnb_data.head(5)
```

    Dimensiones del dataset airbnb_data: (11824, 42)
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Neighbourhood Cleansed</th>
      <th>Neighbourhood Group Cleansed</th>
      <th>City</th>
      <th>State</th>
      <th>Zipcode</th>
      <th>Market</th>
      <th>Country Code</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>...</th>
      <th>Availability 365</th>
      <th>Number of Reviews</th>
      <th>Review Scores Rating</th>
      <th>Review Scores Accuracy</th>
      <th>Review Scores Cleanliness</th>
      <th>Review Scores Checkin</th>
      <th>Review Scores Communication</th>
      <th>Review Scores Location</th>
      <th>Review Scores Value</th>
      <th>Cancellation Policy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jerónimos</td>
      <td>Jerónimos</td>
      <td>Retiro</td>
      <td>Madrid</td>
      <td>Comunidad de Madrid</td>
      <td>28014</td>
      <td>Madrid</td>
      <td>ES</td>
      <td>Spain</td>
      <td>40.407732</td>
      <td>...</td>
      <td>117</td>
      <td>12</td>
      <td>95.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>moderate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Sol</td>
      <td>Centro</td>
      <td>Madrid</td>
      <td>Comunidad de Madrid</td>
      <td>28012</td>
      <td>Madrid</td>
      <td>ES</td>
      <td>Spain</td>
      <td>40.415802</td>
      <td>...</td>
      <td>208</td>
      <td>20</td>
      <td>91.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>flexible</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Carabanchel</td>
      <td>Vista Alegre</td>
      <td>Carabanchel</td>
      <td>Madrid</td>
      <td>Comunidad de Madrid</td>
      <td>28025</td>
      <td>Madrid</td>
      <td>ES</td>
      <td>Spain</td>
      <td>40.389048</td>
      <td>...</td>
      <td>140</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>moderate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Embajadores</td>
      <td>Centro</td>
      <td>Madrid</td>
      <td>Comunidad de Madrid</td>
      <td>28012</td>
      <td>Madrid</td>
      <td>ES</td>
      <td>Spain</td>
      <td>40.412814</td>
      <td>...</td>
      <td>311</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>strict</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gaztambide</td>
      <td>Gaztambide</td>
      <td>Chamberí</td>
      <td>Madrid</td>
      <td>28</td>
      <td>28015</td>
      <td>Madrid</td>
      <td>ES</td>
      <td>Spain</td>
      <td>40.438631</td>
      <td>...</td>
      <td>337</td>
      <td>97</td>
      <td>92.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>strict</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>




```python
airbnb_data.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Accommodates</th>
      <th>Bathrooms</th>
      <th>Bedrooms</th>
      <th>Beds</th>
      <th>Square Feet</th>
      <th>Price</th>
      <th>Weekly Price</th>
      <th>Monthly Price</th>
      <th>...</th>
      <th>Availability 90</th>
      <th>Availability 365</th>
      <th>Number of Reviews</th>
      <th>Review Scores Rating</th>
      <th>Review Scores Accuracy</th>
      <th>Review Scores Cleanliness</th>
      <th>Review Scores Checkin</th>
      <th>Review Scores Communication</th>
      <th>Review Scores Location</th>
      <th>Review Scores Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11824.000000</td>
      <td>11824.000000</td>
      <td>11824.000000</td>
      <td>11780.000000</td>
      <td>11804.00000</td>
      <td>11787.000000</td>
      <td>474.000000</td>
      <td>11809.000000</td>
      <td>2881.000000</td>
      <td>2869.000000</td>
      <td>...</td>
      <td>11824.000000</td>
      <td>11824.000000</td>
      <td>11824.000000</td>
      <td>9163.000000</td>
      <td>9143.000000</td>
      <td>9148.000000</td>
      <td>9136.000000</td>
      <td>9147.000000</td>
      <td>9133.000000</td>
      <td>9132.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.491628</td>
      <td>-3.776863</td>
      <td>3.277486</td>
      <td>1.285229</td>
      <td>1.34429</td>
      <td>2.049122</td>
      <td>396.489451</td>
      <td>73.712592</td>
      <td>378.437348</td>
      <td>1432.390728</td>
      <td>...</td>
      <td>39.803958</td>
      <td>202.217185</td>
      <td>22.664834</td>
      <td>91.628179</td>
      <td>9.410040</td>
      <td>9.320726</td>
      <td>9.623905</td>
      <td>9.647863</td>
      <td>9.534655</td>
      <td>9.211345</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.701030</td>
      <td>14.014695</td>
      <td>2.093973</td>
      <td>0.664691</td>
      <td>0.90518</td>
      <td>1.623489</td>
      <td>671.125823</td>
      <td>71.624844</td>
      <td>194.751472</td>
      <td>1236.992934</td>
      <td>...</td>
      <td>29.663314</td>
      <td>128.006830</td>
      <td>38.092338</td>
      <td>9.137614</td>
      <td>0.938013</td>
      <td>1.004472</td>
      <td>0.804050</td>
      <td>0.765450</td>
      <td>0.770421</td>
      <td>0.963131</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-37.851182</td>
      <td>-123.124429</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>70.000000</td>
      <td>250.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>40.409758</td>
      <td>-3.707538</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>34.000000</td>
      <td>220.000000</td>
      <td>720.000000</td>
      <td>...</td>
      <td>11.000000</td>
      <td>78.000000</td>
      <td>1.000000</td>
      <td>89.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>40.419331</td>
      <td>-3.700763</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>2.000000</td>
      <td>90.000000</td>
      <td>55.000000</td>
      <td>350.000000</td>
      <td>1200.000000</td>
      <td>...</td>
      <td>38.000000</td>
      <td>240.000000</td>
      <td>7.000000</td>
      <td>94.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>40.430778</td>
      <td>-3.683917</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>2.00000</td>
      <td>2.000000</td>
      <td>624.000000</td>
      <td>87.000000</td>
      <td>500.000000</td>
      <td>1750.000000</td>
      <td>...</td>
      <td>65.000000</td>
      <td>319.000000</td>
      <td>27.000000</td>
      <td>98.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>55.966912</td>
      <td>153.371427</td>
      <td>16.000000</td>
      <td>8.000000</td>
      <td>10.00000</td>
      <td>16.000000</td>
      <td>6997.000000</td>
      <td>969.000000</td>
      <td>999.000000</td>
      <td>25000.000000</td>
      <td>...</td>
      <td>90.000000</td>
      <td>365.000000</td>
      <td>356.000000</td>
      <td>100.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 28 columns</p>
</div>



# Analizaremos la información por localización, para ver la distribución de datos

Los valores sobre el número de propiedades por ciudad y porcentajes nos pueden ayudar a tomar decisiones.


```python
# Contar el número de propiedades por ciudad
city_counts = airbnb_data['City'].value_counts()

# Mostrar el número de propiedades para todas las ciudades
print("Número de propiedades por ciudad:")
print(city_counts)

# Mostrar el número total de propiedades
total_properties = city_counts.sum()
print(f"\nNúmero total de propiedades: {total_properties}")

# Calcular el porcentaje de propiedades en Madrid
madrid_percentage = (city_counts['Madrid'] / total_properties) * 100
print(f"\nPorcentaje de propiedades en Madrid: {madrid_percentage:.2f}%")

# Mostrar las ciudades con más de 5 propiedades
print("\nCiudades con más de 20 propiedades:")
print(city_counts[city_counts >=20])

```

    Número de propiedades por ciudad:
    City
    Madrid              10567
    Barcelona             235
    London                104
    Paris                  85
    Palma                  44
                        ...  
    Berlín                  1
    Mile End / Bow          1
    Dorroughby              1
    Templeogue              1
    Aravaca (Madrid)        1
    Name: count, Length: 222, dtype: int64
    
    Número total de propiedades: 11820
    
    Porcentaje de propiedades en Madrid: 89.40%
    
    Ciudades con más de 20 propiedades:
    City
    Madrid         10567
    Barcelona        235
    London           104
    Paris             85
    Palma             44
    马德里               43
    Roma              33
    Berlin            32
    Alcúdia           31
    Dublin            29
    New York          27
    Los Angeles       26
    Brooklyn          22
    Wien              20
    Name: count, dtype: int64
    

Efectivamente **10567 propiedades en Madrid** representan el **89.40%**. Vamos a quedarnos con estos datos, teniendo en cuenta que las otras ciudades no tienen una cantidad representativa para realizar estimaciones.


```python
airbnb_madrid = airbnb_data[airbnb_data['City'] == 'Madrid']
print(airbnb_madrid['City'].value_counts())
```

    City
    Madrid    10567
    Name: count, dtype: int64
    

Vamos a separar las variables numéricas y categóricas


```python
numeric_airbnb_madrid = airbnb_madrid.select_dtypes(include=[np.number])
categorical_airbnb_madrid = airbnb_madrid.select_dtypes(exclude=[np.number])
```


```python
#  Análisis de valores nulos
null_percentages = (numeric_airbnb_madrid.isnull().sum() / len(numeric_airbnb_madrid)) * 100
print("\nPorcentaje de valores nulos:")
print(null_percentages[null_percentages > 0].sort_values(ascending=False))
```

    
    Porcentaje de valores nulos:
    Square Feet                    96.091606
    Monthly Price                  74.704268
    Weekly Price                   74.477146
    Security Deposit               56.950885
    Cleaning Fee                   40.550771
    Review Scores Location         21.869973
    Review Scores Value            21.860509
    Review Scores Checkin          21.841582
    Review Scores Accuracy         21.784802
    Review Scores Communication    21.746948
    Review Scores Cleanliness      21.737485
    Review Scores Rating           21.633387
    Bathrooms                       0.378537
    Beds                            0.350147
    Bedrooms                        0.170342
    Price                           0.075707
    dtype: float64
    

# Analisis:

1. Price: Porcentaje de nulos 0.076
- Decisión: Eliminar las filas donde el valor esnulo
2. Weekly Price y Monthly Price:
Tienen  un alto porcentaje de valores nulos (75%).
- Decisión: Eliminar estas columnas, ya que probablemente son derivadas del precio diario y tienen demasiados valores faltantes.
3. Square Feet:
Tiene  un 96.09% de valores nulos.
- Decisión: Eliminar esta columna debido al alto porcentaje de valores faltantes.
4. Cleaning Fee y Security Deposit:
Tienen porcentajes de valores nulos manejables.
- Decisión: Mantener estas columnas e imputar los valores faltantes.
5. Review Scores:
Podrían ser importantes para otros aspectos del análisis.
- Decisión: Mantener estas columnas e imputar los valores faltantes.
6. Otras variables numéricas:
Mantener todas las demás variables numéricas, ya que tienen pocos o ningún valor nulo.


```python
# Eliminar filas donde Price es nulo
numeric_airbnb_madrid = numeric_airbnb_madrid.dropna(subset=['Price'])  

# Columnas a eliminar
columns_to_drop = ['Weekly Price', 'Monthly Price', 'Square Feet']
numeric_airbnb_madrid = numeric_airbnb_madrid.drop(columns=columns_to_drop)

# Imputar valores faltantes en Cleaning Fee y Security Deposit
numeric_airbnb_madrid['Cleaning Fee'] = numeric_airbnb_madrid['Cleaning Fee'].fillna(numeric_airbnb_madrid['Cleaning Fee'].median())
numeric_airbnb_madrid['Security Deposit'] = numeric_airbnb_madrid['Security Deposit'].fillna(numeric_airbnb_madrid['Security Deposit'].median())

# Imputar valores faltantes en Review Scores
review_score_columns = [col for col in numeric_airbnb_madrid if col.startswith('Review Scores')]
numeric_airbnb_madrid[review_score_columns] = numeric_airbnb_madrid[review_score_columns].fillna(numeric_airbnb_madrid[review_score_columns].mean())

# Imputar valores faltantes en las columnas restantes con pocos nulos
columns_few_nulls = ['Bathrooms', 'Bedrooms', 'Beds', 'Price']
numeric_airbnb_madrid[columns_few_nulls] = numeric_airbnb_madrid[columns_few_nulls].fillna(numeric_airbnb_madrid[columns_few_nulls].median())

# Verificar que no quedan valores nulos
print(numeric_airbnb_madrid.isnull().sum())
```

    Latitude                       0
    Longitude                      0
    Accommodates                   0
    Bathrooms                      0
    Bedrooms                       0
    Beds                           0
    Price                          0
    Security Deposit               0
    Cleaning Fee                   0
    Guests Included                0
    Extra People                   0
    Minimum Nights                 0
    Maximum Nights                 0
    Availability 30                0
    Availability 60                0
    Availability 90                0
    Availability 365               0
    Number of Reviews              0
    Review Scores Rating           0
    Review Scores Accuracy         0
    Review Scores Cleanliness      0
    Review Scores Checkin          0
    Review Scores Communication    0
    Review Scores Location         0
    Review Scores Value            0
    dtype: int64
    

## 3. Visualización (y más análisis)

Una buena práctica es intentar resumir toda la información posible de los datos. Habitualmente nos interesa saber la media y desviación estándar, posiblemente cuartiles de cada una de las variables. Esto nos permitirá, por una lado, tener una idea de cómo son las ditribuciones de cada una de las variables y por otra, nos permitirá verificar si existen datos anómalos, también conocidos como [**outliers**]. 

Además, conviene siempre hacer representaciones gráficas, que nos ofrecen, en general un mejor entendimiento de los datos.


```python
numeric_airbnb_madrid.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Accommodates</th>
      <th>Bathrooms</th>
      <th>Bedrooms</th>
      <th>Beds</th>
      <th>Price</th>
      <th>Security Deposit</th>
      <th>Cleaning Fee</th>
      <th>Guests Included</th>
      <th>...</th>
      <th>Availability 90</th>
      <th>Availability 365</th>
      <th>Number of Reviews</th>
      <th>Review Scores Rating</th>
      <th>Review Scores Accuracy</th>
      <th>Review Scores Cleanliness</th>
      <th>Review Scores Checkin</th>
      <th>Review Scores Communication</th>
      <th>Review Scores Location</th>
      <th>Review Scores Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.00000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>...</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
      <td>10559.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.420433</td>
      <td>-3.697121</td>
      <td>3.186855</td>
      <td>1.255233</td>
      <td>1.293967</td>
      <td>1.988730</td>
      <td>66.18515</td>
      <td>164.561038</td>
      <td>27.981722</td>
      <td>1.569467</td>
      <td>...</td>
      <td>39.954352</td>
      <td>205.686902</td>
      <td>23.176058</td>
      <td>91.585286</td>
      <td>9.404866</td>
      <td>9.320673</td>
      <td>9.626453</td>
      <td>9.648076</td>
      <td>9.546589</td>
      <td>9.207778</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.019807</td>
      <td>0.023272</td>
      <td>1.988657</td>
      <td>0.605364</td>
      <td>0.832286</td>
      <td>1.527163</td>
      <td>56.16416</td>
      <td>74.425089</td>
      <td>21.032267</td>
      <td>1.072429</td>
      <td>...</td>
      <td>29.317242</td>
      <td>127.067217</td>
      <td>38.374709</td>
      <td>8.068067</td>
      <td>0.829619</td>
      <td>0.892015</td>
      <td>0.698548</td>
      <td>0.665729</td>
      <td>0.671402</td>
      <td>0.853697</td>
    </tr>
    <tr>
      <th>min</th>
      <td>40.332908</td>
      <td>-3.835498</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>9.00000</td>
      <td>70.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>40.410091</td>
      <td>-3.707764</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>31.00000</td>
      <td>150.000000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>12.000000</td>
      <td>83.000000</td>
      <td>1.000000</td>
      <td>90.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>9.626453</td>
      <td>9.648076</td>
      <td>9.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>40.418455</td>
      <td>-3.701573</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>53.00000</td>
      <td>150.000000</td>
      <td>25.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>38.000000</td>
      <td>248.000000</td>
      <td>7.000000</td>
      <td>91.585286</td>
      <td>9.404866</td>
      <td>9.320673</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>9.207778</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>40.427682</td>
      <td>-3.693877</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>80.00000</td>
      <td>150.000000</td>
      <td>30.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>65.000000</td>
      <td>320.000000</td>
      <td>28.000000</td>
      <td>97.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>40.514247</td>
      <td>-3.575142</td>
      <td>16.000000</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>16.000000</td>
      <td>875.00000</td>
      <td>990.000000</td>
      <td>500.000000</td>
      <td>16.000000</td>
      <td>...</td>
      <td>90.000000</td>
      <td>365.000000</td>
      <td>356.000000</td>
      <td>100.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 25 columns</p>
</div>




```python
#  Correlación con el precio (solo para variables numéricas)
correlation_with_price = numeric_airbnb_madrid.corr()['Price'].sort_values(ascending=False)
print("\nCorrelaciones con el precio:")
print(correlation_with_price)
```

    
    Correlaciones con el precio:
    Price                          1.000000
    Accommodates                   0.579816
    Cleaning Fee                   0.537200
    Bedrooms                       0.517584
    Beds                           0.477581
    Guests Included                0.368170
    Bathrooms                      0.343744
    Security Deposit               0.274494
    Extra People                   0.118086
    Review Scores Location         0.112262
    Availability 365               0.075401
    Latitude                       0.069960
    Review Scores Cleanliness      0.065704
    Review Scores Accuracy         0.047825
    Review Scores Rating           0.045811
    Minimum Nights                 0.034217
    Availability 30                0.018862
    Review Scores Value            0.015229
    Availability 60                0.004393
    Review Scores Communication    0.003058
    Maximum Nights                 0.002366
    Review Scores Checkin         -0.010238
    Availability 90               -0.012012
    Longitude                     -0.030818
    Number of Reviews             -0.033247
    Name: Price, dtype: float64
    

Vamos a realizar un análisis visual con tres tipos de gráficos que nos proporcionan una visión completa y detallada de las características de cada variable y su relación con el precio.


```python
# Seleccionar las columnas más relevantes para visualizar
columns_to_plot = ['Price', 'Accommodates', 'Bedrooms', 'Bathrooms', 'Cleaning Fee', 'Security Deposit','Guests Included']

# Crear una figura con subplots
fig, axs = plt.subplots(len(columns_to_plot), 3, figsize=(24, 6*len(columns_to_plot)))
fig.suptitle('Análisis de Variables de Airbnb', fontsize=16)

for i, column in enumerate(columns_to_plot):
    # Boxplot
    axs[i, 0].boxplot(numeric_airbnb_madrid[column].dropna())
    axs[i, 0].set_title(f'Boxplot de {column}')
    axs[i, 0].set_xlabel(column)
    
    # Histograma
    axs[i, 1].hist(numeric_airbnb_madrid[column].dropna(), bins=30, edgecolor='black')
    axs[i, 1].set_title(f'Histograma de {column}')
    axs[i, 1].set_xlabel(column)
    
    # Gráfico de dispersión (solo para variables que no sean 'Price')
    if column != 'Price':
        axs[i, 2].scatter(numeric_airbnb_madrid[column], numeric_airbnb_madrid['Price'], alpha=0.5)
        axs[i, 2].set_title(f'{column} vs Price')
        axs[i, 2].set_xlabel(column)
        axs[i, 2].set_ylabel('Price')
        
        # Añadir una línea de tendencia al gráfico de dispersión
        z = np.polyfit(numeric_airbnb_madrid[column], numeric_airbnb_madrid['Price'], 1)
        p = np.poly1d(z)
        axs[i, 2].plot(numeric_airbnb_madrid[column], p(numeric_airbnb_madrid[column]), "r--")
    else:
        # Para 'Price', eliminamos el tercer gráfico
        fig.delaxes(axs[i, 2])

# Ajustar el layout
plt.tight_layout()
plt.show()
```


    
![Variables numéricas](https://github.com/GONZALOCACERES2004/Machine-Learning/blob/main/Variables%20num%C3%A9ricas.png)
    


Análisis de Outliers
1. Precio (Price)
Media: $66.18
Mediana (50%): $53.
Percentil 75% : $80.
Máximo: $875.
Hay claros indicios de outliers en el precio:
- El máximo ($875) es significativamente más alto que el 75% percentil ($80).
La diferencia entre la media y la mediana sugiere una distribución sesgada con posibles outliers en el extremo superior.
2. Capacidad (Accommodates)
Media: 3.18.
Máximo: 16.
- El máximo de 16 personas podría considerarse un outlier, siendo casi 5 veces la media.
3. Baños (Bathrooms)
Media: 1.26.
Máximo: 8.
- Propiedades con 8 baños son atípicas y pueden considerarse outliers, siendo más de 6 veces la media.
4. Dormitorios (Bedrooms)
Media: 1.29.
Máximo: 10.
- Propiedades con 10 dormitorios son outliers potenciales, siendo más de 7 veces la media.
5. Número de Reseñas (Number of Reviews)
Media: 23.16.
Máximo: 356.
- Propiedades con 356 reseñas son outliers claros, siendo más de 15 veces la media.
6. Puntuaciones de Reseñas (Review Scores)
- La mayoría de las puntuaciones tienen un mínimo de 2, que podría considerarse un outlier en el extremo inferior, dado que la media está por encima de 9 para la mayoría de las categorías.

# Analizaremos la distribución de los precios


```python
# Crear bins de 100 en 100
max_price = numeric_airbnb_madrid['Price'].max()
bins = range(0, int(max_price) + 101, 100)

# Crear una serie con la distribución de precios
price_distribution = pd.cut(numeric_airbnb_madrid['Price'], bins=bins).value_counts().sort_index()

# Renombrar los índices para mayor claridad
price_distribution.index = [f'{i}-{i+99}' for i in range(0, int(max_price) + 1, 100)]

# Mostrar la distribución
print("Distribución de alojamientos por rango de precios:")
print(price_distribution)

# Calcular y mostrar el total
total = price_distribution.sum()
print(f"\nTotal de alojamientos: {total}")
```

    Distribución de alojamientos por rango de precios:
    0-99       9144
    100-199    1166
    200-299     164
    300-399      41
    400-499      26
    500-599      13
    600-699       1
    700-799       3
    800-899       1
    Name: count, dtype: int64
    
    Total de alojamientos: 10559
    

Después de realizar validaciones determinamos quedarnos con los valores menores de 300 


```python
airbnb_madrid_no_outliers = numeric_airbnb_madrid[numeric_airbnb_madrid['Price'] <= 300]

# Imprimir el número de registros eliminados

print(
    f'Original: {numeric_airbnb_madrid.shape[0]} // '
    f'Modificado: {airbnb_madrid_no_outliers.shape[0]}\nDiferencia: {numeric_airbnb_madrid.shape[0] - airbnb_madrid_no_outliers.shape[0]}'
)
print(f'Variación: {((numeric_airbnb_madrid.shape[0] - airbnb_madrid_no_outliers.shape[0])/numeric_airbnb_madrid.shape[0])*100:2f}%')

```

    Original: 10559 // Modificado: 10474
    Diferencia: 85
    Variación: 0.805000%
    

Revisemos las variables categóricas


```python
categorical_airbnb_madrid.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 10567 entries, 0 to 11823
    Data columns (total 14 columns):
     #   Column                        Non-Null Count  Dtype 
    ---  ------                        --------------  ----- 
     0   Neighbourhood                 7040 non-null   object
     1   Neighbourhood Cleansed        10567 non-null  object
     2   Neighbourhood Group Cleansed  10567 non-null  object
     3   City                          10567 non-null  object
     4   State                         10530 non-null  object
     5   Zipcode                       10220 non-null  object
     6   Market                        10528 non-null  object
     7   Country Code                  10567 non-null  object
     8   Country                       10567 non-null  object
     9   Property Type                 10567 non-null  object
     10  Room Type                     10567 non-null  object
     11  Bed Type                      10567 non-null  object
     12  Amenities                     10477 non-null  object
     13  Cancellation Policy           10567 non-null  object
    dtypes: object(14)
    memory usage: 1.2+ MB
    


```python
categorical_airbnb_madrid.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Neighbourhood Cleansed</th>
      <th>Neighbourhood Group Cleansed</th>
      <th>City</th>
      <th>State</th>
      <th>Zipcode</th>
      <th>Market</th>
      <th>Country Code</th>
      <th>Country</th>
      <th>Property Type</th>
      <th>Room Type</th>
      <th>Bed Type</th>
      <th>Amenities</th>
      <th>Cancellation Policy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7040</td>
      <td>10567</td>
      <td>10567</td>
      <td>10567</td>
      <td>10530</td>
      <td>10220</td>
      <td>10528</td>
      <td>10567</td>
      <td>10567</td>
      <td>10567</td>
      <td>10567</td>
      <td>10567</td>
      <td>10477</td>
      <td>10567</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>65</td>
      <td>125</td>
      <td>21</td>
      <td>1</td>
      <td>13</td>
      <td>68</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>5</td>
      <td>9030</td>
      <td>5</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Malasaña</td>
      <td>Embajadores</td>
      <td>Centro</td>
      <td>Madrid</td>
      <td>Comunidad de Madrid</td>
      <td>28012</td>
      <td>Madrid</td>
      <td>ES</td>
      <td>Spain</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>Real Bed</td>
      <td>TV,Internet,Wireless Internet,Air conditioning...</td>
      <td>strict</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>693</td>
      <td>1479</td>
      <td>5419</td>
      <td>10567</td>
      <td>8495</td>
      <td>1660</td>
      <td>10520</td>
      <td>10567</td>
      <td>10567</td>
      <td>8741</td>
      <td>6333</td>
      <td>10335</td>
      <td>38</td>
      <td>3984</td>
    </tr>
  </tbody>
</table>
</div>



# Análisis.
1. Neighbourhood,Neighbourhood Cleansed,Neighbourhood Group Cleansed:
Para el análisis de precios utilizaremos Neighbourhood Group Cleansed por las siguientes razones:
- Cobertura completa: Tiene datos para todos los registros, lo que evita la pérdida de información.
- Nivel de granularidad adecuado: Con 21 categorías, ofrece un buen balance entre detalle y generalización. Esto facilitará el análisis sin ser demasiado específico ni demasiado amplio.
- Datos limpios: Al ser una versión "limpia", es probable que haya pasado por un proceso de estandarización, reduciendo errores o inconsistencias.
- Facilidad de interpretación: Un número manejable de categorías permitirá una interpretación más clara de los resultados y la creación de visualizaciones más comprensibles.
2. City	State,	Zipcode, Market, Country Code, Country, Bed Type: Estas variables no son útiles para el análisis de precios por las siguientes razones:
- Falta de variabilidad: La mayoría de estas variables tienen un único valor o muy poca variación, lo que no permite diferenciar entre los alojamientos.
- Irrelevancia geográfica: Para un análisis centrado en Madrid, variables como país o comunidad autónoma no aportan información significativa.
- Redundancia: La información geográfica relevante ya está mejor capturada en la variable Neighbourhood Group Cleansed que hemos seleccionado anteriormente.
3. Property Type, Room Type, Cancellation Policy:Utilizaremos estas variables ya que ofrecen una variedad significativa de tipos de propiedades que pueden influir en el precio o son un factor crucial en la determinación del precio.
4. Amenities: La alta variabilidad hace que sea extremadamente difícil categorizar o analizar de manera efectiva sin un procesamiento extenso y  el formato de texto libre hace que sea complicado cuantificar o comparar amenidades entre propiedades de manera sistemática.


```python
list_categorical_columns=['Neighbourhood Group Cleansed','Property Type','Room Type','Cancellation Policy']

for variable in list_categorical_columns: # recorremos las variable categóricas
        plt.figure() # creamos la figura
        categorical_airbnb_madrid[variable].value_counts().sort_index().plot(kind='bar', title=variable) # rellenamos la figura con un gráfico de barras
#plot_categorical_variables(categorical_airbnb_madrid,columns_categorical)
```


    
![png]([Pr%C3%A1ctica_ML_files/Pr%C3%A1ctica_ML_39_0.png](https://github.com/GONZALOCACERES2004/Machine-Learning/blob/main/Variables%20categ%C3%B3ricas.png)
    
    
    


## 4. Generación de nuevas características

Crearemos nuevas características con el  objetivo de mejorar la precisión del modelo predictivo al capturar aspectos relevantes sobre la comodidad, el lujo y las relaciones espaciales dentro del inmueble.


```python
airbnb_data['Bathrooms_per_person'] = airbnb_data['Bathrooms'] / airbnb_data['Accommodates']
airbnb_data['Beds_per_person'] = airbnb_data['Beds'] / airbnb_data['Accommodates']
airbnb_data['Bedrooms_squared'] = airbnb_data['Bedrooms'].apply(lambda x: x**2)
airbnb_data['Bed_bath_rooms']   = airbnb_data['Bedrooms']*airbnb_data['Bathrooms']
```

## 5. Modelado, cross-validation y estudio de resultados en train y test

Antes de modelar, tenemos que cargar los datos de test y aplicar exactamente las mismas transformaciones. Es buena práctica, llegado este momento, combinar todo nuestro preprocesamiento en una única celda:


```python
# Cargar los datos de train
airbnb_data = pd.read_csv('airbnb_data_cleaned_train.csv', sep=';', decimal='.')

# 1. Filtrar los datos para mantener solo las entradas de Madrid y los precios
airbnb_data = airbnb_data[airbnb_data['City'] == 'Madrid']
airbnb_data = airbnb_data[airbnb_data['Price'] < 300]

# 2. Manejo de valores nulos
airbnb_data = airbnb_data.dropna(subset=['Price'])  # Eliminar filas donde Price es NaN

#  Si deseamos podemos aplicar el logaritmo al precio
#airbnb_data = airbnb_data[airbnb_data['Price'] > 0]  # Filtrar precios no válidos
#airbnb_data['log_price'] = np.log(airbnb_data['Price'])  # Crear columna logarítmica

# Imputar valores faltantes en Review Scores
review_score_columns = [col for col in airbnb_data.columns if col.startswith('Review Scores')]
airbnb_data[review_score_columns] = airbnb_data[review_score_columns].fillna(airbnb_data[review_score_columns].mean())

# Imputar valores nulos en columnas importantes
columns_to_impute = ['Bathrooms', 'Bedrooms', 'Beds']
for col in columns_to_impute:
    airbnb_data[col] = airbnb_data[col].fillna(airbnb_data[col].median())

# 3. Crear características derivadas
airbnb_data['Bathrooms_per_person'] = airbnb_data['Bathrooms'] / airbnb_data['Accommodates']
airbnb_data['Beds_per_person'] = airbnb_data['Beds'] / airbnb_data['Accommodates']
airbnb_data['Bedrooms_squared'] = airbnb_data['Bedrooms'].apply(lambda x: x**2)
airbnb_data['Bed_bath_rooms']   = airbnb_data['Bedrooms']*airbnb_data['Bathrooms']

# Manejar Security Deposit y Cleaning Fee
airbnb_data['Security Deposit'] = airbnb_data['Security Deposit'].fillna(0)
airbnb_data['Cleaning Fee'] = airbnb_data['Cleaning Fee'].fillna(0)
airbnb_data['Total Additional Cost'] = airbnb_data['Security Deposit'] + airbnb_data['Cleaning Fee']

# 4. Eliminar características con muchos valores nulos
airbnb_data = airbnb_data.drop(['Square Feet', 'Weekly Price', 'Monthly Price'], axis=1)

# 5. Codificar variables categóricas
# Crear variables dummy para Room Type, Property Type y Cancellation Policy
columns_to_dummify = ['Room Type', 'Property Type', 'Cancellation Policy']
airbnb_data = pd.get_dummies(airbnb_data, columns=columns_to_dummify, drop_first=True)

# Manejar Neighbourhood Group Cleansed por separado
airbnb_data = pd.get_dummies(airbnb_data, columns=['Neighbourhood Group Cleansed'], prefix='NGC')

# Identificar las columnas dummy creadas
dummy_columns = [col for col in airbnb_data.columns if col.startswith(('Room Type_', 'Property Type_', 'Cancellation Policy_', 'NGC_'))]

# Convertir solo las columnas dummy a int8
airbnb_data[dummy_columns] = airbnb_data[dummy_columns].astype('int8')

```


```python
# Cargar los datos de test
airbnb_data_test = pd.read_csv('airbnb_data_cleaned_test.csv', sep=';', decimal='.')

# 1. Filtrar los datos para mantener solo las entradas de Madrid y los precios
airbnb_data_test = airbnb_data_test[airbnb_data_test['City'] == 'Madrid']
airbnb_data_test = airbnb_data_test[airbnb_data_test['Price'] < 300]

# 2. Manejo de valores nulos
airbnb_data_test = airbnb_data_test.dropna(subset=['Price'])  # Eliminar filas donde Price es NaN

# Si deseamos podemos aplicar el logaritmo al precio
#airbnb_data_test = airbnb_data_test[airbnb_data_test['Price'] > 0]  # Filtrar precios no válidos
#airbnb_data_test['log_price'] = np.log(airbnb_data_test['Price'])  # Crear columna logarítmica

# Imputar valores faltantes en Review Scores usando la media del conjunto de entrenamiento
review_score_columns = [col for col in airbnb_data_test.columns if col.startswith('Review Scores')]
# Calcular medias del conjunto de entrenamiento
review_score_means = airbnb_data[review_score_columns].mean()  
airbnb_data_test[review_score_columns] = airbnb_data_test[review_score_columns].fillna(review_score_means)

# Imputar valores nulos en columnas importantes usando la mediana del conjunto de entrenamiento
columns_to_impute = ['Bathrooms', 'Bedrooms', 'Beds']
for col in columns_to_impute:
    # Usar la mediana del conjunto de entrenamiento para imputar
    airbnb_data_test[col] = airbnb_data_test[col].fillna(airbnb_data[col].median())

# 3. Crear características derivadas
airbnb_data_test['Bathrooms_per_person'] = airbnb_data_test['Bathrooms'] / airbnb_data_test['Accommodates']
airbnb_data_test['Beds_per_person'] = airbnb_data_test['Beds'] / airbnb_data_test['Accommodates']
airbnb_data_test['Bedrooms_squared'] = airbnb_data_test['Bedrooms'].apply(lambda x: x**2)
airbnb_data_test['Bed_bath_rooms']   = airbnb_data_test['Bedrooms']*airbnb_data_test['Bathrooms']

# Manejar Security Deposit y Cleaning Fee
airbnb_data_test['Security Deposit'] = airbnb_data_test['Security Deposit'].fillna(0)
airbnb_data_test['Cleaning Fee'] = airbnb_data_test['Cleaning Fee'].fillna(0)
airbnb_data_test['Total Additional Cost'] = airbnb_data_test['Security Deposit'] + airbnb_data_test['Cleaning Fee']

# 4. Eliminar características con muchos valores nulos
airbnb_data_test = airbnb_data_test.drop(['Square Feet', 'Weekly Price', 'Monthly Price'], axis=1)

# 5. Codificar variables categóricas
# Crear variables dummy para Room Type, Property Type y Cancellation Policy
columns_to_dummify = ['Room Type', 'Property Type', 'Cancellation Policy']
airbnb_data_test = pd.get_dummies(airbnb_data_test, columns=columns_to_dummify, drop_first=True)

# Manejar Neighbourhood Group Cleansed por separado
airbnb_data_test = pd.get_dummies(airbnb_data_test, columns=['Neighbourhood Group Cleansed'], prefix='NGC')

# Identificar las columnas dummy creadas
dummy_columns = [col for col in airbnb_data_test.columns if col.startswith(('Room Type_', 'Property Type_', 'Cancellation Policy_', 'NGC_'))]

# Convertir solo las columnas dummy a int8
airbnb_data_test[dummy_columns] = airbnb_data_test[dummy_columns].astype('int8')

# Añadir columnas faltantes y eliminar columnas extra
train_columns = set(airbnb_data.columns)
test_columns = set(airbnb_data_test.columns)
for col in train_columns - test_columns:
    airbnb_data_test[col] = 0
airbnb_data_test = airbnb_data_test[airbnb_data.columns]

```

Ahora podemos preparar los datos para sklearn:


```python
# Dataset de train
# Preparar los datos para el modelado
features = airbnb_data.drop(['Price'],axis=1).select_dtypes(include=[np.number]).columns

y_train = airbnb_data['Price']  # nos quedamos con la columna, Price o log_price
X_train = airbnb_data[features]  # nos quedamos con el resto


# Dataset de test

y_test = airbnb_data_test['Price']  # nos quedamos con la columna, Price o log_price
X_test = airbnb_data_test[features]  # nos quedamos con el resto


# Verificar las formas de los conjuntos resultantes
print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de y_test:", y_test.shape)

# Verificar las columnas en X_train y X_test
print("\nColumnas en X_train:")
print(X_train.columns.tolist())

print("\nColumnas en X_test:")
print(X_test.columns.tolist())

# Verificar los primeros registros de X_train y y_train
print("\nPrimeros registros de X_train:")
print(X_train.head())

print("\nPrimeros registros de y_train:")
print(y_train.head())
```

    Forma de X_train: (10456, 75)
    Forma de y_train: (10456,)
    Forma de X_test: (2617, 75)
    Forma de y_test: (2617,)
    
    Columnas en X_train:
    ['Latitude', 'Longitude', 'Accommodates', 'Bathrooms', 'Bedrooms', 'Beds', 'Security Deposit', 'Cleaning Fee', 'Guests Included', 'Extra People', 'Minimum Nights', 'Maximum Nights', 'Availability 30', 'Availability 60', 'Availability 90', 'Availability 365', 'Number of Reviews', 'Review Scores Rating', 'Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin', 'Review Scores Communication', 'Review Scores Location', 'Review Scores Value', 'Bathrooms_per_person', 'Beds_per_person', 'Bedrooms_squared', 'Bed_bath_rooms', 'Total Additional Cost', 'Room Type_Private room', 'Room Type_Shared room', 'Property Type_Bed & Breakfast', 'Property Type_Boutique hotel', 'Property Type_Bungalow', 'Property Type_Camper/RV', 'Property Type_Casa particular', 'Property Type_Chalet', 'Property Type_Condominium', 'Property Type_Dorm', 'Property Type_Earth House', 'Property Type_Guest suite', 'Property Type_Guesthouse', 'Property Type_Hostel', 'Property Type_House', 'Property Type_Loft', 'Property Type_Other', 'Property Type_Serviced apartment', 'Property Type_Tent', 'Property Type_Townhouse', 'Property Type_Villa', 'Cancellation Policy_moderate', 'Cancellation Policy_strict', 'Cancellation Policy_super_strict_30', 'Cancellation Policy_super_strict_60', 'NGC_Arganzuela', 'NGC_Barajas', 'NGC_Carabanchel', 'NGC_Centro', 'NGC_Chamartín', 'NGC_Chamberí', 'NGC_Ciudad Lineal', 'NGC_Fuencarral - El Pardo', 'NGC_Hortaleza', 'NGC_Latina', 'NGC_Moncloa - Aravaca', 'NGC_Moratalaz', 'NGC_Puente de Vallecas', 'NGC_Retiro', 'NGC_Salamanca', 'NGC_San Blas - Canillejas', 'NGC_Tetuán', 'NGC_Usera', 'NGC_Vicálvaro', 'NGC_Villa de Vallecas', 'NGC_Villaverde']
    
    Columnas en X_test:
    ['Latitude', 'Longitude', 'Accommodates', 'Bathrooms', 'Bedrooms', 'Beds', 'Security Deposit', 'Cleaning Fee', 'Guests Included', 'Extra People', 'Minimum Nights', 'Maximum Nights', 'Availability 30', 'Availability 60', 'Availability 90', 'Availability 365', 'Number of Reviews', 'Review Scores Rating', 'Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin', 'Review Scores Communication', 'Review Scores Location', 'Review Scores Value', 'Bathrooms_per_person', 'Beds_per_person', 'Bedrooms_squared', 'Bed_bath_rooms', 'Total Additional Cost', 'Room Type_Private room', 'Room Type_Shared room', 'Property Type_Bed & Breakfast', 'Property Type_Boutique hotel', 'Property Type_Bungalow', 'Property Type_Camper/RV', 'Property Type_Casa particular', 'Property Type_Chalet', 'Property Type_Condominium', 'Property Type_Dorm', 'Property Type_Earth House', 'Property Type_Guest suite', 'Property Type_Guesthouse', 'Property Type_Hostel', 'Property Type_House', 'Property Type_Loft', 'Property Type_Other', 'Property Type_Serviced apartment', 'Property Type_Tent', 'Property Type_Townhouse', 'Property Type_Villa', 'Cancellation Policy_moderate', 'Cancellation Policy_strict', 'Cancellation Policy_super_strict_30', 'Cancellation Policy_super_strict_60', 'NGC_Arganzuela', 'NGC_Barajas', 'NGC_Carabanchel', 'NGC_Centro', 'NGC_Chamartín', 'NGC_Chamberí', 'NGC_Ciudad Lineal', 'NGC_Fuencarral - El Pardo', 'NGC_Hortaleza', 'NGC_Latina', 'NGC_Moncloa - Aravaca', 'NGC_Moratalaz', 'NGC_Puente de Vallecas', 'NGC_Retiro', 'NGC_Salamanca', 'NGC_San Blas - Canillejas', 'NGC_Tetuán', 'NGC_Usera', 'NGC_Vicálvaro', 'NGC_Villa de Vallecas', 'NGC_Villaverde']
    
    Primeros registros de X_train:
        Latitude  Longitude  Accommodates  Bathrooms  Bedrooms  Beds  \
    0  40.407732  -3.684819             4        1.0       1.0   2.0   
    1  40.415802  -3.705340             4        1.0       1.0   2.0   
    2  40.389048  -3.740374             1        1.5       1.0   8.0   
    3  40.412814  -3.703052             2        3.0       1.0   1.0   
    4  40.438631  -3.713716             2        1.0       1.0   1.0   
    
       Security Deposit  Cleaning Fee  Guests Included  Extra People  ...  \
    0               0.0          25.0                1             0  ...   
    1               0.0          15.0                1             0  ...   
    2               0.0           5.0                1             0  ...   
    3               0.0           0.0                1             0  ...   
    4               0.0           0.0                1            10  ...   
    
       NGC_Moratalaz  NGC_Puente de Vallecas  NGC_Retiro  NGC_Salamanca  \
    0              0                       0           1              0   
    1              0                       0           0              0   
    2              0                       0           0              0   
    3              0                       0           0              0   
    4              0                       0           0              0   
    
       NGC_San Blas - Canillejas  NGC_Tetuán  NGC_Usera  NGC_Vicálvaro  \
    0                          0           0          0              0   
    1                          0           0          0              0   
    2                          0           0          0              0   
    3                          0           0          0              0   
    4                          0           0          0              0   
    
       NGC_Villa de Vallecas  NGC_Villaverde  
    0                      0               0  
    1                      0               0  
    2                      0               0  
    3                      0               0  
    4                      0               0  
    
    [5 rows x 75 columns]
    
    Primeros registros de y_train:
    0    60.0
    1    50.0
    2    10.0
    3    30.0
    4    32.0
    Name: Price, dtype: float64
    

Y si queremos, podemos normalizar, pero con los datos de train!


```python
# Escalamos (con los datos de train)
scaler = StandardScaler().fit(X_train)
XtrainScaled = scaler.transform(X_train)

# Esta normalización/escalado la realizo con el scaler anterior, basado en los datos de training!
XtestScaled = scaler.transform(X_test) 
```

Ahora vendría lo-de-siempre: cross validation, búsqueda de los parámetros óptimos, visualización de performance vs complejidad...

#  Entrenamos un Lasso usando Grid Search.


```python
alpha_vector =  np.logspace(-1.5,0,20)
param_grid = {'alpha': alpha_vector }
grid = GridSearchCV(Lasso(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv = 5, verbose=1)
grid.fit(XtrainScaled, y_train)
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

#-1 porque es negado
scores = -1*np.array(grid.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('5-Fold MSE')
plt.show()
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits
    best mean cross-validation score: -669.816
    best parameters: {'alpha': 0.06543189129712969}
    


    
![png](Pr%C3%A1ctica_ML_files/Pr%C3%A1ctica_ML_52_1.png)
    



```python
# Obtener el mejor alpha del GridSearchCV
alpha_optimo = grid.best_params_['alpha']

# Entrenar el modelo Lasso con el alpha óptimo
lasso = Lasso(alpha=alpha_optimo)
lasso.fit(XtrainScaled, y_train)

# Hacer predicciones
y_train_pred = lasso.predict(XtrainScaled)
y_test_pred = lasso.predict(XtestScaled)

# Calcular métricas
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Análisis de coeficientes
feature_names = X_train.columns
coefficients = lasso.coef_

# Crear un DataFrame con las características y sus coeficientes
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Ordenar los coeficientes por su valor absoluto
coef_df['Abs_Coefficient'] = abs(coef_df['Coefficient'])
coef_df_sorted = coef_df.sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)

top = coef_df_sorted.head(20)
# Visualizar los coeficientes

plt.figure(figsize=(14, 10))
plt.bar(top['Feature'], top['Coefficient'])
plt.xticks(rotation=90)
plt.title(f'Top {len(top)} Lasso Coefficients  calculado con [Price]')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')

# Crear el cuadro de texto con las métricas
metrics_text = f"""Métricas del modelo:
MSE (train): {mse_train:.2f}
MSE (test): {mse_test:.2f}
RMSE (train): {rmse_train:.2f}
RMSE (test): {rmse_test:.2f}
R2 (train): {r2_train:.4f}
R2 (test): {r2_test:.4f}
Alpha óptimo: {alpha_optimo:.6f}
Total variables: {len(feature_names)}
Variables no nulas: {np.sum(coefficients != 0)}"""

# Añadir el cuadro de texto al gráfico 
plt.text(0.95, 0.98, metrics_text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
```


    
![png](Pr%C3%A1ctica_ML_files/Pr%C3%A1ctica_ML_53_0.png)
    


#  Entrenamos un BaggingRegressor usando Grid Search.


```python
max_depth_vector = range(1,15)
param_grid = {'estimator__max_depth': max_depth_vector}
grid = GridSearchCV(
    BaggingRegressor(estimator=DecisionTreeRegressor(random_state=0)),
    scoring= 'neg_mean_squared_error',
    param_grid=param_grid,
    cv = 5,
    verbose=1
).fit(X_train, y_train)

print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

#-1 porque es negado
scores = -1*np.array(grid.cv_results_['mean_test_score'])
plt.plot(max_depth_vector,scores,'-o')
plt.xlabel('max depth',fontsize=16)
plt.ylabel('5-Fold MSE')
plt.show()
```

    Fitting 5 folds for each of 14 candidates, totalling 70 fits
    best mean cross-validation score: -537.026
    best parameters: {'estimator__max_depth': 13}
    


    
![png](Pr%C3%A1ctica_ML_files/Pr%C3%A1ctica_ML_55_1.png)
    



```python
maxDepthOptimo = 10  #grid.best_params_['estimator__max_depth']
baggingModel = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=maxDepthOptimo),
    n_estimators=200
).fit(X_train,y_train)

print("Train: ", baggingModel.score(X_train,y_train))
print("Test: ", baggingModel.score(X_test,y_test))
```

    Train:  0.8296581648547419
    Test:  0.7026621074363322
    


```python
# Número de características a mostrar
num_features_to_show = 30

# Calcular las importancias de las características
importances = np.mean([tree.feature_importances_ for tree in baggingModel.estimators_], axis=0)
importances = importances / np.max(importances)
feature_names = features
indices = np.argsort(importances)[::-1]

# Calcular métricas adicionales
train_score = baggingModel.score(X_train, y_train)
test_score = baggingModel.score(X_test, y_test)
y_train_pred = baggingModel.predict(X_train)
y_test_pred = baggingModel.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Crear el gráfico
plt.figure(figsize=(12, 10))
plt.barh(range(num_features_to_show), importances[indices][:num_features_to_show])
plt.yticks(range(num_features_to_show), [features[i] for i in indices[:num_features_to_show]])
plt.xlabel('Importancia Relativa')
plt.title(f'Top {num_features_to_show} Características más Importantes en el Modelo Bagging calculado con [Price]')

# Crear el cuadro de texto con las métricas
metrics_text = f"""Métricas del modelo:
MSE (train): {mse_train:.2f}
MSE (test): {mse_test:.2f}
RMSE (train): {rmse_train:.2f}
RMSE (test): {rmse_test:.2f}
R² (train): {train_score:.4f}
R² (test): {test_score:.4f}
Max Depth: {maxDepthOptimo}
N° Estimadores: 200
Total variables: {len(feature_names)}"""

# Añadir el cuadro de texto al gráfico
plt.text(0.75, 0.65, metrics_text, transform=plt.gca().transAxes, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=12)

plt.tight_layout()
plt.show()

```


    
![png](Pr%C3%A1ctica_ML_files/Pr%C3%A1ctica_ML_57_0.png)
    


## Entrenamos un RandomForestRegressor usando Grid Search.


```python
maxDepth = range(1,15)
tuned_parameters = {'max_depth': maxDepth}

grid = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=200, max_features='sqrt'),
       param_grid=tuned_parameters,cv=5, verbose=1,scoring='neg_mean_squared_error')  
grid.fit(X_train, y_train)

print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

scores = -np.array(grid.cv_results_['mean_test_score']) # Negamos para convertir a MSE
plt.plot(maxDepth,scores,'-o')
plt.xlabel('max_depth')
plt.ylabel('5-Fold MSE')

plt.show()
```

    Fitting 5 folds for each of 14 candidates, totalling 70 fits
    best mean cross-validation score: -524.755
    best parameters: {'max_depth': 14}
    


    
![png](Pr%C3%A1ctica_ML_files/Pr%C3%A1ctica_ML_59_1.png)
    



```python
maxDepthOptimo = 12  #grid.best_params_['max_depth']
randomForest = RandomForestRegressor(max_depth=maxDepthOptimo,n_estimators=200,max_features='sqrt').fit(X_train,y_train)

print("Train: ",randomForest.score(X_train,y_train))
print("Test: ",randomForest.score(X_test,y_test))
```

    Train:  0.8456715855937735
    Test:  0.7019537555231137
    


```python
# Número de características a mostrar
num_features_to_show = 30

# Calcular métricas (si no están ya calculadas)
train_score = randomForest.score(X_train, y_train)
test_score = randomForest.score(X_test, y_test)
y_train_pred = randomForest.predict(X_train)
y_test_pred = randomForest.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Calcular importancias de características
importances = randomForest.feature_importances_
importances = importances / np.max(importances)
indices = np.argsort(importances)[::-1]

# Crear el gráfico para las primeras num_features_to_show características
plt.figure(figsize=(14, 12))
plt.barh(range(num_features_to_show), importances[indices][:num_features_to_show])
plt.yticks(range(num_features_to_show), [features[i] for i in indices[:num_features_to_show]])
plt.xlabel('Importancia Relativa', fontsize=14)
plt.title(f'Top {num_features_to_show} Características más Importantes en el Modelo Random Forest calculado con[Price]', fontsize=16)

# Crear el cuadro de texto con las métricas
metrics_text = f"""Métricas del modelo:
MSE (train): {mse_train:.2f}
MSE (test): {mse_test:.2f}
RMSE (train): {rmse_train:.2f}
RMSE (test): {rmse_test:.2f}
R² (train): {train_score:.4f}
R² (test): {test_score:.4f}
Max Depth: {randomForest.max_depth}
N° Estimadores: {randomForest.n_estimators}
Max Features: {randomForest.max_features}
Total variables: {len(feature_names)}"""

# Añadir el cuadro de texto al gráfico
plt.text(0.75, 0.65, metrics_text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=14)

plt.tight_layout()
plt.show()

```


    
![png](Pr%C3%A1ctica_ML_files/Pr%C3%A1ctica_ML_61_0.png)
    


# Seleccion de características usando RandomForestRegressor y Validación cruzada KFold


```python
num_features_to_show = 30
N, Nfeatures = X_train.shape
maxDepthOptimo = 10
rf = RandomForestRegressor(max_depth=maxDepthOptimo, n_estimators=20, max_features='sqrt')
kf = KFold(n_splits=5, shuffle=True, random_state=1)
cv_error = []
cv_std = []

all_importances = []  # Para almacenar las importancias de cada iteración

for nfeatures in range(Nfeatures, 0, -1):
    error_i = []
    
    for idxTrain, idxVal in kf.split(X_train):
        Xt = X_train.iloc[idxTrain, :]
        yt = y_train.iloc[idxTrain]
        Xv = X_train.iloc[idxVal, :]
        yv = y_train.iloc[idxVal]
        
        rf.fit(Xt, yt)
        
        ranking = rf.feature_importances_
        all_importances.append(ranking)  # Agregamos las importancias de esta iteración
        
        # Usamos el promedio de importancias acumuladas hasta ahora
        average_importances = np.mean(all_importances, axis=0)
        indices = np.argsort(average_importances)[::-1]
        
        selected = indices[0:(Nfeatures-nfeatures+1)]
        
        Xs = Xt.iloc[:, selected]
        
        rf.fit(Xs, yt)
        error = (1.0 - rf.score(Xv.iloc[:, selected], yv))
        error_i.append(error)
    
    cv_error.append(np.mean(error_i))
    cv_std.append(np.std(error_i))
    
    #print(f'# features {len(selected)} error {np.mean(error_i):.4f} +/- {np.std(error_i):.4f}')

# Calculamos el ranking final de importancias
final_importances = np.mean(all_importances, axis=0)
final_indices = np.argsort(final_importances)[::-1]

# Seleccionamos las num_features_to_show más importantes

selected_features = final_indices[:num_features_to_show]

# Entrenamos un modelo final con las 30 características más importantes
final_rf = RandomForestRegressor(max_depth=12, n_estimators=200)
final_rf.fit(X_train.iloc[:, selected_features], y_train)

# Evaluamos el modelo final
print(f'Modelo con {num_features_to_show} features ')
print("Train score:", final_rf.score(X_train.iloc[:, selected_features], y_train))
print("Test score:", final_rf.score(X_test.iloc[:, selected_features], y_test))

# 30 características más importantes:
top_features = X_train.columns[selected_features]
print(f"Top {num_features_to_show} features:", top_features.tolist())
```

    Modelo con 30 features 
    Train score: 0.8772140376146786
    Test score: 0.7125826552032535
    Top 30 features: ['Room Type_Private room', 'Accommodates', 'Bed_bath_rooms', 'Bedrooms_squared', 'Bedrooms', 'Cleaning Fee', 'Bathrooms_per_person', 'Beds', 'Bathrooms', 'Total Additional Cost', 'Guests Included', 'Latitude', 'Security Deposit', 'Beds_per_person', 'Longitude', 'Extra People', 'Availability 365', 'Minimum Nights', 'Availability 30', 'Availability 90', 'Availability 60', 'Number of Reviews', 'Review Scores Location', 'Review Scores Rating', 'NGC_Centro', 'Review Scores Accuracy', 'Review Scores Cleanliness', 'Maximum Nights', 'Room Type_Shared room', 'Review Scores Value']
    


```python
feature_names = features

# Crear el gráfico
plt.figure(figsize=(12, 6))  # Aumentado el tamaño para mejor visibilidad

# Plotear el error CV solo para las primeras 30 características
plt.plot(range(1, num_features_to_show + 1), cv_error[:num_features_to_show], '-o')
plt.errorbar(range(1, num_features_to_show + 1), cv_error[:num_features_to_show], 
             yerr=cv_std[:num_features_to_show], fmt='o')

# Preparar las etiquetas del eje X para las primeras 30 características
x_ticks = range(1, num_features_to_show + 1)
x_labels = [feature_names[final_indices[i-1]] for i in x_ticks]

# Configurar el eje x
plt.xticks(x_ticks, x_labels, rotation=90, ha='right')
plt.xlabel('Características (ordenadas por importancia)')
plt.ylabel('Error CV')
plt.title(f'Error de Validación Cruzada para las Top {num_features_to_show} Características calculado con [Price]\n'
          f'(Total de {len(feature_names)} características, profundidad máxima: {maxDepthOptimo})')
# Ajustar el diseño para evitar que las etiquetas se corten
plt.tight_layout()

plt.show()
```


    
![png](Pr%C3%A1ctica_ML_files/Pr%C3%A1ctica_ML_64_0.png)
    



```python
# Calcular métricas
train_score = final_rf.score(X_train.iloc[:, selected_features], y_train)
test_score = final_rf.score(X_test.iloc[:, selected_features], y_test)
y_train_pred = final_rf.predict(X_train.iloc[:, selected_features])
y_test_pred = final_rf.predict(X_test.iloc[:, selected_features])
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Calcular importancias de características
importances = final_rf.feature_importances_
importances = importances / np.max(importances)
indices = np.argsort(importances)[::-1]

# Crear el gráfico
plt.figure(figsize=(14, 12))
plt.barh(range(num_features_to_show), importances[indices])
plt.yticks(range(num_features_to_show), [top_features[i] for i in indices])
plt.xlabel('Importancia Relativa', fontsize=14)
plt.title(f'Top {num_features_to_show} Características más Importantes en el Modelo Random Forest Final', fontsize=16)

# Crear el cuadro de texto con las métricas
metrics_text = f"""Métricas del modelo:
MSE (train): {mse_train:.2f}
MSE (test): {mse_test:.2f}
RMSE (train): {rmse_train:.2f}
RMSE (test): {rmse_test:.2f}
R² (train): {train_score:.4f}
R² (test): {test_score:.4f}
Max Depth: {final_rf.max_depth}
N° Estimadores: {final_rf.n_estimators}
Max Features: {final_rf.max_features}
Total variables: {len(top_features)}"""

# Añadir el cuadro de texto al gráfico
plt.text(0.75, 0.65, metrics_text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=14)

plt.tight_layout()
plt.show()
```


    
![png](Pr%C3%A1ctica_ML_files/Pr%C3%A1ctica_ML_65_0.png)
    

