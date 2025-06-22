# Predicción de Precios de autos

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto final del curso de Machine Learning Engineering 1


# 1. Problema de ML

El mercado de automóviles presenta una amplia variedad de modelos, marcas, condiciones y características técnicas que influyen en el precio de un vehículo. Sin embargo, establecer un precio justo y preciso puede ser complejo debido a la gran cantidad de variables involucradas. Esta incertidumbre puede afectar tanto a vendedores como a compradores al momento de valorar correctamente un automóvil.
Desde el punto de vista del aprendizaje automático, este desafío se puede abordar como un problema de regresión supervisada, en el que el objetivo es predecir un valor numérico continuo (el precio) a partir de atributos del vehícul

## Objetivo del Proyecto

Desarrollar un modelo de Machine Learning capaz de predecir con precisión el precio de un automóvil en función de sus características, tales como marca, modelo, año, kilometraje, transmisión, tipo de combustible y otras variables relevantes. El modelo deberá ser entrenado sobre un conjunto de datos históricos, validado rigurosamente y evaluado mediante métricas como RMSE, MAE o R².

# 2. Diagrama de Flujo de Proyecto

flowchart LR
    A["Datos en csv"] --> B["Preprocesamiento de datos"]
    B --> C["Feature Engineering"]
    C --> D["Entrenamiento del modelo"]
    D --> E["Evaluación del modelo"]
    E --> F["Modelo .joblib"]


# 3. Descripción del Dataset

Para este laboratorio, utilizaremos el conjunto de datos de ventas de automóviles, alojado en Kaggle. Este conjunto de datos se puede encontrar y descargar desde [kaggle.com](https://www.kaggle.com/datasets/goyalshalini93/car-data), una fuente de datos pública y abierta.
El conjunto de datos contiene toda la información sobre los automóviles, el nombre del fabricante, todos los parámetros técnicos y el precio de venta.


## 📚 Diccionario de Datos

| Nº | Columna            | Descripción                                                                 | Tipo de dato     |
|----|--------------------|-----------------------------------------------------------------------------|------------------|
| 1  | `Car_ID`           | ID único de cada observación                                                | Entero           |
| 2  | `Symboling`        | Riesgo de seguro asignado (+3 = riesgoso, -3 = seguro)                      | Categórico       |
| 3  | `carCompany`       | Nombre de la compañía del auto                                              | Categórico       |
| 4  | `fueltype`         | Tipo de combustible (gasolina o diésel)                                     | Categórico       |
| 5  | `aspiration`       | Tipo de aspiración del motor                                                | Categórico       |
| 6  | `doornumber`       | Número de puertas del vehículo                                              | Categórico       |
| 7  | `carbody`          | Tipo de carrocería                                                          | Categórico       |
| 8  | `drivewheel`       | Tipo de tracción (rueda motriz)                                             | Categórico       |
| 9  | `enginelocation`   | Ubicación del motor                                                         | Categórico       |
| 10 | `wheelbase`        | Distancia entre ejes                                                        | Numérico         |
| 11 | `carlength`        | Longitud del auto                                                           | Numérico         |
| 12 | `carwidth`         | Ancho del auto                                                              | Numérico         |
| 13 | `carheight`        | Altura del auto                                                             | Numérico         |
| 14 | `curbweight`       | Peso del vehículo sin ocupantes ni equipaje                                | Numérico         |
| 15 | `enginetype`       | Tipo de motor                                                               | Categórico       |
| 16 | `cylindernumber`   | Número de cilindros                                                         | Categórico       |
| 17 | `enginesize`       | Tamaño del motor                                                            | Numérico         |
| 18 | `fuelsystem`       | Sistema de combustible                                                      | Categórico       |
| 19 | `boreratio`        | Relación diámetro del cilindro / carrera                                   | Numérico         |
| 20 | `stroke`           | Carrera del pistón                                                          | Numérico         |
| 21 | `compressionratio` | Relación de compresión del motor                                            | Numérico         |
| 22 | `horsepower`       | Caballos de fuerza del motor                                                | Numérico         |
| 23 | `peakrpm`          | Revoluciones por minuto máximas                                             | Numérico         |
| 24 | `citympg`          | Consumo de combustible en ciudad (millas por galón)                         | Numérico         |
| 25 | `highwaympg`       | Consumo de combustible en carretera (millas por galón)                      | Numérico         |
| 26 | `price`            | Precio del automóvil (**variable dependiente**)                             | Numérico         |

# 4. Model Card

# 5. Resultados

# 6. Conclusiones

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         package-mle1 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── package-mle1   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes package-mle1 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

