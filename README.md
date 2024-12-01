# Proyecto8-Prediccion de la probabilidad de un empleado de irse de la Empresa



## Entrenamiento del mejor modelo predictivo de la probabilidad de un empleado de irse de la Empresa
![Predicción de la probabilidad de un empleado de irse de la Empresa](https://github.com/jgilsu11/Proyecto8-Predicci-n-de-Retenci-n-de-Empleados/blob/main/Imagen/imagen_oficina.webp)  
  

***Descripción:***
El proyecto 8 consiste en la especificación e iteración de modelos predictivos hasta obtener un modelo predictivo óptimo para la probabilidad de un empleado de irse de la Empresa haciendo uso de archivos.py y Jupyter notebook.

Las técnicas usadas durante el proyecto son en su mayoría aquellas enseñadas durante la octava semana de formación (Preprocesamiento(Gestión de nulos, Duplicados,Encoding, Estandarizacióny Gestión de Outliers) , generación y entrenamiento de modelos de clasificación).

Adicionalmente, se usaron recursos obtenidos mediante research en documentación especializada, vídeos de YouTube e IA como motor de búsqueda y apoyo al aprendizaje.


***Estructura del Proyecto:***

El desarrollo del proyecto se gestionó de la siguiente manera:

- _En Primer lugar_, haciendo uso de JupyterNotebook como primer paso donde realizar ensayos con el código.  

- _En Segundo Lugar_, se creó una presentación basada en los datos.

- _Finalmente_, se realizó la documentación del proyecto en un archivo README (documento actual).

Por todo lo anterior, el usuario tiene acceso a:

        ├── datos/                                       # Donde se guardan los csv que se van generando en cada modelo 
        ├── Imagen/                                      # Imagen para su uso en streamlit       
        ├── Modelos/                                     # Notebooks de Jupyter donde se han ido desarrollando los modelos
        ├── transformers/                                # Donde se han guardado los pickles para su uso en streamlit     
        |          ├──modelos                            # Donde se han guardado los pickles de los modelos para su uso en streamlit
        |          |    ├──modelo                        # Donde se han guardado los pickles de los mejores modelos de cada método
        |          |    ├──basura                        # Donde se han guardado los pickles de los modelos que no son los mejores
        |          ├──preprocesamiento                   # Donde se han guardado los picklesdel preprocesamineto para su uso en streamlit
        ├── src/                                         # Scripts (.py)
        ├── README.md                                    # Descripción del proyecto
        ├── main                                         # Script donde desarrollar streamlit (.py) 
        ├── base.yml/                                    # Entorno con todas las bibliotecas y versiones necesarias                  
        
***Requisitos e Instalación🛠️:***

Este proyecto usa Python 3.11.9 y bibliotecas que se necesitarán importar al principio del código como:
- [pandas](https://pandas.pydata.org/docs/)
- [numpy](https://numpy.org/doc/2.1/)
- [matplotlib](https://matplotlib.org/stable/index.html)
- [matplotlib-inline](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html)
- [seaborn](https://seaborn.pydata.org/)
- [requests](https://requests.readthedocs.io/en/latest/)
- [sys](https://docs.python.org/3/library/sys.html)
- [os](https://docs.python.org/3/library/os.html)
- [sklearn](https://scikit-learn.org/stable/)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [tqdm](https://tqdm.github.io/)
- [warnings](https://docs.python.org/3/library/warnings.html)
- [pandas.options.display](https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html)
- [shap](https://shap.readthedocs.io/en/latest/)
  
Además se dispone del archivo base.yml donde se puede clonar el entorno necesario para el funcionamiento del proyecto.     
***Tabla Resumen***  
  
| **Modelo**    | **Algoritmo**       | **accuracy**   | **precision**    | **recall**    | **kappa**       | **Descripción**                                    |
|---------------|---------------------|----------------|------------------|---------------|-----------------|----------------------------------------------------|
| **Modelo 1**  | XGBoost             | 0.99           | 0.99             | 0.99          | 0.97            | Muy buenas metricas (no válido) porduplicados sin tratar     |
| **Modelo 2**  | XGBoost             | 0.86           | 0.84             | 0.86          | 0.34            | Desempeño significativamente más bajo.           |
| **Modelo 3**  | Regresión Logística | 0.91           | 0.91             | 0.91          | 0.81            | Mejor modelo.                                      |
   
  
***Aportación al Usuario🤝:***

El doble fin de este proyecto incluye tanto el propio aprendizaje y formación como la intención de crear un modelo predictivo de la probabilidad de irse de la empresa de un empleado que el usuario pueda usar.


***Próximos pasos:***

En un futuro, se recomienda ser más exhaustivo y variado en el preprocesamiento y especificar más modelos para poder hacer un predicción más precisa. La herramientas que más útiles pueden ser son el uso de otras formas de machine learning, inteligencia artificial u otras opciones.
