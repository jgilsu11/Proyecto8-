import streamlit as st
import pandas as pd
import pickle
import numpy as np


st.set_page_config(
    page_title="Predicción de la probabilidad de partida de un empleado",
    page_icon="📠",
    layout="centered",
)

# Título y descripción
st.title("📠 Predicción de la probabilidad de partida de un empleado")
st.write("Usa esta aplicación para predecir la probabilidad de partida de un empleado de tu empresa basandote en sus respuestas a encuestas y características demográficas.")

# Mostrar una imagen
st.image(
    "Proyecto 8\\Proyecto8-Predicci-n-de-Retenci-n-de-Empleados\\Imagen\\imagen_oficina.webp",  # URL de la imagen
    caption="Retén a tus empleados.",
    use_column_width=True,
)


# Cargar los modelos y transformadores entrenados
def load_models():
    with open('transformers/preprocesamiento3/one_hot_encoder.pkl', 'rb') as f:
        one_hot = pickle.load(f)    
    with open('transformers/preprocesamiento3/target_encoder.pkl', 'rb') as t:
        target_encoder = pickle.load(t)
    with open('transformers/preprocesamiento3/scaler.pkl', 'rb') as s:
        scaler = pickle.load(s)
    with open('transformers/modelos3/MODELO_DEFINITIVO.pkl', 'rb') as l:
        model = pickle.load(l)
    return one_hot,target_encoder, scaler, model

one_hot,target_encoder, scaler, model = load_models()

st.header("Datos y características del empleado 🧔")
col1, col2 ,col3= st.columns(3)

with col1:
    Age = st.number_input("Edad", min_value=18,max_value=60 , value=40, step=1, help="Elige la edad del trabajador entre 18 y 60")
    Education = st.selectbox("Educación", ['Below College', 'College', 'Bachelor', 'Master', 'Doctor'], help="Elige la educación del trabajador")
    EducationField= st.selectbox("Campo de Estudio", ['Life Sciences', 'Other', 'Medical', 'Marketing','Technical Degree', 'Human Resources'])
    Gender= st.selectbox("Género", ['Female', 'Male'])
    MaritalStatus = st.selectbox("Estado civil", ['Married', 'Single', 'Divorced'], help="Elige el estado civil del empleado")
    NumCompaniesWorked = st.selectbox('Número de compañías', [0,1,2,3,4,5,6,7,8,9], help="Elige el numero de compañías donde el empleado ha trabajado")
    TotalWorkingYears= st.selectbox('Años trabajados',[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                                       21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])


with col2:
    JobLevel = st.selectbox("Nivel del puesto", ['Intern', 'Junior', 'Senior', 'Manager', 'Head'],help="Elige el nivel del puesto del empleado")
    JobRole = st.selectbox('Rol', ['Healthcare Representative', 'Research Scientist','Sales Executive', 'Human Resources', 'Research Director','Laboratory Technician', 'Manufacturing Director','Sales Representative', 'Manager'], help='Elije el rol del empleado')
    BusinessTravel = st.selectbox('Frecuencia de viaje', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], help='Elige la frecuencia de viaje del trabajador') #min,max,predeterminado
    Department = st.selectbox("Departamento", ['Sales', 'Research & Development', 'Human Resources'], help="Elige el departamento del trabajador")
    DistanceFromHome = st.selectbox("Distancia de casa al trabajo", ['entre 1 y 4',  'entre 5 y 8', 'entre 9 y 12','entre 13 y 18', 'entre 19 y 23', 'entre 24 y 29'], help="Eligela distancia a casa del trabajador")
    MonthlyIncome= st.slider("Salario mensual",114,2260,500 )
    PercentSalaryHike= st.slider('Porcentaje de subida de salario', 11,25,15)
    StockOptionLevel= st.selectbox('Nivel de reparto de acciones', ['Bad', 'Good', 'Better', 'Best'])
    
with col3:
    TrainingTimesLastYear= st.number_input('Número de formaciones en el último año', min_value=0,max_value=6 , value=1, step=1)
    YearsAtCompany= st.slider('Años en la compañía actual', 0,40,15)
    YearsSinceLastPromotion= st.number_input('Años desde la última promoción', min_value=0,max_value=15 , value=1, step=1)
    EnvironmentSatisfaction= st.selectbox('Nivel de satisfacción con el entorno', ['Low','Medium','High', 'Very High' ])
    JobSatisfaction= st.selectbox('Nivel de satisfacción en el trabajo', ['Low','Medium','High', 'Very High' ])
    WorkLifeBalance= st.selectbox('Nivel de satisfacción con el balance vida-trabajo', ['Bad', 'Good', 'Better', 'Best'])
    JobInvolvement= st.selectbox('Nivel de implicación en el trabajo', ['Low','Medium','High', 'Very High' ])





# Botón para realizar la predicción
if st.button("Predecir si el empleado se va 🤞"):
    # Crear DataFrame con los datos ingresados    #Comprobar que sean las mismas que la mía y el mismo orden
    new_employee = pd.DataFrame({
        'Age': [Age],
        'BusinessTravel': [str(BusinessTravel)],
        'Department': [str(Department)],
        'DistanceFromHome': [str(DistanceFromHome)],
        'Education': [str(Education)],
        'EducationField' : [str(EducationField)],
        'Gender' : [str(Gender)],
        'JobLevel': [str(JobLevel)],
        'JobRole': [str(JobRole)],
        'MaritalStatus': [str(MaritalStatus)],
        'MonthlyIncome': [MonthlyIncome],
        'NumCompaniesWorked': [str(NumCompaniesWorked)],
        'PercentSalaryHike': [PercentSalaryHike],
        'StockOptionLevel': [str(StockOptionLevel)],
        'TotalWorkingYears': [str(TotalWorkingYears)],
        'TrainingTimesLastYear': [TrainingTimesLastYear],
        'YearsAtCompany': [YearsAtCompany],
        'YearsSinceLastPromotion': [YearsSinceLastPromotion],
        'EnvironmentSatisfaction' : [str(EnvironmentSatisfaction)],
        'JobSatisfaction' : [str(JobSatisfaction)],
        'WorkLifeBalance': [str(WorkLifeBalance)],
        'JobInvolvement': [str(JobInvolvement)]       
    })
  
    new_employee=pd.DataFrame(new_employee)
    
    col_encode = ["Gender", "DistanceFromHome", "Education", "JobLevel", "StockOptionLevel", "JobRole", "TrainingTimesLastYear", "JobInvolvement"]
    onehot = one_hot.transform(new_employee[col_encode])
    # Obtenemos los nombres de las columnas del codificador
    column_names = one_hot.get_feature_names_out(col_encode)
    # Convertimos a un DataFrame
    onehot_df = pd.DataFrame(onehot.toarray(), columns=column_names)
    #onehot_df.drop(columns=col_encode,inplace=True)                    
    # Columnas categóricas y numéricas                    #Poner aquí mis columnas concretas
    # categorical_columns = ['propertyType', 'exterior', 'rooms', 'bathrooms', 'status', 'floor', 'hasLift']
    numerical_columns = ['price', 'size', 'bathrooms', 'province', 'municipality', 'distance',
             'district', 'propertyType_chalet', 'propertyType_countryHouse',
             'propertyType_duplex', 'propertyType_flat', 'propertyType_penthouse',
             'propertyType_studio', 'exterior_False', 'exterior_True', 'rooms_0',
             'rooms_1', 'rooms_2', 'rooms_3', 'rooms_4', 'status_desconocido',
             'status_good', 'status_newdevelopment', 'status_renew', 'floor_1',
             'floor_14', 'floor_2', 'floor_3', 'floor_4', 'floor_5', 'floor_6',
             'floor_7', 'floor_8', 'floor_bj', 'floor_desconocido', 'floor_en',
             'floor_ss', 'floor_st', 'hasLift_False', 'hasLift_True',
             'hasLift_desconocido', 'parkingSpace_False', 'parkingSpace_True',
             'parkingSpace_desconocido']
   
    
    # # Aplicar el OneHotEncoder, TargetEncoder y StandardScaler
    # diccionario_encoding = {"onehot": ["propertyType", "exterior", "rooms", "status", "floor", "hasLift", "parkingSpace"], 
    #                     "dummies": [], # no metemos ninguna
    #                     'ordinal' : {}, #no metemos ninguna
    #                     "label": [] , # no metemos ninguna columna porque no queremos en ningún caso que se asignen las categorías de forma aleatoria
    #                     "frequency": [], # no metemos ninguna columna porque no coincide el orden del value counts con las categorias y la variable respuesta
    #                     "target": ["bathrooms", "province", "municipality", "district"]  
    #                     }
    # col_encode=diccionario_encoding.get("onehot", [])
    # st.write(col_encode)
    
             #Añadir el Onehot
    # Codificación de las columnas categóricas
    new_employee.drop(columns = col_encode,inplace=True)
    new_employee_encoded = pd.concat([new_employee, onehot_df], axis=1)
    #new_house_encoded.drop(columns=col_encode,inplace=True)
    # new_house_encoded = pd.DataFrame()
    new_employee_encoded["Attrition"] = np.nan
    new_employee_encoded = target_encoder.transform(new_employee_encoded)
    

    new_employee_encoded2=new_employee_encoded.copy()
    new_employee_encoded3=new_employee_encoded.copy()
    # One-Hot Encoding
    # Hacemos el OneHot Encoder

    # st.write(onehot_df)

    # concatenamos los resultados obtenidos en la transformación con el DataFrame original
    #new_house.drop(columns=col_encode,inplace=True)



    # # Target Encoding (corrige el uso)
    # encoded_target = target_encoder.transform(new_house_encoded[["bathrooms", "province", "municipality", "district"]])

    # # Combina los datos codificados
    # new_house_encoded = pd.concat([new_house_encoded, encoded_target], axis=1)
    
    # Filtra las columnas numéricas y escala
    new_employee_encoded2.drop(columns="Attrition", inplace=True)
    new_employee_encoded = scaler.transform(new_employee_encoded2)
    new_employee_encoded = pd.DataFrame(new_employee_encoded)
    # new_house_encoded.drop(columns=6,inplace=True)
    new_employee_encoded3["Attrition"]=new_employee_encoded[6]   #EL DATA FRAME AL PASAR POR EL SCALER NO PUEDE TENER PRICE PERO EN EL MODLEO SI QUE TIENE QUE ESTAR PRICE
    
    # Realizar la predicción
    prediction = model.predict(new_employee_encoded)[0]
    if prediction ==0:
        pred="No se irá de la empresa"
    else:
        pred="Se irá de la empresa :("
    # dicc_pred={0:"No se irá de la empresa",
    #            1:"Se irá de la empresa"}
    # prediction_encoded=prediction.map(dicc_pred)
    # y_pred=modelo_final.predict(x)
    # Mostrar el resultado
    st.success(f"El empleado que has consultado {pred}")
    if pred=="No se irá de la empresa":
        st.balloons()
    else:
        pass
    

st.markdown(
    """
    ---
    **Proyecto creado con el potencial de la ciencia de datos.**  
    Desarrollado con ❤️ usando Streamlit.
    """,
    unsafe_allow_html=True,
)
