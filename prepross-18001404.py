import datetime
import time
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.outliers import Winsorizer
from sklearn.tree  import DecisionTreeClassifier

import joblib

def train_model(train_data, target):
    start_time = time.time()
#=================================== CONFIGURACION PIPELINE ======================================================
    
    #imputación de variables categoricas con indicador de Faltante (Missing)
    CATEGORICAL_VARS_WITH_NA_MISSING = ['Credit_Mix','Occupation']
    #Imputación de variables numéricas
    NUMERICAL_VARS_WITH_NA_MEAN = ['Num_Credit_Inquiries','Monthly_Balance']
    #Variables para transfomraicón Yeo
    NUMERICAL_YEON_VARS =['Annual_Income','Interest_Rate','Outstanding_Debt','Num_Bank_Accounts','Num_Credit_Card',
    'Num_Credit_Inquiries','Delay_from_due_date','Monthly_Balance']
    #Variables para codificación por frecuencia
    CATEGORICAL_VARS = ['SSN','Occupation','Payment_of_Min_Amount','Payment_Behaviour','Month','Credit_Mix']
    #Variables a utilzar en el entrenamiento
    FEATURES = ['Annual_Income','Interest_Rate','Outstanding_Debt','Num_Bank_Accounts','Num_Credit_Card',
    'Num_Credit_Inquiries','Delay_from_due_date','Monthly_Balance','Month','SSN','Occupation','Credit_Mix',
    'Payment_of_Min_Amount','Payment_Behaviour']

    train_data = train_data[FEATURES]

    for c in NUMERICAL_YEON_VARS:
        minimo=train_data[c].min()
        if minimo<=0:
            train_data[c]=train_data[c]+abs(minimo)+1
        else:
            train_data[c]=train_data[c]


 #=================================== CONSTRUCCION DE PIPELINE ======================================================  
 #  
    # Crear el pipeline de ingeniería de características y modelo
    pipeline = Pipeline([
    
    #=========== IMPUTACIONES ===============
    
    #1. Imputacion de variables categóricas con indicador de faltante
    ('missing_imputation',
        CategoricalImputer(imputation_method='missing', variables=CATEGORICAL_VARS_WITH_NA_MISSING)
    ),
    
 
        #3. Indicador faltane en variables numericas para imputación
    ('missing_indicator_numeric',
        AddMissingIndicator(variables=NUMERICAL_VARS_WITH_NA_MEAN)
    ),

    
    #4. Imputación de variables numéricas
    ('mean_imputation',
         MeanMedianImputer(imputation_method='mean', variables=NUMERICAL_VARS_WITH_NA_MEAN)
    ),
    
    #============= CODIFICACIÓN DE VARIABLES CATEGORICAS NOMINALES ==================
    ('rare_label_encoder',
        RareLabelEncoder(n_categories=1, tol=0.01, variables=CATEGORICAL_VARS)
    ),
    
    ('categorical_encoder',
        OrdinalEncoder(encoding_method='ordered', variables=CATEGORICAL_VARS)
    ),

    #=============== TRATAMIENTO DE OUTLIERS ============
    ('Tratamiento_outliers',
        Winsorizer(capping_method='iqr', variables=NUMERICAL_YEON_VARS)),
        
    #=============== TRANSFORMACIÓN DE VARIABLES CONTINUAS ============
    
    ('yeo_transformer',
        YeoJohnsonTransformer(variables=NUMERICAL_YEON_VARS)
    ),

     #=============== SCALER ============
    ('scaler',
        StandardScaler(),
    ),
         #=============== MODELO ============
    ('modelo',
       DecisionTreeClassifier(max_depth= 10, max_features=14)
    )
    ])
    
    # Entrenar el modelo
    pipeline.fit(train_data, target)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Obtener las predicciones en los datos de entrenamiento
    predictions = pipeline.predict(train_data)
    
    # Calcular las métricas de entrenamiento
    accuracy = accuracy_score(target, predictions)
    # Haciendo la matriz de confusion
    cm = confusion_matrix(target, predictions)
    cm_df = pd.DataFrame(cm, index = ["Poor","Standard","Good"], columns = ["Poor","Standard","Good"])

    #=============================APLICANDO METODO OVR PARA OBTENER PROMEDIOS GENERALES =========================

    #caso32991
    TP32991=cm_df.iloc[0,0]
    FN32991=cm_df.iloc[0,1]+cm_df.iloc[0,2]
    FP32991=cm_df.iloc[1,0]+cm_df.iloc[2,0]
    TN32991=cm_df.iloc[1,1]+cm_df.iloc[1,2]+cm_df.iloc[2,1]+cm_df.iloc[2,2]
    #caso16445
    TP16445=cm_df.iloc[1,1]
    FN16445=cm_df.iloc[1,0]+cm_df.iloc[1,2]
    FP16445=cm_df.iloc[0,1]+cm_df.iloc[2,1]
    TN16445=cm_df.iloc[0,0]+cm_df.iloc[0,2]+cm_df.iloc[2,0]+cm_df.iloc[2,2]
    #caso26773
    TP26773=cm_df.iloc[2,2]
    FN26773=cm_df.iloc[2,0]+cm_df.iloc[2,1]
    FP26773=cm_df.iloc[0,2]+cm_df.iloc[1,2]
    TN26773=cm_df.iloc[0,0]+cm_df.iloc[0,1]+cm_df.iloc[1,0]+cm_df.iloc[1,1]
    #Sensibilidad para cada clase y general
    Sen32991=TP32991/(TP32991+FN32991)
    Sen16445=TP16445/(TP16445+FN16445)
    Sen26773=TP26773/(TP26773+FN26773)
    sensitivity=(Sen32991+Sen16445+Sen26773)/3
    #Specifity
    Sp32991=TN32991/(TN32991+FP32991)
    Sp16445=TN16445/(TN16445+FP16445)
    Sp26773=TN26773/(TN26773+FP26773)
    specificity=(Sp32991+Sp16445+Sp26773)/3

    #ROC AUC
    y_proba = pipeline.predict_proba(train_data)
    classes = pipeline.classes_
    #calculo
    roc_auc=roc_auc_score(target, y_proba, labels = classes, multi_class = 'ovr', average = 'macro')
    # Guardar las métricas y detalles del entrenamiento en un archivo de texto
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"metricas_train.txt"
    with open(filename, 'w') as file:
        file.write(f"Fecha y hora: {timestamp}\n")
        file.write(f"Tiempo de entrenamiento: {training_time} segundos\n")
        file.write(f"Métricas de entrenamiento:\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Specificity: {specificity}\n")
        file.write(f"Sensitivity: {sensitivity}\n")
        file.write(f"ROC-AUC: {roc_auc}\n")
    
    print(f"Entrenamiento finalizado. Métricas guardadas en {filename}")
#========================================PREDICT=============================================================

def predict(test_data, output_file,train_data,target):
    
#=================================== CONFIGURACION PIPELINE ======================================================
    
    #imputación de variables categoricas con indicador de Faltante (Missing)
    CATEGORICAL_VARS_WITH_NA_MISSING = ['Credit_Mix','Occupation']
    #Imputación de variables numéricas
    NUMERICAL_VARS_WITH_NA_MEAN = ['Num_Credit_Inquiries','Monthly_Balance']
    #Variables para transfomraicón Yeo
    NUMERICAL_YEON_VARS =['Annual_Income','Interest_Rate','Outstanding_Debt','Num_Bank_Accounts','Num_Credit_Card',
    'Num_Credit_Inquiries','Delay_from_due_date','Monthly_Balance']
    #Variables para codificación por frecuencia
    CATEGORICAL_VARS = ['SSN','Occupation','Payment_of_Min_Amount','Payment_Behaviour','Month','Credit_Mix']
    #Variables a utilzar en el entrenamiento
    FEATURES = ['Annual_Income','Interest_Rate','Outstanding_Debt','Num_Bank_Accounts','Num_Credit_Card',
    'Num_Credit_Inquiries','Delay_from_due_date','Monthly_Balance','Month','SSN','Occupation','Credit_Mix',
    'Payment_of_Min_Amount','Payment_Behaviour']

    train_data = train_data[FEATURES]

    for c in NUMERICAL_YEON_VARS:
        minimo=train_data[c].min()
        if minimo<=0:
            train_data[c]=train_data[c]+abs(minimo)+1
        else:
            train_data[c]=train_data[c]
    
    test_data = test_data[FEATURES]

    for c in NUMERICAL_YEON_VARS:
        minimo=test_data[c].min()
        if minimo<=0:
            test_data[c]=test_data[c]+abs(minimo)+1
        else:
            test_data[c]=test_data[c]

 #=================================== CONSTRUCCION DE PIPELINE ======================================================  
 #  
    # Crear el pipeline de ingeniería de características y modelo
    pipeline2= Pipeline([
    
    #=========== IMPUTACIONES ===============
    
    #1. Imputacion de variables categóricas con indicador de faltante
    ('missing_imputation',
        CategoricalImputer(imputation_method='missing', variables=CATEGORICAL_VARS_WITH_NA_MISSING)
    ),
    
 
        #3. Indicador faltane en variables numericas para imputación
    ('missing_indicator_numeric',
        AddMissingIndicator(variables=NUMERICAL_VARS_WITH_NA_MEAN)
    ),

    
    #4. Imputación de variables numéricas
    ('mean_imputation',
         MeanMedianImputer(imputation_method='mean', variables=NUMERICAL_VARS_WITH_NA_MEAN)
    ),
    
    #============= CODIFICACIÓN DE VARIABLES CATEGORICAS NOMINALES ==================
    ('rare_label_encoder',
        RareLabelEncoder(n_categories=1, tol=0.01, variables=CATEGORICAL_VARS)
    ),
    
    ('categorical_encoder',
        OrdinalEncoder(encoding_method='ordered', variables=CATEGORICAL_VARS)
    ),

    #=============== TRATAMIENTO DE OUTLIERS ============
    ('Tratamiento_outliers',
        Winsorizer(capping_method='iqr', variables=NUMERICAL_YEON_VARS)),
        
    #=============== TRANSFORMACIÓN DE VARIABLES CONTINUAS ============
    
    ('yeo_transformer',
        YeoJohnsonTransformer(variables=NUMERICAL_YEON_VARS)
    ),

     #=============== SCALER ============
    ('scaler',
        StandardScaler(),
    ),
         #=============== MODELO ============
    ('modelo',
       DecisionTreeClassifier(max_depth= 10, max_features=14)
    )
    ])
    
    # Entrenar el modelo
    pipeline2.fit(train_data, target)

    # Realizar las predicciones en los datos de prueba
    predictions = pipeline2.predict(test_data)
    
    # Crear un dataframe con las predicciones
    df_predictions = pd.DataFrame({'Prediction': predictions})
    creditMapp = {'Prediction': {1:"Poor",2:"Standard", 3:"Good"}}
    df_predictions = df_predictions.replace(creditMapp)
    # Guardar las predicciones en un archivo CSV
    df_predictions.to_csv(output_file, index=False)
    
    print(f"Predicciones guardadas en {output_file}")

#==================================================PREPARACION DATASET========================================================================

#========================== LECTURA DATASET ==========================================================================
dataTrain = pd.read_csv("train.csv")

#========================== CONVERSION NUMERO ==========================================================================

# En este caso 'numero_' debe ser 'numero' por tanto se hace el cambio '_' = ''
dataTrain['Annual_Income'] = dataTrain['Annual_Income'].str.replace('_', '')
dataTrain['Outstanding_Debt'] = dataTrain['Outstanding_Debt'].str.replace('_', '')
dataTrain['Age'] = dataTrain['Age'].str.replace('_', '')
dataTrain['Num_of_Loan'] = dataTrain['Num_of_Loan'].str.replace('_', '')
# En este caso '_' debe ser '0' porque significa sin cambio en limite de credito, por tanto se reemplaza '_' = '0'
dataTrain['Changed_Credit_Limit'] = dataTrain['Changed_Credit_Limit'].str.replace('_', '0')
# En este caso solo hay un numero con problema: ' __10000__' por tanto se sustituye con 10000
dataTrain['Amount_invested_monthly'] = dataTrain['Amount_invested_monthly'].str.replace('__10000__', '10000')
# En este caso solo hay un numero con problema: _-3333333_ por tanto se sustituye con -3333333
dataTrain['Monthly_Balance'] = dataTrain['Monthly_Balance'].str.replace('__-333333333333333333333333333__', '33333333333333333')
#TIPO FLOAT
dataTrain['Annual_Income'] = dataTrain['Annual_Income'].astype(float)
dataTrain['Changed_Credit_Limit'] = dataTrain['Changed_Credit_Limit'].astype(float)
dataTrain['Outstanding_Debt'] = dataTrain['Outstanding_Debt'].astype(float)
dataTrain['Amount_invested_monthly'] = dataTrain['Amount_invested_monthly'].astype(float)
dataTrain['Monthly_Balance'] = dataTrain['Monthly_Balance'].astype(float)
#TIPO INT
dataTrain['Age'] = dataTrain['Age'].astype(int)
dataTrain['Num_of_Loan'] = dataTrain['Num_of_Loan'].astype(int)

#========================== BALANCEO TARGET ==========================================================================

df_standard = dataTrain[dataTrain['Credit_Score'] == 'Standard']
df_poor = dataTrain[dataTrain['Credit_Score'] == 'Poor']
df_good = dataTrain[dataTrain['Credit_Score'] == 'Good']
cantidadstandard = 2*df_good.shape[0]
df_standard = df_standard.sample(n=cantidadstandard, random_state=2023)
dataTrain = pd.concat([df_standard, df_poor, df_good])

#========================== COMPLETE CASE ANALYSIS ====================================================================

dataTrain = dataTrain[dataTrain['Payment_Behaviour'].str.contains('!@9#%8') == False]

#========================== CODIFICACION TARGET Y SEPARACION TRAIN - TEST ==============================================

creditMapp = {'Credit_Score': {"Poor":1,"Standard": 2, "Good": 3}}
dataTrain = dataTrain.replace(creditMapp)
X_train, X_test, y_train, y_test = train_test_split(
    dataTrain.drop(['ID', 'Credit_Score','Changed_Credit_Limit','Credit_Utilization_Ratio','Total_EMI_per_month',
                    'Amount_invested_monthly'], axis=1),
    dataTrain['Credit_Score'],
    test_size=0.3,
    random_state=2022)

#====================================///////// === MAIN ===////////========================================================

# Ejecutar las funciones según la opción seleccionada por el usuario
flag=True
while flag== True:
    option = input("Seleccione la opción (train/predict/salir): ")
    if option == "train":
            train_model(X_train, y_train)
            
    elif option == "predict":
            # Especificar el nombre del archivo de salida para las predicciones
            output_file = input("Ingrese el nombre del archivo de salida para las predicciones (ejemplo.csv): ")
            predict(X_test, output_file,X_train, y_train)

    elif option == "salir":
            flag=False

    elif option == "ver":
            print(f"test:{X_test.columns}")
            print(f"train:{X_train.columns}")

    else:
            print("Opción inválida.")
quit()
