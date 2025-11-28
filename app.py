import streamlit as st 
import numpy as np 
import pandas as pd
import plotly.offline as py 
import plotly.graph_objs as go 
import plotly.tools as tls 

# librerias de modelado
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# librerias de redes neuronales
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Configuración inicial de Streamlit
st.set_page_config(layout="wide", page_title="Credit Card App")


# --- Funciones de Análisis y Preparación de Datos ---

def get_eda(dataset):
    """Genera y muestra gráficos de distribución para variables clave."""
    st.subheader("Distribuciones de Variables Clave")

    # Lista de columnas a graficar
    cols_to_plot = {
        "housing": "Housing Distribution",
        "sex": "Gender Distribution",
        "job": "Job Distribution",
        "saving_accounts": "Saving Accounts Distribution",
        "checking account": "Checking Account Distribution",
        "duration": "Duration Distribution (Months)",
        "purpose": "Purpose Distribution"
    }

    for col, title in cols_to_plot.items():
        if col not in dataset.columns:
            st.warning(f"La columna '{col}' no está presente en el dataset.")
            continue
            
        # Distribución de Creditos por Columna
        try:
            trace0 = go.Bar(
                x=dataset[dataset["risk"] == 'good'][col].value_counts().index.values,
                y=dataset[dataset["risk"] == 'good'][col].value_counts().values,
                name='Good credit'
            )

            trace1 = go.Bar(
                x=dataset[dataset["risk"] == 'bad'][col].value_counts().index.values,
                y=dataset[dataset["risk"] == 'bad'][col].value_counts().values,
                name="Bad Credit"
            )

            data = [trace0, trace1]

            layout = go.Layout(title=title)
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar gráfico para la columna {col}: {e}")


def feature_engineering(dataset):
    """Aplica ingeniería de características y codificación One-Hot."""
    
    # Usar una copia para evitar SettingWithCopyWarning
    df = dataset.copy()

    # crear categorias por edad
    interval = (18, 25, 35, 60, 120)
    cats = ['Student', 'Young', 'Adult', 'Senior']
    df['Age_cat'] = pd.cut(df.age, interval, labels=cats, right=False)

    # reemplazar los valores nan
    df['saving_accounts'] = df['saving_accounts'].fillna('no_inf')
    df['checking account'] = df['checking account'].fillna('no_inf')

    # convertir a dummies las variables categoricas
    df = df.merge(pd.get_dummies(df.purpose, drop_first=True, prefix='purpose'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df.sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df.housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df["saving_accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df.risk, prefix='Risk'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df["checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)
    
    # Asegurar que las columnas existan antes de eliminarlas
    cols_to_drop = ["Unnamed: 0", "saving_accounts", "checking account", "purpose", "sex", 
                    "housing", "Age_cat", "risk"]
    
    for col in cols_to_drop:
        if col in df.columns:
            del df[col]

    # La columna 'Risk_good' será redundante ya que 'Risk_bad' es la variable objetivo
    if "Risk_good" in df.columns:
        del df["Risk_good"]

    return df


def modelling(dataset):
    """Entrena modelos clásicos y realiza validación cruzada."""
    df = dataset.copy()
    
    # aplicamos una funcion logaritmo para ajustar los valores
    # Asegurarse de que el nombre de la columna es correcto
    if 'credit amount' in df.columns:
        df['credit amount'] = np.log(df['credit amount'])

    # separamos la variable objetivo (y) de las variables predictoras (X)
    X = df.drop('Risk_bad', axis=1) # Mantenemos como DataFrame para obtener los nombres de las columnas
    y = df['Risk_bad'].values
    
    # Guardamos los nombres de las características para usarlas en la predicción
    feature_names = X.columns.tolist()

    # Spliting X and y into train and test version
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.30, random_state=42)

    # Prepapar los modelos
    models = []
    models.append(('LGR', LogisticRegression(max_iter=500)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier(random_state=42)))
    models.append(('SVM', SVC(gamma='auto', random_state=42)))

    # Entrenamos y validamos cada modelo
    results = []
    names = []
    scoring = 'recall' # Usando recall como métrica de evaluación

    st.subheader("Resultados de Validación Cruzada (Métrica: Recall)")
    for name, model in models:
        with st.spinner(f"Entrenando {name}..."):
            kfold = KFold(n_splits=10, random_state=None, shuffle=True)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            st.write(f"**{name}:** Media={cv_results.mean():.4f}, Desv.Std.={cv_results.std():.4f}")

    # Graficar resultados (BoxPlot)
    resultsBox = pd.DataFrame(np.array(results).T, columns=names)
    
    fig = go.Figure()
    for name in names:
        fig.add_trace(go.Box(y=resultsBox[name], name=name))
    
    fig.update_layout(title_text='Comparación de Modelos por Recall', yaxis_title='Recall Score')
    st.plotly_chart(fig, use_container_width=True)
    
    # Devolvemos también los nombres de las características
    return X_train, X_test, y_train, y_test, feature_names


# --- Funciones de Redes Neuronales ---

def nn_model(learning_rate, y_train_categorical, X_train):
    """Define la arquitectura de la Red Neuronal."""
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal', input_dim=X_train.shape[1], activation='relu', dtype='float32'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu', dtype='float32'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu', dtype='float32'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu', dtype='float32'))

    # The Output Layer :
    # La salida tiene 2 neuronas (una para good, otra para bad) con activación sigmoid
    NN_model.add(Dense(2, kernel_initializer='normal', activation='sigmoid', dtype='float32'))

    # Compile the network :
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Se usa 'binary_crossentropy' debido a la salida de 2 clases con one-hot encoding
    NN_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    st.text("Resumen del Modelo de Red Neuronal:")
    # Usar una función lambda o un StringIO para capturar el resumen y mostrarlo
    import io
    buffer = io.StringIO()
    NN_model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    st.text(buffer.getvalue())
    
    return NN_model

def TrainningNN(X_train, X_test, y_train, y_test):
    """Entrena la Red Neuronal y evalúa los resultados."""
    
    # 1. Preparación de Datos
    # Convertir X_train a numpy array de tipo float32
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    
    # Convertir las etiquetas a formato categórico (One-Hot Encoding)
    y_test_categorical = to_categorical(y_test, num_classes=2)
    y_train_categorical = to_categorical(y_train, num_classes=2)

    # Semilla para aleatorios
    np.random.seed(7)

    # 2. Definición y Entrenamiento
    NN_model = nn_model(1e-4, y_train_categorical, X_train)
    nb_epochs = 100
    batch_size = 50
    
    with st.spinner(f"Entrenando Red Neuronal por {nb_epochs} épocas..."):
        # verbose=0 para suprimir la salida durante el entrenamiento en Streamlit
        history = NN_model.fit(
            X_train, y_train_categorical, 
            epochs=nb_epochs, 
            batch_size=batch_size, 
            verbose=0,
            validation_data=(X_test, y_test_categorical)
        )
    
    st.success(f"✅ Entrenamiento de la Red Neuronal finalizado después de {nb_epochs} épocas.")

    # 3. Evaluación
    # Predicción de probabilidades (ej: [0.9, 0.1] para clase 0)
    y_pred_nn_prob = NN_model.predict(X_test, verbose=0)
    # Convertir probabilidades a la clase predicha (0 o 1)
    y_pred_nn = np.argmax(y_pred_nn_prob, axis=1)

    # Métricas de Evaluación
    accuracy = accuracy_score(y_test, y_pred_nn)
    report = classification_report(y_test, y_pred_nn, output_dict=True, zero_division=0)

    st.subheader("Evaluación del Modelo de Red Neuronal")
    st.metric(label="Accuracy", value=f"{accuracy:.4f}")
    
    st.markdown("**Reporte de Clasificación:**")
    st.dataframe(pd.DataFrame(report).transpose().round(2))
    
    return NN_model

def predictionForm():
    """Formulario de entrada de datos para la predicción."""
    # Asegurarse de que el modelo y las features estén disponibles
    if 'NN_model' not in st.session_state or st.session_state['NN_model'] is None:
        st.warning("El modelo de Red Neuronal no está entrenado. Por favor, ejecute la sección 'Neural Network'.")
        return
    if 'feature_names' not in st.session_state:
        st.warning("Los nombres de las características de entrenamiento no están disponibles. Por favor, ejecute la sección 'Modelado'.")
        return
    
    # Opciones de categorías (corregidas para ser más completas)
    option_sex = ['male', 'female']
    option_job = [0, 1, 2, 3] # La variable job es numérica/ordinal
    option_housing = ['own', 'rent', 'free'] # 'free' es otra categoría común
    # Incluimos 'no_inf' ya que se usa para rellenar NaNs en feature_engineering
    option_saving = ['moderate', 'quite rich', 'rich', 'little', 'no_inf'] 
    option_checking = ['moderate', 'rich', 'little', 'no_inf'] 
    option_duration = [6, 7, 8, 9, 10, 11, 12, 15, 18, 24, 27, 30, 36, 42, 45, 48, 60]
    option_purpose = ['radio/TV', 'education', 'furniture/equipment', 'car',
        'domestic appliances', 'repairs', 'vacation/others']

    st.markdown("Ingrese los detalles del cliente para predecir el riesgo crediticio:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('Edad', min_value=18, max_value=120, value=30, step=1)
            select_sex = st.selectbox('Género', option_sex)
            select_housing = st.selectbox('Vivienda', option_housing)
            # Nota: la columna original se llama 'checking account'
            select_checking = st.selectbox('Cuenta de Crédito', option_checking) 
            select_duration = st.selectbox('Duración (meses)', option_duration)
        with col2:
            select_job = st.selectbox('Trabajo', option_job)
            select_saving = st.selectbox('Cuenta de Ahorro', option_saving)
            # Nota: la columna original se llama 'credit amount'
            credit_amount = st.number_input('Monto del Crédito', min_value=100, max_value=20000, value=4000, step=100) 
            select_purpose = st.selectbox('Propósito', option_purpose)
        
        submitted = st.form_submit_button('Predecir Riesgo')
        
        if submitted:
            # Recuperar el modelo y los nombres de las features de session state
            modelNN = st.session_state['NN_model']
            feature_names = st.session_state['feature_names']
            
            prediction(age, select_sex, select_job, select_housing, select_saving, select_checking, credit_amount, select_duration, select_purpose, modelNN, feature_names)
    
def prediction(age, sex, job, housing, saving, checking, amount, duration, purpose, modelNN, feature_names):
    """
    Prepara la entrada de datos del usuario, la transforma para que coincida con 
    el entrenamiento y realiza la predicción con el modelo NN.
    """
    try:
        # 1. Crear el DataFrame de entrada con datos crudos (con nombres originales de columnas)
        data = {
            "age": [age], 
            "sex": [sex], 
            "job": [job],
            "housing": [housing], 
            "saving_accounts": [saving],
            "checking account": [checking], 
            "credit amount": [amount], 
            "duration": [duration], 
            "purpose": [purpose]
        }

        df_user = pd.DataFrame(data)
        
        # Aplicamos logaritmo al monto del crédito, igual que en el entrenamiento
        df_user['credit amount'] = np.log(df_user['credit amount'])

        # 2. Replicar el Feature Engineering
        
        # A. Categoría de Edad
        interval = (18, 25, 35, 60, 120)
        cats = ['Student', 'Young', 'Adult', 'Senior']
        # pd.cut usa el rango, y .astype(str) lo convierte a la etiqueta para dummies
        df_user['Age_cat'] = pd.cut(df_user['age'], interval, labels=cats, right=False).astype(str)
        
        # B. Aplicar One-Hot Encoding
        cols_to_dummy = ["purpose", "sex", "housing", "saving_accounts", "checking account", "Age_cat"]
        
        # Crear Dummies (igual que en feature_engineering: drop_first=True)
        df_dummies = pd.get_dummies(df_user[cols_to_dummy], drop_first=True, prefix={
            "purpose": "purpose", 
            "sex": "Sex", 
            "housing": "Housing", 
            "saving_accounts": "Savings", 
            "checking account": "Check", 
            "Age_cat": "Age_cat"
        })
        
        # C. Limpieza del DataFrame de Usuario
        # Eliminar las columnas categóricas originales y 'age' original (ya usada para Age_cat)
        # Ojo: 'age' y 'job' se mantienen como numéricas. Solo eliminamos las columnas que se convirtieron a dummies.
        cols_to_drop_from_base = cols_to_dummy
        df_processed = df_user.drop(columns=cols_to_drop_from_base)
        
        # D. Combinar numéricos y dummies
        df_processed = pd.concat([df_processed, df_dummies], axis=1)

        # 3. Normalizar y Ordenar las Features (¡La Corrección Clave!)
        
        # Crear un DataFrame de todas las características esperadas inicializadas a 0
        # Esto garantiza que TODAS las features de entrenamiento existan y estén en orden
        X_predict_final = pd.DataFrame(0.0, index=[0], columns=feature_names)
        
        # Actualizar las columnas con los valores calculados
        # Usamos .columns.intersection para solo copiar las columnas que realmente existen
        cols_to_update = df_processed.columns.intersection(X_predict_final.columns)
        
        for col in cols_to_update:
            # Asegurar que el tipo de dato sea float para la red neuronal
            X_predict_final.loc[0, col] = df_processed.loc[0, col]
            
        # El vector final de predicción (en el orden correcto)
        # ¡IMPORTANTE! Asegurar el dtype correcto (float32)
        X_predict_values = X_predict_final.values.astype(np.float32)

        # 4. Mostrar predicción
        st.write("### Resultado de la Predicción")
        st.write("Vector de Features (Ordenado para el Modelo):")
        st.dataframe(X_predict_final)

        # Calcular Prediccion
        valuePredict = modelNN.predict(X_predict_values, verbose=0)
        
        # valuePredict es un array de probabilidades, ej: [[Prob_Good, Prob_Bad]]
        prob_good = valuePredict[0][0]
        prob_bad = valuePredict[0][1]

        st.markdown(f"**Probabilidad de Crédito Bueno (Clase 0):** `{prob_good:.4f}`")
        st.markdown(f"**Probabilidad de Crédito Malo (Clase 1):** `{prob_bad:.4f}`")
        
        if prob_bad > prob_good:
            st.error("#### 🚫 Predicción del Modelo: CRÉDITO MALO")
        else:
            st.success("#### ✅ Predicción del Modelo: CRÉDITO BUENO")
            
    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")
        st.warning("Asegúrese de que todos los campos del formulario sean válidos (ej: Monto de Crédito debe ser un número).")


# --- Aplicación Principal de Streamlit ---

st.title("Credit Card Risk Prediction App")

# Inicialización de Session State para persistir datos
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
# Nuevos estados para la predicción
if 'NN_model' not in st.session_state:
    st.session_state['NN_model'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None


# Definir las opciones de página
pages = ["Cargar Datos", "Explorar Datos", "Feature Engineering", "Modelado", "Neural Network", "Prediccion"]

# Mostrar un menú para seleccionar la página
selected_page = st.sidebar.multiselect("Seleccione una página", pages, default=["Cargar Datos"]) # Default to Cargar Datos

# Condicionales para mostrar la página seleccionada
if "Cargar Datos" in selected_page:
    st.header("Cargar Datos")
    # Cargar archivo CSV usando file uploader
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state['dataset'] = pd.read_csv(uploaded_file)
        st.subheader("Vista Previa de Datos Originales")
        st.dataframe(st.session_state['dataset'].head())
    elif st.session_state['dataset'] is not None:
        st.subheader("Datos Cargados Previamente")
        st.dataframe(st.session_state['dataset'].head())

if "Explorar Datos" in selected_page:
    if st.session_state['dataset'] is not None:
        st.header("Explorar Datos")
        get_eda(st.session_state['dataset'].copy())
    else:
        st.warning("⚠️ Por favor, cargue los datos en la sección 'Cargar Datos' primero.")

if "Feature Engineering" in selected_page:
    if st.session_state['dataset'] is not None:
        st.header("Feature Engineering")
        st.write("Aplicando transformaciones y codificación One-Hot...")
        
        processed_dataset = feature_engineering(st.session_state['dataset'].copy())
        st.session_state['dataset_processed'] = processed_dataset
        
        st.subheader("Dataset Procesado (Variables One-Hot)")
        st.dataframe(processed_dataset.head())
        st.write(f"Shape del Dataset Procesado: {processed_dataset.shape}")
    else:
        st.warning("⚠️ Por favor, cargue los datos y ejecute 'Cargar Datos' primero.")

if "Modelado" in selected_page:
    if 'dataset_processed' in st.session_state and st.session_state['dataset_processed'] is not None:
        st.header("Entrenamiento con Modelos Clásicos")
        
        # --- CORRECCIÓN CLAVE: Capturar los nombres de las features ---
        X_train, X_test, y_train, y_test, feature_names = modelling(st.session_state['dataset_processed'].copy())
        
        # Almacenar los splits de datos en session_state y los nombres de las features
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['feature_names'] = feature_names # <<< Almacenado
        
        st.success("✅ Splits de datos de entrenamiento y prueba almacenados para la Red Neuronal.")

    else:
        st.warning("⚠️ Por favor, ejecute los pasos 'Cargar Datos' y 'Feature Engineering' primero.")

if "Neural Network" in selected_page:
    st.header("Neural Network")
    
    if st.session_state['X_train'] is not None:
        st.write(f"TensorFlow Version: {tf.__version__}")
        
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        
        modelNN = TrainningNN(X_train, X_test, y_train, y_test)
        st.session_state['NN_model'] = modelNN
        
    else:
        st.warning("⚠️ Por favor, ejecute el paso **Modelado** primero para preparar los datos de entrenamiento.")

if "Prediccion" in selected_page:
    # Comprobar que el modelo y las features estén en session_state
    if 'NN_model' in st.session_state and st.session_state['NN_model'] is not None:
        if 'feature_names' in st.session_state and st.session_state['feature_names'] is not None:
            st.header("Prediccion (Usando el Modelo NN)")
            predictionForm() # Llamar sin argumentos
        else:
            st.warning("⚠️ Los nombres de las características no están cargados. Ejecute 'Modelado' primero.")
    else:
        st.warning("⚠️ Por favor, entrene la Red Neuronal en la sección 'Neural Network' primero.")
