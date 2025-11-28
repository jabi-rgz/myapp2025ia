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
    df['credit amount'] = np.log(df['credit amount'])

    # separamos la variable objetivo (y) de las variables predictoras (X)
    X = df.drop('Risk_bad', axis=1).values
    y = df['Risk_bad'].values
    
    # Spliting X and y into train and test version
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

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
    
    return X_train, X_test, y_train, y_test


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
    # Se usa 'binary_crossentropy' aunque la salida sea 2 clases, debido al uso de 'sigmoid' y el one-hot encoding.
    NN_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    st.text("Resumen del Modelo de Red Neuronal:")
    NN_model.summary(print_fn=lambda x: st.text(x))
    
    return NN_model

def TrainningNN(X_train, X_test, y_train, y_test):
    """Entrena la Red Neuronal y evalúa los resultados."""
    
    # 1. Preparación de Datos
    # Convertir X_train a numpy array de tipo float32 (¡La clave para el error!)
    # Esto asegura la compatibilidad con Keras/TensorFlow.
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


# --- Aplicación Principal de Streamlit ---

st.title("Credit Card App")

# Inicialización de Session State para persistir datos
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None


# Definir las opciones de página
pages = ["Cargar Datos", "Explorar Datos", "Feature Engineering", "Modelado", "Neural Network", "Prediccion"]

# Mostrar un menú para seleccionar la página
selected_page = st.sidebar.multiselect("Seleccione una página", pages)

# Condicionales para mostrar la página seleccionada
if "Cargar Datos" in selected_page:
    st.header("Cargar Datos")
    # Cargar archivo CSV usando file uploader
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    
    # Si el archivo se cargó correctamente
    if uploaded_file is not None:
        # Leer archivo CSV usando Pandas y almacenarlo en session_state
        st.session_state['dataset'] = pd.read_csv(uploaded_file)
        st.subheader("Vista Previa de Datos Originales")
        st.dataframe(st.session_state['dataset'].head())
    elif st.session_state['dataset'] is not None:
        st.subheader("Datos Cargados Previamente")
        st.dataframe(st.session_state['dataset'].head())

if "Explorar Datos" in selected_page:
    if st.session_state['dataset'] is not None:
        st.header("Explorar Datos")
        # Usar una copia para el EDA para no modificar el original
        get_eda(st.session_state['dataset'].copy())
    else:
        st.warning("⚠️ Por favor, cargue los datos en la sección 'Cargar Datos' primero.")

if "Feature Engineering" in selected_page:
    if st.session_state['dataset'] is not None:
        st.header("Feature Engineering")
        st.write("Aplicando transformaciones y codificación One-Hot...")
        
        # Aplicar Feature Engineering y actualizar el dataset en session_state
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
        
        X_train, X_test, y_train, y_test = modelling(st.session_state['dataset_processed'].copy())
        
        # Almacenar los splits de datos en session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        
        st.success("✅ Splits de datos de entrenamiento y prueba almacenados para la Red Neuronal.")

    else:
        st.warning("⚠️ Por favor, ejecute los pasos 'Cargar Datos' y 'Feature Engineering' primero.")

if "Neural Network" in selected_page:
    st.header("Neural Network")
    
    if st.session_state['X_train'] is not None:
        st.write(f"TensorFlow Version: {tf.__version__}")
        
        # Recuperar los datos de session_state
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        
        # Llamar a la función con las correcciones de dtype
        modelNN = TrainningNN(X_train, X_test, y_train, y_test)
        st.session_state['NN_model'] = modelNN
        
    else:
        st.warning("⚠️ Por favor, ejecute el paso **Modelado** primero para preparar los datos de entrenamiento.")

if "Prediccion" in selected_page:
    if 'NN_model' in st.session_state and st.session_state['NN_model'] is not None:
        st.header("Prediccion (Usando el Modelo NN)")
        st.info("Aquí iría la interfaz de usuario para ingresar nuevos datos y obtener una predicción del riesgo de crédito.")
        # Se podría implementar la lógica de predicción aquí.
    else:
        st.warning("⚠️ Por favor, entrene la Red Neuronal en la sección 'Neural Network' primero.")
