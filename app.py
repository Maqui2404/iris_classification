import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

st.set_page_config(
    page_title="Clasificación de Flores 🌸",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Estilos generales */
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stApp {
            background-color: #0E1117;
        }
        h1, h2, h3 {
            color: #E0E0E0;
        }
        .title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            background: linear-gradient(90deg, #9C27B0, #3F51B5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
            padding: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #BDBDBD;
            margin-bottom: 30px;
        }
        .metric-box {
            background: linear-gradient(145deg, #1A1C25, #2A2D3A);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0px 6px 10px rgba(0,0,0,0.2);
            transition: transform 0.3s, box-shadow 0.3s;
            height: 100%;
        }
        .metric-box:hover {
            transform: translateY(-5px);
            box-shadow: 0px 10px 15px rgba(0,0,0,0.3);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: bold;
            background: linear-gradient(90deg, #00BCD4, #4CAF50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .metric-label {
            font-size: 1rem;
            color: #E0E0E0;
        }
        .stButton button {
            background: linear-gradient(90deg, #673AB7, #3F51B5);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0px 6px 8px rgba(0,0,0,0.2);
        }
        .prediction-box {
            background: linear-gradient(145deg, #1A1C25, #2A2D3A);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 6px 10px rgba(0,0,0,0.2);
            margin: 20px 0;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
            }
        }
        div[data-testid="stDataFrame"] {
            background-color: #1A1C25;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }
        div[data-testid="stNumberInput"] label {
            color: #E0E0E0;
        }
        div[data-testid="stNumberInput"] input {
            background-color: #1A1C25;
            color: #E0E0E0;
            border: 1px solid #3F51B5;
            border-radius: 5px;
        }
        .stProgress div {
            background-color: #3F51B5;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            border-top: 1px solid #333;
            color: #BDBDBD;
            opacity: 0.8;
            transition: opacity 0.3s;
        }
        .footer:hover {
            opacity: 1;
        }
        .feature-importance {
            background: linear-gradient(145deg, #1A1C25, #2A2D3A);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 6px 10px rgba(0,0,0,0.2);
        }
        .section-header {
            background: linear-gradient(145deg, #1A1C25, #2A2D3A);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }
        /* Mejoras en los tabs */
        .stTabs [role="tablist"] {
            background-color: #1A1C25;
            border-radius: 10px;
            padding: 5px;
            margin-bottom: 20px;
        }
        .stTabs [role="tab"] {
            color: #BDBDBD;
            background-color: #1A1C25;
            border-radius: 10px;
            padding: 10px 20px;
            margin: 0 5px;
            transition: all 0.3s;
        }
        .stTabs [role="tab"]:hover {
            background-color: #2A2D3A;
            color: #E0E0E0;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, #673AB7, #3F51B5);
            color: white;
            font-weight: bold;
        }
        /* Botón flotante */
        .float-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(90deg, #673AB7, #3F51B5);
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 24px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .float-button:hover {
            transform: translateY(-5px);
            box-shadow: 0px 6px 8px rgba(0,0,0,0.2);
        }
    </style>
    <script>
        // Función para el botón flotante
        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    </script>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_datos():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    return X, y, feature_names, class_names

X, y, feature_names, class_names = cargar_datos()
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [class_names[i] for i in y]

especies_colores = {
    'setosa': '#FF6D00',
    'versicolor': '#2979FF',
    'virginica': '#00C853'
}

if 'inicio' not in st.session_state:
    st.session_state['inicio'] = True
    barra_progreso = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        barra_progreso.progress(i + 1)
    barra_progreso.empty()

st.markdown('# 🌸 Clasificación Inteligente de Iris 🌿', unsafe_allow_html=True)
st.markdown('### Explora y predice especies de flores mediante Machine Learning avanzado',
            unsafe_allow_html=True)


st.markdown("""
<div style="text-align: right; margin-bottom: 25px;">
    <p>Desarrollado por <strong>Marco Mayta</strong></p>
    <p>
        <a href="https://github.com/Maqui2404" target="_blank" style="text-decoration: none; margin-right: 10px;">
            <i class="fab fa-github"></i> GitHub
        </a> 
        <a href="https://www.linkedin.com/in/marco-mayta-835781170/" target="_blank" style="text-decoration: none; margin-right: 10px;">
            <i class="fab fa-linkedin"></i> LinkedIn
        </a> 
        <a href="https://maqui2404.github.io/PortafolioMarco.github.io/" target="_blank" style="text-decoration: none;">
            <i class="fas fa-globe"></i> Portafolio
        </a>
    </p>
</div>

<!-- Cargar Font Awesome -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
""", unsafe_allow_html=True)


tabs = st.tabs(["📊 Exploración", "🤖 Modelo", "🔮 Predicción", "📈 Visualizaciones"])

st.markdown("""
    <button class="float-button" onclick="scrollToTop()">↑</button>
""", unsafe_allow_html=True)


with tabs[0]:
    st.markdown('<div class="section-header"><h2>📊 Exploración del Dataset</h2></div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(df.head(10), use_container_width=True)

        st.write("Estadísticas básicas:")
        stats_df = df.drop('species', axis=1).describe().round(2)
        st.dataframe(stats_df, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="species", title="Distribución de Clases",
                           color="species", color_discrete_map=especies_colores,
                           template="plotly_dark")
        fig.update_layout(
            plot_bgcolor='rgba(26, 28, 37, 0.8)',
            paper_bgcolor='rgba(26, 28, 37, 0.8)',
            font=dict(color='#E0E0E0'),
            title_font=dict(size=24, color='#E0E0E0'),
            legend_title_font=dict(size=14),
            title_x=0.5,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlación entre características")
    corr = df.drop('species', axis=1).corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis',
                    template="plotly_dark", zmin=-1, zmax=1)
    fig.update_layout(
        plot_bgcolor='rgba(26, 28, 37, 0.8)',
        paper_bgcolor='rgba(26, 28, 37, 0.8)',
        font=dict(color='#E0E0E0'),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.markdown('<div class="section-header"><h2>🤖 Entrenamiento del Modelo</h2></div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.slider("Número de árboles", 10, 200, 100, 10)
    with col2:
        max_depth = st.slider("Profundidad máxima", 2, 20, 10, 1)
    with col3:
        test_size = st.slider("Porcentaje de test", 0.1, 0.4, 0.2, 0.05)

    train_button = st.button("🔄 Entrenar Modelo Avanzado")

    if train_button or 'model' in st.session_state:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Entrenando modelo: {i+1}%")
            time.sleep(0.01)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        status_text.success(
            f"✅ Modelo entrenado con éxito | Precisión: {accuracy:.2%}")

        report = classification_report(y_test, y_pred, output_dict=True)
        f1_scores = []
        for label in range(len(class_names)):
            if str(label) in report:
                f1_scores.append(report[str(label)]['f1-score'])
        f1 = np.mean(f1_scores) if f1_scores else 0.0

        st.markdown(
            "<h3 style='text-align: center; margin-top: 20px;'>📊 Métricas del Modelo</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-value">{n_estimators}</div>
                    <div class="metric-label">Árboles</div>
                </div>
            ''', unsafe_allow_html=True)

        with col2:
            st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-value">{accuracy:.2%}</div>
                    <div class="metric-label">Precisión</div>
                </div>
            ''', unsafe_allow_html=True)

        with col3:
            st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-value">{len(np.unique(y))}</div>
                    <div class="metric-label">Clases</div>
                </div>
            ''', unsafe_allow_html=True)

        with col4:
            st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-value">{f1:.2%}</div>
                    <div class="metric-label">F1-Score</div>
                </div>
            ''', unsafe_allow_html=True)

        st.subheader("📊 Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm,
                        x=[name for name in class_names],
                        y=[name for name in class_names],
                        text_auto=True,
                        color_continuous_scale='Viridis',
                        template="plotly_dark")
        fig.update_layout(
            plot_bgcolor='rgba(26, 28, 37, 0.8)',
            paper_bgcolor='rgba(26, 28, 37, 0.8)',
            font=dict(color='#E0E0E0'),
            xaxis_title="Predicción",
            yaxis_title="Real",
            title_font=dict(size=20),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🌟 Importancia de Características")
        feature_importance = model.feature_importances_
        feature_names_clean = [name.replace(
            '(cm)', '').strip() for name in feature_names]

        fig = px.bar(
            x=feature_importance,
            y=feature_names_clean,
            orientation='h',
            color=feature_importance,
            color_continuous_scale='Viridis',
            template="plotly_dark"
        )
        fig.update_layout(
            plot_bgcolor='rgba(26, 28, 37, 0.8)',
            paper_bgcolor='rgba(26, 28, 37, 0.8)',
            font=dict(color='#E0E0E0'),
            xaxis_title="Importancia",
            yaxis_title="Característica",
            height=400,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.session_state["model"] = model
        st.session_state["accuracy"] = accuracy
        st.session_state["feature_importance"] = feature_importance

with tabs[2]:
    st.markdown('<div class="section-header"><h2>🔮 Predicción en Tiempo Real</h2></div>',
                unsafe_allow_html=True)
    st.markdown("<p style='margin-bottom: 20px;'>Ajusta los controles deslizantes para ver la predicción actualizada instantáneamente.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider(
            "📏 Longitud del Sépalo (cm)", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("📐 Ancho del Sépalo (cm)", 2.0, 4.5, 3.5, 0.1)
    with col2:
        petal_length = st.slider(
            "🌿 Longitud del Pétalo (cm)", 1.0, 7.0, 1.4, 0.1)
        petal_width = st.slider("🌸 Ancho del Pétalo (cm)", 0.1, 2.5, 0.2, 0.1)

    input_features = [sepal_length, sepal_width, petal_length, petal_width]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=input_features,
        theta=feature_names,
        fill='toself',
        name='Entrada',
        line=dict(color='#9C27B0', width=2),
        fillcolor='rgba(156, 39, 176, 0.3)'
    ))

    for species in class_names:
        species_df = df[df['species'] == species]
        avg_values = species_df.drop('species', axis=1).mean().values

        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=feature_names,
            fill='toself',
            name=f'Promedio {species}',
            line=dict(width=1),
            opacity=0.7
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(input_features), df.drop(
                    'species', axis=1).max().max()) + 0.5]
            )
        ),
        showlegend=True,
        legend=dict(x=0, y=-0.2, orientation='h'),
        template="plotly_dark",
        plot_bgcolor='rgba(26, 28, 37, 0.8)',
        paper_bgcolor='rgba(26, 28, 37, 0.8)',
        font=dict(color='#E0E0E0'),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    predict_button = st.button("🔮 Predecir Especie")
    prediction_container = st.empty()

    if "model" in st.session_state:
        model = st.session_state["model"]
        nueva_muestra = np.array(
            [[sepal_length, sepal_width, petal_length, petal_width]])
        prediccion = model.predict(nueva_muestra)
        probabilidades = model.predict_proba(nueva_muestra)[0]
        especie_predicha = class_names[prediccion[0]]

        with prediction_container.container():
            st.markdown(f"""
                <div class="prediction-box">
                    <h2 style='text-align: center; color: #4CAF50;'>🌺 Especie Predicha: <span style='color: #FF9800;'>{especie_predicha.capitalize()}</span></h2>
                    <p style='text-align: center;'>Confianza: {probabilidades[prediccion[0]]:.2%}</p>
                </div>
            """, unsafe_allow_html=True)

            fig = px.bar(
                x=class_names,
                y=probabilidades,
                color=class_names,
                color_discrete_map=especies_colores,
                template="plotly_dark",
                labels={'x': 'Especie', 'y': 'Probabilidad'}
            )
            fig.update_layout(
                title="Probabilidades por especie",
                title_x=0.5,
                plot_bgcolor='rgba(26, 28, 37, 0.8)',
                paper_bgcolor='rgba(26, 28, 37, 0.8)',
                font=dict(color='#E0E0E0'),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Primero debes entrenar el modelo en la pestaña 'Modelo'.")

with tabs[3]:
    st.markdown('<div class="section-header"><h2>📈 Visualizaciones Avanzadas</h2></div>',
                unsafe_allow_html=True)

    viz_type = st.radio(
        "Selecciona el tipo de visualización:",
        ["Gráfico 3D", "Pair Plot", "Distribuciones", "Análisis PCA"],
        horizontal=True
    )

    if viz_type == "Gráfico 3D":
        st.subheader("🔍 Visualización 3D de Características")

        col1, col2, col3 = st.columns(3)
        with col1:
            x_feature = st.selectbox("Eje X", feature_names, 0)
        with col2:
            y_feature = st.selectbox("Eje Y", feature_names, 1)
        with col3:
            z_feature = st.selectbox("Eje Z", feature_names, 2)

        fig = px.scatter_3d(
            df,
            x=x_feature,
            y=y_feature,
            z=z_feature,
            color='species',
            color_discrete_map=especies_colores,
            size_max=10,
            opacity=0.8,
            template="plotly_dark",
            labels={
                x_feature: x_feature.replace('(cm)', '').strip(),
                y_feature: y_feature.replace('(cm)', '').strip(),
                z_feature: z_feature.replace('(cm)', '').strip()
            }
        )

        fig.update_layout(
            scene=dict(
                xaxis_title=x_feature.replace('(cm)', '').strip(),
                yaxis_title=y_feature.replace('(cm)', '').strip(),
                zaxis_title=z_feature.replace('(cm)', '').strip(),
            ),
            plot_bgcolor='rgba(26, 28, 37, 0.8)',
            paper_bgcolor='rgba(26, 28, 37, 0.8)',
            font=dict(color='#E0E0E0'),
            height=700,
        )

        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Pair Plot":
        st.subheader("🔄 Matriz de Dispersión")

        fig = px.scatter_matrix(
            df,
            dimensions=feature_names,
            color="species",
            color_discrete_map=especies_colores,
            template="plotly_dark",
            opacity=0.8
        )

        fig.update_layout(
            plot_bgcolor='rgba(26, 28, 37, 0.8)',
            paper_bgcolor='rgba(26, 28, 37, 0.8)',
            font=dict(color='#E0E0E0'),
            height=800,
        )

        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Distribuciones":
        st.subheader("📊 Distribuciones por Especie")

        feature = st.selectbox("Selecciona característica:", feature_names)

        fig = px.histogram(
            df,
            x=feature,
            color="species",
            marginal="box",
            color_discrete_map=especies_colores,
            template="plotly_dark",
            opacity=0.7,
            barmode="overlay"
        )

        fig.update_layout(
            plot_bgcolor='rgba(26, 28, 37, 0.8)',
            paper_bgcolor='rgba(26, 28, 37, 0.8)',
            font=dict(color='#E0E0E0'),
            height=500,
            xaxis_title=feature.replace('(cm)', '').strip(),
            yaxis_title="Frecuencia"
        )

        st.plotly_chart(fig, use_container_width=True)

        fig = px.violin(
            df,
            y=feature,
            x="species",
            color="species",
            box=True,
            points="all",
            color_discrete_map=especies_colores,
            template="plotly_dark"
        )

        fig.update_layout(
            plot_bgcolor='rgba(26, 28, 37, 0.8)',
            paper_bgcolor='rgba(26, 28, 37, 0.8)',
            font=dict(color='#E0E0E0'),
            height=500,
            yaxis_title=feature.replace('(cm)', '').strip(),
            xaxis_title="Especie"
        )

        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Análisis PCA":
        st.subheader("🔍 Análisis de Componentes Principales (PCA)")

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        pca_df['species'] = [class_names[i] for i in y]

        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='species',
            color_discrete_map=especies_colores,
            template="plotly_dark",
            labels={'PC1': 'Componente Principal 1',
                    'PC2': 'Componente Principal 2'}
        )

        loading = pca.components_.T * np.sqrt(pca.explained_variance_)

        for i, feature in enumerate(feature_names):
            fig.add_shape(
                type='line',
                x0=0, y0=0,
                x1=loading[i, 0],
                y1=loading[i, 1],
                line=dict(color='white', width=1, dash='dot'),
            )
            fig.add_annotation(
                x=loading[i, 0] * 1.1,
                y=loading[i, 1] * 1.1,
                ax=0, ay=0,
                text=feature.replace('(cm)', '').strip(),
                font=dict(color='yellow', size=12),
                arrowhead=2,
            )

        fig.update_layout(
            plot_bgcolor='rgba(26, 28, 37, 0.8)',
            paper_bgcolor='rgba(26, 28, 37, 0.8)',
            font=dict(color='#E0E0E0'),
            height=600,
            title=f"Varianza explicada: {sum(pca.explained_variance_ratio_):.2%}",
            title_x=0.5,
        )

        st.plotly_chart(fig, use_container_width=True)

        explained_var = pca.explained_variance_ratio_

        fig = px.bar(
            x=[f"PC{i+1}" for i in range(len(explained_var))],
            y=explained_var,
            labels={'x': 'Componente Principal', 'y': 'Varianza Explicada'},
            template="plotly_dark",
            text=[f"{v:.1%}" for v in explained_var]
        )

        fig.update_layout(
            plot_bgcolor='rgba(26, 28, 37, 0.8)',
            paper_bgcolor='rgba(26, 28, 37, 0.8)',
            font=dict(color='#E0E0E0'),
            height=400,
            title="Varianza explicada por componente",
            title_x=0.5,
        )

        st.plotly_chart(fig, use_container_width=True)


st.markdown("""
    <div class="footer">
        <p>🌿 Aplicación interactiva creada con <strong>Streamlit</strong>, <strong>Plotly</strong> y <strong>Scikit-learn</strong>.</p>
        <p>📌 Diseño oscuro moderno con visualizaciones avanzadas</p>
    </div>
""", unsafe_allow_html=True)
