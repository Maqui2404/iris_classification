import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_lottie import st_lottie
import json
import requests
from streamlit_echarts import st_echarts
from streamlit_option_menu import option_menu

# App setup and configuration
st.set_page_config(
    page_title="Iris Flower Classification | AI Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load animations


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Imágenes de reemplazo
flower_img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/800px-Iris_versicolor_3.jpg"
analysis_img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Iris_floral_diagram.jpg/800px-Iris_floral_diagram.jpg"
ml_img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Random_forest_diagram.png/800px-Random_forest_diagram.png"

# Custom CSS and themes
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #4B4B4B;
        --secondary-color: #fff;
        --accent-color: #43C6AC;
        --background-color: #121212;
        --text-color: #fff;
        --card-color: #1F1F1F;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--background-color) 0%, #1E1E1E 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 600;
    }
    
    .main-title {
        font-size: 3rem;
        background: linear-gradient(120deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-weight: 700;
        text-align: center;
    }
    
    .subtitle {
        text-align: center;
        color: #B0BEC5;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: var(--card-color);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.5);
    }
    
    .metric-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        padding: 1rem;
        border-radius: 15px;
        background: rgba(40, 40, 40, 0.8);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: var(--text-color);
        font-weight: 500;
    }
    
    .stButton button {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--accent-color) 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] {
        background-color: #181818;
        border-right: none;
        box-shadow: 5px 0 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom slider */
    [data-testid="stSlider"] > div {
        height: 5px;
        background-color: #444;
        border-radius: 10px;
    }
    
    [data-testid="stSlider"] > div > div {
        background-color: var(--primary-color);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem;
        color: #B0BEC5;
        background: rgba(40, 40, 40, 0.9);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.8s ease-in-out;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: var(--primary-color);
        border-radius: 10px;
    }
    
    /* Info boxes */
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: rgba(30, 136, 229, 0.1);
        border-left: 5px solid var(--primary-color);
        margin-bottom: 1rem;
    }
    
    /* Tooltip custom */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted var(--primary-color);
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #263238;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Data table styling */
    [data-testid="stTable"] {
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid #444;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 136, 229, 0.2);
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* Input fields */
    [data-testid="stNumberInput"] input {
        border-radius: 10px;
        border: 1px solid #444;
        background-color: #222;
        color: white;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.4);
    }
</style>

""", unsafe_allow_html=True)

# Initialize session state variables
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'training_progress' not in st.session_state:
    st.session_state['training_progress'] = 0
if 'show_prediction_animation' not in st.session_state:
    st.session_state['show_prediction_animation'] = False
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'model_metrics' not in st.session_state:
    st.session_state['model_metrics'] = None
if 'feature_importance' not in st.session_state:
    st.session_state['feature_importance'] = None
if 'confusion_mat' not in st.session_state:
    st.session_state['confusion_mat'] = None
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = "Dashboard"

# Load Iris dataset


@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # Rename features to be more readable
    readable_features = [
        "Sepal Length (cm)",
        "Sepal Width (cm)",
        "Petal Length (cm)",
        "Petal Width (cm)"
    ]

    # Create DataFrame
    df = pd.DataFrame(X, columns=readable_features)
    df['Species'] = [class_names[i] for i in y]

    return df, X, y, readable_features, class_names


df, X, y, feature_names, class_names = load_data()

# Sidebar navigation
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 30px;"><h2>🌸 Iris Classifier</h2></div>',
                unsafe_allow_html=True)

    st.image(flower_img_url, use_column_width=True)

    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Data Explorer",
                 "Model Training", "Real-time Prediction"],
        icons=["speedometer2", "table", "gear", "magic"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#6C63FF", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#F0F2F6"},
            "nav-link-selected": {"background-color": "rgba(108, 99, 255, 0.2)", "color": "#6C63FF"},
        }
    )

    st.session_state['current_tab'] = selected

    st.markdown("---")
    st.markdown("### Model Parameters")

    n_estimators = st.slider(
        "Number of Trees", min_value=10, max_value=200, value=100, step=10)
    max_depth = st.slider("Max Depth", min_value=1,
                          max_value=20, value=10, step=1)

    with st.expander("Advanced Parameters"):
        criterion = st.selectbox("Criterion", ["gini", "entropy"])
        min_samples_split = st.slider("Min Samples Split", 2, 10, 2)

    st.markdown("---")
    st.markdown("<div class='footer'>Powered by Streamlit & Scikit-learn</div>",
                unsafe_allow_html=True)

# Main content area


def dashboard_page():
    st.markdown('<h1 class="main-title fade-in">Iris Flower Classification AI</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle fade-in">Interactive machine learning dashboard to classify iris flowers</p>',
                unsafe_allow_html=True)

    # Top metrics
    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.markdown(
            """
            <div class="metric-container">
                <div class="metric-value">150</div>
                <div class="metric-label">Total Samples</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metrics_cols[1]:
        st.markdown(
            """
            <div class="metric-container">
                <div class="metric-value">3</div>
                <div class="metric-label">Flower Species</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metrics_cols[2]:
        if st.session_state['trained_model'] is not None and 'model_metrics' in st.session_state:
            accuracy = st.session_state['model_metrics']['accuracy']
            accuracy_display = f"{accuracy:.2%}"
        else:
            accuracy_display = "N/A"

        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value">{accuracy_display}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Main dashboard content
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("🌿 Iris Dataset Overview")

    dashboard_tabs = st.tabs(
        ["📊 Data Distribution", "🔍 Species Comparison", "🌱 Feature Correlations"])

    with dashboard_tabs[0]:
        fig = px.histogram(
            df,
            x="Species",
            color="Species",
            title="Distribution of Iris Flower Species",
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with dashboard_tabs[1]:
        # Radar chart for species comparison
        categories = feature_names

        species_means = df.groupby('Species').mean().reset_index()

        # Prepare data for radar chart
        radar_data = []
        for i, species in enumerate(class_names):
            species_data = species_means[species_means['Species'] == species].iloc[0].drop(
                'Species').values.tolist()
            radar_data.append({
                'value': species_data,
                'name': species
            })

        # ECharts radar chart
        radar_option = {
            'title': {
                'text': 'Iris Species Characteristics'
            },
            'legend': {
                'data': class_names.tolist(),
                'orient': 'horizontal',
                'bottom': 0
            },
            'radar': {
                'indicator': [{'name': name, 'max': df[name].max() * 1.1} for name in feature_names],
                'shape': 'circle',
            },
            'series': [{
                'type': 'radar',
                'data': radar_data,
                'emphasis': {
                    'lineStyle': {
                        'width': 4
                    }
                }
            }]
        }

        st_echarts(options=radar_option, height="400px")

    with dashboard_tabs[2]:
        correlation_fig = px.scatter_matrix(
            df,
            dimensions=feature_names,
            color="Species",
            title="Feature Correlations by Species",
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        correlation_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )

        st.plotly_chart(correlation_fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)  # End card

    # Quick access cards
    st.markdown("<h3 style='margin-top: 2rem;'>Quick Actions</h3>",
                unsafe_allow_html=True)

    action_cols = st.columns(3)
    with action_cols[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Explore Data")
        st.write("Dive into the dataset and examine relationships between features.")
        if st.button("Go to Data Explorer"):
            st.rerun()()
        st.markdown('</div>', unsafe_allow_html=True)

    with action_cols[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🤖 Train Model")
        st.write("Train a Random Forest model to classify iris flowers.")
        if st.button("Go to Model Training"):
            st.session_state['current_tab'] = "Model Training"
            st.rerun()()
        st.markdown('</div>', unsafe_allow_html=True)

    with action_cols[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔮 Make Predictions")
        st.write("Use your trained model to predict iris species in real-time.")
        if st.button("Go to Prediction"):
            st.session_state['current_tab'] = "Real-time Prediction"
            st.rerun()()
        st.markdown('</div>', unsafe_allow_html=True)


def data_explorer_page():
    st.markdown('<h1 class="main-title fade-in">Data Explorer</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle fade-in">Analyze and visualize the Iris dataset</p>',
                unsafe_allow_html=True)

    explorer_tabs = st.tabs(["📋 Dataset", "📊 Visualizations", "📈 Statistics"])

    with explorer_tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Iris Dataset")

        # Search and filter functionality
        search_term = st.text_input("🔍 Search in dataset", "")

        filtered_df = df
        if search_term:
            filtered_df = df[df.astype(str).apply(
                lambda row: row.str.contains(search_term, case=False).any(), axis=1)]

        # Species filter
        selected_species = st.multiselect(
            "Filter by Species", df['Species'].unique(), df['Species'].unique())
        if selected_species:
            filtered_df = filtered_df[filtered_df['Species'].isin(
                selected_species)]

        st.dataframe(filtered_df, use_container_width=True)

        # Download button
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='iris_filtered_data.csv',
            mime='text/csv',
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with explorer_tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Interactive Data Visualization")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Scatter Plot", "Box Plot", "Violin Plot",
                "3D Scatter Plot", "Parallel Coordinates"]
        )

        if chart_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", feature_names, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis", feature_names, index=2)

            scatter_fig = px.scatter(
                df, x=x_axis, y=y_axis, color="Species",
                size_max=12, opacity=0.8,
                title=f"{y_axis} vs {x_axis} by Species",
                color_discrete_sequence=px.colors.qualitative.Bold,
                hover_data=feature_names
            )

            scatter_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                height=500
            )

            st.plotly_chart(scatter_fig, use_container_width=True)

        elif chart_type == "Box Plot":
            feature = st.selectbox("Select Feature", feature_names)

            box_fig = px.box(
                df, x="Species", y=feature, color="Species",
                title=f"Distribution of {feature} by Species",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            box_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                height=500
            )

            st.plotly_chart(box_fig, use_container_width=True)

        elif chart_type == "Violin Plot":
            feature = st.selectbox("Select Feature", feature_names)

            violin_fig = px.violin(
                df, x="Species", y=feature, color="Species",
                box=True, points="all",
                title=f"Distribution of {feature} by Species",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            violin_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                height=500
            )

            st.plotly_chart(violin_fig, use_container_width=True)

        elif chart_type == "3D Scatter Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_axis = st.selectbox("X-axis", feature_names, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis", feature_names, index=1)
            with col3:
                z_axis = st.selectbox("Z-axis", feature_names, index=2)

            scatter_3d_fig = px.scatter_3d(
                df, x=x_axis, y=y_axis, z=z_axis, color="Species",
                opacity=0.8,
                title=f"3D Plot of {x_axis}, {y_axis}, and {z_axis} by Species",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            scatter_3d_fig.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                height=600
            )

            st.plotly_chart(scatter_3d_fig, use_container_width=True)

        elif chart_type == "Parallel Coordinates":
            parallel_fig = px.parallel_coordinates(
                df, color="Species",
                labels=feature_names,
                color_continuous_scale=px.colors.diverging.Tealrose,
                title="Parallel Coordinates Plot of Iris Dataset"
            )

            parallel_fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=500
            )

            st.plotly_chart(parallel_fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with explorer_tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset Statistics")

        stat_tabs = st.tabs(["Summary", "By Species", "Correlations"])

        with stat_tabs[0]:
            st.write("Overall dataset statistics:")
            st.dataframe(df.describe(), use_container_width=True)

        with stat_tabs[1]:
            species = st.selectbox("Select Species", df['Species'].unique())
            st.write(f"Statistics for {species}:")
            st.dataframe(df[df['Species'] == species].describe(),
                         use_container_width=True)

        with stat_tabs[2]:
            st.write("Correlation Matrix:")
            corr = df.drop('Species', axis=1).corr()

            # Generate a heatmap of correlations
            heatmap_fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Feature Correlation Matrix"
            )

            heatmap_fig.update_layout(
                height=500
            )

            st.plotly_chart(heatmap_fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)


def model_training_page():
    st.markdown('<h1 class="main-title fade-in">Model Training</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle fade-in">Train and evaluate a Random Forest Classifier</p>',
                unsafe_allow_html=True)

    # Training setup card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🛠️ Model Configuration")

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
        st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)  # Espacio entre sliders
        random_state = st.slider("Random Seed", 0, 100, 42, 1)
        st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)  # Espacio entre sliders



    with col2:
        col2.image(ml_img_url, width=180)

    # Train button
    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            # Simulated training progress
            progress_bar = st.progress(0)
            for i in range(101):
                time.sleep(0.01)  # Simulated delay
                progress_bar.progress(i)
                st.session_state['training_progress'] = i

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )

            # Train model
            model = RandomForestClassifier(
                n_estimators=st.session_state.get('n_estimators', 100),
                max_depth=st.session_state.get('max_depth', 10),
                criterion=st.session_state.get('criterion', 'gini'),
                min_samples_split=st.session_state.get('min_samples_split', 2),
                random_state=random_state
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Store in session state
            st.session_state['trained_model'] = model
            st.session_state['model_metrics'] = {
                'accuracy': accuracy,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred
            }
            st.session_state['confusion_mat'] = conf_matrix
            st.session_state['feature_importance'] = model.feature_importances_

            st.success(
                f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")

    st.markdown('</div>', unsafe_allow_html=True)  # End card

    # Model evaluation (only if model is trained)
    if st.session_state['trained_model'] is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Model Performance")

            # Metrics
            accuracy = st.session_state['model_metrics']['accuracy']
            st.markdown(
                f"<div class='metric-container'><div class='metric-value'>{accuracy:.2%}</div><div class='metric-label'>Accuracy</div></div>", unsafe_allow_html=True)

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            conf_mat = st.session_state['confusion_mat']

            fig = make_subplots(rows=1, cols=1)

            heatmap = go.Heatmap(
                z=conf_mat,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                showscale=False,
                text=conf_mat,
                texttemplate="%{text}",
                hoverinfo="text",
            )

            fig.add_trace(heatmap)
            fig.update_layout(
                title="Confusion Matrix",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Predicted Label",
                yaxis_title="True Label"
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)  # End card

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔍 Feature Importance")

            # Feature importance chart
            importance = st.session_state['feature_importance']
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

            fig = px.bar(
                feature_importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance',
                color='Importance',
                color_continuous_scale='Blues'
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)  # End card

        # Advanced evaluation
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧪 Prediction Results Analysis")

        # Plot correct vs incorrect predictions
        X_test = st.session_state['model_metrics']['X_test']
        y_test = st.session_state['model_metrics']['y_test']
        y_pred = st.session_state['model_metrics']['y_pred']

        # Convert to DataFrame for easier handling
        test_results = pd.DataFrame(X_test, columns=feature_names)
        test_results['True Species'] = [class_names[i] for i in y_test]
        test_results['Predicted Species'] = [class_names[i] for i in y_pred]
        test_results['Correct'] = test_results['True Species'] == test_results['Predicted Species']

        # Visualization of correct vs incorrect predictions
        vis_cols = st.columns(2)

        with vis_cols[0]:
            # Select features to visualize
            x_feature = st.selectbox(
                "X-axis Feature", feature_names, index=0, key="viz_x_feature")
            y_feature = st.selectbox(
                "Y-axis Feature", feature_names, index=2, key="viz_y_feature")

        with vis_cols[1]:
            # Filter options
            show_correct = st.checkbox("Show Correct Predictions", True)
            show_incorrect = st.checkbox("Show Incorrect Predictions", True)

        # Filter the results
        filtered_results = test_results.copy()
        if not show_correct:
            filtered_results = filtered_results[~filtered_results['Correct']]
        if not show_incorrect:
            filtered_results = filtered_results[filtered_results['Correct']]

        # Create scatter plot
        fig = px.scatter(
            filtered_results,
            x=x_feature,
            y=y_feature,
            color='True Species',
            symbol='Correct',
            title=f"Model Predictions ({x_feature} vs {y_feature})",
            color_discrete_sequence=px.colors.qualitative.Bold,
            symbol_map={True: 'circle', False: 'x'},
            size_max=12,
            hover_data=['True Species', 'Predicted Species']
        )

        # Add decision boundaries (simplified approach for visualization)
        if st.checkbox("Show Approximate Decision Boundaries"):
            # Create a meshgrid for the selected features
            feature_idx_x = feature_names.index(x_feature)
            feature_idx_y = feature_names.index(y_feature)

            x_min, x_max = X_test[:, feature_idx_x].min(
            ) - 0.5, X_test[:, feature_idx_x].max() + 0.5
            y_min, y_max = X_test[:, feature_idx_y].min(
            ) - 0.5, X_test[:, feature_idx_y].max() + 0.5

            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100)
            )

            # Create a test dataset using mean values for other features
            mean_values = X_test.mean(axis=0)
            test_points = np.tile(mean_values, (xx.ravel().shape[0], 1))
            test_points[:, feature_idx_x] = xx.ravel()
            test_points[:, feature_idx_y] = yy.ravel()

            # Predict on the test points
            Z = st.session_state['trained_model'].predict(test_points)
            Z = Z.reshape(xx.shape)

            # Add contours to the plot
            for i, species in enumerate(class_names):
                fig.add_trace(
                    go.Contour(
                        z=Z == i,
                        x=np.linspace(x_min, x_max, 100),
                        y=np.linspace(y_min, y_max, 100),
                        showscale=False,
                        opacity=0.2,
                        line=dict(width=0),
                        colorscale=[[0, 'rgba(255,255,255,0)'], [
                            1, px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]]]
                    )
                )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show misclassified examples
        if st.checkbox("Show Misclassified Examples"):
            misclassified = test_results[~test_results['Correct']]
            if misclassified.empty:
                st.success(
                    "🎉 Perfect classification! No misclassified examples.")
            else:
                st.write(f"Found {len(misclassified)} misclassified examples:")
                st.dataframe(misclassified, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)  # End card
    else:
        st.info("👆 Train a model to see evaluation metrics and visualizations.")


def prediction_page():
    st.markdown('<h1 class="main-title fade-in">Real-time Prediction</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle fade-in">Predict iris species using your trained model</p>',
                unsafe_allow_html=True)

    if st.session_state['trained_model'] is None:
        st.warning(
            "⚠️ You need to train a model first. Go to the Model Training page.")
        if st.button("Go to Model Training"):
            st.session_state['current_tab'] = "Model Training"
            st.rerun()()
    else:
        # Prediction card
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🌺 Enter Flower Measurements")

            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["Manual Entry", "Interactive Sliders"],
                horizontal=True
            )

            if input_method == "Manual Entry":
                input_cols = st.columns(2)
                with input_cols[0]:
                    sepal_length = st.number_input(
                        f"{feature_names[0]}", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
                    sepal_width = st.number_input(
                        f"{feature_names[1]}", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
                with input_cols[1]:
                    petal_length = st.number_input(
                        f"{feature_names[2]}", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
                    petal_width = st.number_input(
                        f"{feature_names[3]}", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
            else:
                # Interactive sliders with range indicators
                sepal_length = st.slider(
                    f"{feature_names[0]}",
                    min_value=float(df[feature_names[0]].min()),
                    max_value=float(df[feature_names[0]].max()),
                    value=5.1,
                    step=0.1
                )

                st.markdown(f"""
                <div class='info-box'>
                    <strong>Typical ranges:</strong><br>
                    <small>Setosa: {df[df['Species']=='setosa'][feature_names[0]].mean():.1f} | 
                    Versicolor: {df[df['Species']=='versicolor'][feature_names[0]].mean():.1f} | 
                    Virginica: {df[df['Species']=='virginica'][feature_names[0]].mean():.1f}</small>
                </div>
                """, unsafe_allow_html=True)

                sepal_width = st.slider(
                    f"{feature_names[1]}",
                    min_value=float(df[feature_names[1]].min()),
                    max_value=float(df[feature_names[1]].max()),
                    value=3.5,
                    step=0.1
                )

                petal_length = st.slider(
                    f"{feature_names[2]}",
                    min_value=float(df[feature_names[2]].min()),
                    max_value=float(df[feature_names[2]].max()),
                    value=1.4,
                    step=0.1
                )

                petal_width = st.slider(
                    f"{feature_names[3]}",
                    min_value=float(df[feature_names[3]].min()),
                    max_value=float(df[feature_names[3]].max()),
                    value=0.2,
                    step=0.1
                )

        with col2:
            col2.image(analysis_img_url, width=180)

        # Buttons
        predict_col, reset_col = st.columns([1, 1])
        with predict_col:
            predict_button = st.button(
                "🔮 Predict Species", use_container_width=True)
        with reset_col:
            if st.button("🔄 Reset Values", use_container_width=True):
                st.session_state['prediction_result'] = None
                st.session_state['show_prediction_animation'] = False
                st.rerun()()

        # Make prediction
        if predict_button:
            # Animated prediction effect
            st.session_state['show_prediction_animation'] = True

            with st.spinner("Analyzing flower characteristics..."):
                # Simulate processing time
                progress_bar = st.progress(0)
                for i in range(101):
                    time.sleep(0.01)
                    progress_bar.progress(i)

                # Make prediction
                input_features = np.array(
                    [[sepal_length, sepal_width, petal_length, petal_width]])
                prediction = st.session_state['trained_model'].predict(
                    input_features)
                probabilities = st.session_state['trained_model'].predict_proba(input_features)[
                    0]

                # Store in session state
                st.session_state['prediction_result'] = {
                    'species': class_names[prediction[0]],
                    'probabilities': {class_names[i]: prob for i, prob in enumerate(probabilities)},
                    'input_features': input_features[0]
                }

        # Show prediction result
        if st.session_state['prediction_result'] is not None:
            result = st.session_state['prediction_result']

            st.markdown("---")

            # Result display
            result_cols = st.columns([1, 2])

            with result_cols[0]:
                # Show prediction with nice styling
                species = result['species']

                species_colors = {
                    'setosa': '#FF6584',
                    'versicolor': '#6C63FF',
                    'virginica': '#43C6AC'
                }

                species_color = species_colors.get(species, '#6C63FF')

                st.markdown(f"""
                <div style="text-align: center; margin-top: 20px;">
                    <div style="font-size: 1.2rem; color: #5C6585;">Predicted Species</div>
                    <div style="font-size: 2.5rem; font-weight: 700; color: black;">{species}</div>
                </div>
                """, unsafe_allow_html=True)

            with result_cols[1]:
                # Display prediction probabilities as gauge charts
                probabilities = result['probabilities']

                gauge_options = {
                    "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
                    "series": [
                        {
                            "name": "Probability",
                            "type": "gauge",
                            "startAngle": 180,
                            "endAngle": 0,
                            "progress": {"show": True},
                            "radius": "100%",
                            "itemStyle": {"color": species_colors.get(species, '#6C63FF')},
                            "axisLine": {
                                "lineStyle": {
                                    "width": 15,
                                    "color": [
                                        [0.3, "#FF6584"],
                                        [0.7, "#FFCE56"],
                                        [1, "#43C6AC"]
                                    ]
                                }
                            },
                            "pointer": {
                                "icon": "path://M12.8,0.7l12,40.1H0.7L12.8,0.7z",
                                "length": "12%",
                                "width": 8,
                                "offsetCenter": [0, "-60%"],
                                "itemStyle": {"color": "auto"}
                            },
                            "axisTick": {"length": 12, "lineStyle": {"color": "auto", "width": 2}},
                            "splitLine": {"length": 20, "lineStyle": {"color": "auto", "width": 5}},
                            "title": {"offsetCenter": [0, "-20%"], "fontSize": 16},
                            "detail": {
                                "fontSize": 30,
                                "offsetCenter": [0, "0%"],
                                "valueAnimation": True,
                                "formatter": "{value}%",
                                "color": "auto"
                            },
                            "data": [{"value": round(probabilities[species] * 100), "name": "Confidence"}]
                        }
                    ]
                }

                st_echarts(options=gauge_options, height="200px")

            # Display all probabilities
            st.markdown("##### Probability Distribution")
            probs_data = [{"species": species, "probability": prob * 100}
                          for species, prob in probabilities.items()]
            probs_df = pd.DataFrame(probs_data)

            fig = px.bar(
                probs_df,
                x='species',
                y='probability',
                color='species',
                title='Probability by Species (%)',
                color_discrete_map=species_colors,
                text=probs_df['probability'].apply(lambda x: f"{x:.1f}%")
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis_range=[0, 100],
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

            # Nearest neighbors visualization
            if st.checkbox("Show Similar Flowers"):
                st.subheader("Similar Flowers in Dataset")

                # Find nearest neighbors
                from sklearn.neighbors import NearestNeighbors

                nn = NearestNeighbors(n_neighbors=5)
                nn.fit(X)

                distances, indices = nn.kneighbors(
                    result['input_features'].reshape(1, -1))

                neighbors_df = pd.DataFrame({
                    feature_names[0]: X[indices[0], 0],
                    feature_names[1]: X[indices[0], 1],
                    feature_names[2]: X[indices[0], 2],
                    feature_names[3]: X[indices[0], 3],
                    'Species': [class_names[y[i]] for i in indices[0]],
                    'Distance': distances[0]
                })

                st.dataframe(neighbors_df, use_container_width=True)

                # Plot the input point with its nearest neighbors
                fig = px.scatter_3d(
                    neighbors_df,
                    x=feature_names[0],
                    y=feature_names[1],
                    z=feature_names[2],
                    color='Species',
                    size='Distance',
                    size_max=15,
                    opacity=0.8,
                    title=f"Your Flower and Its 5 Nearest Neighbors",
                    color_discrete_map=species_colors
                )

                # Add the input point
                fig.add_scatter3d(
                    x=[result['input_features'][0]],
                    y=[result['input_features'][1]],
                    z=[result['input_features'][2]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='black',
                        symbol='diamond',
                        line=dict(color='white', width=2)
                    ),
                    name='Your Flower'
                )

                fig.update_layout(
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom",
                                y=1.02, xanchor="right", x=1),
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)  # End card


# Render the appropriate page based on selected tab
if st.session_state['current_tab'] == "Dashboard":
    dashboard_page()
elif st.session_state['current_tab'] == "Data Explorer":
    data_explorer_page()
elif st.session_state['current_tab'] == "Model Training":
    model_training_page()
elif st.session_state['current_tab'] == "Real-time Prediction":
    prediction_page()

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; background-color: rgba(0, 0, 0, 1); border-radius: 10px; color: white;">
    <p>🌿 Application created with <strong>Streamlit</strong> and <strong>Scikit-learn</strong></p>
    <p>📌 Made with 💙 for Data Science and Machine Learning</p>
</div>
""", unsafe_allow_html=True)
