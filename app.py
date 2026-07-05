import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib

# ---------------------------------------------------------------------------
# Configuración general y de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Riesgo Actuarial",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Estilos CSS Avanzados (UI/UX Limpia y Corporativa)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Configuración de fuente global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Encabezado Principal */
    .header-contenedor {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .titulo-app {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.025em;
    }
    .subtitulo-app {
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
        opacity: 0.9;
    }
    .autor-app {
        font-size: 0.85rem;
        margin-top: 1rem;
        opacity: 0.7;
        font-weight: 500;
    }

    /* Tarjetas de Resultados K-Means */
    .tarjeta-kpi {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    .kpi-titulo {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .kpi-valor {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }

    /* Alertas de Riesgo Personalizadas */
    .badge-riesgo {
        padding: 0.75rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 1rem;
    }
    .riesgo-bajo { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .riesgo-medio { background-color: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .riesgo-alto { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }

    /* Contenedor de gráficos y datos */
    .contenedor-blanco {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Carga de Recursos (Modelo, Metadata y Dataset)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_recursos():
    ruta_kmeans = "models/kmeans_riesgo_actuarial.pkl"
    ruta_metadata = "models/model_metadata.json"
    ruta_csv = "insurance.csv"

    modelo_kmeans = joblib.load(ruta_kmeans) if os.path.exists(ruta_kmeans) else None

    metadata = {}
    if os.path.exists(ruta_metadata):
        with open(ruta_metadata, "r") as f:
            metadata = json.load(f)

    df_clean = pd.read_csv(ruta_csv) if os.path.exists(ruta_csv) else None

    return modelo_kmeans, metadata, df_clean

modelo, metadata, df = cargar_recursos()

# Verificar que el modelo principal esté cargado
if modelo is None:
    st.error("No se encontró el modelo K-means en `models/kmeans_riesgo_actuarial.pkl`. Por favor verifica la ruta.", icon="🚨")
    st.stop()

# ---------------------------------------------------------------------------
# Barra Lateral - Captura de Datos Estilizada
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📋 Datos del Asegurado")
    st.caption("Modifique los parámetros para evaluar el riesgo técnico en tiempo real.")
    st.markdown("---")

    age = st.slider("Edad del Titular", min_value=18, max_value=100, value=35)

    sexo_opciones = {"Femenino": "female", "Masculino": "male"}
    sex_es = st.selectbox("Sexo Biológico", options=list(sexo_opciones.keys()))
    sex = sexo_opciones[sex_es]

    bmi = st.slider("Índice de Masa Corporal (IMC / BMI)", min_value=15.0, max_value=60.0, value=28.5, step=0.1)
    children = st.number_input("Número de dependientes (Hijos)", min_value=0, max_value=10, value=1, step=1)

    fumador_opciones = {"No": "no", "Sí": "yes"}
    smoker_es = st.selectbox("¿Declara Tabaquismo?", options=list(fumador_opciones.keys()))
    smoker = fumador_opciones[smoker_es]

    region_opciones = {
        "Suroeste (Southwest)": "southwest",
        "Sureste (Southeast)": "southeast",
        "Noroeste (Northwest)": "northwest",
        "Noreste (Northeast)": "northeast"
    }
    region_es = st.selectbox("Región de Cobertura", options=list(region_opciones.keys()))
    region = region_opciones[region_es]

    charges = st.number_input("Siniestralidad Estimada / Cargos Médicos Anuales ($)", min_value=100.0, max_value=100000.0, value=13000.0, step=500.0)

# Construcción de la matriz para el clasificador (Pipeline)
cliente_dict = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
    "charges": charges
}
df_cliente = pd.DataFrame([cliente_dict])

# ---------------------------------------------------------------------------
# Ejecución del Modelo de Segmentación
# ---------------------------------------------------------------------------
datos_prediccion = df_cliente.copy()
try:
    cluster_asignado = int(modelo.predict(datos_prediccion)[0])
except ValueError:
    columnas_ordenadas = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    datos_prediccion = datos_prediccion[columnas_ordenadas]
    cluster_asignado = int(modelo.predict(datos_prediccion)[0])

# Mapeos actuariales
riesgo_mapeo = {0: "Bajo", 1: "Medio", 2: "Alto"}
explicacion_mapeo = {
    0: "Segmento de riesgo controlado. Clientes con hábitos de vida saludables (no fumadores) y niveles estables de cargos asistenciales.",
    1: "Riesgo intermedio o transitorio. Clientes con edades avanzadas o IMC en rango de sobrepeso que reflejan un gasto médico base estandarizado.",
    2: "Riesgo Técnico Crítico. Clientes con alta exposición (tabaquismo activo y/o índices de masa corporal elevados) correlacionados con curvas de costos de alta volatilidad."
}

nivel_riesgo = riesgo_mapeo.get(cluster_asignado, "No definido")
explicacion_cluster = explicacion_mapeo.get(cluster_asignado, "Segmento de clientes analizado por patrones de costos y comportamiento.")

# Define el estilo CSS en base al clúster asignado
clase_riesgo = "riesgo-bajo" if nivel_riesgo == "Bajo" else ("riesgo-medio" if nivel_riesgo == "Medio" else "riesgo-alto")

# ---------------------------------------------------------------------------
# Renderizado de Interfaz Principal
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="header-contenedor">
        <p class="titulo-app">🏥 Sistema de Segmentación y Riesgo Actuarial</p>
        <p class="subtitulo-app">Pipeline Inteligente K-Means enfocado en tarificación de carteras y análisis técnico de seguros.</p>
        <p class="autor-app"><b>Desarrollado por:</b> Erik Guillen · <b>Cuenta:</b> 20211920287 · <b>Módulo:</b> Inteligencia Artificial</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Panel Analítico de Dos Columnas
col_kpi, col_grafico = st.columns([1, 2], gap="large")

with col_kpi:
    st.markdown("#### 🎯 Clasificación Estructurada")

    # Tarjeta 1: Cluster Asignado
    st.markdown(
        f"""
        <div class="tarjeta-kpi">
            <div class="kpi-titulo">Segmentación de Carteras</div>
            <div class="kpi-valor">Cluster {cluster_asignado}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("")

    # Tarjeta 2: Nivel de Riesgo Actuarial
    st.markdown(
        f"""
        <div class="tarjeta-kpi">
            <div class="kpi-titulo">Calificación de Riesgo</div>
            <div class="badge-riesgo {clase_riesgo}">Riesgo {nivel_riesgo}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Descripción analítica
    st.markdown("")
    st.caption("**Comportamiento Técnico del Segmento:**")
    st.info(explicacion_cluster)

with col_grafico:
    st.markdown("#### 📊 Análisis Espacial de la Población")

    if df is not None:
        with st.container(border=True):
            # Estilo del Gráfico Integrado con el Framework de la UI
            fig, ax = plt.subplots(figsize=(7, 4.2))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8fafc')

            sns.scatterplot(
                data=df,
                x="bmi",
                y="charges",
                hue="smoker",
                alpha=0.6,
                palette=["#0d9488", "#f43f5e"],
                edgecolor="none",
                ax=ax
            )

            # Destacar la posición del cliente ingresado de forma Premium
            ax.scatter(
                bmi, charges,
                color="#1e3a8a",
                s=250,
                marker="*",
                edgecolor="white",
                linewidth=2,
                label="Cliente Evaluado",
                zorder=5
            )

            # Limpieza estética del gráfico
            ax.set_title("Distribución Poblacional: IMC vs Cargos Médicos", fontsize=11, fontweight='bold', color='#1e293b', pad=12)
            ax.set_xlabel("Índice de Masa Corporal (BMI / IMC)", fontsize=9, color='#475569')
            ax.set_ylabel("Cargos Médicos Totales ($)", fontsize=9, color='#475569')
            ax.grid(True, linestyle="--", alpha=0.3, color="#cbd5e1")

            # Formatear la leyenda
            ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=8)
            sns.despine(left=True, bottom=True)

            st.pyplot(fig)
    else:
        st.info("Para habilitar las proyecciones visuales avanzadas y mapas de dispersión, coloque el archivo `insurance.csv` en la raíz del entorno.", icon="ℹ️")

# ---------------------------------------------------------------------------
# Resumen de Datos de Entrada (Estructuración Limpia de Tabla)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### 📋 Auditoría de Datos de Entrada (Muestra Transaccional)")

df_bonito = pd.DataFrame([{
    "Edad": f"{age} años",
    "Sexo Biológico": sex_es,
    "Índice IMC (BMI)": f"{bmi:.1f}",
    "Dependientes": children,
    "Condición Tabaquismo": smoker_es,
    "Jurisdicción / Región": region_es,
    "Siniestralidad / Cargos": f"${charges:,.2f}"
}])

st.dataframe(df_bonito, use_container_width=True, hide_index=True)
