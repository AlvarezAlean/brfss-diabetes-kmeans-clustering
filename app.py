import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Segmentación BRFSS 2015 - K-Means",
    page_icon="📊",
    layout="wide",
)

# Aviso académico obligatorio para proyectos relacionados con salud
st.warning(
    "⚠️ **Aviso académico:** Esta aplicación forma parte de un proyecto universitario de "
    "Minería de Datos con fines exclusivamente académicos e ilustrativos. Los resultados "
    "representan agrupamientos estadísticos en datos históricos (BRFSS 2015) y **no constituyen "
    "un diagnóstico médico ni sustituyen la valoración de un profesional de la salud**."
)

st.title("Segmentación de Perfiles de Salud y Estilo de Vida (K-Means)")
st.write(
    "Aplicación vinculada al pipeline de escalamiento (`scaler_kmeans.joblib`) y al modelo "
    "de agrupamiento (`modelo_kmeans.joblib`) entrenados sobre el conjunto limpio BRFSS 2015."
)


@st.cache_resource
def cargar_modelos():
  scaler = joblib.load("scaler_kmeans.joblib")
  modelo = joblib.load("modelo_kmeans.joblib")
  return scaler, modelo


try:
  scaler, modelo = cargar_modelos()
except Exception as e:
  st.error(
      "No se pudieron cargar `scaler_kmeans.joblib` o `modelo_kmeans.joblib`. "
      "Asegúrate de que ambos archivos estén en la misma carpeta que `app.py`."
  )
  st.exception(e)
  st.stop()

# Obtener nombres exactos de variables usadas en el escalador
if hasattr(scaler, "feature_names_in_"):
  columnas_modelo = list(scaler.feature_names_in_)
else:
  columnas_modelo = [
      "HighBP",
      "HighChol",
      "CholCheck",
      "BMI",
      "Smoker",
      "Stroke",
      "HeartDiseaseorAttack",
      "PhysActivity",
      "Fruits",
      "Veggies",
      "HvyAlcoholConsump",
      "AnyHealthcare",
      "NoDocbcCost",
      "GenHlth",
      "MentHlth",
      "PhysHlth",
      "DiffWalk",
      "Sex",
      "Age",
      "Education",
      "Income",
  ]

# Metadatos básicos y rangos habituales del BRFSS 2015
config_vars = {
    "HighBP": {
        "label": "Presión arterial alta (HighBP)",
        "tipo": "binaria",
        "default": 1,
    },
    "HighChol": {
        "label": "Colesterol alto (HighChol)",
        "tipo": "binaria",
        "default": 1,
    },
    "CholCheck": {
        "label": "Control de colesterol en 5 años (CholCheck)",
        "tipo": "binaria",
        "default": 1,
    },
    "BMI": {
        "label": "Índice de Masa Corporal - BMI",
        "tipo": "float",
        "min": 12.0,
        "max": 98.0,
        "default": 28.0,
        "step": 0.5,
    },
    "Smoker": {
        "label": "Fumador de al menos 100 cigarrillos (Smoker)",
        "tipo": "binaria",
        "default": 0,
    },
    "Stroke": {
        "label": "Antecedente de derrame cerebral (Stroke)",
        "tipo": "binaria",
        "default": 0,
    },
    "HeartDiseaseorAttack": {
        "label": "Enfermedad coronaria o infarto (HeartDiseaseorAttack)",
        "tipo": "binaria",
        "default": 0,
    },
    "PhysActivity": {
        "label": "Actividad física en últimos 30 días (PhysActivity)",
        "tipo": "binaria",
        "default": 1,
    },
    "Fruits": {
        "label": "Consume frutas diariamente (Fruits)",
        "tipo": "binaria",
        "default": 1,
    },
    "Veggies": {
        "label": "Consume verduras diariamente (Veggies)",
        "tipo": "binaria",
        "default": 1,
    },
    "HvyAlcoholConsump": {
        "label": "Consumo alto de alcohol (HvyAlcoholConsump)",
        "tipo": "binaria",
        "default": 0,
    },
    "AnyHealthcare": {
        "label": "Cobertura de salud (AnyHealthcare)",
        "tipo": "binaria",
        "default": 1,
    },
    "NoDocbcCost": {
        "label": "No fue al médico por costo (NoDocbcCost)",
        "tipo": "binaria",
        "default": 0,
    },
    "GenHlth": {
        "label": "Salud general (1=Excelente ... 5=Mala) [GenHlth]",
        "tipo": "int",
        "min": 1,
        "max": 5,
        "default": 3,
    },
    "MentHlth": {
        "label": "Días de mala salud mental últimos 30d [MentHlth]",
        "tipo": "float",
        "min": 0.0,
        "max": 30.0,
        "default": 0.0,
        "step": 1.0,
    },
    "PhysHlth": {
        "label": "Días de mala salud física últimos 30d [PhysHlth]",
        "tipo": "float",
        "min": 0.0,
        "max": 30.0,
        "default": 0.0,
        "step": 1.0,
    },
    "DiffWalk": {
        "label": "Dificultad para caminar o subir escaleras (DiffWalk)",
        "tipo": "binaria",
        "default": 0,
    },
    "Sex": {
        "label": "Sexo (0=Mujer, 1=Hombre) [Sex]",
        "tipo": "binaria",
        "default": 0,
    },
    "Age": {
        "label": "Categoría de edad (1=18-24 ... 13=80+) [Age]",
        "tipo": "int",
        "min": 1,
        "max": 13,
        "default": 7,
    },
    "Education": {
        "label": "Nivel educativo (1=Sin escolaridad ... 6=Universitario) [Education]",
        "tipo": "int",
        "min": 1,
        "max": 6,
        "default": 4,
    },
    "Income": {
        "label": "Nivel de ingresos (1=Menor ... 8=Mayor) [Income]",
        "tipo": "int",
        "min": 1,
        "max": 8,
        "default": 5,
    },
}

# Inicializar valores en session_state
if "valores_entrada" not in st.session_state:
  st.session_state.valores_entrada = {
      col: config_vars.get(col, {"default": 0.0})["default"]
      for col in columnas_modelo
  }

col_btn1, col_btn2, _ = st.columns([1.2, 1.2, 3])
with col_btn1:
  if st.button("Cargar perfil por defecto"):
    st.session_state.valores_entrada = {
        col: config_vars.get(col, {"default": 0.0})["default"]
        for col in columnas_modelo
    }
    st.rerun()

with col_btn2:
  if st.button("Cargar perfil de prueba B"):
    perfil_b = {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 34.0,
        "Smoker": 1,
        "Stroke": 0,
        "HeartDiseaseorAttack": 1,
        "PhysActivity": 0,
        "Fruits": 0,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 4,
        "MentHlth": 10.0,
        "PhysHlth": 15.0,
        "DiffWalk": 1,
        "Sex": 1,
        "Age": 10,
        "Education": 4,
        "Income": 4,
    }
    for col in columnas_modelo:
      if col in perfil_b:
        st.session_state.valores_entrada[col] = perfil_b[col]
    st.rerun()

st.subheader("Ingresa los datos del registro")

datos_usuario = {}
c1, c2, c3 = st.columns(3)
columnas_ui = [c1, c2, c3]

for idx, col in enumerate(columnas_modelo):
  contenedor = columnas_ui[idx % 3]
  info = config_vars.get(
      col,
      {
          "label": f"{col} (valor numérico)",
          "tipo": "float",
          "min": -1000.0,
          "max": 1000.0,
          "default": 0.0,
          "step": 1.0,
      },
  )
  valor_actual = st.session_state.valores_entrada.get(
      col, info.get("default", 0.0)
  )

  with contenedor:
    if info.get("tipo") == "binaria":
      val = st.selectbox(
          info["label"],
          options=[0, 1],
          index=0 if int(valor_actual) == 0 else 1,
          key=f"input_{col}",
      )
      datos_usuario[col] = float(val)
    elif info.get("tipo") == "int":
      val = st.number_input(
          info["label"],
          min_value=int(info.get("min", 0)),
          max_value=int(info.get("max", 100)),
          value=int(
              np.clip(
                  int(valor_actual),
                  int(info.get("min", 0)),
                  int(info.get("max", 100)),
              )
          ),
          step=1,
          key=f"input_{col}",
      )
      datos_usuario[col] = float(val)
    else:
      val = st.number_input(
          info["label"],
          min_value=float(info.get("min", -1000.0)),
          max_value=float(info.get("max", 1000.0)),
          value=float(
              np.clip(
                  float(valor_actual),
                  float(info.get("min", -1000.0)),
                  float(info.get("max", 1000.0)),
              )
          ),
          step=float(info.get("step", 1.0)),
          key=f"input_{col}",
      )
      datos_usuario[col] = float(val)

st.markdown("---")

if st.button("Asignar clúster", type="primary"):
  df_entrada = pd.DataFrame([datos_usuario])[columnas_modelo]
  X_entrada_scaled = scaler.transform(df_entrada)
  cluster_pred = int(modelo.predict(X_entrada_scaled)[0])

  st.success(f"### Clúster asignado: **Grupo {cluster_pred}**")

  st.write("**Resumen de la asignación:**")
  st.write(
      f"- El registro fue escalado con el mismo `StandardScaler` del notebook y asignado "
      f"al **Clúster {cluster_pred}** por el modelo **K-Means** (`n_clusters={modelo.n_clusters}`)."
  )
  st.write(
      "- Recuerda consultar en tu notebook la tabla de promedios y distribución de "
      "`Diabetes_012` del **Clúster "
      f"{cluster_pred}** para describir los patrones observados en ese grupo."
  )

  with st.expander("Ver registro enviado al pipeline"):
    st.dataframe(df_entrada, use_container_width=True)