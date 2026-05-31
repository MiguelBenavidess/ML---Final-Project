import os
import io
import openpyxl
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost
import os
import io
import unicodedata
import openpyxl
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ── funciones de preprocesamiento ──────────────────────────────
def remove_accents_and_lowercase(text):
    if isinstance(text, str):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower().replace('.', '')
    return text
def definir_variables(data):
  # -------------------------------------------
  #     Definición de variables inicial
  # -------------------------------------------

  #Como el 99% de entradas eran de Colombia, vamos filtrar sólo los del país y eliminar la columna
  data= data[data["pais"].isin(["COL", "Colombia"])].copy()
  #Dado que ciudad, estado y posicion de empleo tienen una alta cardinalidad, vamos a eliminarlas
  data.drop(columns=["pais", "ciudad", "state", "employement_position"], inplace=True)
  data["genero"] = data["genero"].str.lower()
  #La fecha de registro estaba en formato UTC, mientras que las demás fechas están UTC -5.
  data["fecha_registro"] = pd.to_datetime(data["fecha_registro"]).dt.tz_convert("America/Bogota").dt.tz_localize(None)
  #convertir en tipo fecha
  data["fecha_onboarding_completed"] = pd.to_datetime(data["fecha_onboarding_completed"], format="mixed")
  data["fecha_apertura_cuenta"] = pd.to_datetime(data["fecha_apertura_cuenta"])
  data["fecha_nacimiento"] = pd.to_datetime(data["fecha_nacimiento"])
  #Configuramos nuestra variable objetivo si alguien fondeó o no
  #data["fondeo"]= data["fecha_fondeo"].notna().astype(int)
  #data.drop(columns=["fecha_fondeo"], inplace=True)
  # Obtener las columnas que son strings
  string_columns = data.select_dtypes(include='object').columns
  # Limpiar las columnas de tildes y volverlas en minúsculas
  for col in string_columns:
      data[col] = data[col].apply(remove_accents_and_lowercase)
  data["rango-ingresos"] = data["annual_income_min"].astype(str)+ " - " + data["annual_income_max"].astype(str)
  data["rango-patrimonio"] = data["total_net_worth_min"].astype(str)+ " - " + data["total_net_worth_max"].astype(str)
  data.drop(columns=["liquid_net_worth_max", "annual_income_min", "annual_income_max", "total_net_worth_min", "total_net_worth_max"], inplace=True)
  #Se toman estas tipo estatus de empleo
  data = data[data["employement_status"].isin(["employed", "student", "unemployed", "retired"])]
  data.drop(columns=["Nivel_inversionista"], inplace=True)

  data = data[data["funding_source"].isin(["employment_income", "investments", "business_income"])]

  # -------------------------------------------
  #     Creación de variables temporales
  # -------------------------------------------

  data["edad"] = (data["fecha_apertura_cuenta"] - data["fecha_nacimiento"]).dt.days // 365
  data = data.dropna(subset=["edad", "fecha_onboarding_completed", "genero"])
  data["edad"] = data["edad"].astype(int)
  data["minutos_onboarding_registro"] = (data["fecha_onboarding_completed"] - data["fecha_registro"]).dt.total_seconds() / (60)
  data["minutos_onboarding_registro"] = data["minutos_onboarding_registro"].astype(int)
  data["minutos_apertura_cuenta_onboarding"] = (data["fecha_apertura_cuenta"] - data["fecha_onboarding_completed"]).dt.total_seconds() / (60)
  data["minutos_apertura_cuenta_onboarding"] = data["minutos_apertura_cuenta_onboarding"].astype(int)
  data["minutos_apertura_cuenta_registro"] = (data["fecha_apertura_cuenta"] - data["fecha_registro"]).dt.total_seconds() / (60)
  data["minutos_apertura_cuenta_registro"] = data["minutos_apertura_cuenta_registro"].astype(int)

  # -------------------------------------------
  #   Enriquecimiento de variables temporales
  # -------------------------------------------

  # Requerimos la fecha_apertura_cuenta antes del drop
  data['dia_semana_apertura'] = data['fecha_apertura_cuenta'].dt.dayofweek

  # 0 = lunes, 5 = sábado, 6 = domingo

  data['es_fin_de_semana'] = data['fecha_apertura_cuenta'].dt.dayofweek.isin([5, 6]).astype(int).astype("object")
  data['dia_de_la_semana'] = data['fecha_apertura_cuenta'].dt.dayofweek.astype("object")

  data['mes_apertura'] = data['fecha_apertura_cuenta'].dt.month.astype("object")

  data['trimestre_apertura'] = data['fecha_apertura_cuenta'].dt.quarter.astype("object")

  data['dia_mes_apertura'] = data['fecha_apertura_cuenta'].dt.day.astype("object")

  data['dia_mes_onboarding'] = data['fecha_onboarding_completed'].dt.day.astype("object")

  data['hora_registro'] = data['fecha_registro'].dt.hour

  #Captura de horario

  data['es_horario_laboral'] = data['hora_registro'].between(8, 18).astype(int).astype("object")

  data.drop(columns=["fecha_registro", "fecha_onboarding_completed", "fecha_apertura_cuenta", "fecha_nacimiento"], inplace=True)
  return data



def variables_definitivas(data):
  data["fondeo"]= data["fecha_fondeo"].notna().astype(int)
  data= data.drop(columns=['Unnamed: 0', "Pauta", "fecha_fondeo"])
  data= definir_variables(data)
  data = data.drop(columns=[ 'minutos_onboarding_registro', "es_horario_laboral"])
  X = pd.get_dummies(data.drop(columns=['fondeo']), drop_first=True)
  X = X.reindex(columns=feature_columns, fill_value=0)
  y = data['fondeo']
  return X, y
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'modelo_random_forest.joblib')
    columns_path = os.path.join(base_path, 'feature_columns.joblib')
    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)
    return model, feature_columns

model, feature_columns = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📈 Predictor de Fondeo de Clientes")
st.markdown("Carga un archivo CSV con la información de los clientes a analizar.")

uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    X, y = variables_definitivas(input_df)

    upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_key"] = upload_key
        st.session_state["results"] = None

    st.write("### Vista previa de los datos")
    st.dataframe(X.head())
    if y is not None:
        st.write("### Variable objetivo")
        st.dataframe(y.head())

    if st.button("Ejecutar predicción"):
        try:
            probabilities = model.predict_proba(X)[:, 1]
            predictions = model.predict(X)

            results_df = X.copy()

            # Ajusta los umbrales según tu modelo
            results_df["Prediccion"] = [
                "Revisión Manual" if 0.4 <= p <= 0.6
                else "Fondeará" if p > 0.6
                else "No Fondeará"
                for p in probabilities
            ]
            results_df["Probabilidad_Fondeo"] = [f"{p:.2%}" for p in probabilities]

            st.session_state["results"] = {
                "results_df": results_df,
                "input_columns": list(input_df.columns),
                "y": y,
                "predictions": predictions,
                "probabilities": probabilities,
            }
        except Exception as e:
            st.session_state["results"] = None
            st.error(f"Error durante el procesamiento: {e}")

    cached = st.session_state.get("results")
    if cached is not None:
        results_df    = cached["results_df"]
        input_columns = cached["input_columns"]
        y_cached      = cached["y"]
        predictions   = cached["predictions"]

        # ── Métricas (solo si hay etiquetas reales) ───────────────────────────
        if y_cached is not None:
            st.write("### Matriz de Confusión")
            cm = confusion_matrix(y_cached, predictions, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["No Fondeó", "Fondeó"],
                        yticklabels=["No Fondeó", "Fondeó"],
                        annot_kws={"size": 14})
            ax.set_title("Matriz de Confusión")
            ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
            st.pyplot(fig)

            st.write("### Reporte de Clasificación")
            st.code(classification_report(y_cached, predictions,
                                          target_names=["No Fondeó", "Fondeó"]))

            st.write("### Importancia de Variables (Top 10)")
            # Random Forest expone feature_importances_ directamente
            rf = model.named_steps["model"]          # si usas Pipeline
            # rf = model                             # si no usas Pipeline
            feat_imp = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": rf.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feat_imp, x="Importance", y="Feature",
                        ax=ax2, palette="viridis")
            ax2.set_title("Top 10 Variables más Influyentes")
            sns.despine(left=True, bottom=True)
            st.pyplot(fig2)

        # ── Resultados ────────────────────────────────────────────────────────
        st.success("¡Análisis completo!")
        st.write("### Resultados de Predicción")
        vc = results_df["Prediccion"].value_counts(normalize=True)
        for label in ["Fondeará", "No Fondeará", "Revisión Manual"]:
            if label in results_df["Prediccion"].value_counts():
                n = results_df["Prediccion"].value_counts()[label]
                st.write(f"**{label}:** {n} ({vc[label]:.2%})")

        st.dataframe(
            results_df[["Prediccion", "Probabilidad_Fondeo"] +
                        [c for c in input_columns if c not in ("fondeo", "fecha_fondeo")]]
        )

        # ── Descarga ──────────────────────────────────────────────────────────
        download_option = st.selectbox("Descargar", ["Todas las predicciones", "Solo Revisión Manual"],
                                       key="dl_option")
        format_option   = st.selectbox("Formato", ["CSV", "Excel"], key="dl_format")

        out_df    = results_df if download_option == "Todas las predicciones" \
                    else results_df[results_df["Prediccion"] == "Revisión Manual"]
        base_name = "predicciones_fondeo" if download_option == "Todas las predicciones" \
                    else "revision_manual"

        if format_option == "CSV":
            file_data = out_df.to_csv(index=False).encode("utf-8")
            file_name, mime = f"{base_name}.csv", "text/csv"
        else:
            buf = io.BytesIO()
            out_df.to_excel(buf, index=False, engine="openpyxl")
            file_data = buf.getvalue()
            file_name = f"{base_name}.xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        st.download_button(f"Descargar {download_option} ({format_option})",
                           data=file_data, file_name=file_name, mime=mime,
                           key=f"dl_{download_option}_{format_option}")
