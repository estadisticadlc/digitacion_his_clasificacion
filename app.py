
import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="Clasificación de Riesgo de Digitación HISMINSA",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

RUTA_MODELO = BASE_DIR / "modelo_clasificacion_digitacion_his.joblib"
RUTA_CONFIG = BASE_DIR / "config_modelo_clasificacion_his.json"
RUTA_CATALOGO = BASE_DIR / "catalogo_eess_clasificacion.csv"
RUTA_HISTORIAL = BASE_DIR / "historial_eess_clasificacion.csv"

NOMBRE_AUTOR = "Claudia A."
CURSO = "Curso de Despliegue Web"

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1rem;
            color: #666;
            margin-bottom: 1.3rem;
        }
        .risk-card {
            border-radius: 16px;
            padding: 22px;
            text-align: center;
            border: 1px solid rgba(128,128,128,0.20);
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .risk-high {
            background: rgba(255, 75, 75, 0.08);
            border-left: 7px solid #ff4b4b;
        }
        .risk-low {
            background: rgba(0, 150, 136, 0.08);
            border-left: 7px solid #009688;
        }
        .risk-label {
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0.3rem 0;
        }
        .probability {
            font-size: 2.4rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .small-note {
            color: #666;
            font-size: 0.9rem;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            color: #777;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)

@st.cache_data
def cargar_config():
    with open(RUTA_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def cargar_catalogo():
    return pd.read_csv(RUTA_CATALOGO, encoding="utf-8-sig")

@st.cache_data
def cargar_historial():
    df = pd.read_csv(RUTA_HISTORIAL, encoding="utf-8-sig")
    if "FECHA_CIERRE" in df.columns:
        df["FECHA_CIERRE"] = pd.to_datetime(df["FECHA_CIERRE"], errors="coerce")
    return df

try:
    modelo = cargar_modelo()
    config = cargar_config()
    catalogo = cargar_catalogo()
    historial = cargar_historial()
except Exception as e:
    st.error("No se pudieron cargar los archivos del proyecto.")
    st.exception(e)
    st.stop()

def periodo_siguiente(periodo: int) -> int:
    anio = int(periodo) // 100
    mes = int(periodo) % 100
    return (anio + 1) * 100 + 1 if mes == 12 else anio * 100 + mes + 1

def nombre_periodo(periodo: int) -> str:
    meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    anio = int(periodo) // 100
    mes = int(periodo) % 100
    return f"{meses[mes]} {anio}"

def cierre_oficial(periodo: int):
    cierres = {
        202501: "2025-02-04",
        202502: "2025-03-04",
        202503: "2025-04-05",
        202504: "2025-05-05",
        202505: "2025-06-04",
        202506: "2025-07-05",
        202507: "2025-08-04",
        202508: "2025-09-04",
        202509: "2025-10-04",
        202510: "2025-11-04",
        202511: "2025-12-04",
        202512: "2026-01-05",
        202601: "2026-02-06",
        202602: "2026-03-04",
        202603: "2026-04-05",
        202604: "2026-05-05",
        202605: "2026-06-04",
        202606: "2026-07-05",
        202607: "2026-08-04",
        202608: "2026-09-04",
        202609: "2026-10-04",
        202610: "2026-11-04",
        202611: "2026-12-04",
        202612: "2027-01-05",
    }
    valor = cierres.get(int(periodo))
    return None if valor is None else pd.to_datetime(valor).date()

def fin_mes_periodo(periodo: int):
    anio = int(periodo) // 100
    mes = int(periodo) % 100
    inicio = pd.Timestamp(anio, mes, 1)
    return (inicio + pd.offsets.MonthEnd(0)).date()

def construir_entrada(hist_eess, fila_catalogo, periodo_objetivo, fecha_cierre):
    hist_eess = hist_eess.sort_values("PERIODO").copy()
    if hist_eess.empty:
        raise ValueError("No existe historial para el establecimiento seleccionado.")

    ultimo_periodo = int(hist_eess["PERIODO"].max())
    esperado = periodo_siguiente(ultimo_periodo)

    if periodo_objetivo != esperado:
        raise ValueError(
            f"El modelo está diseñado para predecir el siguiente mes. "
            f"El último periodo disponible es {ultimo_periodo}, por lo que corresponde predecir {esperado}."
        )

    ultima = hist_eess.iloc[-1].copy()
    ultimas3 = hist_eess.tail(3).copy()

    dias_hasta_cierre = (
        pd.Timestamp(fecha_cierre) - pd.Timestamp(fin_mes_periodo(periodo_objetivo))
    ).days

    entrada = pd.DataFrame([{
        "DIAS_HASTA_CIERRE": dias_hasta_cierre,
        "TOTAL_ATENCIONES_LAG1": ultima["TOTAL_ATENCIONES"],
        "N_REGISTRADORES_LAG1": ultima["N_REGISTRADORES"],
        "N_UPS_LAG1": ultima["N_UPS"],
        "ATENCIONES_POR_REGISTRADOR_LAG1": ultima["ATENCIONES_POR_REGISTRADOR"],
        "PCT_ATENCIONES_TARDIAS_LAG1": ultima["PCT_ATENCIONES_TARDIAS"],
        "PCT_ATENCIONES_TARDIAS_MA3": ultimas3["PCT_ATENCIONES_TARDIAS"].mean(),
        "PCT_ULTIMOS_3_DIAS_LAG1": ultima["PCT_ULTIMOS_3_DIAS"],
        "PCT_ULTIMOS_3_DIAS_MA3": ultimas3["PCT_ULTIMOS_3_DIAS"].mean(),
        "PCT_DIA_CIERRE_LAG1": ultima["PCT_DIA_CIERRE"],
        "PCT_APP_1_LAG1": ultima["PCT_APP_1"],
        "TOTAL_ATENCIONES_MA3": ultimas3["TOTAL_ATENCIONES"].mean(),
        "DESC_ESTAB": str(fila_catalogo["DESC_ESTAB"]),
        "RIS": str(fila_catalogo["RIS"]),
        "CAT_ESTAB": str(fila_catalogo["CAT_ESTAB"]),
        "DESC_DIST": str(fila_catalogo["DESC_DIST"]),
        "MES_CAT": str(int(periodo_objetivo) % 100).zfill(2),
    }])

    return entrada[config["features"]], ultimo_periodo, ultimas3

st.markdown(
    '<div class="main-title">📊 Clasificación de Riesgo de Digitación HISMINSA</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Predicción del riesgo de que un establecimiento presente 10% o más de atenciones
    registradas después del cierre oficial.
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    ris_disponibles = sorted(
        catalogo["RIS"].dropna().astype(str).unique().tolist()
    )
    ris_seleccionada = st.selectbox("RIS", ris_disponibles)

catalogo_ris = catalogo[
    catalogo["RIS"].astype(str) == str(ris_seleccionada)
].copy().sort_values("DESC_ESTAB")

with col2:
    establecimiento = st.selectbox(
        "Establecimiento de salud",
        catalogo_ris["DESC_ESTAB"].astype(str).tolist()
    )

fila_catalogo = catalogo_ris[
    catalogo_ris["DESC_ESTAB"].astype(str) == str(establecimiento)
].iloc[0]

id_establecimiento = fila_catalogo["ID_ESTABLECIMIENTO"]

hist_eess = historial[
    historial["ID_ESTABLECIMIENTO"] == id_establecimiento
].copy()

if hist_eess.empty:
    st.warning("No se encontró historial para el establecimiento seleccionado.")
    st.stop()

ultimo_periodo = int(hist_eess["PERIODO"].max())
periodo_objetivo = periodo_siguiente(ultimo_periodo)

st.info(
    f"Último periodo disponible: **{nombre_periodo(ultimo_periodo)}**. "
    f"La predicción corresponde a **{nombre_periodo(periodo_objetivo)}**."
)

fecha_cierre_default = cierre_oficial(periodo_objetivo)

if fecha_cierre_default is None:
    fecha_cierre_default = date.today()
    texto_cierre = "No se encontró una fecha oficial cargada para este periodo. Verifícala antes de predecir."
else:
    texto_cierre = "Fecha tomada del calendario oficial cargado en la aplicación."

fecha_cierre = st.date_input(
    "Fecha de cierre oficial del periodo",
    value=fecha_cierre_default
)

st.caption(texto_cierre)

if st.button("Generar predicción", type="primary", width="stretch"):
    try:
        entrada, _, ultimas3 = construir_entrada(
            hist_eess,
            fila_catalogo,
            periodo_objetivo,
            fecha_cierre
        )

        prediccion = int(modelo.predict(entrada)[0])
        probabilidad = float(modelo.predict_proba(entrada)[0, 1])

        etiqueta = "ALTO RIESGO" if prediccion == 1 else "RIESGO BAJO"
        clase_css = "risk-high" if prediccion == 1 else "risk-low"

        st.markdown("---")
        st.subheader(f"Resultado para {nombre_periodo(periodo_objetivo)}")

        st.markdown(
            f"""
            <div class="risk-card {clase_css}">
                <div class="small-note">Predicción del modelo</div>
                <div class="risk-label">{etiqueta}</div>
                <div class="small-note">Probabilidad estimada de alto riesgo</div>
                <div class="probability">{probabilidad*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "ALTO RIESGO = clase 1. RIESGO BAJO = clase 0."
        )

        st.markdown("---")
        st.subheader("Contexto histórico del establecimiento")

        ultima = hist_eess.sort_values("PERIODO").iloc[-1]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Atenciones del último mes",
            f"{int(ultima['TOTAL_ATENCIONES']):,}"
        )
        c2.metric(
            "% tardías último mes",
            f"{float(ultima['PCT_ATENCIONES_TARDIAS']):.2f}%"
        )
        c3.metric(
            "Promedio tardías últimos 3 meses",
            f"{float(ultimas3['PCT_ATENCIONES_TARDIAS'].mean()):.2f}%"
        )
        c4.metric(
            "Registradores último mes",
            f"{int(ultima['N_REGISTRADORES']):,}"
        )

        historico_grafico = hist_eess[
            ["PERIODO", "PCT_ATENCIONES_TARDIAS"]
        ].sort_values("PERIODO").copy()

        historico_grafico["PERIODO"] = historico_grafico["PERIODO"].astype(str)

        st.line_chart(
            historico_grafico.set_index("PERIODO"),
            y="PCT_ATENCIONES_TARDIAS",
            width="stretch"
        )

        with st.expander("Ver variables utilizadas por el modelo"):
            detalle = entrada.copy().T
            detalle.columns = ["Valor"]
            # Convertimos a texto solo para visualización para evitar
            # errores de serialización Arrow por mezclar números y cadenas.
            detalle["Valor"] = detalle["Valor"].astype(str)
            st.dataframe(detalle, width="stretch")

        st.markdown("---")
        st.subheader("Rendimiento del modelo")

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric("Accuracy", f"{config.get('accuracy_test', np.nan)*100:.1f}%")
        m2.metric("Precision", f"{config.get('precision_test', np.nan)*100:.1f}%")
        m3.metric("Recall", f"{config.get('recall_test', np.nan)*100:.1f}%")
        m4.metric("F1", f"{config.get('f1_test', np.nan)*100:.1f}%")
        m5.metric("AUC", f"{config.get('auc_test', np.nan):.2f}")

        st.caption(
            "Estas métricas corresponden a la evaluación temporal realizada antes "
            "del reentrenamiento final del modelo."
        )

    except Exception as e:
        st.error("No se pudo generar la predicción.")
        st.exception(e)

st.markdown(
    f"""
    <div class="footer">
        <b>{NOMBRE_AUTOR}</b><br>
        {CURSO}<br>
        Uso académico y analítico.
    </div>
    """,
    unsafe_allow_html=True
)
