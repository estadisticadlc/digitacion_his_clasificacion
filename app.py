import json
from pathlib import Path

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
RUTA_VALIDACION = BASE_DIR / "registro_validacion_test.csv"

AUTOR = "Ing. Claudia Alocén"
CURSO = "ENEI 2026-G4-612491-MACHINE LEARNING EN PRODUCCIÓN - DESPLIEGUE WEB"

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.15rem;
            font-weight: 750;
            margin-bottom: 0.15rem;
        }

        .subtitle {
            color: #666666;
            font-size: 1rem;
            margin-bottom: 1.25rem;
        }

        .result-low {
            background-color: #d9f8d9;
            border-left: 7px solid #28a745;
            padding: 18px 22px;
            border-radius: 8px;
            margin-top: 12px;
            margin-bottom: 12px;
        }

        .result-high {
            background-color: #ffdede;
            border-left: 7px solid #dc3545;
            padding: 18px 22px;
            border-radius: 8px;
            margin-top: 12px;
            margin-bottom: 12px;
        }

        .result-title {
            font-size: 1.5rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .result-prob {
            font-size: 1.35rem;
            font-weight: 750;
            margin-top: 7px;
        }

        .info-box {
            background-color: #eef6ff;
            border-left: 6px solid #3693ff;
            padding: 14px 18px;
            border-radius: 8px;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .footer {
            text-align: center;
            color: #777777;
            font-size: 0.88rem;
            margin-top: 2rem;
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
    return pd.read_csv(RUTA_HISTORIAL, encoding="utf-8-sig")

@st.cache_data
def cargar_validacion():
    if RUTA_VALIDACION.exists():
        return pd.read_csv(RUTA_VALIDACION, encoding="utf-8-sig")
    return None

try:
    modelo = cargar_modelo()
    config = cargar_config()
    catalogo = cargar_catalogo()
    historial = cargar_historial()
    validacion = cargar_validacion()
except Exception as e:
    st.error("No se pudieron cargar los archivos del proyecto.")
    st.exception(e)
    st.stop()

MESES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

CIERRES = {
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

def nombre_periodo(periodo):
    periodo = int(periodo)
    return f"{MESES[periodo % 100]} {periodo // 100}"

def dias_hasta_cierre(periodo):
    periodo = int(periodo)
    fecha_cierre = pd.Timestamp(CIERRES[periodo])
    anio = periodo // 100
    mes = periodo % 100
    fin_mes = pd.Timestamp(anio, mes, 1) + pd.offsets.MonthEnd(0)
    return int((fecha_cierre - fin_mes).days)

def construir_entrada_futura(hist_eess, fila_catalogo, periodo_objetivo):
    hist_eess = hist_eess.sort_values("PERIODO").copy()
    ultima = hist_eess.iloc[-1].copy()
    ultimas3 = hist_eess.tail(3).copy()

    entrada = pd.DataFrame([{
        "DIAS_HASTA_CIERRE": dias_hasta_cierre(periodo_objetivo),
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
        "MES_CAT": str(int(periodo_objetivo) % 100).zfill(2)
    }])

    entrada = entrada[config["features"]]

    return entrada, ultima, ultimas3

def mostrar_resultado(clase, prob_alto, descripcion=None):
    es_alto = str(clase).upper() == "ALTO RIESGO"
    css = "result-high" if es_alto else "result-low"

    if descripcion is None:
        if es_alto:
            descripcion = (
                "El modelo clasifica al establecimiento dentro del grupo "
                "con 10% o más de atenciones registradas después del cierre oficial."
            )
        else:
            descripcion = (
                "El modelo clasifica al establecimiento dentro del grupo "
                "con menos de 10% de atenciones registradas después del cierre oficial."
            )

    html = (
        f'<div class="{css}">'
        f'<div class="result-title">Predicción: {clase}</div>'
        f'<div>{descripcion}</div>'
        f'<div class="result-prob">'
        f'Probabilidad estimada de RIESGO: {prob_alto*100:.1f}%'
        f'</div>'
        f'<div style="margin-top:6px;">'
        f'Exactitud global del modelo (Accuracy): {config.get("accuracy_test", np.nan)*100:.1f}%'
        f'</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📊 Clasificación de Riesgo de Digitación HISMINSA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Clasificación del riesgo de que un establecimiento presente '
    '10% o más de atenciones registradas después del cierre oficial.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    ris_disponibles = sorted(
        catalogo["RIS"].dropna().astype(str).unique().tolist()
    )
    ris_seleccionada = st.selectbox("RIS", ris_disponibles)

catalogo_ris = (
    catalogo[
        catalogo["RIS"].astype(str) == str(ris_seleccionada)
    ]
    .copy()
    .sort_values("DESC_ESTAB")
)

with col2:
    establecimiento = st.selectbox(
        "Establecimiento de salud",
        catalogo_ris["DESC_ESTAB"].astype(str).tolist()
    )

fila_catalogo = (
    catalogo_ris[
        catalogo_ris["DESC_ESTAB"].astype(str) == str(establecimiento)
    ]
    .iloc[0]
)

id_establecimiento = fila_catalogo["ID_ESTABLECIMIENTO"]

hist_eess = (
    historial[
        historial["ID_ESTABLECIMIENTO"] == id_establecimiento
    ]
    .copy()
)

if hist_eess.empty:
    st.warning("No se encontró historial para el establecimiento seleccionado.")
    st.stop()

periodos_disponibles = [
    202607,
    202608,
    202609,
    202610,
    202611,
    202612
]

periodo_objetivo = st.selectbox(
    "Mes a evaluar",
    options=periodos_disponibles,
    format_func=nombre_periodo
)

es_test_temporal = periodo_objetivo in [202607, 202608]

if es_test_temporal:
    st.info(
        f"**{nombre_periodo(periodo_objetivo)}** pertenece al conjunto "
        f"de **TEST temporal** utilizado para evaluar el modelo."
    )
elif periodo_objetivo == 202609:
    st.info(
        "El último periodo real disponible es **Agosto 2026**. "
        "Por ello, **Septiembre 2026** corresponde a una predicción directa "
        "utilizando información histórica real."
    )
else:
    st.warning(
        f"**{nombre_periodo(periodo_objetivo)}** se presenta como una "
        f"proyección de escenario utilizando el comportamiento histórico "
        f"real más reciente disponible."
    )

st.caption(
    "Cierre oficial del periodo: "
    + pd.Timestamp(CIERRES[periodo_objetivo]).strftime("%d/%m/%Y")
)

if st.button(
    "Generar predicción",
    type="primary",
    width="stretch"
):
    try:
        if es_test_temporal:
            if validacion is None:
                st.error(
                    "Falta el archivo `registro_validacion_test.csv` "
                    "en la misma carpeta que `app.py`."
                )
                st.stop()

            val = validacion.copy()
            val["PERIODO"] = pd.to_numeric(
                val["PERIODO"],
                errors="coerce"
            ).astype("Int64")

            caso = val[
                (val["PERIODO"] == int(periodo_objetivo))
                &
                (
                    val["ID_ESTABLECIMIENTO"].astype(str)
                    == str(id_establecimiento)
                )
            ].copy()

            if caso.empty:
                st.warning(
                    "El establecimiento seleccionado no tiene una observación "
                    "disponible en ese periodo del TEST temporal."
                )
                st.stop()

            caso = caso.iloc[0]

            clase_predicha = str(caso["CLASE_PREDICHA"])
            prob_alto = float(caso["PROB_ALTO_RIESGO"])

            st.markdown("---")
            st.subheader(
                "Resultado para "
                + nombre_periodo(periodo_objetivo)
            )

            mostrar_resultado(
                clase_predicha,
                prob_alto,
                descripcion=(
                    "Resultado obtenido durante la evaluación "
                    "del TEST temporal del modelo."
                )
            )

            pct_real = float(
                caso["PCT_ATENCIONES_TARDIAS"]
            )

            clase_real = str(
                caso["CLASE_REAL"]
            )

            st.markdown(
                f"""
                <div class="info-box">
                    <b>Dato observado del periodo:</b><br>
                    Porcentaje real de atenciones tardías:
                    <b>{pct_real:.2f}%</b><br>
                    Clase real:
                    <b>{clase_real}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            entrada, ultima, ultimas3 = construir_entrada_futura(
                hist_eess,
                fila_catalogo,
                periodo_objetivo
            )

            prediccion = int(
                modelo.predict(entrada)[0]
            )

            prob_alto = float(
                modelo.predict_proba(entrada)[0, 1]
            )

            clase_predicha = (
                "ALTO RIESGO"
                if prediccion == 1
                else "RIESGO BAJO"
            )

            st.markdown("---")
            st.subheader(
                "Resultado para "
                + nombre_periodo(periodo_objetivo)
            )

            mostrar_resultado(
                clase_predicha,
                prob_alto
            )

            st.markdown("---")
            st.subheader(
                "Contexto histórico utilizado"
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Atenciones último mes",
                f"{int(ultima['TOTAL_ATENCIONES']):,}"
            )

            m2.metric(
                "% tardías último mes",
                f"{float(ultima['PCT_ATENCIONES_TARDIAS']):.2f}%"
            )

            m3.metric(
                "Promedio tardías últimos 3 meses",
                f"{float(ultimas3['PCT_ATENCIONES_TARDIAS'].mean()):.2f}%"
            )

            m4.metric(
                "Registradores último mes",
                f"{int(ultima['N_REGISTRADORES']):,}"
            )

            with st.expander(
                "Ver variables utilizadas por el modelo"
            ):
                detalle = entrada.copy().T
                detalle.columns = ["Valor"]
                detalle["Valor"] = detalle["Valor"].astype(str)

                st.dataframe(
                    detalle,
                    width="stretch"
                )

        st.markdown("---")
        st.subheader(
            "Rendimiento del modelo"
        )

        a1, a2, a3, a4, a5 = st.columns(5)

        a1.metric(
            "Accuracy",
            f"{config.get('accuracy_test', np.nan) * 100:.1f}%"
        )

        a2.metric(
            "Precision",
            f"{config.get('precision_test', np.nan) * 100:.1f}%"
        )

        a3.metric(
            "Recall",
            f"{config.get('recall_test', np.nan) * 100:.1f}%"
        )

        a4.metric(
            "F1",
            f"{config.get('f1_test', np.nan) * 100:.1f}%"
        )

        a5.metric(
            "AUC",
            f"{config.get('auc_test', np.nan):.2f}"
        )

        st.caption(
            "Métricas obtenidas sobre el TEST temporal "
            "formado por los periodos más recientes reservados "
            "para evaluación."
        )

    except Exception as e:
        st.error(
            "No se pudo generar el resultado."
        )
        st.exception(e)

st.markdown(
    f"""
    <div class="footer">
        <b>Elaborado por:</b> {AUTOR}<br>
        <b>Curso:</b> {CURSO}<br>
        Uso académico y analítico.
    </div>
    """,
    unsafe_allow_html=True
)
