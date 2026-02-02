import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from fpdf import FPDF

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PROVA 6830 - Expert Energy Analytics", layout="wide")

# --- 2. BLOQUE DE CSS PARA VISIBILIDAD TOTAL ---
st.markdown(""" 
<style>
    .stApp { background-color: white !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #1f3b4d !important; }
    [data-testid="stFileUploader"] {
        background-color: #f8f9fa !important;
        border: 2px dashed #1f3b4d !important;
        border-radius: 10px; padding: 20px !important;
    }
    [data-testid="stFileUploader"] section { background-color: #f8f9fa !important; }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] small {
        color: #1f3b4d !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #ffffff !important; color: #1f3b4d !important; border: 1px solid #1f3b4d !important;
    }
    .stDownloadButton > button {
        background-color: #ffffff !important; color: #1f3b4d !important;
        border: 2px solid #1f3b4d !important; width: 100% !important; font-weight: bold !important; height: 3em !important;
    }
    .stDownloadButton > button:hover { background-color: #1f3b4d !important; color: white !important; }
    [data-testid="stSidebar"] { background-color: #f1f3f6 !important; border-right: 1px solid #d1d1d1; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #1f3b4d !important; }
    .stMetric { background-color: #ffffff !important; border: 1px solid #dee2e6 !important; padding: 15px !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. DICCIONARIO DE VARIABLES ---
PROVA_MAP = {
    'v1': 'Voltaje Fase 1 (V)', 'v2': 'Voltaje Fase 2 (V)', 'v3': 'Voltaje Fase 3 (V)',
    'v12': 'Voltaje L1-L2 (V)', 'v23': 'Voltaje L2-L3 (V)', 'v31': 'Voltaje L3-L1 (V)',
    'i1': 'Corriente Fase 1 (A)', 'i2': 'Corriente Fase 2 (A)', 'i3': 'Corriente Fase 3 (A)',
    'p1': 'Potencia Activa 1 (kW)', 'p2': 'Potencia Activa 2 (kW)', 'p3': 'Potencia Activa 3 (kW)',
    'w_sys': 'Potencia Activa Total (kW)', 'va_sys': 'Potencia Aparente Total (kVA)',
    'var_sys': 'Potencia Reactiva Total (kVAR)', 'pf_sys': 'Factor de Potencia Total',
    'wh_sys': 'Energía Activa (kWh)', 'vah_sys': 'Energía Aparente (kVAh)',
    'varh_sys': 'Energía Reactiva (kVARh)'
}

# --- 4. FUNCIONES CORE ---
@st.cache_data
def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=';')
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col.replace(' ', '')) for col in df.columns]
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.str.contains('unnamed')]
    cols_to_clean = df.columns[2:]
    for col in cols_to_clean:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True)
    df.set_index('timestamp', inplace=True)
    return df.drop(columns=['date','time']).ffill(limit=2)

def generar_heatmap(df, columna):
    df_heat = df[[columna]].copy()
    df_heat['Hora'] = df_heat.index.hour
    df_heat['Fecha_Etiqueta'] = df_heat.index.strftime('%a %d %b')
    fechas_ordenadas = df_heat['Fecha_Etiqueta'].unique()
    pivot = df_heat.groupby(['Fecha_Etiqueta', 'Hora'])[columna].mean().unstack().reindex(fechas_ordenadas)
    fig = px.imshow(pivot, labels=dict(x="Hora del Día", y="Día de Medición", color="Valor"),
                    title=f"Distribución Horaria: {PROVA_MAP.get(columna, columna)}",
                    color_continuous_scale='YlOrRd')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color="black", size=12),
                    coloraxis_colorbar=dict(tickfont=dict(color="black"), title_font=dict(color="black")))
    fig.update_xaxes(tickfont=dict(color="black"), title_font=dict(color="black"), gridcolor='lightgray')
    fig.update_yaxes(tickfont=dict(color="black"), title_font=dict(color="black"), gridcolor='lightgray')
    return fig

# --- 5. MOTOR DE ANÁLISIS TÉCNICO (DINÁMICO) ---
def analizar_sistema(df, v_nom, umbral):
    conclusiones = []
    
    # 1. Análisis de Voltaje (Dinámico para cualquier fase presente)
    v_cols = [c for c in ['v1', 'v2', 'v3'] if c in df.columns]
    v_min, v_max = v_nom*(1-umbral), v_nom*(1+umbral)
    
    eventos_v = 0
    for vc in v_cols:
        eventos_v += len(df[df[vc] < v_min]) + len(df[df[vc] > v_max])
    
    if eventos_v > 0:
        conclusiones.append(f"CALIDAD DE VOLTAJE: Se detectaron {eventos_v} registros fuera de rango en las fases presentes. Esto indica inestabilidad que puede afectar la vida útil de equipos electrónicos.")
    else:
        conclusiones.append("CALIDAD DE VOLTAJE: Los niveles de tensión se mantienen estables dentro del umbral definido en todas las fases.")

    # 2. Análisis de Factor de Potencia
    if 'pf_sys' in df.columns:
        pf_avg = df['pf_sys'].mean()
        if pf_avg < 0.90:
            conclusiones.append(f"EFICIENCIA: PF promedio de {pf_avg:.2f}. Riesgo de penalización por energía reactiva.")
    
    # 3. Análisis de Desbalance de Carga (Dinámico)
    i_cols = [c for c in ['i1', 'i2', 'i3'] if c in df.columns]
    if len(i_cols) > 1:
        means = [df[c].mean() for c in i_cols]
        avg_i = np.mean(means)
        if avg_i > 0:
            imb = (max(means) - min(means)) / avg_i * 100
            if imb > 15:
                conclusiones.append(f"DESBALANCE: Existe un desequilibrio de corriente del {imb:.1f}%. Esto puede saturar el neutro.")

    return conclusiones

# --- 6. LÓGICA DE LA APP ---
st.sidebar.header("📁 Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo", type=["csv"])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)

    # Identificar fases disponibles para el resto de la app
    #fases_potencia = [c for c in ['p1', 'p2', 'p3'] if c in df.columns]

    # Identificar qué variables existen en este archivo específico
    opciones_disponibles = {PROVA_MAP.get(col, col): col for col in df.columns if col in PROVA_MAP}
    
    # SIDEBAR
    st.sidebar.markdown("---")
    st.sidebar.header("💸 Costos y Umbrales")
    costo_kwh = st.sidebar.number_input("Costo por kWh ($)", value=0.20)
    v_nom = st.sidebar.number_input("Voltaje Nominal (V)", value=120)
    umbral = st.sidebar.slider("Tolerancia (%)", 1, 20, 10) / 100
    
    st.sidebar.header("📊 Filtros")
    # Seleccionar por defecto V1, V2, V3 si existen
    default_vals = [n for n in opciones_disponibles.keys() if 'L-N' in n or 'Fase' in n][:3]
    #opciones = {PROVA_MAP.get(col, col): col for col in df.columns if col in PROVA_MAP}
    seleccionadas = st.sidebar.multiselect("Variables", options=list(opciones_disponibles.keys()), default=default_vals)
    freq = st.sidebar.select_slider("Suavizado", options=['2min', '10min', '1h', '1d'], value='10min')

    # RESUMEN SUPERIOR
    st.title("⚡ Estudio Energético")

    col_a, col_b, col_c = st.columns(3)
    energia = df['wh_sys'].max() - df['wh_sys'].min() if 'wh_sys' in df.columns else 0
    pf_medio = df['pf_sys'].mean() if 'pf_sys' in df.columns else 0
    
    col_a.metric("Energía Consumida", f"{energia:.2f} kWh")
    col_b.metric("Costo Estimado", f"${(energia * costo_kwh):.2f}")
    col_c.metric("PF Promedio", f"{pf_medio:.2f}")
    
    if pf_medio >= 0.95: pf_status, pf_col = "Excelente", "green"
    elif pf_medio >= 0.90: pf_status, pf_col = "Aceptable", "orange"
    else: pf_status, pf_col = "Riesgo / Penalización", "red"
    
    with col_c:
        st.markdown(f"**Salud Energética (PF)**<br><h2 style='color:{pf_col}; margin:0;'>{pf_status}</h2>", unsafe_allow_html=True)

    # --- SECCIÓN DE AUDITORÍA TÉCNICA ---
    st.markdown("---")
    st.subheader("🔬 Análisis Técnico")
    analisis = analizar_sistema(df, v_nom, umbral)
    
    for item in analisis:
        st.write(f"🔹 {item}")

    # GRÁFICA PRINCIPAL
    if seleccionadas:
        st.markdown("---")
        cols_t = [opciones_disponibles[n] for n in seleccionadas]
        df_res = df[cols_t].resample(freq).mean()

        #fig = go.Figure()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

    for c in cols_t:
        # Lógica: Si la columna es voltaje (empieza por 'v'), va al eje principal (izq)
        # Si es corriente ('i'), potencia ('p', 'w', 'va', 'var') o PF, va al eje secundario (der)
        es_secundario = any(tipo in c for tipo in ['i', 'p', 'w', 'va', 'var', 'pf'])
        
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res[c], name=PROVA_MAP.get(c,c)), secondary_y=es_secundario)
        
    fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='white', height=500,
        font=dict(color="black"), hovermode="x unified",
        legend=dict(font=dict(color="black"), orientation="h", y=1.12, bordercolor="gray", borderwidth=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text="Línea de Tiempo", title_font=dict(color="black"), tickfont=dict(color="black"), gridcolor='lightgray', linecolor='black', rangeslider=dict(visible=True, bgcolor="#f1f3f6"))
    fig.update_yaxes(title_text="Magnitud Medida", title_font=dict(color="black"), tickfont=dict(color="black"), gridcolor='lightgray', linecolor='black')
    
    st.plotly_chart(fig, use_container_width=True)

    # TABLA ESTADÍSTICA
    st.subheader("📊 Resumen Estadístico")
    stats = df[cols_t].describe().T[['mean', 'min', 'max']]
    stats.index = [PROVA_MAP.get(i, i) for i in stats.index]
    st.dataframe(stats.style.format("{:.2f}").set_properties(**{'background-color': '#ffffff', 'color': 'black'}), use_container_width=True)



    # SECCIÓN INFERIOR: CALOR Y REPARTO
    st.markdown("---")
    col_heatmap, col_pie = st.columns(2)
    
    with col_heatmap:
        st.subheader("🔥 Análisis Horario de Consumo")
        var_h = st.selectbox("Variable para Mapa", options=list(opciones_disponibles.keys()), index=0)
        st.plotly_chart(generar_heatmap(df, opciones_disponibles[var_h]), use_container_width=True)

    with col_pie:
        st.subheader("⚖️ Reparto de Carga (Corrientes)")
        # Lógica dinámica para priorizar corriente en el reparto
        cols_i = [c for c in ['i1', 'i2', 'i3'] if c in df.columns]
        if cols_i:
            i_means = [df[c].mean() for c in cols_i]
            nombres_i = [PROVA_MAP.get(c, c) for c in cols_i]
            fig_pie = px.pie(values=i_means, names=nombres_i, hole=0.4, 
                             color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_layout(paper_bgcolor='white', font=dict(color="black", size=14), legend=dict(font=dict(color="black")))
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No se detectaron columnas de corriente (I1, I2, I3).")

    # --- SECCIÓN DE DESCARGA DE INFORME TÉCNICO PDF ---
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Exportar Informe Final")

    # Función para generar el PDF (Simplificada para Streamlit)
    def create_pdf_report(analisis_text, stats_df):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Informe de Auditoria Electrica", ln=True, align='C')
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, f"Fecha del Reporte: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "1. Conclusiones y Diagnostico:", ln=True)
        pdf.set_font("Arial", size=11)
        for line in analisis_text:
            pdf.multi_cell(0, 7, f"- {line}")
            pdf.ln(2)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "2. Estadisticas de Variables Criticas:", ln=True)
        pdf.set_font("Arial", size=10)
        for index, row in stats_df.iterrows():
            pdf.cell(0, 7, f"{index}: Media={row['mean']:.2f}, Min={row['min']:.2f}, Max={row['max']:.2f}", ln=True)
        return pdf.output(dest='S').encode('latin-1')

    # Solo generamos el botón si hay datos seleccionados para las estadísticas
    if seleccionadas:
        pdf_bytes = create_pdf_report(analisis, stats)
        st.sidebar.download_button(label="📥 Descargar Informe PDF", data=pdf_bytes,
            file_name=f"Informe_Tecnico_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

else:
    st.title("⚡ Energy Intelligence Platform")
    st.info("Suba el archivo CSV para iniciar el análisis.")