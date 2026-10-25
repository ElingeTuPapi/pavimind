import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="PaviMind - Inspección de Pavimentos",
    page_icon="🛣️",
    layout="wide"
)

# ==============================================================================
# CARGA DEL MODELO YOLOv8 (PESOS LOCALES)
# ==============================================================================
@st.cache_resource
def cargar_modelo():
    rutas_candidatas = [
        os.path.join("runs", "detect", "train-2", "weights", "best.pt"),
        os.path.join("runs", "detect", "train", "weights", "best.pt"),
        "modelo_pavimind.pt"
    ]
    for ruta in rutas_candidatas:
        if os.path.exists(ruta):
            if os.path.getsize(ruta) > 1_000_000:
                try:
                    return YOLO(ruta)
                except Exception:
                    continue
    return None

modelo_yolo = cargar_modelo()

# ==============================================================================
# MOTOR HÍBRIDO (CON SUPRESIÓN DE PINTURA Y BACHES)
# ==============================================================================
def analizar_pavimento(image_np, escala_m_px, conf_yolo=0.15, corte_roi_superior=0.35, sens_fisura=35, largo_min_px=110, umbral_pintura=170):
    img_dibujo = image_np.copy()
    alto_img, ancho_img = image_np.shape[:2]
    linea_limite_y = int(alto_img * corte_roi_superior)

    lista_fisuras = []
    lista_baches = []
    cajas_baches_coords = []

    # --------------------------------------------------------------------------
    # 1. DETECCIÓN DE BACHES CON YOLOv8
    # --------------------------------------------------------------------------
    if modelo_yolo is not None:
        try:
            resultados = modelo_yolo(image_np, conf=conf_yolo, verbose=False)[0]
            nombres = resultados.names

            for box in resultados.boxes:
                cls_id = int(box.cls[0].item())
                clase = str(nombres.get(cls_id, cls_id)).lower()
                confianza = float(box.conf[0].item())

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w_px = max(1, x2 - x1)
                h_px = max(1, y2 - y1)
                centro_y = (y1 + y2) / 2

                if centro_y < linea_limite_y or w_px > (ancho_img * 0.70):
                    continue

                if "bache" in clase or "pothole" in clase or clase == "0":
                    ancho_m = round(w_px * escala_m_px, 2)
                    largo_m = round(h_px * escala_m_px, 2)
                    area_m2 = round(ancho_m * largo_m, 2)

                    if area_m2 < 0.10:
                        severidad = "Baja"
                        dv_base = 20
                    elif area_m2 <= 0.25:
                        severidad = "Media"
                        dv_base = 45
                    else:
                        severidad = "Alta"
                        dv_base = 80

                    lista_baches.append({
                        "ID": f"B-{len(lista_baches)+1}",
                        "Patología": "Bache",
                        "Severidad": severidad,
                        "Cantidad": area_m2, # Se mide en m2
                        "Unidad": "m²",
                        "DV_Base": dv_base
                    })

                    cajas_baches_coords.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                    cv2.rectangle(img_dibujo, (x1, y1), (x2, y2), (230, 30, 30), 3)
                    cv2.putText(img_dibujo, f"Bache {confianza*100:.0f}%", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 30, 30), 2)
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # 2. DETECCIÓN DE FISURAS (ANTI-RIPIO Y ANTI-PINTURA VIAL)
    # --------------------------------------------------------------------------
    roi = image_np[linea_limite_y:alto_img, 0:ancho_img]
    gris_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # MÁSCARA ANTI-PINTURA TERMOPLÁSTICA: Aísla las zonas muy claras (líneas de carril)
    _, mascara_pintura = cv2.threshold(gris_roi, umbral_pintura, 255, cv2.THRESH_BINARY)
    kernel_pintura = np.ones((9, 9), np.uint8) # Dilatar para cubrir el borde difuso de la pintura
    mascara_pintura = cv2.dilate(mascara_pintura, kernel_pintura, iterations=2)

    difuminado = cv2.GaussianBlur(gris_roi, (13, 13), 0)
    kernel_morf = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    blackhat = cv2.morphologyEx(difuminado, cv2.MORPH_BLACKHAT, kernel_morf)
    _, binaria = cv2.threshold(blackhat, sens_fisura, 255, cv2.THRESH_BINARY)

    # EXCLUSIÓN: Restar la máscara de pintura a la detección de fisuras
    binaria_limpia = cv2.bitwise_and(binaria, cv2.bitwise_not(mascara_pintura))

    kernel_conectar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binaria_conectada = cv2.morphologyEx(binaria_limpia, cv2.MORPH_CLOSE, kernel_conectar, iterations=2)

    contornos, _ = cv2.findContours(binaria_conectada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area_px = cv2.contourArea(c)
        if area_px < 650:
            continue

        x, y, w, h = cv2.boundingRect(c)
        longitud_px = max(w, h)
        ancho_px = max(1, min(w, h))

        if longitud_px >= largo_min_px and (float(longitud_px) / ancho_px) >= 1.8:
            y_real = y + linea_limite_y

            # Supresión MTC: ignorar fisuras dentro de baches
            centro_x = x + (w / 2)
            centro_y = y_real + (h / 2)
            if any((b["x1"] - 15 <= centro_x <= b["x2"] + 15) and (b["y1"] - 15 <= centro_y <= b["y2"] + 15) for b in cajas_baches_coords):
                continue

            longitud_m = round(longitud_px * escala_m_px, 2)
            ancho_mm = ancho_px * escala_m_px * 1000

            if ancho_mm < 3.0:
                severidad, ancho_inf, dv_base = "Baja", 0.15, 10
            elif ancho_mm <= 10.0:
                severidad, ancho_inf, dv_base = "Media", 0.35, 25
            else:
                severidad, ancho_inf, dv_base = "Alta", 0.60, 50

            area_afectada_m2 = round(longitud_m * ancho_inf, 2)

            lista_fisuras.append({
                "ID": f"F-{len(lista_fisuras)+1}",
                "Patología": "Fisura Longitudinal",
                "Severidad": severidad,
                "Cantidad": longitud_m, # Se mide en metros lineales
                "Unidad": "m",
                "Área Afectada (m²)": area_afectada_m2,
                "DV_Base": dv_base
            })

            cv2.rectangle(img_dibujo, (x, y_real), (x + w, y_real + h), (255, 140, 0), 2)
            cv2.putText(img_dibujo, f"Fisura {severidad}", (x, max(22, y_real - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2)

    return img_dibujo, lista_fisuras, lista_baches

# ==============================================================================
# MEMORIA DE CÁLCULO MTC / ASTM D6433 (LÓGICA EXCEL)
# ==============================================================================
def calcular_memoria_pci(lista_fisuras, lista_baches, area_muestra_m2):
    consolidado = lista_fisuras + lista_baches
    if not consolidado:
        return 100, pd.DataFrame(), 0, 0, "Excelente", "Mantenimiento rutinario."

    df_pci = pd.DataFrame(consolidado)
    
    # Agrupar por Patología y Severidad (Paso 1 del Excel)
    resumen = df_pci.groupby(["Patología", "Severidad", "Unidad"]).agg(
        Total_Cantidad=("Cantidad", "sum"),
        DV_Maximo=("DV_Base", "max") # Proxy del DV obtenido en curvas
    ).reset_index()

    # Cálculo de Densidad % (Paso 2 del Excel)
    resumen["Densidad (%)"] = round((resumen["Total_Cantidad"] / area_muestra_m2) * 100, 2)
    
    # Valor Deducido (DV) dinámico ajustado por densidad
    resumen["DV_Calculado"] = resumen.apply(lambda row: min(100, round(row["DV_Maximo"] * (1 + (row["Densidad (%)"] / 100)), 1)), axis=1)

    # Ordenar DVs de mayor a menor para el cálculo del CDV
    dvs_ordenados = resumen["DV_Calculado"].sort_values(ascending=False).tolist()
    
    # Cálculo del TDV (Total Deduct Value)
    tdv = sum(dvs_ordenados)

    # Cálculo del CDV (Corrected Deduct Value) - Lógica de curvas de corrección
    if len(dvs_ordenados) == 1:
        cdv = tdv
    else:
        # Aproximación matemática a las curvas de corrección MTC para muestras múltiples
        max_dv = dvs_ordenados[0]
        resto_dv = sum(dvs_ordenados[1:])
        cdv = round(max_dv + (0.35 * resto_dv), 1)
        cdv = min(100.0, cdv)

    pci = max(0, int(100 - cdv))

    if pci >= 85: estado, rec = "Excelente / Bueno", "Limpieza y monitoreo."
    elif pci >= 70: estado, rec = "Satisfactorio", "Sellado preventivo de fisuras aisladas."
    elif pci >= 55: estado, rec = "Regular", "Sellado de grietas y bacheo superficial."
    elif pci >= 40: estado, rec = "Malo", "Bacheo profundo y fresado puntual."
    else: estado, rec = "Muy Malo / Fallado", "Rehabilitación estructural."

    return pci, resumen, tdv, cdv, estado, rec

# ==============================================================================
# INTERFAZ STREAMLIT
# ==============================================================================
st.title("🛣️ PaviMind: Plataforma de Inspección de Pavimentos")
st.caption("Inspección automatizada y memoria de cálculo de condición (ASTM D6433).")

st.sidebar.header("⚙️ Geometría de Inspección")
ancho_calzada = st.sidebar.number_input("Ancho Visible (m):", 1.00, 15.00, 3.60, 0.10)
long_tramo = st.sidebar.number_input("Longitud del Tramo (m):", 1.00, 50.00, 6.00, 0.50)
area_muestra = ancho_calzada * long_tramo

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Calibración de Detección")
conf_baches = st.sidebar.slider("Confianza Baches (YOLO):", 0.05, 0.90, 0.20, 0.05)
sens_fisuras = st.sidebar.slider("Sensibilidad Fisuras:", 15, 60, 32, 2)
filtro_pintura = st.sidebar.slider("Filtro Pintura Vial (Umbral Brillo):", 120, 240, 170, 5, help="Valores menores filtran más pintura blanca/amarilla. Súbelo si borra asfalto claro.")
largo_min = st.sidebar.slider("Longitud Mínima Fisura (px):", 50, 300, 140, 10)
excluir_fondo = st.sidebar.slider("Excluir Fondo (% superior):", 0.0, 0.60, 0.00, 0.05)

uploaded_file = st.file_uploader("Sube una fotografía de la superficie del pavimento (JPG / PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    img_w = image_np.shape[1]
    escala = ancho_calzada / float(img_w)

    img_procesada, fisuras, baches = analizar_pavimento(
        image_np, escala, conf_yolo=conf_baches, corte_roi_superior=excluir_fondo,
        sens_fisura=sens_fisuras, largo_min_px=largo_min, umbral_pintura=filtro_pintura
    )

    pci, df_memoria, tdv, cdv, estado, recomendacion = calcular_memoria_pci(fisuras, baches, area_muestra)

    st.subheader("📸 Imagen Analizada")
    st.image(img_procesada, width="stretch")

    st.markdown("### 📊 Diagnóstico de Condición del Pavimento")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Índice PCI Estimado", f"{pci} / 100")
    k2.metric("Estado Operacional", estado)
    k3.metric("Baches Identificados", len(baches))
    k4.metric("Fisuras Identificadas", len(fisuras))
    st.info(f"💡 **Recomendación Técnica:** {recomendacion}")

    # EL PROCEDIMIENTO "EXCEL" DESPLEGABLE
    with st.expander("🧮 Ver Memoria de Cálculo PCI Detallada (Procedimiento MTC)", expanded=True):
        st.markdown(f"**Área de la Unidad de Muestra:** {area_muestra:.2f} m²")
        if not df_memoria.empty:
            st.dataframe(df_memoria.style.format({
                "Total_Cantidad": "{:.2f}",
                "Densidad (%)": "{:.2f}%",
                "DV_Calculado": "{:.1f}"
            }), width="stretch")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor Deducido Total (TDV)", f"{tdv:.1f}")
            c2.metric("Valor Deducido Corregido (CDV)", f"{cdv:.1f}")
            c3.metric("Cálculo Final PCI (100 - CDV)", f"{pci}")
        else:
            st.success("Pavimento sin daños deductibles. PCI = 100.")
else:
    st.info("Sube una imagen de prueba para iniciar la inspección automática.")