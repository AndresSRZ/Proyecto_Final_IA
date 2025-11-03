import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import requests
import json

# =======================
# CARGAR VARIABLES DE ENTORNO
# =======================
load_dotenv()

def get_secret(key: str):
    """Get secret from st.secrets (Streamlit Cloud) or os.getenv (local)"""
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return os.getenv(key)

GROQ_API_KEY = get_secret("GROQ_API_KEY")
HF_API_KEY = get_secret("HUGGINGFACE_API_KEY")

# =======================
# CONFIG PAGINA
# =======================
st.set_page_config(page_title="Taller IA: OCR + LLM", page_icon="🧠")
st.title("🧠 Taller IA: OCR + LLM")
st.write("Aplicacion multimodal: leer texto de imagen (OCR) y luego analizarlo con un LLM.")

# =======================
# MODULO 1 - OCR
# =======================
st.subheader("Módulo 1: Lector de Imágenes (OCR)")
st.write("Sube una imagen con texto y la aplicacion intentara extraerlo usando OCR.")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["es", "en"], gpu=False, detector=True, recognizer=True)

reader = load_ocr_reader()

uploaded_file = st.file_uploader(
    "Sube una imagen (.png, .jpg, .jpeg)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    if st.button("🔎 Extraer texto con OCR"):
        with st.spinner("Leyendo la imagen..."):
            image_np = np.array(image)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            image_np = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            results = reader.readtext(image_np, detail=0)

            extracted_text = " ".join([t.strip() for t in results])
            extracted_text = (
                extracted_text
                .replace("`", "'")
                .replace(" ,", ",")
                .replace(" .", ".")
            )

            st.session_state["ocr_text"] = extracted_text

            st.success("Texto extraido:")
            st.text_area("Texto OCR", value=extracted_text, height=200)
else:
    st.info("Aun no has subido una imagen.")

# =======================
# MODULO 2 + 3 - NLP
# =======================
st.markdown("---")
st.subheader("Módulo 2 y 3: Analisis del texto con LLM")

ocr_text = st.session_state.get("ocr_text", "")
if not ocr_text:
    st.info("👆 Primero extrae texto con el Modulo 1.")

# 1) Elegir proveedor
provider = st.radio(
    "Proveedor de LLM",
    ["GROQ", "Hugging Face"],
    index=0,
    horizontal=True
)

# 2) Parametros comunes
st.markdown("### Parámetros del modelo")
temperature = st.slider(
    "Creatividad (temperature)",
    0.0, 1.5, 0.4, 0.1
)
max_tokens = st.slider(
    "Longitud maxima de respuesta",
    64, 2048, 512, 32
)

# 3) Tarea
task = st.selectbox(
    "Que quieres hacer con el texto?",
    [
        "Resumir en 3 puntos clave",
        "Traducir al ingles",
        "Identificar las entidades principales",
        "Explicar el texto de forma sencilla",
    ],
)

# 4) Modelos segun proveedor
if provider == "GROQ":
    groq_model = st.selectbox(
        "Modelo de GROQ",
        [
            "llama-3.1-8b-instant",
            "llama3-8b-8192",
            "gemma-7b-it",
        ],
        index=0,
    )
else:
    HF_DEFAULT_MODEL = "facebook/bart-large-cnn"  # público y gratuito
    st.info(f"Usando modelo de Hugging Face: **{HF_DEFAULT_MODEL}**")

# 5) Boton de analisis
if st.button("🧠 Analizar texto"):
    if not ocr_text:
        st.error("No hay texto para analizar. Primero usa el Modulo 1.")
    else:
        # ========== GROQ ==========
        if provider == "GROQ":
            if not GROQ_API_KEY:
                st.error("❌ No se encontro GROQ_API_KEY. Verifica que este configurada en Secrets.")
            else:
                try:
                    from groq import Groq
                    
                    client = Groq(api_key=GROQ_API_KEY)

                    if task == "Resumir en 3 puntos clave":
                        system_msg = "Eres un asistente que resume textos en espanol en exactamente 3 viñetas claras."
                        user_msg = f"Resume el siguiente texto en 3 puntos clave:\n\n{ocr_text}"
                    elif task == "Traducir al ingles":
                        system_msg = "Eres un traductor profesional de espanol a ingles. Mantienes el sentido y el tono."
                        user_msg = f"Traduce al ingles el siguiente texto:\n\n{ocr_text}"
                    elif task == "Identificar las entidades principales":
                        system_msg = "Eres un extractor de informacion. Devuelves personas, lugares, organizaciones y fechas en formato de lista."
                        user_msg = f"Extrae las entidades principales del siguiente texto:\n\n{ocr_text}"
                    else:
                        system_msg = "Eres un profesor que explica textos en espanol usando lenguaje sencillo."
                        user_msg = f"Explica en terminos sencillos el siguiente texto:\n\n{ocr_text}"

                    with st.spinner("Consultando GROQ..."):
                        resp = client.chat.completions.create(
                            model=groq_model,
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": user_msg},
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        answer = resp.choices[0].message.content
                        st.markdown("### ✅ Respuesta del modelo (GROQ)")
                        st.markdown(answer)
                        st.session_state["last_llm_answer"] = answer
                except Exception as e:
                    st.error(f"❌ Error al llamar a GROQ: {str(e)}")
                    st.info("Verifica que GROQ_API_KEY sea válida en los Secrets.")

        # ========== HUGGING FACE ==========
        else:
            if not HF_API_KEY:
                st.error("❌ No se encontro HUGGINGFACE_API_KEY. Verifica que este configurada en Secrets.")
            else:
                st.success("✅ Clave de Hugging Face cargada correctamente")

                HF_DEFAULT_MODEL = "facebook/bart-large-cnn"
                API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_DEFAULT_MODEL}"

                # preparar entrada según tarea
                if task == "Resumir en 3 puntos clave":
                    inputs = f"Resume el siguiente texto en 3 puntos clave:\n\n{ocr_text}"
                elif task == "Traducir al ingles":
                    inputs = f"Traduce al ingles el siguiente texto:\n\n{ocr_text}"
                elif task == "Identificar las entidades principales":
                    inputs = (
                        "Extrae personas, lugares, organizaciones y fechas del siguiente texto. "
                        "Devuelve el resultado en formato de lista en espanol.\n\n"
                        f"Texto:\n{ocr_text}"
                    )
                else:
                    inputs = f"Explica el siguiente texto de forma sencilla:\n\n{ocr_text}"

                headers = {
                    "Authorization": f"Bearer {HF_API_KEY}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "inputs": inputs,
                    "parameters": {
                        "temperature": temperature,
                        "max_new_tokens": max_tokens,
                    },
                }

                with st.spinner("Consultando Hugging Face..."):
                    try:
                        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

                        if response.status_code == 200:
                            data = response.json()
                            if isinstance(data, list) and len(data) > 0:
                                answer = data[0].get("summary_text") or data[0].get("generated_text") or str(data)
                            elif isinstance(data, dict):
                                answer = data.get("summary_text") or data.get("generated_text") or str(data)
                            else:
                                answer = str(data)

                            st.markdown("### ✅ Respuesta del modelo (Hugging Face)")
                            st.markdown(answer)
                            st.session_state["last_llm_answer"] = answer
                        else:
                            st.error(f"❌ Error {response.status_code}: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error al llamar a Hugging Face: {str(e)}")

# mostrar ultima respuesta
if "last_llm_answer" in st.session_state:
    with st.expander("Ver ultima respuesta del LLM"):
        st.markdown(st.session_state["last_llm_answer"])
