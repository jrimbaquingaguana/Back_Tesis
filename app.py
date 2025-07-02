from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import re
import os
import json
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from datetime import datetime
from cryptography.fernet import Fernet

app = Flask(__name__)
CORS(app)

# Rutas y carga modelo
ruta_actual = os.path.dirname(__file__)
ruta_modelo = r"C:\Users\richa\Desktop\tesis_bien\PervertedJusticeDataset"
ruta_historial = os.path.join(ruta_actual, "historial_mensajes.json")
ruta_usuarios = os.path.join(ruta_actual, "usuarios_enc.json")

modelo = load(os.path.join(ruta_modelo, "modelo_lgbm_bert_gpu_mejorado.joblib"))
encoder = load(os.path.join(ruta_modelo, "bert_encoder.joblib"))

# Clave para cifrado usuarios
SECRET_KEY = b"tEjU4djvW9Op5HgKTXMAnWLIhUHyAJJsPxv7lL_NLR4="
fernet = Fernet(SECRET_KEY)

# Parámetros umbral y confianza para clasificación mejorada
UMBRAL = 0.40
CONF_BAJO = 0.10
CONF_ALTO = 0.90

def guardar_usuarios(usuarios):
    json_data = json.dumps(usuarios).encode("utf-8")
    encrypted = fernet.encrypt(json_data)
    with open(ruta_usuarios, "wb") as f:
        f.write(encrypted)

def obtener_usuarios():
    if not os.path.exists(ruta_usuarios):
        return []
    try:
        with open(ruta_usuarios, "rb") as f:
            encrypted = f.read()
        decrypted = fernet.decrypt(encrypted)
        usuarios = json.loads(decrypted.decode("utf-8"))
        return usuarios
    except Exception as e:
        print("Error leyendo usuarios cifrados:", e)
        return []

def limpiar_texto(texto):
    try:
        if not isinstance(texto, str):
            return ""
        texto = unidecode(texto.lower().strip())
        texto = re.sub(r"http\S+|www\S+|https\S+|@\w+", '', texto)
        texto = re.sub(r'\d{1,2}\s*/\s*[mf]\b', 'edad_genero', texto)
        texto = re.sub(r'\b\d+\b', 'num', texto)

        abreviaturas = {
            r'\bu\b': 'you', r'\bur\b': 'your', r'\br\b': 'are',
            r'\bthx\b': 'thanks', r'\bplz\b': 'please',
            r'\bwanna\b': 'want to', r'\bgonna\b': 'going to',
            r'\blol\b': 'laugh', r'\bomg\b': 'oh my god'
        }
        for pat, repl in abreviaturas.items():
            texto = re.sub(pat, repl, texto)

        texto = re.sub(r"[^\w\s'?]", '', texto)
        tokens = word_tokenize(texto)
        return ' '.join(tokens).strip()
    except Exception:
        return ""

def es_texto_simple(texto):
    palabras_basicas = {
        "hello", "hi", "hey", "what", "yes", "no", "ok", "okay", "thanks", "thank",
        "please", "bye", "good", "bad", "sure", "welcome"
    }
    tokens = word_tokenize(texto)
    return 1 <= len(tokens) <= 2 and all(token in palabras_basicas for token in tokens)

# Nueva función para clasificación con ajuste de umbral y confianza
def clasificar_con_ajuste(prob):
    pred = 1 if prob >= UMBRAL else 0
    if pred == 1 and prob < CONF_ALTO:
        # En zona ambigua para predicción positiva, baja a negativo para reducir falsos positivos
        pred = 0
    elif pred == 0 and prob > CONF_BAJO:
        # En zona ambigua para negativo, mantener negativo
        pred = 0
    return pred

def clasificar_str(pred):
    return "Depredador" if pred == 1 else "Normal"

# Manejo historial
def obtener_historial():
    if os.path.exists(ruta_historial):
        with open(ruta_historial, "r", encoding="utf-8") as f:
            try:
                historial = json.load(f)
                return historial
            except json.JSONDecodeError:
                return []
    return []

def guardar_historial(historial):
    with open(ruta_historial, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=4)

def guardar_en_historial(registro):
    historial = obtener_historial()
    nuevo_id = 1
    if historial:
        ids = [r.get("id", 0) for r in historial]
        nuevo_id = max(ids) + 1
    registro["id"] = nuevo_id
    registro["fecha_creacion"] = datetime.now().isoformat()
    historial.append(registro)
    guardar_historial(historial)

# Endpoint principal de predicción
@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        data = request.get_json()
        if not data or "mensaje" not in data:
            return jsonify({"error": "Falta el campo 'mensaje' en el JSON."}), 400

        texto = data["mensaje"]
        limpio = limpiar_texto(texto)
        fecha_actual = datetime.now().isoformat()

        if es_texto_simple(limpio):
            resultado = {
                "texto_original": texto,
                "probabilidad": 0.0,
                "clasificacion": "Normal",
                "fecha": fecha_actual
            }
            guardar_en_historial(resultado)
            return jsonify(resultado)

        vector = encoder.encode([limpio])
        prob = modelo.predict_proba(vector)[0][1]
        pred = clasificar_con_ajuste(prob)
        resultado = {
            "texto_original": texto,
            "probabilidad": round(float(prob), 4),
            "clasificacion": clasificar_str(pred),
            "fecha": fecha_actual
        }

        guardar_en_historial(resultado)
        return jsonify(resultado)

    except Exception as e:
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

# Aquí agregas los demás endpoints que ya tienes (historial, usuarios, login, etc.)
# ...

if __name__ == '__main__':
    usuarios = obtener_usuarios()
    if not any(u["username"] == "admin" for u in usuarios):
        usuarios.append({
            "username": "admin",
            "email": "admin@example.com",
            "password": "admin",
            "role": "admin"
        })
        guardar_usuarios(usuarios)

    app.run(debug=True, port=5000)
