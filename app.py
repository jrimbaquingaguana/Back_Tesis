from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
from pymongo import MongoClient
import bcrypt
import re
import os
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from datetime import datetime
import torch
app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------

# Ruta de modelos (desde variable de entorno o por defecto)
ruta_modelo = os.getenv("RUTA_MODELO", "./modelos")
modelo_path = os.path.join(ruta_modelo, "modelo_lgbm_bert_cpu.joblib")
encoder_path = os.path.join(ruta_modelo, "bert_encoder.joblib")

# Verificar existencia de archivos
if not os.path.isfile(modelo_path):
    raise FileNotFoundError(f"Modelo no encontrado en {modelo_path}")
if not os.path.isfile(encoder_path):
    raise FileNotFoundError(f"Encoder no encontrado en {encoder_path}")

# Forzar uso de CPU
device = torch.device("cpu")

# Cargar modelo LGBM (entrenado con CPU)
modelo = load(modelo_path)

# Cargar encoder BERT entrenado (guardado con joblib)
encoder = load(encoder_path)

client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017/"))
db = client["tesis"]
col_usuarios = db["usuarios"]
col_historial = db["historial_mensajes"]
col_auditoria = db["auditoria"]

# ---------------- UTILITIES ----------------

def registrar_auditoria(usuario_actor, accion, detalle="", usuario_afectado="-"):
    evento = {
        "usuario": usuario_actor,
        "accion": accion,
        "usuario_afectado": usuario_afectado,
        "detalle": detalle,
        "fecha": datetime.now().isoformat(),
        "ip": request.remote_addr
    }
    col_auditoria.insert_one(evento)

def limpiar_texto(texto):
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

def eliminar_id_mongo(documento):
    if isinstance(documento, list):
        return [eliminar_id_mongo(d) for d in documento]
    elif isinstance(documento, dict):
        return {k: eliminar_id_mongo(v) for k, v in documento.items() if k != "_id"}
    return documento

# ---------------- ENDPOINTS ----------------

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.get_json()
    texto = data.get("mensaje", "")
    usuario = request.headers.get("usuario") or "unidentified_user"
    
    texto_limpio = limpiar_texto(texto)
    vector = encoder.encode([texto_limpio])
    prob = modelo.predict_proba(vector)[0][1]

    if len(texto_limpio.split()) <= 2:
        clasificacion = "Normal"
    elif prob < 0.25:
        clasificacion = "Normal"
    elif prob < 0.85:
        clasificacion = "Cibersexting"
    else:
        clasificacion = "CiberGrooming"

    registro = {
        "texto_original": texto,
        "clasificacion": clasificacion,
        "fecha": datetime.now().isoformat()
    }

    last = col_historial.find_one(sort=[("id", -1)])
    registro["id"] = (last["id"] if last else 0) + 1
    registro["fecha_creacion"] = datetime.now().isoformat()
    col_historial.insert_one(registro)

    registrar_auditoria(usuario, "prediction", f"Classified message as {clasificacion}", usuario_afectado="-")
    return jsonify(eliminar_id_mongo(registro)), 200

@app.route('/historial', methods=['GET'])
def obtener_historial():
    historial = list(col_historial.find())
    return jsonify(eliminar_id_mongo(historial)), 200

@app.route('/historial/<int:registro_id>', methods=['PUT'])
def actualizar_historial(registro_id):
    data = request.get_json()
    usuario_actor = request.headers.get("usuario") or "unidentified_user"
    original = col_historial.find_one({"id": registro_id})

    if not original:
        return jsonify({"error": "Record not found"}), 404

    cambios = {k: v for k, v in data.items() if k in original and original[k] != v}
    if not cambios:
        return jsonify({"message": "No changes detected"}), 200

    detalles_cambios = []
    for campo, nuevo_valor in cambios.items():
        valor_anterior = original.get(campo, "<not present>")
        detalles_cambios.append(f"'{campo}': '{valor_anterior}' -> '{nuevo_valor}'")
    detalle_str = "Changes: " + ", ".join(detalles_cambios)

    col_historial.update_one({"id": registro_id}, {"$set": cambios})
    registrar_auditoria(usuario_actor, "edit_history", detalle_str, usuario_afectado=str(registro_id))
    return jsonify({"message": "Record updated"}), 200

@app.route('/historial/<int:registro_id>', methods=['DELETE'])
def eliminar_historial(registro_id):
    usuario_actor = request.headers.get("usuario") or "unidentified_user"
    original = col_historial.find_one({"id": registro_id})
    if not original:
        return jsonify({"error": "Record not found"}), 404

    texto = original.get("texto_original", "<no text>")
    clasificacion = original.get("clasificacion", "<no classification>")

    result = col_historial.delete_one({"id": registro_id})
    if result.deleted_count:
        detalle = f"Deleted message '{texto}' with classification '{clasificacion}'"
        registrar_auditoria(usuario_actor, "delete_history", detalle, usuario_afectado=str(registro_id))
        return jsonify({"message": "Record deleted"}), 200

    return jsonify({"error": "Record not found"}), 404

@app.route('/usuarios', methods=['POST'])
def crear_usuario():
    data = request.get_json()
    if not all(k in data for k in ("username", "email", "password", "role")):
        return jsonify({"error": "Missing required fields."}), 400

    if col_usuarios.find_one({"username": data["username"]}):
        return jsonify({"error": "User already exists."}), 409

    hashed_password = bcrypt.hashpw(data["password"].encode("utf-8"), bcrypt.gensalt())
    usuario = {
        "username": data["username"],
        "email": data["email"],
        "password": hashed_password.decode("utf-8"),
        "role": data["role"]
    }

    col_usuarios.insert_one(usuario)
    registrar_auditoria(data["username"], "create_user", "Registered new user", usuario_afectado=data["username"])
    return jsonify({"message": "User created successfully"}), 201

@app.route('/usuarios/login', methods=['POST'])
def login_usuario():
    data = request.get_json()
    usuario = col_usuarios.find_one({"username": data.get("username")})
    if usuario and bcrypt.checkpw(data.get("password").encode("utf-8"), usuario["password"].encode("utf-8")):
        user_out = eliminar_id_mongo(usuario)
        user_out.pop("password", None)
        registrar_auditoria(data.get("username"), "login", "Successful login", usuario_afectado="-")
        return jsonify({"message": "Login successful", "user": user_out}), 200
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/usuarios/<username>', methods=['PUT'])
def actualizar_usuario(username):
    data = request.get_json()
    usuario_actor = request.headers.get("usuario") or "unidentified_user"
    original = col_usuarios.find_one({"username": username})
    if not original:
        return jsonify({"error": "User not found"}), 404

    cambios = {}
    for k, v in data.items():
        if k == "password":
            hashed = bcrypt.hashpw(v.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            if not bcrypt.checkpw(v.encode("utf-8"), original["password"].encode("utf-8")):
                cambios["password"] = hashed
        elif original.get(k) != v:
            cambios[k] = v

    if not cambios:
        return jsonify({"message": "No changes detected"}), 200

    col_usuarios.update_one({"username": username}, {"$set": cambios})
    cambios_legibles = {k: "*****" if k == "password" else v for k, v in cambios.items()}
    registrar_auditoria(usuario_actor, "update_user", f"Updated fields: {cambios_legibles}", usuario_afectado=username)
    return jsonify({"message": "User updated successfully"}), 200

@app.route('/usuarios/<username>', methods=['DELETE'])
def eliminar_usuario(username):
    usuario_actor = request.headers.get("usuario") or "unidentified_user"
    result = col_usuarios.delete_one({"username": username})
    if result.deleted_count:
        registrar_auditoria(usuario_actor, "delete_user", f"Deleted user '{username}'", usuario_afectado=username)
        return jsonify({"message": "User deleted"}), 200
    return jsonify({"error": "User not found"}), 404

@app.route('/usuarios', methods=['GET'])
def listar_usuarios():
    usuarios = eliminar_id_mongo(list(col_usuarios.find()))
    for u in usuarios:
        u.pop("password", None)
    return jsonify(usuarios), 200

@app.route('/auditoria', methods=['GET'])
def ver_auditoria():
    eventos = eliminar_id_mongo(list(col_auditoria.find().sort("fecha", -1)))
    return jsonify(eventos), 200

# ---------------- APP START ----------------

if __name__ == '__main__':
    for nombre in ["usuarios", "historial_mensajes", "auditoria"]:
        if nombre not in db.list_collection_names():
            db[nombre].insert_one({"_temp": True})
            db[nombre].delete_many({"_temp": True})

    if not col_usuarios.find_one({"username": "admin"}):
        hashed_password = bcrypt.hashpw("admin".encode("utf-8"), bcrypt.gensalt())
        col_usuarios.insert_one({
            "username": "admin",
            "email": "admin@example.com",
            "password": hashed_password.decode("utf-8"),
            "role": "admin"
        })

    app.run(debug=True, port=5000)
