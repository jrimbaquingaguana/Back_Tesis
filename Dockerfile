FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# ✅ Copiar requirements.txt separado para aprovechar cache
COPY requirements.txt .

# ✅ Instalar dependencias solo si requirements.txt cambia
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m nltk.downloader punkt

# ✅ Luego copiar el resto del código
COPY modelos /app/modelos
COPY app.py /app/

EXPOSE 5000

CMD ["python", "app.py"]
