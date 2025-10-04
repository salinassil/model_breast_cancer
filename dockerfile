# Usa una imagen base ligera de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias e instalarlas
# Esto es una capa separada, lo que mejora la eficiencia del caché de Docker [cite: 42]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el modelo y el código de la aplicación
COPY model.pkl .
COPY app.py .

# Exponer el puerto en el que corre Flask
EXPOSE 5001

# Comando para ejecutar la aplicación
# Usa Gunicorn o un servidor de producción real en producción, pero Flask es suficiente para la evaluación local
CMD ["python", "app.py"]