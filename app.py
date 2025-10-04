from flask import Flask, request, jsonify
from joblib import load
import logging
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Cargar el modelo guardado
try:
    MODELO = load('model.pkl')
    logging.info("Modelo cargado exitosamente.")
except Exception as e:
    logging.error(f"Error al cargar el modelo: {e}")
    MODELO = None

# Dimensiones esperadas de entrada (ejemplo: 30 características para Breast Cancer)
NUM_CARACTERISTICAS = 30

@app.route('/', methods=['GET'])
def estado_servicio():
    """Ruta GET / para verificar el estado del servicio[cite: 28]."""
    if MODELO:
        return jsonify({
            "status": "OK",
            "message": "Servicio de predicción activo."
        }), 200
    else:
        return jsonify({
            "status": "Error",
            "message": "Servicio inactivo: El modelo no pudo ser cargado."
        }), 500

@app.route('/predict', methods=['POST'])
def predecir():
    """Ruta POST /predict para recibir JSON y retornar predicción[cite: 30]."""
    if not MODELO:
        return jsonify({"error": "Modelo no disponible"}), 503

    datos = request.get_json(force=True)
    logging.info(f"Datos recibidos para predicción: {datos}")

    # === Validación de entradas [cite: 31] ===
    try:
        # Los datos deben ser una lista de números para el modelo
        caracteristicas = datos['features']
    except (KeyError, TypeError):
        logging.error("JSON de entrada inválido. Se espera {'features': [valores...]} ")
        return jsonify({
            "error": "Formato JSON incorrecto",
            "expected_format": "{'features': [valor1, valor2, ..., valor30]}"
        }), 400

    if len(caracteristicas) != NUM_CARACTERISTICAS:
        logging.error(f"Número de características incorrecto. Se esperaban {NUM_CARACTERISTICAS}.")
        return jsonify({
            "error": "Número incorrecto de características",
            "expected": NUM_CARACTERISTICAS,
            "received": len(caracteristicas)
        }), 400

    # === Realizar la predicción ===
    try:
        # El modelo espera un array 2D: [[valor1, valor2, ...]]
        entrada = np.array(caracteristicas).reshape(1, -1)
        prediccion = MODELO.predict(entrada)[0]
        probabilidad = MODELO.predict_proba(entrada)[0].tolist()

        # Mapear la predicción (0 o 1) a una etiqueta (Benigno o Maligno)
        etiqueta = "Benigno" if prediccion == 1 else "Maligno"

        logging.info(f"Predicción generada: {etiqueta}")

        return jsonify({
            "prediccion": int(prediccion), # Convertir a int para JSON
            "etiqueta": etiqueta,
            "probabilidad_clases": probabilidad
        }), 200

    except Exception as e:
        # Manejo de errores generales durante la predicción [cite: 31]
        logging.exception("Error inesperado durante la predicción.")
        return jsonify({"error": "Error interno del servidor", "details": str(e)}), 500

if __name__ == '__main__':
    # Usar el puerto 5000 por defecto
    app.run(host='0.0.0.0', port=5001, debug=False)