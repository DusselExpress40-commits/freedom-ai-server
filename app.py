
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = "openai/gpt-oss-120b"

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "FREEDOM AI",
        "status": "online",
        "version": "1.0"
    })

@app.route("/chat", methods=["POST"])
def chat():
    if not HF_TOKEN:
        return jsonify({"error": "KI-Server nicht konfiguriert"}), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Nachricht fehlt"}), 400

    messages = [
        {
            "role": "system",
            "content": (
                "Du bist FREEDOM AI, ein hilfreicher, "
                "intelligenter und freundlicher KI-Assistent. "
                "Du antwortest klar, ehrlich und auf Deutsch, "
                "wenn der Nutzer Deutsch spricht."
            )
        }
    ]

    for item in history[-20:]:
        if item.get("role") in ("user", "assistant"):
            messages.append({
                "role": item["role"],
                "content": str(item.get("content", ""))
            })

    messages.append({"role": "user", "content": message})

    try:
        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 1000
            },
            timeout=90
        )

        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"]

        return jsonify({"reply": reply})

    except requests.RequestException as error:
        return jsonify({
            "error": "KI-Verbindung fehlgeschlagen",
            "details": str(error)
        }), 502

    except (KeyError, IndexError, TypeError):
        return jsonify({"error": "Ungültige Antwort vom KI-Modell"}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
