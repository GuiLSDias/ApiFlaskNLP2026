from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as generativeai
import pickle
import os

from geminiFunctions import gerarBuscarConsulta, melhorarResposta

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Configuração ─────────────────────────────────────────────────────────────
chave_secreta = os.getenv('GEMINI_API_KEY')
if not chave_secreta:
    raise ValueError("GEMINI_API_KEY não encontrada no .env")

generativeai.configure(api_key=chave_secreta)

# Carrega dataset com embeddings gerado pelo gerar_embeddings.py
modeloEmbeddings = pickle.load(open('datasetEmbeddings.pkl', 'rb'))
print(f"✅ Dataset carregado: {len(modeloEmbeddings)} registros")


# ── Rota de teste ─────────────────────────────────────────────────────────────
@app.route("/")
def home():
    consulta  = "Quem é você?"
    contexto  = gerarBuscarConsulta(consulta, modeloEmbeddings)
    resposta  = melhorarResposta(consulta, contexto)
    return resposta


# ── Rota principal da API ─────────────────────────────────────────────────────
@app.route("/api", methods=["POST"])
def results():
    # ── Verificação de autorização ──
    auth_header = request.headers.get("Authorization", "").strip()
    auth_key    = auth_header.removeprefix("Bearer ").strip()

    if auth_key != chave_secreta:
        return jsonify({"error": "Unauthorized"}), 401

    # ── Leitura do corpo da requisição ──
    data = request.get_json(force=True)

    if not data or "consulta" not in data:
        return jsonify({"error": "Campo 'consulta' é obrigatório no corpo JSON"}), 400

    consulta = data["consulta"].strip()
    if not consulta:
        return jsonify({"error": "A consulta não pode ser vazia"}), 400

    # ── Pipeline RAG ──
    contexto = gerarBuscarConsulta(consulta, modeloEmbeddings)
    resposta = melhorarResposta(consulta, contexto)

    return jsonify({
        "mensagem": resposta,
        # metadados opcionais — úteis para debug e para o frontend exibir a fonte
        "contexto": {
            "conceito":     contexto["conceito"],
            "categoria":    contexto["categoria"],
            "nivel_ensino": contexto["nivel_ensino"],
            "fonte":        contexto["fonte"],
            "score":        round(contexto["score"], 4),
        }
    })


# ── Rota de saúde ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "registros": len(modeloEmbeddings),
        "colunas":   list(modeloEmbeddings.columns),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)