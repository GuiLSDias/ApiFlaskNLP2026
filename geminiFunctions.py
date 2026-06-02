import os
import numpy as np
import google.generativeai as generativeai
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Modelos ──────────────────────────────────────────────────────────────────
MODELO_EMBEDDING  = 'models/gemini-embedding-001'
MODELO_GERACAO    = 'gemini-2.0-flash'   # gemini-3-flash-preview ainda não é público;
                                          # troque para 'gemini-1.5-flash' se der erro

# ── Colunas do dataset gamificacao_ensino_dataset.xlsx ───────────────────────
# ID | Categoria | Conceito | Descricao | Exemplo_Pratico | Beneficio | Nivel_Ensino | Fonte
TITLE_COL   = 'Conceito'
CONTENT_COL = 'Descricao'


def gerarBuscarConsulta(consulta: str, dataset) -> dict:
    """
    Gera embedding da consulta e retorna o registro mais similar do dataset
    via produto escalar (similaridade de cosseno aproximada).

    Retorna um dict com todos os campos do registro encontrado.
    """
    embedding_consulta = generativeai.embed_content(
        model=MODELO_EMBEDDING,
        content=consulta,
        task_type="retrieval_query",
    )

    # Produto escalar entre a consulta e todos os embeddings do dataset
    produtos_escalares = np.dot(
        np.stack(dataset["Embeddings"]),
        embedding_consulta['embedding']
    )

    indice = np.argmax(produtos_escalares)
    linha  = dataset.iloc[indice]

    return {
        'conceito':     linha['Conceito'],
        'categoria':    linha['Categoria'],
        'descricao':    linha['Descricao'],
        'exemplo':      linha['Exemplo_Pratico'],
        'beneficio':    linha['Beneficio'],
        'nivel_ensino': linha['Nivel_Ensino'],
        'fonte':        linha['Fonte'],
        'score':        float(produtos_escalares[indice]),
    }


def melhorarResposta(consulta: str, contexto: dict) -> str:
    """
    Recebe a consulta original e o dict de contexto recuperado pelo RAG,
    e usa o Gemini para gerar uma resposta natural e coerente.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Monta o contexto de forma estruturada para o modelo
    contexto_formatado = (
        f"Conceito: {contexto['conceito']}\n"
        f"Categoria: {contexto['categoria']}\n"
        f"Descrição: {contexto['descricao']}\n"
        f"Exemplo prático: {contexto['exemplo']}\n"
        f"Benefício: {contexto['beneficio']}\n"
        f"Nível de ensino: {contexto['nivel_ensino']}\n"
        f"Fonte: {contexto['fonte']}"
    )

    input_text = f"Consulta: {consulta}\n\nContexto recuperado:\n{contexto_formatado}"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=input_text)],
        ),
    ]

    config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""
Você é o GamiBot 🎮, um assistente especializado em Gamificação aplicada ao Ensino e Educação.
Você é baseado em RAG (Retrieval-Augmented Generation).

Regras:
- Utilize EXCLUSIVAMENTE o conteúdo recuperado da base de conhecimento para responder.
- Gere uma resposta clara, didática e empolgante, reescrevendo as informações de forma natural.
- Não invente informações que não estejam no contexto fornecido.
- Use emojis relacionados a jogos e educação para tornar a resposta mais dinâmica.
- Ao final, sempre cite a fonte do conteúdo (campo Fonte do contexto).
- Se a consulta for algo como "Quem é você?", apresente-se como GamiBot, especialista em gamificação no ensino.
            """),
        ],
    )

    response = client.models.generate_content(
        model=MODELO_GERACAO,
        contents=contents,
        config=config,
    )

    return response.text