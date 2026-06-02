import os
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as generativeai

load_dotenv()

# ── Configuração da API ──────────────────────────────────────────────────────
chave_secreta = os.environ.get('GEMINI_API_KEY', '')
if not chave_secreta:
    raise ValueError(
        "GEMINI_API_KEY não encontrada. Crie um arquivo .env com:\n"
        "GEMINI_API_KEY=sua_chave_aqui"
    )

generativeai.configure(api_key=chave_secreta)

# ── Carregar planilha ────────────────────────────────────────────────────────
# Opção 1: arquivo local (recomendado para uso com este projeto)
XLSX_LOCAL = 'gamificacao_ensino_dataset.xlsx'

# Opção 2: Google Sheets público exportado como CSV (substitua o ID se necessário)
SHEETS_URL = (
    'https://docs.google.com/spreadsheets/d/1Fyj-xUPnszOyI806-Qa6j-iusc9sBbja/edit?gid=2099250555#gid=2099250555'
    
)

df = None

# Tenta carregar o .xlsx local primeiro
if os.path.exists(XLSX_LOCAL):
    df = pd.read_excel(XLSX_LOCAL)
    print(f"✅ Planilha carregada localmente: {XLSX_LOCAL}")
else:
    # Fallback: tenta baixar do Google Sheets
    try:
        df = pd.read_csv(SHEETS_URL)
        print("✅ Planilha carregada via Google Sheets")
    except Exception as e:
        raise FileNotFoundError(
            f"Não foi possível carregar a planilha.\n"
            f"Coloque o arquivo '{XLSX_LOCAL}' na mesma pasta deste script.\n"
            f"Erro: {e}"
        )

print(f"\nShape: {df.shape}")
print("Colunas:", df.columns.tolist())
print(df.head(3).to_string())

# ── Colunas do dataset gamificacao_ensino_dataset ────────────────────────────
# ID | Categoria | Conceito | Descricao | Exemplo_Pratico | Beneficio | Nivel_Ensino | Fonte
#
# title_col   → 'Conceito'   (nome/título de cada registro)
# content_col → texto rico composto por múltiplas colunas (melhor para RAG)

TITLE_COL   = 'Conceito'
CONTENT_COL = 'Descricao'   # coluna principal de busca

def montar_texto_completo(row):
    """Combina todas as colunas relevantes em um único texto para embedding."""
    return (
        f"Categoria: {row['Categoria']}. "
        f"Conceito: {row['Conceito']}. "
        f"Descrição: {row['Descricao']}. "
        f"Exemplo prático: {row['Exemplo_Pratico']}. "
        f"Benefício: {row['Beneficio']}. "
        f"Nível de ensino: {row['Nivel_Ensino']}. "
        f"Fonte: {row['Fonte']}."
    )

# Cria coluna de texto completo (usada para o embedding)
df['Texto_Completo'] = df.apply(montar_texto_completo, axis=1)

print("\n--- Exemplo de texto que será embedado ---")
print(df['Texto_Completo'].iloc[0])

# ── Funções de embedding ─────────────────────────────────────────────────────
MODELO_EMBEDDING = 'models/gemini-embedding-001'

def gerar_embedding(title: str, text: str) -> list:
    """Gera embedding de um documento usando task_type retrieval_document."""
    resultado = generativeai.embed_content(
        model=MODELO_EMBEDDING,
        content=str(text),
        task_type="retrieval_document",
        title=str(title),
    )
    return resultado['embedding']

def buscar_por_consulta(consulta: str, dataset: pd.DataFrame) -> dict:
    """
    Gera embedding da consulta e retorna o documento mais similar
    via produto escalar (similaridade de cosseno aproximada).
    """
    embedding_consulta = generativeai.embed_content(
        model=MODELO_EMBEDDING,
        content=consulta,
        task_type="retrieval_query",
    )

    # Produto escalar entre a consulta e todos os embeddings do dataset
    scores = np.dot(
        np.stack(dataset['Embeddings']),
        embedding_consulta['embedding']
    )

    indice = np.argmax(scores)
    resultado = dataset.iloc[indice]

    print(f"\n🔍 Consulta: '{consulta}'")
    print(f"📊 Score mais alto: {scores[indice]:.4f}")
    print(f"✅ Conceito encontrado: {resultado[TITLE_COL]}")

    return {
        'conceito':      resultado[TITLE_COL],
        'categoria':     resultado['Categoria'],
        'descricao':     resultado[CONTENT_COL],
        'exemplo':       resultado['Exemplo_Pratico'],
        'beneficio':     resultado['Beneficio'],
        'nivel_ensino':  resultado['Nivel_Ensino'],
        'fonte':         resultado['Fonte'],
        'score':         float(scores[indice]),
    }

# ── Gerar embeddings para todo o dataset ─────────────────────────────────────
print(f"\n⏳ Gerando embeddings para {len(df)} registros...")
print("   (Isso pode levar alguns segundos devido ao rate limit da API)\n")

try:
    df['Embeddings'] = df.apply(
        lambda row: gerar_embedding(row[TITLE_COL], row['Texto_Completo']),
        axis=1
    )
    print(f"✅ Embeddings gerados com sucesso para {len(df)} registros!")
except Exception as e:
    print(f"❌ Erro ao gerar embeddings: {e}")
    raise

# ── Salvar dataset com embeddings ────────────────────────────────────────────
PICKLE_PATH = 'datasetEmbeddings.pkl'
pickle.dump(df, open(PICKLE_PATH, 'wb'))
print(f"\n💾 Dataset com embeddings salvo em: {PICKLE_PATH}")

# ── Teste de busca ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TESTE DE BUSCA SEMÂNTICA")
print("="*60)

consultas_teste = [
    "Como usar badges para motivar alunos?",
    "Quais ferramentas digitais posso usar em sala de aula?",
    "Teoria do Flow no contexto educacional",
]

for consulta in consultas_teste:
    resultado = buscar_por_consulta(consulta, df)
    print(f"   → {resultado['descricao'][:120]}...")
    print(f"   → Fonte: {resultado['fonte']}\n")

print("\n✅ Script finalizado! Use 'datasetEmbeddings.pkl' no seu chatbot RAG.")