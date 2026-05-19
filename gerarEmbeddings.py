import os
from dotenv import load_dotenv
import google.generativeai as generativeai
import pandas as pd
import numpy as np

load_dotenv()

chave_secreta = os.environ.get('GEMINI_API_KEY', '')
if not chave_secreta:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it with your Gemini API key.")

generativeai.configure(api_key=chave_secreta)

csv_url = 'https://docs.google.com/spreadsheets/d/11QU1ibjUAlNKLwLWF1s-kSpRH2UBOiVbLyl1pJIyeSk/export?format=csv&id=11QU1ibjUAlNKLwLWF1s-kSpRH2UBOiVbLyl1pJIyeSk'
df = pd.read_csv(csv_url)
print(df.head())

model = 'models/gemini-embedding-001'
def gerarEmbeddings(title, text):
  result = generativeai.embed_content(model=model,
                                content=text,
                                task_type="retrieval_document",
                                title=title)
  return result['embedding']

def gerarBuscarConsulta(consulta,dataset):
    embedding_consulta = generativeai.embed_content(model=model,
                                content=consulta,
                                task_type="retrieval_query",
                                )
    produtos_escalares = np.dot(np.stack(dataset["Embeddings"]), embedding_consulta['embedding']) # Calculo de distancia entre consulta e a base
    #maior_escalar = np.argmax(produtos_escalares)
    print(embedding_consulta)
    print(produtos_escalares)
    indice = np.argmax(produtos_escalares)
    print(produtos_escalares[indice])
    return dataset.iloc[indice]['Conteúdo']

df["Embeddings"] = df.apply(lambda row: gerarEmbeddings(row["Titulo"],row["Conteúdo"]), axis=1)
print(df)


import pickle
pickle.dump(df, open('datasetEmbeddings.pkl','wb'))