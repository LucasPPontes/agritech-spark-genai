# Modelagem do Crescimento de Agriculturas em Diferentes Condições Usando PySpark e LLM

# Imports
import os
import json
import urllib.request
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, sum, col
from pyspark.sql import functions as F

# ==============================================================================
# CONFIGURAÇÃO DO PROVEDOR DE LLM (Google Gemini Gratuito vs OpenAI)
# ==============================================================================
# Escolha o provedor: "gemini" (Gratuito via Google AI Studio) ou "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Insira sua chave de API abaixo ou utilize as variáveis de ambiente:
# GEMINI_API_KEY -> Obtenha gratuitamente em: https://aistudio.google.com/
# OPENAI_API_KEY -> Chave paga da OpenAI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "coloque-aqui-sua-chave-gemini-gratuita")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "coloque-aqui-sua-chave-openai")

if LLM_PROVIDER == "gemini":
    print("-> Provedor LLM ativado: Google Gemini (Gratuito via Requisição HTTP Nativa)")
elif LLM_PROVIDER == "openai":
    print("-> Provedor LLM ativado: OpenAI (gpt-4o via Requisição HTTP Nativa)")
else:
    raise ValueError(f"Provedor LLM inválido: '{LLM_PROVIDER}'. Escolha 'gemini' ou 'openai'.")

# Inicializando sessão Spark
spark = SparkSession.builder.appName("Main").getOrCreate()

# Caminho do arquivo CSV
csv_file_path = "/opt/spark/dados/dataset.csv"

# Função para carregar os dados 
def dsa_carrega_dados(spark, file_path):
    dsa_dados = spark.read.csv(file_path, header = True)
    print(f"\nTotal de registros carregados: {dsa_dados.count()}\n")
    return dsa_dados

# Carrega os dados
df_spark = dsa_carrega_dados(spark, csv_file_path)

# Crescimento total acumulado por cultura
df_spark.groupBy("cultura").agg(sum("crescimento").alias("crescimento_acumulado")).show()

# Média de crescimento por tipo de solo
df_spark.groupBy("solo").agg(avg("crescimento").alias("media_crescimento")).show()

# Função para coletar estatísticas e preparar o prompt para o mês de julho
def dsa_prepara_dados_prompt(df):
    
    # Extrai o mês da coluna de data e filtra para o mês de julho
    df_julho = df.withColumn("mes", F.month("data")).filter(F.col("mes") == 7)
    
    # Calcula a média de temperatura, umidade e crescimento por cultura e tipo de solo para julho
    stats_df = df_julho.groupBy("cultura", "solo").agg(
        F.avg("temperatura").alias("media_temperatura"),
        F.avg("umidade").alias("media_umidade"),
        F.avg("crescimento").alias("media_crescimento")
    ).collect()
    
    # Formatando os dados coletados em uma string para o prompt
    dados_formatados = "Estatísticas de crescimento para o mês de Julho:\n"
    for row in stats_df:
        dados_formatados += (f"Cultura: {row['cultura']}, Solo: {row['solo']}, "
                             f"Média de Temperatura: {row['media_temperatura']:.2f}°C, "
                             f"Média de Umidade: {row['media_umidade']:.2f}%, "
                             f"Média de Crescimento: {row['media_crescimento']:.2f} cm/dia\n")
    
    return dados_formatados

# Extraindo e formatando dados para o prompt
dados_para_prompt = dsa_prepara_dados_prompt(df_spark)

# Requisição HTTP nativa ao Gemini para evitar conflitos de bibliotecas (httpx/gRPC)
def _chama_gemini(prompt, api_key):
    modelos = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
    ultimo_erro = None
    
    for modelo in modelos:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2}
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                texto = res_data["candidates"][0]["content"]["parts"][0]["text"]
                print(f"-> Resposta gerada com sucesso usando o modelo: {modelo}")
                return texto
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            ultimo_erro = f"HTTP {e.code}: {err_body}"
            continue
        except Exception as e:
            ultimo_erro = str(e)
            continue
            
    raise RuntimeError(f"Erro ao comunicar com a API do Gemini. Verifique se a sua chave GEMINI_API_KEY é válida.\nDetalhes: {ultimo_erro}")

# Requisição HTTP nativa à OpenAI
def _chama_openai(prompt, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8')
        raise RuntimeError(f"Erro ao comunicar com a API da OpenAI.\nHTTP {e.code}: {err_body}")
    except Exception as e:
        raise RuntimeError(f"Erro ao comunicar com a API da OpenAI: {e}")

# Função para gerar uma pergunta sobre os dados e obter insights do LLM
def dsa_analisa_dados_com_llm(texto):

    # Construindo o prompt com dados formatados e a pergunta
    prompt = f"""
    Você está analisando um conjunto de dados sobre o crescimento de diferentes culturas agrícolas (milho, soja, trigo).
    Cada registro inclui variáveis como temperatura, umidade, tipo de solo e crescimento diário.
    
    Dados resumidos para análise:
    {dados_para_prompt}
    
    Baseado nesses dados, responda a pergunta abaixo fornecendo sugestões:
    {texto}
    """
    
    if LLM_PROVIDER == "gemini":
        return _chama_gemini(prompt, GEMINI_API_KEY)
    else:
        return _chama_openai(prompt, OPENAI_API_KEY)

# Pergunta sobre os dados
pergunta = "Quais condições de temperatura e umidade são ideais para o crescimento do milho?"

# Obtém a resposta a partir do LLM
resposta = dsa_analisa_dados_com_llm(pergunta)
print("\nResposta do LLM:", resposta)

# Fim
