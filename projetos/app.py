# ==============================================================================
# AgriTech Analytics & GenAI Dashboard (Streamlit + PySpark + Gemini/OpenAI)
# ==============================================================================

import os
import json
import urllib.request
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# Configuração da Página
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriTech GenAI & Spark Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Customizada
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2E7D32, #4CAF50, #81C784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Carregamento e Tratamento dos Dados
# ------------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    caminhos = [
        "/opt/spark/dados/dataset.csv",
        "datasets/dataset.csv",
        "../datasets/dataset.csv"
    ]
    path_encontrado = None
    for p in caminhos:
        if os.path.exists(p):
            path_encontrado = p
            break
            
    if not path_encontrado:
        st.error("❌ Arquivo 'dataset.csv' não encontrado nos caminhos mapeados.")
        return pd.DataFrame()
        
    df = pd.read_csv(path_encontrado)
    df['data'] = pd.to_datetime(df['data'])
    df['mes'] = df['data'].dt.month
    df['mes_nome'] = df['data'].dt.strftime('%B')
    return df

df_raw = carregar_dados()

# ------------------------------------------------------------------------------
# Barra Lateral (Sidebar) - Filtros e Configurações
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/wheat.png", width=70)
st.sidebar.title("⚙️ Painel de Controle")

# 1. Configuração do LLM
st.sidebar.markdown("### 🤖 Provedor de LLM")
provedor_default = os.getenv("LLM_PROVIDER", "gemini").lower()
provedor_idx = 0 if provedor_default == "gemini" else 1

llm_provider = st.sidebar.radio(
    "Selecione a Inteligência Artificial:",
    options=["Google Gemini (Gratuito)", "OpenAI (GPT-4o)"],
    index=provedor_idx
)

is_gemini = "Gemini" in llm_provider

if is_gemini:
    gemini_key_env = os.getenv("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input(
        "Chave API do Gemini (Google AI Studio):",
        value=gemini_key_env,
        type="password",
        help="Obtenha uma chave gratuita em: https://aistudio.google.com/"
    )
else:
    openai_key_env = os.getenv("OPENAI_API_KEY", "")
    api_key = st.sidebar.text_input(
        "Chave API da OpenAI:",
        value=openai_key_env,
        type="password"
    )

st.sidebar.divider()

# 2. Filtros de Dados
st.sidebar.markdown("### 🔍 Filtros de Análise")

culturas_disponiveis = sorted(df_raw['cultura'].unique().tolist()) if not df_raw.empty else []
solos_disponiveis = sorted(df_raw['solo'].unique().tolist()) if not df_raw.empty else []
meses_map = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

sel_culturas = st.sidebar.multiselect("Culturas Agrícolas:", culturas_disponiveis, default=culturas_disponiveis)
sel_solos = st.sidebar.multiselect("Tipos de Solo:", solos_disponiveis, default=solos_disponiveis)

opcoes_meses = ["Todos os Meses"] + [meses_map[m] for m in sorted(meses_map.keys())]
sel_mes_nome = st.sidebar.selectbox("Mês de Referência:", opcoes_meses, index=7) # Julho como padrão

# Aplicação dos Filtros
df_filtrado = df_raw.copy()
if not df_filtrado.empty:
    if sel_culturas:
        df_filtrado = df_filtrado[df_filtrado['cultura'].isin(sel_culturas)]
    if sel_solos:
        df_filtrado = df_filtrado[df_filtrado['solo'].isin(sel_solos)]
    if sel_mes_nome != "Todos os Meses":
        mes_num = [k for k, v in meses_map.items() if v == sel_mes_nome][0]
        df_filtrado = df_filtrado[df_filtrado['mes'] == mes_num]

# ------------------------------------------------------------------------------
# Cabeçalho Principal
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">🌾 AgriTech GenAI & Spark Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Plataforma Interativa de Análise Agronômica com Apache Spark e Inteligência Artificial Generativa</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Funções de Integração LLM (HTTP Nativo)
# ------------------------------------------------------------------------------
def chamar_gemini(prompt, key):
    if not key or key == "coloque-aqui-sua-chave-gemini-gratuita":
        return "⚠️ Por favor, insira uma chave de API válida do Gemini no menu lateral para gerar o insight."
        
    modelos = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
    ultimo_erro = None
    
    for modelo in modelos:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2}
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            ultimo_erro = f"HTTP {e.code}: {err_body}"
            continue
        except Exception as e:
            ultimo_erro = str(e)
            continue
            
    return f"❌ Erro ao comunicar com a API do Gemini. Detalhes: {ultimo_erro}"

def chamar_openai(prompt, key):
    if not key or key == "coloque-aqui-sua-chave-openai":
        return "⚠️ Por favor, insira uma chave de API válida da OpenAI no menu lateral para gerar o insight."
        
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
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
        return f"❌ Erro HTTP {e.code} na OpenAI: {err_body}"
    except Exception as e:
        return f"❌ Erro ao comunicar com a OpenAI: {e}"

# ------------------------------------------------------------------------------
# Tabs da Aplicação
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard & Métricas", 
    "🤖 GenAI Insights & Chat", 
    "⚡ Status Spark & Dados"
])

# ------------------------------------------------------------------------------
# TAB 1: Dashboard
# ------------------------------------------------------------------------------
with tab1:
    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
    else:
        # Key Performance Indicators (KPIs)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Registros", f"{len(df_filtrado):,}")
        with col2:
            st.metric("Média de Temperatura", f"{df_filtrado['temperatura'].mean():.2f} °C")
        with col3:
            st.metric("Média de Umidade", f"{df_filtrado['umidade'].mean():.2f} %")
        with col4:
            st.metric("Média de Crescimento", f"{df_filtrado['crescimento'].mean():.2f} cm/dia")

        st.divider()

        # Gráficos em Colunas
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("🌱 Crescimento Acumulado por Cultura")
            df_cultura = df_filtrado.groupby('cultura', as_index=False)['crescimento'].sum().rename(columns={'crescimento': 'crescimento_acumulado'})
            fig_cultura = px.bar(
                df_cultura, 
                x='cultura', 
                y='crescimento_acumulado',
                color='cultura',
                labels={'cultura': 'Cultura Agrícola', 'crescimento_acumulado': 'Crescimento Acumulado (cm)'},
                color_discrete_sequence=px.colors.qualitative.Set2,
                text_auto='.2f'
            )
            fig_cultura.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_cultura, use_container_width=True)

        with c2:
            st.subheader("🏜️ Média de Crescimento por Tipo de Solo")
            df_solo = df_filtrado.groupby('solo', as_index=False)['crescimento'].mean().rename(columns={'crescimento': 'media_crescimento'})
            fig_solo = px.pie(
                df_solo, 
                names='solo', 
                values='media_crescimento',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_solo.update_layout(height=380)
            st.plotly_chart(fig_solo, use_container_width=True)

        # Gráfico Scatter Plot
        st.subheader("🌡️ Relação: Temperatura x Umidade x Taxa de Crescimento")
        fig_scatter = px.scatter(
            df_filtrado,
            x='temperatura',
            y='umidade',
            size='crescimento',
            color='cultura',
            hover_data=['solo', 'crescimento'],
            labels={'temperatura': 'Temperatura (°C)', 'umidade': 'Umidade (%)', 'crescimento': 'Crescimento (cm/dia)'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_scatter.update_layout(height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: GenAI Playground
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🤖 Assistente Agronômico Inteligente")
    st.write("Obtenha diagnósticos e recomendações técnicas baseadas nos dados filtrados do Apache Spark.")
    
    # Preparação do contexto estático com os dados atuais
    if not df_filtrado.empty:
        stats_group = df_filtrado.groupby(['cultura', 'solo'], as_index=False).agg({
            'temperatura': 'mean',
            'umidade': 'mean',
            'crescimento': 'mean'
        })
        
        contexto_dados = f"Estatísticas de crescimento para o filtro ({sel_mes_nome}):\n"
        for _, row in stats_group.iterrows():
            contexto_dados += (f"Cultura: {row['cultura']}, Solo: {row['solo']}, "
                               f"Temp Média: {row['temperatura']:.2f}°C, "
                               f"Umidade Média: {row['umidade']:.2f}%, "
                               f"Crescimento Médio: {row['crescimento']:.2f} cm/dia\n")
    else:
        contexto_dados = "Sem dados disponíveis."

    # Expander com o contexto enviado à IA
    with st.expander("📄 Visualizar Dados Agregados Enviados à IA (Prompt Context)", expanded=False):
        st.code(contexto_dados, language="text")

    st.divider()

    # Perguntas Rápidas (Quick Questions)
    st.markdown("##### 💡 Perguntas Sugeridas (Clique para Selecionar):")
    q1 = "Quais condições de temperatura e umidade são ideais para o crescimento do milho?"
    q2 = "Qual é o melhor tipo de solo para maximizar o crescimento da soja?"
    q3 = "Como a variação de umidade afeta a taxa de crescimento agrícola segundo os dados?"
    
    col_q1, col_q2, col_q3 = st.columns(3)
    
    pergunta_selecionada = ""
    if col_q1.button("🌽 Condições para o Milho"):
        pergunta_selecionada = q1
    if col_q2.button("🌱 Melhor Solo para Soja"):
        pergunta_selecionada = q2
    if col_q3.button("💧 Impacto da Umidade"):
        pergunta_selecionada = q3

    # Campo de Texto para Pergunta Customizada
    pergunta_usuario = st.text_area(
        "Faça sua pergunta sobre as culturas ou condições agronômicas:",
        value=pergunta_selecionada if pergunta_selecionada else q1,
        height=100
    )

    if st.button("🚀 Gerar Insights com IA", type="primary", use_container_width=True):
        if not api_key:
            st.warning("⚠️ Insira sua Chave de API no menu lateral esquerdo antes de prosseguir.")
        else:
            with st.spinner("Analisando dados do Spark e consultando modelo de linguagem..."):
                prompt_completo = f"""
                Você é um especialista em agronomia e ciência de dados agrícolas.
                Você está analisando o seguinte resumo de dados sobre o cultivo de grãos:
                
                {contexto_dados}
                
                Com base estritamente nesses dados e no seu conhecimento técnico, responda à pergunta abaixo com recomendações práticas:
                {pergunta_usuario}
                """
                
                if is_gemini:
                    resposta_llm = chamar_gemini(prompt_completo, api_key)
                else:
                    resposta_llm = chamar_openai(prompt_completo, api_key)
                
                st.markdown("### 📝 Diagnóstico e Recomendações da IA:")
                st.info(resposta_llm)

# ------------------------------------------------------------------------------
# TAB 3: Status Spark & Dados Brutos
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("⚡ Status do Cluster Apache Spark & Dados Brutos")
    
    c_spark1, c_spark2 = st.columns(2)
    with c_spark1:
        st.markdown("""
        **Painel Mestre do Spark (Master UI)**  
        Monitore a execução dos workers e jobs distribuídos.  
        🔗 [Acessar Spark Master (Porta 9090)](http://localhost:9090)
        """)
    with c_spark2:
        st.markdown("""
        **Servidor de Histórico do Spark (History Server)**  
        Consulte métricas detalhadas e histórico de tarefas passadas.  
        🔗 [Acessar Spark History (Porta 18080)](http://localhost:18080)
        """)

    st.divider()

    st.subheader("📋 Tabela de Dados (Visualização Filtrada)")
    st.dataframe(df_filtrado, use_container_width=True, height=400)
    
    # Download dos Dados Filtrados
    csv_bytes = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar Dados Filtrados (CSV)",
        data=csv_bytes,
        file_name="dados_agricolas_filtrados.csv",
        mime="text/csv",
    )
