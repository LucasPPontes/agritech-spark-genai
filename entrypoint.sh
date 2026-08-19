#!/bin/bash

SPARK_WORKLOAD=$1

echo "SPARK_WORKLOAD: $SPARK_WORKLOAD"

# Garante que o diretório de eventos do Spark exista
mkdir -p /opt/spark/spark-events

if [ "$SPARK_WORKLOAD" == "master" ];
then
  start-master.sh -p 7077
elif [ "$SPARK_WORKLOAD" == "worker" ];
then
  start-worker.sh spark://spark-master-dsa:7077
elif [ "$SPARK_WORKLOAD" == "history" ];
then
  start-history-server.sh
elif [ "$SPARK_WORKLOAD" == "streamlit" ];
then
  # Garante que o streamlit e plotly estejam instalados no container
  if ! python3 -c "import streamlit" &>/dev/null; then
    echo "==> Instalando pacotes do Streamlit e Plotly..."
    pip3 install streamlit plotly
  fi

  echo "==> 1/2 Executando processamento de dados via Apache Spark (main.py)..."
  spark-submit --deploy-mode client /opt/spark/jobs/main.py || true

  echo "==> 2/2 Inicializando o Dashboard Web Streamlit..."
  python3 -m streamlit run /opt/spark/jobs/app.py --server.port=8501 --server.address=0.0.0.0
fi