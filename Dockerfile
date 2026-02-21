# Imagem Oficial do Python 3.11 limpa (slim-buster ou slim-bookworm para ocupar menos RAM)
FROM python:3.11-slim

# Impede o Python de gerar arquivos pyc .pyc e de manter o print em buffer (melhor para logs de nuvem)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala ferramentas basicas de compilacao de C++ caso alguma biblioteca de Reranker exija dlib/numpy build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Configura o diretorio base dentro do container Docker
WORKDIR /app

# Copia os arquivos de configuração de pacotes primeiro (Técnica de Cache do Docker)
COPY requirements.txt .

# Instala as dependencias pip (O flag --no-cache-dir deixa a maquina virtual mais leve)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos do projeto para o Container
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Comando default do Docker! (Se você subir na nuvem sem especificar, ele roda o Streamlit por padrao)
# OBS: O RENDER, KOYEB ou GCP vai pedir o seu "Start Command". 
# - Para a interface Web (Streamlit), o comando será: streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
# - Para a interface do Telegram, o comando será: python telegram_bot.py
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
