# Use an official Python runtime as the base image
FROM python:3.12-slim

# Install system-level dependencies, including git
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instala PyTorch CPU-only primeiro (imagem menor, sem necessidade de GPU no container)
RUN pip install --no-cache-dir \
    torch==2.11.0+cpu \
    torchvision==0.26.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Set the working directory inside the container
WORKDIR /dgm

# Copy the entire repository into the container
COPY . .

# Inicializa repositório git (necessário para self_improve_step gerar diffs)
RUN git init && git add --all && \
    git -c user.name='dgm' -c user.email='dgm@local' commit -m 'initial'

# Install remaining Python dependencies (torch/torchvision já instalados acima)
RUN pip install --no-cache-dir -r requirements.txt

# Cria diretório de dados e pré-baixa datasets small (MNIST + Fashion-MNIST)
# Evita download em runtime a cada avaliação (~40 MB por container)
RUN mkdir -p /tmp/data && python -c "\
import torchvision; \
torchvision.datasets.MNIST('/tmp/data', download=True); \
torchvision.datasets.FashionMNIST('/tmp/data', download=True)"

# Keep the container running by default
CMD ["tail", "-f", "/dev/null"]
