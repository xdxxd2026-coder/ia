#!/bin/bash
# install_neurogenesis.sh

# Atualizar sistema
sudo apt-get update
sudo apt-get upgrade -y

# Instalar dependências do sistema
sudo apt-get install -y python3-pip python3-venv git ffmpeg

# Criar ambiente virtual
python3 -m venv neurogenesis_env
source neurogenesis_env/bin/activate

# Instalar PyTorch (escolha baseado no seu sistema)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8

# Instalar dependências principais
pip3 install transformers diffusers accelerate
pip3 install opencv-python pillow numpy scipy
pip3 install TTS soundfile librosa
pip3 install ftfy regex tqdm sentencepiece protobuf

# Para GPU NVIDIA
pip3 install xformers flash-attn

# Clonar repositório (se aplicável)
git clone https://github.com/your-repo/neurogenesis-ai.git
cd neurogenesis-ai