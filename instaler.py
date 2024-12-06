import os
import subprocess
import sys

def install_dependencies():
    """Install required dependencies."""
    print("Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Todas as dependências foram instaladas com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao instalar as dependências: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if not os.path.exists("requirements.txt"):
        print("Arquivo requirements.txt não encontrado!")
        sys.exit(1)

    install_dependencies()
