# neuralLink_demo

Este é um projeto de demonstração de uma Rede Neural Artificial construída com TensorFlow e Keras, voltado à previsão de aceitação de programas por clientes com base em dados simulados.

---

## 🚀 Como executar o projeto

### 1. Clone o repositório

git clone https://github.com/mirandaxgui/neuralLink_demo.git
cd neuralLink_demo

### 2. (Opcional) Crie e ative um ambiente virtual com Python 3.11

py -3.11 -m venv venv
.
env\Scripts ctivate         # Para PowerShell no Windows
source venv/bin/activate       # Para Linux/Mac

⚠️ Se o PowerShell bloquear a ativação, execute como administrador:
Set-ExecutionPolicy RemoteSigned

### 3. Instale as dependências

pip install -r requirements.txt

### 4. Execute o script principal

python rede_neural_farm_demo_melhorado.py

---

## ✅ Resultado Esperado

- Acurácia final superior a 90%
- Gráfico de curva de aprendizado exibido via Matplotlib
- Testes manuais com dois perfis de cliente:
  - Probabilidade próxima de 1 para cliente que deve aceitar
  - Probabilidade próxima de 0 para cliente que não deve aceitar

---

## 📦 Estrutura

├── rede_neural_farm_demo_melhorado.py  # Script principal da rede neural
├── requirements.txt                    # Dependências do projeto
├── README.txt                          # Instruções e documentação

---

## 🧠 Sobre

Desenvolvido como projeto de demonstração de machine learning aplicável ao setor de vendas consultivas e análise de clientes.

Criado por @mirandaxgui
