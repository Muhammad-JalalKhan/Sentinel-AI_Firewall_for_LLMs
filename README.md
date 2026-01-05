# 🛡️ Sentinel-AI: Firewall for LLMs 🤖

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sentinel-AI** is a lightweight, high-speed security layer designed to protect Large Language Models (LLMs) from malicious prompt injections and jailbreak attempts.

---

## 🚀 Interactive Demo
> [!TIP]
> **View the live app here:** [Link to your Streamlit Cloud - See Step 3 below]

---

## 🧠 How it Works
This project uses a two-stage detection pipeline:
1. **Semantic Embedding**: Uses `all-MiniLM-L6-v2` to convert text into mathematical vectors.
2. **AI Classification**: An `XGBoost` model trained on security datasets classifies the input as `Safe` or `Malicious`.

## 📁 Project Structure
- `app.py`: The main Streamlit dashboard.
- `models/`: Contains the trained XGBoost model and local embeddings.
- `notebook/`: Google Colab notebooks showing the training process.
- `requirements.txt`: List of dependencies for easy setup.

## 🛠️ Installation & Setup
1. **Clone the repo**:
   ```bash
   git clone https://github.com/Muhammad-JalalKhan/Sentinel-AI_Firewall_for_LLMs.git
   ---

