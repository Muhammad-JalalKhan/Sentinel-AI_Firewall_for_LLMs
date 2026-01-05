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

### 2. Add Visuals (The "Wow" Factor)
A reader will spend only 30 seconds on your page. Visuals make them stay.

*   **Take a Screenshot:** Run your Streamlit app locally, take a screenshot of the dashboard.
*   **Make a GIF:** Use a free tool like **[ScreenToGif](https://www.screentogif.com/)** or **[GIPHY Capture](https://giphy.com/apps/giphycapture)**. Record yourself typing a "Malicious Prompt" and showing how the AI blocks it.
*   **How to add it:** Upload the image/GIF to your GitHub repository and add this line to your README:
    `![Demo](demo.gif)`

---

### 3. Make the App "Live" (Streamlit Cloud)
The best way to make it interactive is to let the reader **use it** without downloading anything.

1.  Go to **[share.streamlit.io](https://share.streamlit.io/)**.
2.  Connect your GitHub account.
3.  Select your `Sentinel-AI_Firewall_for_LLMs` repository.
4.  Click **Deploy**.
5.  **Copy the URL** they give you and paste it into the "Interactive Demo" section of your README.

---

### 4. Use GitHub "Social Preview"
Give your project a thumbnail.
1.  On your GitHub repo page, go to **Settings**.
2.  Under **General**, look for **Social Preview**.
3.  Upload a cool image (use Canva to make a "Sentinel-AI" logo).
*   **Benefit:** When you share the link on LinkedIn or WhatsApp, it will show a professional image instead of a plain link.

---

### 5. Final Push Commands
Once you have updated your README and added your images, push them:

```bash
git add .
git commit -m "Enhance README with badges and instructions"
git push origin main