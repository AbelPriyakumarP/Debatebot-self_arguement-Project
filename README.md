# 🤖 DebateBot: Self-Arguing LLM with FAISS & Streamlit

DebateBot is an AI-powered debating assistant that simulates structured debates using a Large Language Model (LLM). It generates an **argument**, a **counterargument**, and a **rebuttal** based on a given debate topic and initial idea. The app integrates **Hugging Face Transformers**, **SentenceTransformers**, **FAISS for similarity search**, and a stylish **Streamlit UI**.

---

## 📌 Features

- **Self-arguing debate generation** using DistilGPT-2
- **Semantic similarity retrieval** from a knowledge base via FAISS
- Dynamic argument generation: Argument → Counterargument → Rebuttal
- Customizable debate topics and starting ideas
- Elegant 3D animated starry background UI with custom CSS in Streamlit

---

## 📊 Tech Stack

- Python 🐍
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [SentenceTransformers](https://www.sbert.net/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- PyTorch (CUDA compatible)

---

## 📂 Project Structure

📦 DebateBot/
┣ 📜 app.py # Main Streamlit application with argumentation logic
┣ 📜 requirements.txt # (Recommended) List of Python dependencies
┗ 📜 README.md # Project documentation

yaml
Copy
Edit

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/DebateBot.git
cd DebateBot
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Dependencies (if no requirements.txt):

bash
Copy
Edit
pip install torch transformers sentence-transformers faiss-cpu streamlit
3️⃣ Run the App
bash
Copy
Edit
streamlit run app.py
🎮 How It Works
User inputs a debate topic and an initial argument idea.

FAISS retrieves the most relevant arguments from a small pre-defined knowledge base.

The model:

Generates an argument based on the topic and context.

Generates a counterargument to that argument.

Generates a rebuttal defending the original argument.

Results are displayed on an animated, styled Streamlit interface.

📖 Example Topics
Universal healthcare

Climate Change

Minimum Wage

Artificial Intelligence in Education

📊 Visual UI Preview
Feature	Description
🎨 Animated 3D Background	Starry night sky effect via custom CSS
📋 Inputs	Debate Topic & Initial Argument Idea
📊 Outputs	Retrieved Context, Argument, Counterargument, and Rebuttal

📌 Notes
Supports GPU acceleration if available.

Predefined knowledge base stored inside app.py.

Adjustable number of similar arguments retrieved via k parameter in FAISS search.

Input token limit: 512 tokens for prompt generation.

📃 License
This project is open-source under the MIT License.

🙌 Acknowledgments
Hugging Face for LLM & Sentence Transformers

Facebook AI for FAISS

Streamlit for rapid web UI development

📞 Contact
For queries or collaborations, reach out via:

Email: roshabel001@gmail.com
