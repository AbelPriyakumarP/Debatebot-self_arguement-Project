DebateBot: Self-Arguing LLM
Overview
DebateBot is an interactive web application built with Streamlit that leverages a transformer-based Large Language Model (LLM) to simulate debates. It generates an argument, counterargument, and rebuttal for a user-provided topic and initial argument idea. The application uses Retrieval-Augmented Generation (RAG) with a FAISS vector database to retrieve relevant context from a predefined knowledge base, enhancing the quality of generated responses. The UI features a visually appealing 3D starry sky background, making the experience engaging.
The project is designed for users interested in exploring automated debate generation, suitable for educational purposes, research, or casual experimentation. It uses distilgpt2 as the default LLM for lightweight performance, with options to upgrade to more powerful models like Mistral-7B for better results.

![image alt](https://github.com/AbelPriyakumarP/Debatebot-self_arguement-Project/blob/0addc5a47d71c10387c96f69eb11493c6359159a/Screenshot%202025-05-24%20212547.png)


Project Architecture
The architecture of DebateBot is modular, combining data retrieval, text generation, and a user-friendly interface. Below is an overview of the components:

Knowledge Base:

A predefined list of debate topics with pro and con arguments (e.g., Universal Healthcare, Climate Change, Artificial Intelligence in Education).
Stored as text strings in the format: Topic: <topic>\n<Pro/Con>: <argument>.
Embedded using sentence-transformers (all-MiniLM-L6-v2) and indexed in a FAISS vector database for efficient retrieval.


Retrieval-Augmented Generation (RAG):

Embedding: The SentenceTransformer model converts arguments into dense vectors.
Retrieval: FAISS retrieves the top k relevant arguments based on cosine similarity to the user’s input query and topic.
Context Integration: Retrieved arguments are included in prompts to guide the LLM’s output.


Text Generation:

LLM: Uses distilgpt2 from Hugging Face’s Transformers library for generating arguments, counterarguments, and rebuttals.
Prompt Engineering: Structured prompts include the topic, retrieved context, and initial argument idea to ensure coherent and relevant outputs.
Self-Argumentation: The self_argue function generates:
Argument: Based on the user’s initial idea and context.
Counterargument: Opposes the generated argument using retrieved counter-context.
Rebuttal: Defends the original argument against the counterargument.


Token Management: Uses max_new_tokens to control output length, avoiding input length errors.


User Interface:

Built with Streamlit for a web-based interface.
Features input fields for topic and initial argument idea, a “Generate Debate” button, and sections to display context, argument, counterargument, and rebuttal.
Includes a 3D starry sky background using CSS animations for a visually appealing experience.


Error Handling:

Robust try-except blocks catch issues like token length errors, model loading failures, and import issues.
User-friendly error messages guide users to correct inputs (e.g., shorten prompts if too long).



File Structure

app.py: The main Streamlit application file, handling the UI and user interaction.
debatebot_utils.py: Contains the core logic, including model initialization, knowledge base, and functions for retrieval (retrieve_context), text generation (generate_text), and debate simulation (self_argue).
README.md: This documentation file.

Requirements

Python: 3.8 or higher
Dependencies:pip install transformers==4.44.2 sentence-transformers==3.1.1 torch==2.4.1 tf-keras==2.17.0 streamlit==1.39.0 faiss-cpu==1.9.0


Optional: For GPU support, install the CUDA version of PyTorch:pip install torch==2.4.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html


Hardware: CPU is sufficient, but a GPU (e.g., NVIDIA RTX 3090) is recommended for faster inference.

Setup Instructions

Create a Project Directory:
mkdir debatebot_project
cd debatebot_project


Set Up Anaconda Environment (optional but recommended):
conda create -n debatebot python=3.9
conda activate debatebot


Install Dependencies:
pip install transformers==4.44.2 sentence-transformers==3.1.1 torch==2.4.1 tf-keras==2.17.0 streamlit==1.39.0 faiss-cpu==1.9.0


Save the Code Files:

Save app.py and debatebot_utils.py (provided in previous responses) in the project directory.
Ensure both files are correctly configured to work together (i.e., app.py imports self_argue from debatebot_utils).


Run the Application:
streamlit run app.py


Access the App:

Open your browser and navigate to http://localhost:8501.
If running on a different machine, use the provided network URL.



Usage

Open the App:

Launch the app via streamlit run app.py.
The UI will display a dark theme with a 3D starry sky background.


Enter Inputs:

Debate Topic: Enter a topic (e.g., “Universal healthcare”, “Artificial Intelligence in Education”).
Initial Argument Idea: Provide a concise argument idea (e.g., “Ensures equal access to medical care”). Keep it short to avoid token length errors.


Generate Debate:

Click the “Generate Debate” button.
The app will display:
Retrieved Context: Relevant arguments from the knowledge base.
Argument: The LLM’s generated argument based on the input.
Counterargument: An opposing viewpoint.
Rebuttal: A defense of the original argument.




Troubleshooting:

Token Length Error: If you see an error about input length, shorten the initial argument idea to 1-2 sentences.
Import Errors: Ensure all dependencies are installed (see Requirements). Run pip show transformers to verify version 4.44.2.
Performance: If generation is slow, consider using a GPU or switching to a larger model (update model_name in debatebot_utils.py).



Example

Topic: Artificial Intelligence in Education
Initial Argument Idea: Enhances personalized learning through adaptive technologies.
Output (approximate, depends on model):
Context: Pro: Enhances personalized learning through adaptive technologies. Con: May reduce human interaction and critical thinking opportunities.
Argument: AI in education tailors learning experiences to individual student needs, improving engagement and outcomes.
Counterargument: Overreliance on AI may limit teacher-student interactions, hindering social and critical thinking skills.
Rebuttal: AI complements human teaching by automating routine tasks, allowing teachers to focus on fostering critical thinking and interaction.



Limitations

Model Quality: distilgpt2 is lightweight but may produce less coherent arguments compared to larger models like Mistral-7B.
Knowledge Base: The current knowledge base is small. Expand it with more topics and arguments for better coverage.
Token Limits: Long prompts may trigger errors. Keep inputs concise or increase max_new_tokens in debatebot_utils.py.
Performance: CPU inference is slower; a GPU is recommended for production use.

Future Improvements

Expand Knowledge Base: Scrape real debate data (e.g., from Reddit or debate.org) to enrich the context.
Model Upgrade: Use a more powerful LLM (e.g., mistralai/Mixtral-8x7B-Instruct-v0.1) for improved argument quality.
Evaluation: Implement an LLM-as-a-judge to score arguments for coherence and relevance.
Deployment: Containerize with Docker and deploy on a cloud platform for scalability.

Troubleshooting

ImportError for transformers:pip uninstall transformers -y
pip install transformers==4.44.2


Token Length Error:Shorten the initial argument idea or increase max_new_tokens in debatebot_utils.py.
Slow Performance:Use a GPU or reduce max_new_tokens to 100.

License
This project is licensed under the MIT License.
Contact
For issues or suggestions, please open an issue on the project repository or contact the developer at [roshabel001@gmail.com].
