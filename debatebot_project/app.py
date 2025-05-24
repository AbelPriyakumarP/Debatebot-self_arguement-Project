import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# Initialize LLM and tokenizer
model_name = "distilgpt2"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

# Initialize embedding model and FAISS index
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    dimension = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
except Exception as e:
    raise Exception(f"Error initializing embedding model or FAISS: {e}")

# Updated knowledge base
knowledge_base = [
    "Topic: Universal healthcare\nPro: Ensures equal access to medical care.",
    "Topic: Universal healthcare\nCon: Increases government spending.",
    "Topic: Universal healthcare\nPro: Reduces financial burden on individuals.",
    "Topic: Universal healthcare\nCon: May lead to longer wait times.",
    "Topic: Climate Change\nPro: Requires immediate action through renewable energy.",
    "Topic: Climate Change\nCon: Economic impact of green policies is too high.",
    "Topic: Minimum Wage\nPro: Reduces poverty and inequality.",
    "Topic: Minimum Wage\nCon: Can lead to job losses for low-skilled workers.",
    "Topic: Artificial Intelligence in Education\nPro: Enhances personalized learning through adaptive technologies.",
    "Topic: Artificial Intelligence in Education\nCon: May reduce human interaction and critical thinking opportunities."
]

# Embed the knowledge base
try:
    argument_parts = [kb.split('\n')[1] for kb in knowledge_base]
    embeddings = embedder.encode(argument_parts)
    index.add(np.array(embeddings, dtype='float32'))
except Exception as e:
    raise Exception(f"Error embedding knowledge base: {e}")

def retrieve_context(topic, query, k=2):
    """Retrieve relevant arguments from the knowledge base."""
    try:
        query_embedding = embedder.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype='float32')
        distances, indices = index.search(query_embedding, k)
        relevant_context = []
        for i in indices[0]:
            kb_entry = knowledge_base[i]
            if f"Topic: {topic}" in kb_entry:
                relevant_context.append(kb_entry)
        return relevant_context
    except Exception as e:
        raise Exception(f"Error in retrieve_context: {e}")

def generate_text(prompt, max_new_tokens=200):
    """Generate text using the LLM."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        # Check input length
        input_length = inputs.input_ids.shape[1]
        if input_length > 512:
            raise ValueError(f"Input length ({input_length}) exceeds maximum allowed (512). Please shorten the prompt.")
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,  # Control new tokens generated
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text
    except Exception as e:
        raise Exception(f"Error in generate_text: {e}")

def self_argue(topic, initial_argument_idea):
    """Simulate self-argumentation by generating an argument, counterargument, and rebuttal."""
    try:
        context = retrieve_context(topic, initial_argument_idea)
        context_str = "\n".join(context)

        prompt_argument = f"Topic: {topic}\nContext: {context_str}\nInitial idea: {initial_argument_idea}\nGenerate a concise argument based on the context and initial idea:"
        argument = generate_text(prompt_argument)

        counter_context = retrieve_context(topic, argument)
        counter_context_str = "\n".join(counter_context)
        prompt_counterargument = f"Topic: {topic}\nContext: {counter_context_str}\nArgument: {argument}\nGenerate a counterargument to the argument based on the context:"
        counterargument = generate_text(prompt_counterargument)

        rebuttal_context = retrieve_context(topic, counterargument)
        rebuttal_context_str = "\n".join(rebuttal_context)
        prompt_rebuttal = f"Topic: {topic}\nContext: {rebuttal_context_str}\nCounterargument: {counterargument}\nArgument: {argument}\nGenerate a rebuttal to the counterargument, defending the original argument:"
        rebuttal = generate_text(prompt_rebuttal)

        return argument, counterargument, rebuttal, context_str
    except Exception as e:
        raise Exception(f"Error in self_argue: {e}")

# Custom CSS for 3D background effect (starry sky with depth)
st.markdown("""
    <style>
    body {
        background: #000;
        overflow: hidden;
    }
    .stars {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: transparent;
        z-index: -1;
        animation: moveStars 100s linear infinite;
    }
    .stars::before, .stars::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: transparent url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg"><circle cx="2" cy="2" r="1" fill="white"/></svg>') repeat;
        background-size: 200px 200px;
    }
    .stars::before {
        opacity: 0.5;
        transform: translateZ(-100px);
        animation: moveStars 50s linear infinite;
    }
    .stars::after {
        opacity: 0.3;
        transform: translateZ(-200px);
        animation: moveStars 75s linear infinite;
    }
    @keyframes moveStars {
        from { transform: translateY(0); }
        to { transform: translateY(-1000px); }
    }
    .main {
        background: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    h1, h3 {
        color: #00ffcc;
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background: #1a1a1a;
        color: white;
        border: 1px solid #00ffcc;
    }
    .stButton > button {
        background: #00ffcc;
        color: black;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background: #00cc99;
    }
    </style>
    <div class="stars"></div>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("DebateBot: Self-Arguing LLM")
st.markdown("Enter a debate topic and an initial argument idea to generate a debate with an argument, counterargument, and rebuttal.")

# Input fields
topic = st.text_input("Debate Topic", "Universal healthcare")
initial_argument_idea = st.text_area("Initial Argument Idea", "Ensures equal access to medical care for all citizens.")

# Generate button
if st.button("Generate Debate"):
    if not topic or not initial_argument_idea:
        st.error("Please provide both a topic and an initial argument idea.")
    else:
        with st.spinner("Generating debate..."):
            try:
                # Call the self_argue function (now defined in this file)
                argument, counterargument, rebuttal, context_str = self_argue(topic, initial_argument_idea)

                # Display results
                st.subheader("Retrieved Context")
                st.write(context_str if context_str else "No relevant context found.")

                st.subheader("Argument")
                st.write(argument if argument else "Failed to generate argument.")

                st.subheader("Counterargument")
                st.write(counterargument if counterargument else "Failed to generate counterargument.")

                st.subheader("Rebuttal")
                st.write(rebuttal if rebuttal else "Failed to generate rebuttal.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Instructions
st.markdown("---")
st.markdown("**Instructions**:")
st.markdown("- Enter a debate topic (e.g., 'Universal healthcare', 'Climate Change', 'Artificial Intelligence in Education').")
st.markdown("- Provide an initial argument idea to guide the debate.")
st.markdown("- Click 'Generate Debate' to see the results.")
st.markdown("- Ensure you have a GPU for faster inference, or expect slower performance on CPU.")
st.markdown('</div>', unsafe_allow_html=True)
