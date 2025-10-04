🧠 Vita AI – Intelligent Disease Prediction System Using Symptoms

Vita AI is an AI-driven medical assistant built with Flask, NLP, and Machine Learning that predicts possible diseases from user-reported symptoms.
It uses semantic and syntactic similarity, natural-language understanding (SpaCy + NLTK), and a KNN classifier trained on a medical dataset to provide accurate diagnostic suggestions, descriptions, severity analysis, and medical precautions.

It can converse with users in natural language via a web chat interface — like talking to your personal digital doctor.

✨ Features

🩺 AI Disease Prediction – Predicts diseases from symptom input using trained KNN model.
🧠 NLP Understanding – Understands varied ways users describe symptoms using SpaCy + NLTK.
💬 Conversational Diagnosis – Engages in step-by-step Q&A to refine diagnosis.
📋 Medical Knowledge Base – Includes disease descriptions, symptom severity, and precaution recommendations.
🧾 Data Logging – Stores patient data and predictions in JSON for future analysis.
🌐 Web Interface (Flask) – Interactive, browser-based chatbot interface.
📊 Semantic & Syntactic Matching – Matches user input to closest known symptoms using Jaccard & WordNet similarity.

🧠 Tech Stack
Component	Technology
Programming Language	Python 3
Web Framework	Flask
NLP Libraries	SpaCy (en_core_web_sm), NLTK
Machine Learning	scikit-learn (KNN classifier via Joblib)
Data Handling	NumPy, Pandas
Storage	JSON, CSV
Frontend	HTML + Jinja Templates
Model Serving	Flask + Gunicorn (optional)
🧩 Project Structure
📦 Vita-AI
├── app.py                         # Main Flask application
├── model/
│   └── knn.pkl                    # Pre-trained KNN model
├── Medical_dataset/
│   ├── Training.csv               # Training data
│   ├── Testing.csv                # Test data
│   ├── symptom_Description.csv    # Disease descriptions
│   ├── symptom_severity.csv       # Symptom severity mapping
│   └── symptom_precaution.csv     # Disease precaution mapping
├── templates/
│   └── home.html                  # Web UI template
├── static/                        # Optional CSS/JS/images
├── DATA.json                      # Saved user data (auto-created)
└── requirements.txt               # Project dependencies

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/yourusername/Vita-AI.git
cd Vita-AI

2️⃣ Create and Activate Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Download SpaCy & NLTK Data
python -m spacy download en_core_web_sm

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

5️⃣ Run the App
python app.py


Open your browser at 👉 http://127.0.0.1:5000

💬 How It Works

1️⃣ User opens the web interface and begins a conversation.
2️⃣ Vita AI asks about age, gender, and main symptom.
3️⃣ It compares user input with known symptoms using semantic + syntactic similarity.
4️⃣ Possible diseases are filtered via KNN model.
5️⃣ Vita AI displays:

📖 Disease Description

⚕️ Severity Analysis

💡 Precautions & Recommendations

🧩 Example Conversation
User Input	Vita AI Response
“Hello”	“What is your name?”
“I’m Rahul”	“How old are you?”
“23”	“Can you specify your gender?”
“Male”	“Tell me your main symptom.”
“Headache and fatigue”	“You may have Migraine. Tap D to see description.”
“D”	Displays description + precautions
🩺 Example Output

Predicted Disease: Typhoid Fever
Description: Typhoid is a bacterial infection caused by Salmonella Typhi affecting the intestinal tract.
Severity: High (if symptoms persist > 5 days)
Precautions:
1️⃣ Take antibiotics as prescribed
2️⃣ Drink boiled water
3️⃣ Rest well
4️⃣ Avoid street food

🧪 Machine Learning Model

Algorithm: K-Nearest Neighbors (KNN)

Training Data: Medical Dataset (132 Diseases × 132 Symptoms)

Feature Vector: Binary One-Hot Encoding of Symptoms

Evaluation: Trained on 80%, tested on 20% dataset

Output: Most probable disease based on symptom set

🔮 Future Enhancements

🤖 Integrate voice input for speech-based diagnosis

🌐 Deploy on Render / AWS / Streamlit for cloud access

📈 Add symptom trend visualizations

🧬 Introduce deep-learning models for multi-disease prediction

🧠 Add medical chatbot with LLM integration

🏆 Credits

Developed by [Your Name]
Powered by Python, Flask, SpaCy, NLTK, and scikit-learn.
For educational and research purposes only — not a replacement for professional medical advice.
Healthcare chatbot to predict Diseases based on patient symptoms.



# How to use:
## create a venv 
virtualenv venv 

## activate it and install reqs
source venv/bin/activate
pip install -r requirements.txt 
python -m spacy download en_core_web_sm

## run app file
python app.py


Medical DataSet available !!
---- 
