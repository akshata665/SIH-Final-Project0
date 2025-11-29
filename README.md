# SIH-Final-Project
This project develops an AI-powered diagnostic software that analyses transformer Frequency Response Analysis (FRA) data to detect mechanical and electrical faults such as winding deformation, core displacement, and insulation degradation. It supports multi-format data from vendors like Omicron, Megger, and Doble, unifies them for analysis
# ⚡ AI-Driven FRA Transformer Fault Diagnostic System  

**A unified & intelligent software to diagnose power transformer faults using Frequency Response Analysis (FRA) + Deep Learning + Explainability.**

## 🔎 What is this project  

Transformers in power grids require regular health checks. Using FRA (Frequency Response Analysis), one can detect mechanical/electrical faults — but interpreting raw FRA data is hard.  
This project builds an AI pipeline that:  
- Ingests FRA data from multiple vendors/formats  
- Automatically classifies faults (e.g. axial displacement, shorted turns, core issues)  
- Shows clear prediction results with probability scores  
- Provides explainability via saliency visualization so engineers understand why a particular fault was flagged

It’s specifically built for the Smart India Hackathon (SIH).

## ✅ Key Features  

- Hybrid model: CNN + BiLSTM + Autoencoder for robust feature extraction  
- Multi-vendor, multi-format FRA data support  
- Web-based UI using Streamlit — easy upload & prediction  
- Real-time prediction with confidence score, class probabilities, and explainability graph  
- Dataset overview & model performance (accuracy, confusion matrix, classification report) included  

## 🚀 Quick Start  

1. Clone repository  
   ```bash
   git clone https://github.com/shivarajsg/SIH-Final-Project.git
   cd SIH-Final-Project/streamlit-app
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the app

bash
Copy code
streamlit run streamlit_app.py
📂 Repository Structure
vbnet
Copy code
SIH-Final-Project/
├── streamlit-app/       ← Main app directory
│   ├── streamlit_app.py
│   ├── fra_model.h5      ← Trained model
│   ├── requirements.txt
│   └── (optional dataset / test files)  
├── website/             ← (Optional) placeholder for your website / other deliverables  
└── README.md            ← This file  
🛠️ Usage
Go to “Prediction” tab

Upload any FRA .csv file (or format supported)

View predicted fault, confidence, class-probability chart

Scroll down to see Why the model predicted — saliency plot showing which frequency ranges influenced the decision

📊 Model Performance
Metric	Value
Test Accuracy	~99 %
Fault Classes	6
Vendors	Multiple vendors/formats supported

You can view full confusion matrix and classification report under “Model” tab in the UI.

⚙️ For Developers
Uses Python 3.x, TensorFlow, Streamlit, scikit-learn

FRA preprocessing converts CSV → fixed shape (500×4) → normalization → model input

Explainability implemented via gradient-based saliency

🔮 Future Improvements
Support for more FRA formats (binary, XML)

Batch-mode predictions (multiple files at once)

Export report (PDF / Excel)

Web-deployment via Streamlit Cloud / custom server

👥 Team / Acknowledgement
SIH Team — Power Grid Corporation Project
Built and trained with real FRA data from multiple vendors

yaml
Copy code

---

## ✅ How to Add This to Your Repo  

1. Create file `README.md` at root of `SIH-Final-Project`  
2. Paste the above content and save  
3. Commit and push  

This ensures when someone visits your GitHub repo — judges or collaborators — they immediately understand the purpose, usage, and professionalism of your project.  

---

If you want — I can also prepare a **badge list**, **license section**, and **table-of-contents links** to make README look even more polished.
::contentReference[oaicite:2]{index=2}











ChatGPT can make 
