# CCTV Violence Detection with Explainable AI (XAI)

## 📌 Project Overview
This project aims to build an automated surveillance assistance system that detects violent activities in CCTV footage deployed in crowded public places. Since continuous human monitoring of surveillance cameras is impractical, the system uses computer vision and deep learning to automatically identify suspicious or violent behavior and raise alerts.  

To improve trust and accountability, the system integrates Explainable AI (XAI) techniques that visually and temporally explain why a particular video was classified as violent.

---

## 🎯 Objectives
- Detect violent activities in real-world CCTV footage
- Reduce dependency on 24/7 human monitoring
- Provide visual explanations for model decisions using XAI
- Support timely alerts for authorities in high-risk situations

---

## 📂 Dataset
**RWF-2000 (Real-World Fighting Dataset)**  
- 2000 CCTV-style video clips  
- Two classes: `Fight` and `NonFight`  
- Captured from real-world surveillance scenarios  
- Suitable for crowded and uncontrolled environments  

Dataset is downloaded using `kagglehub`.

---

## 🧠 System Architecture
1. Video input from CCTV footage  
2. Frame extraction and preprocessing  
3. Deep learning-based violence detection model  
4. Alert logic based on prediction confidence  
5. Explainable AI module for decision interpretation  

---

## 🛠️ Technologies Used
- **Python**
- **OpenCV** – video processing and frame extraction  
- **PyTorch** – deep learning model implementation  
- **Scikit-learn** – evaluation metrics  
- **Matplotlib / Seaborn** – visualization  
- **Grad-CAM** – explainable AI visualizations  
- **KaggleHub** – dataset download and management  

---

## 📁 Project Structure
CCTV_Violence_Detection_XAI/
├── data/
│ └── raw/
│ └── RWF-2000/
│ ├── Fight/
│ └── NonFight/
├── models/
├── xai/
├── notebooks/
├── utils/
├── configs/
├── inference/
├── demo/
├── reports/
├── README.md
├── requirements.txt
└── main.py


---

## 🔍 Explainable AI (XAI)
The project uses explainable AI techniques to justify model predictions:
- Grad-CAM heatmaps to highlight important regions
- Temporal analysis to identify critical frames
- Visual explanations for violent activity detection  

These explanations help understand what patterns influenced the model’s decision.

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix  

---

## 🚀 Future Scope
- Detection of other suspicious activities (theft, vandalism, loitering)
- Multi-class classification of abnormal events
- Real-time deployment with live CCTV feeds
- Integration with alert and notification systems
- Improved explainability using advanced temporal XAI methods

---

## 📜 License
This project is intended for academic and educational purposes.
