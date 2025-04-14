# 🗣️ Grammar Scoring Engine for Voice Samples

A **Streamlit-based web application** that evaluates the **grammar quality** of spoken English from `.wav` audio files. The model uses audio features such as **MFCCs**, **Spectral Contrast**, and **Zero Crossing Rate** to predict a **grammar score** ranging from **0 to 5**.

🔗 **Live Demo**: [Try it on Streamlit](https://grammer-scoring-from-voice-samples-engine.streamlit.app/)

---

## 🎯 Project Objective

This project was inspired by a competition aimed at building a **Grammar Scoring Engine** for spoken data samples. Given audio files (45–60 seconds each) labeled with **MOS Likert Grammar Scores**, the goal was to create a model that:

- Accepts a `.wav` file as input
- Outputs a continuous grammar score between 0 and 5
- Provides visual/audio insights and feedback

---

## 🚀 Features

- 🎤 **Voice Input**: Upload `.wav` samples and instantly get grammar scores.
- 📘 **Grammar Feedback**: Text-based feedback aligned with CEFR levels (Beginner → Advanced).
- 📊 **Visual Insights**:
  - Waveform visualization
  - MFCC heatmap
- 📁 **Coming Soon**: Batch scoring for multiple files

---

## 🧠 How It Works

### 📦 Audio Preprocessing

Uses **Librosa** to extract key audio features:
- **MFCCs (Mel-frequency cepstral coefficients)**
- **Spectral Contrast**
- **Zero Crossing Rate (ZCR)**

### 🧮 Model

A **Random Forest Regressor** (Scikit-learn) is trained on extracted features to predict grammar quality.

---

## 🏗️ Tech Stack

- **Python**
- **Streamlit** (UI)
- **Scikit-learn** (ML model)
- **Librosa** (Audio processing)
- **Matplotlib / Seaborn** (Visualization)

---

## 📊 Dataset

The dataset was **manually created** using audio clips from various public datasets (e.g., [Freesound.org](https://freesound.org)). To generate labeled training data, the following tools were used:

- **Whisper**: Transcription of speech to text
- **TextStat**: Calculated readability scores
- **LanguageTool**: Grammar analysis and error detection
- **AudioSegment (pydub)**: Audio slicing and format handling

### 📁 Data Summary:
- **Training Samples**: 884
- **Testing Samples**: 543
- **Score Labels**: Continuous values (0–5) based on grammar quality

---

## 📈 Evaluation

Model performance was assessed using standard regression metrics:
- **Mean Squared Error (MSE)**
- **R² Score**
- **Visual error analysis**

---

## 📌 Future Work

- ✅ Batch upload and scoring
- 🔄 Grammar fluency and coherence evaluation
- 🌍 CEFR-based language learning level estimation
- 💼 Interview-readiness scoring

---

## 🙌 Acknowledgments

Special thanks to:
- [Freesound.org](https://freesound.org) for open audio datasets
- OpenAI Whisper for robust ASR
- [LanguageTool](https://languagetool.org/) for grammar analysis
- [TextStat](https://pypi.org/project/textstat/) for readability scoring

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🔗 Connect

Feel free to fork, star, or contribute! Feedback and ideas are always welcome.
