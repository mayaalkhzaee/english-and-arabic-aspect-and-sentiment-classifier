# Dual-Language Aspect-Based Sentiment Analysis Pipeline

**A modular, efficiency-driven system for extracting aspect-based sentiment insights from mixed English and Arabic customer reviews.**

![Project Status](https://img.shields.io/badge/Status-Complete-green) ![Python](https://img.shields.io/badge/Python-3.9-blue) ![Framework](https://img.shields.io/badge/Framework-Streamlit-red)

## Abstract
This project presents a scalable, commercially viable Aspect-Based Sentiment Analysis (ABSA) system designed for bilingual markets. Addressing the challenges of unstructured text and dialectal variation in Arabic, the system employs a modular architecture that integrates classical machine learning with targeted Large Language Model (LLM) support. The pipeline automatically distinguishes between English, Modern Standard Arabic (MSA), and Dialectal Arabic, routing them to specialized processing tracks to extract granular sentiment insights.

## Key Features

* **Smart Routing Engine:** Automatically detects and routes reviews based on language (English, MSA, or Dialectal Arabic) using a robust Logistic Regression model.
* **Hybrid Dialect Handling:** Utilizes the Qwen LLM to semantically translate diverse Arabic dialects (e.g., Gulf, Egyptian) into standardized MSA before processing, preserving sentiment while unifying linguistic structure.
* **Dual-Track Processing:**
    * **English Track:** Optimized with SpaCy preprocessing and Stacking Ensembles.
    * **Arabic Track:** Specialized for rich morphology using CAMeL Tools and Character N-grams.
* **Aspect-Based Granularity:** Goes beyond document-level sentiment to identify specific product features (e.g., "Battery," "Screen") and their associated polarity.
* **Interactive Dashboard:** A Streamlit-based UI that visualizes sentiment trends and provides generative text summaries of user feedback.

## System Architecture

The system follows a modular pipeline flow consisting of four distinct phases:

### 1. Input & Smart Routing
* **Input:** Raw Amazon product reviews.
* **Smart Router:** A Logistic Regression classifier acts as the gatekeeper, achieving **88.6% test accuracy** in distinguishing between English, MSA, and Dialects.

### 2. Parallel Processing Pipelines
The system processes languages in parallel tracks to maximize efficiency:

| Component | English Pipeline | Arabic Pipeline |
| :--- | :--- | :--- |
| **Preprocessing** | **SpaCy:** Cleaning, lowercasing, tokenization. | **CAMeL Tools:** Orthographic normalization (Alef, Taa Marbuta), diacritic removal. |
| **Normalization** | N/A | **Qwen LLM:** Dialect-to-MSA translation for colloquial text. |
| **Aspect Extraction** | **TF-IDF + Linear SVM:** Extracts aspect terms (e.g., "camera"). | **Arabic TF-IDF + Linear SVM:** Uses character n-grams to handle morphology. |
| **Classification** | **Stacking Ensemble:** Combines Linear SVC, Logistic Regression, SGD, and Ridge. | **Arabic Stacking Ensemble:** Trained specifically on Arabic corpora. |

### 3. Convergence & Aggregation
* **Aggregation Engine:** Merges aspect counts and sentiment labels from both pipelines into a unified dataset.
* **General Polarity:** A separate SVM assigns sentiment scores to sentences lacking specific aspect terms to ensure full coverage.

### 4. Insights & UI
* **Generative Insights:** Qwen LLM generates natural language summaries explaining *why* certain aspects are trending positively or negatively.
* **Visualization:** Real-time data visualization via Streamlit.

## Performance Metrics

We prioritized Green AI principles, choosing optimized linear ensembles over heavy Transformer models for real-time inference speed without sacrificing accuracy.

| Module | Model | Performance |
| :--- | :--- | :--- |
| **Smart Router** | Logistic Regression | **83.92% CV Accuracy** / **88.6% Test Accuracy**  |
| **Arabic Sentiment** | Stacking Ensemble | **90.18% Accuracy**  |
| **English Sentiment** | Stacking Ensemble | **79.98% Accuracy**  |

## Installation & Usage

### Prerequisites
* Python 3.9+
* [Qwen API Key](https://huggingface.co/Qwen) (for dialect translation and insights)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mayaalkhzaee/english-and-arabic-aspect-and-sentiment-classifier.git](https://github.com/mayaalkhzaee/english-and-arabic-aspect-and-sentiment-classifier.git)
    cd english-and-arabic-aspect-and-sentiment-classifier
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries include: `scikit-learn`, `spacy`, `camel-tools`, `imbalanced-learn`, `streamlit`.*

3.  **Download Language Models:**
    ```bash
    python -m spacy download en_core_web_sm
    camel_data -i light-morphology-msa
    ```

4.  **Run the Dashboard:**
    ```bash
    streamlit run app.py
    ```

## Authors

* **Maya Al-Khzaee**
* **Ahmed AbuShawish**
* **Hamzah Alanati**
* **Abderahman Belemine**

## References
* *Aly, M., & Atiya, A. (2013). LABR: A Large Scale Arabic Book Reviews Dataset.*
* *El-Haj, M. (2019). HARD: Hotel Arabic-Reviews Dataset.*
