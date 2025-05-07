# 🎬 User-Based Movie Recommendation System

This project is a **User-Based Movie Recommendation System** powered by collaborative filtering and enhanced with natural language movie suggestions using an LLM (LLaMA 3 via Groq API). It uses Flask as the backend framework and FAISS for similarity search to recommend personalized movies for users based on historical rating patterns.

---

## 📁 Project Structure


- app.py
- model.py
- utils.py
- movies.csv
- ratings.csv
- tags.csv
- README.md

---

## 🚀 Features

- User-based collaborative filtering with SVD
- FAISS for fast similarity search among users
- Movie genre filtering support (`?genre=Action`, etc.)
- LLM-powered summary (LLaMA 3 via Groq API) for friendly movie suggestions
- REST API using Flask

---

## 🧠 How It Works

1. **Embeddings Creation**:
   - Ratings are transformed into a sparse matrix and decomposed using Truncated SVD to get user embeddings.
  
2. **Similarity Search**:
   - FAISS is used to find similar users based on embeddings.

3. **Recommendation Logic**:
   - Finds high-rated movies watched by similar users which the current user hasn't seen.
   - Optionally filters by genre.
  
4. **Friendly Output**:
   - The final list of movie titles is passed to an LLM to generate a human-friendly summary.

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ssarthak0/User-Based-Movie-Recommendation-System.git
   cd User-Based-Movie-Recommendation-System
   
2. **Install dependencies**
pip install -r requirements.txt

3. **Download the dataset**
The dataset used in this project is the MovieLens 25M Dataset.

Download it manually from this link https://www.kaggle.com/datasets/garymk/movielens-25m-dataset

Extract the contents and place the following files in the project root directory:

movies.csv

ratings.csv

tags.csv

4. **Set up your Groq API key**

Replace the placeholder in utils.py with your actual Groq API key for LLM summaries.

## 🛠️ API Endpoints
GET /recommend/<user_id>?genre=<genre_name>
**Description**: Returns recommended movies for a user, optionally filtered by genre.

**Example:**
curl http://127.0.0.1:5000/recommend/10?genre=Comedy

**Response:**
{
  "recommended_movies": [
    "Superbad",
    "The Hangover",
    "Step Brothers",
    "Bridesmaids",
    "Anchorman"
  ],
  "gpt_summary": "If you're in the mood to laugh, check out these hilarious picks: Superbad, The Hangover, and more!"
}

## 🧪 Running the App

python app.py

Visit **http://localhost:5000/recommend/1** in your browser or use tools like Postman/cURL.

## 🧰 Requirements

- Flask
- pandas
- scikit-learn
- faiss-cpu
- numpy
- scipy
- openai

## ✨ Acknowledgments

- MovieLens 25M Dataset
- Groq API
- FAISS
- scikit-learn
