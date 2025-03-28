#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    book_id = int(request.form["book_id"])  # User selects a book

    # Find similar books
    distances, indices = model.kneighbors([[book_id]])

    # Get recommended books
    recommended_books = indices[0]

    return render_template("result.html", books=recommended_books)

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




