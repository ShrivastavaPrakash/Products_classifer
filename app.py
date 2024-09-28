# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# from typing import List

# # Load the saved model and vectorizer
# model = joblib.load('product_classifier_model.pkl')
# tfidf = joblib.load('tfidf_vectorizer.pkl')

# # Initialize FastAPI app
# app = FastAPI()

# # Define a request model
# class Product(BaseModel):
#     description: str

# # Define a bulk request model for predicting multiple products at once
# class BulkProducts(BaseModel):
#     descriptions: List[str]

# # Define the classify endpoint for a single product
# @app.post("/classify/")
# async def classify_product(product: Product):
#     # Vectorize the input description
#     description_tfidf = tfidf.transform([product.description])
    
#     # Predict the product category
#     prediction = model.predict(description_tfidf)
    
#     # Return the predicted category
#     return {"description": product.description, "category": prediction[0]}

# # Define the classify endpoint for multiple products at once
# @app.post("/classify_bulk/")
# async def classify_bulk(products: BulkProducts):
#     # Vectorize the input descriptions
#     descriptions_tfidf = tfidf.transform(products.descriptions)
    
#     # Predict the product categories
#     predictions = model.predict(descriptions_tfidf)
    
#     # Return the list of predictions
#     return {"predictions": [{"description": desc, "category": pred} for desc, pred in zip(products.descriptions, predictions)]}



from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import uvicorn

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model, vectorizer, and label encoder
model = joblib.load('product_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

# Initialize FastAPI
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Product Classifier API!"}

# Pydantic model to define the request body for the /predict endpoint
class ProductDescription(BaseModel):
    description: str

# Prediction endpoint
@app.post("/predict")
def predict_category(item: ProductDescription):
    logging.info(f"Received description: {item.description}")
    
    # Transform the input description using the pre-trained TF-IDF vectorizer
    transformed_description = tfidf.transform([item.description])
    logging.info(f"Transformed description: {transformed_description}")
    
    # Predict the category using the pre-trained model
    predicted_category = model.predict(transformed_description)
    logging.info(f"Predicted category (encoded): {predicted_category}")
    
    # Inverse transform the label encoded output back to the original category label
    category = le.inverse_transform(predicted_category)[0]
    logging.info(f"Predicted category (decoded): {category}")

    return {"description": item.description, "predicted_category": category}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

