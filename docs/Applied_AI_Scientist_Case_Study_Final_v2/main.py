from fastapi import FastAPI, Request
import pickle

app = FastAPI()

with open('complex_price_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/recommend")
async def recommend_properties(request: Request):
    user_input = await request.json()
    # For demo purposes, apply scoring logic using the model
    score = model.predict(user_input)
    return {
        "recommendations": [
            {
                "address": "123 Main St",
                "score": score,
                "reason": "Matches budget, school rating, and size requirements"
            }
        ]
    }
