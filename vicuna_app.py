from fastapi import FastAPI, HTTPException

from llama_cpp import Llama

# Define the model path
vicuna_model_path = "./model/ggml-vicuna-13b-4bit.bin"

#Load the model
vicuna_model = Llama(model_path=vicuna_model_path)

app = FastAPI()

# Define the default route
@app.get("/")
def home():
    return {"Message": "Welcome to the Vicuna Demo FastAPI"}

@app.post("/vicuna_says")
def answer_prompt(user_prompt):

    if (not (user_prompt)):
        raise HTTPException(status_code=400, 
                            detail="Please Provide a valid text message")

    response = vicuna_model(user_prompt)

    return {"Answer": response}