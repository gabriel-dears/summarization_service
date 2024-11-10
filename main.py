from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Load Hugging Face summarization model (using BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # -1 for CPU, 0 for GPU


# Define a request model for the text to be processed
class TextRequest(BaseModel):
    text: str
    top_k: int = 3  # Number of top summaries to return (you can modify as per your needs)


# Define the response model
class TextResponse(BaseModel):
    summaries: List[str]


# Endpoint to process text and summarize
@app.post("/process-text", response_model=TextResponse)
async def process_text(request: TextRequest):
    try:
        # Ensure the input text is not empty
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty")

        # Summarize the provided text (you can adjust max_length, min_length as per your requirements)
        summaries = summarizer(
            request.text,
            max_length=200,  # Increase max length to allow for a more complete summary
            min_length=50,   # Ensure a minimum summary length
            do_sample=False,
            num_beams=5,  # Use beam search for more coherent summaries
            top_k=request.top_k
        )

        # Extract the summary texts
        summarized_texts = [summary['summary_text'] for summary in summaries]

        # Return the summaries based on the top_k parameter
        return TextResponse(summaries=summarized_texts)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# To run the app with uvicorn:
# uvicorn main:app --reload
