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

        # Adjust max_length to be smaller than the input length to avoid the max_length warning
        input_length = len(request.text.split())  # Count words in the input text

        # Summarize the provided text (you can adjust max_length, min_length as per your requirements)
        summaries = summarizer(
            request.text,
            max_length=input_length,  # Adjust max_length based on input size
            min_length=30,  # Ensure a minimum summary length
            do_sample=False,  # Disable sampling (use beam search instead)
            num_beams=5  # Use beam search for more coherent summaries
        )

        # Extract the summary texts
        summarized_texts = [summary['summary_text'] for summary in summaries]

        # Return the summaries
        return TextResponse(summaries=summarized_texts)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# To run the app with uvicorn:
# uvicorn main:app --reload
