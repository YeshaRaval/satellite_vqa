# Satellite VQA (Visual Question Answering)

This is a Streamlit web application for Visual Question Answering (VQA) on satellite images using large language models (LLMs) via Replicate's API. Users can upload a satellite image, ask questions about it, and receive intelligent responses from the model.

## Features
- Upload satellite images (PNG, JPG, JPEG)
- Chat interface for asking questions about the image
- Uses Replicate's API to run advanced LLMs (e.g., Claude 3.5 Sonnet)
- Adjustable model parameters (temperature, top_p)
- Chat history management

## Requirements
- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [Replicate Python client](https://pypi.org/project/replicate/)
- [transformers](https://pypi.org/project/transformers/)

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd VQA
   ```
2. **Install dependencies:**
   ```bash
   pip install streamlit replicate transformers
   ```

## Replicate API Token
You need a Replicate API token to use this app. Get your token from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens).

Set your token as an environment variable before running the app:
```bash
export REPLICATE_API_TOKEN=<your_token>
```

Alternatively, you can modify the code to input your token directly (not recommended for production).

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

- Upload a satellite image using the file uploader.
- Ask questions about the image in the chat interface.
- Adjust model parameters in the sidebar as needed.
- Use the sidebar button to clear chat history.

## Notes
- The app uses the model `anthropic/claude-3.5-sonnet` by default. You can change this in the code.
- The maximum conversation length is limited to 3072 tokens.
- For best results, use high-quality satellite images.

## License
This project is for educational and research purposes only. 
