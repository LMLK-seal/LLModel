# LLModel: 100% Private-Offline chat!

![LLModel Chat Demo](https://raw.githubusercontent.com/LMLK-seal/LLModel/main/LLModel1.gif)

LLModel Chat is a CPU only PyQt5-based graphical user interface (GUI) application that allows users to interact with a local large language model (LLM) using the Llama library. This application provides a user-friendly interface for loading and using LLM models, engaging in conversations, and even processing uploaded text or PDF files as context for the AI responses.

## Concept and Features

The main concept behind LLModel Chat is to provide an easy-to-use interface for interacting with local LLM models. It combines the power of LLMs with a clean, intuitive GUI, making it accessible for users who may not be comfortable with command-line interfaces.

Key features include:

1. **Model Loading**: Users can load a local GGUF (GPT-Generated Unified Format) model file.
2. **File Upload**: Ability to upload and process text or PDF files as additional context for the AI.
3. **Conversation Interface**: A chat-like interface for interacting with the AI model.
4. **Adjustable Parameters**: Users can modify max tokens and temperature settings.
5. **Code Extraction**: Automatically detects Python code in AI responses and allows downloading.
6. **Progress Indication**: Shows the progress of AI response generation.
7. **Conversation Management**: Option to clear the conversation history.
8. **Text2Speech**: Option to play the AI response.
9. **Copy text**: Option to copy the text of the AI response.
10. **Save/load conversation**: Option to save and load previous conversations. 

## Libraries and Dependencies

To run this application, you'll need to install the following libraries:

1. PyQt5: For the graphical user interface
2. llama-cpp-python: For interfacing with the Llama models
3. PyMuPDF (fitz): For handling PDF files.
4. pyttsx3: Text to speech.

You can install these libraries using pip: 
```
   pip install PyQt5 llama-cpp-python PyMuPDF pyttsx3
   ```

## How to Run the Code

1. Ensure you have Python 3.6+ installed on your system.
2. Install the required libraries as mentioned above.
3. Download the `LLModel.py` file to your local machine.
4. Open a terminal or command prompt and navigate to the directory containing `LLModel.py`.
5. Run the script using Python:

6. The LLModel Chat window should appear.

## Using the Application

1. **Load Model**: Click the "Load Model" button and select your GGUF model file.
2. **Upload Context File** (Optional): You can upload a text or PDF file to provide additional context for the AI responses.
3. **Start Chatting**: Type your message in the input field and click "Send" or press Enter.
4. **Adjust Settings**: You can modify the max tokens and temperature using the controls at the bottom of the window.
5. **Download Code**: If the AI generates Python code, you can download it using the "Download Code" button.
6. **Clear Conversation**: Use the "Clear Conversation" button to start a new chat session.
7. **Auto/Menual mode**: Use auto or menual mode for tokens count.

## Note

This application requires a compatible GGUF model file to function. Make sure you have a suitable model before running the application. The model should be compatible with the llama-cpp-python library.

GGUF libraries: 
https://huggingface.co/models?library=gguf

Recommended GGUF models:
1. Phi-3-mini-4k-instruct-gguf (Q5_K_M)
https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF

2. Meta-Llama-3.1-8B-Instruct-GGUF (Q5_K_M)
https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF

## Contribution

Contributions to improve the application are welcome. Please feel free to submit issues or pull requests to enhance functionality, fix bugs, or improve the user interface.
