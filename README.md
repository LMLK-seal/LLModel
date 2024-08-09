# LLModel Chat

![LLModel Chat Demo](https://github.com/LMLK-seal/LLModel/blob/main/LLModel.gif?raw=true)

LLModel Chat is a PyQt5-based graphical user interface (GUI) application that allows users to interact with a local large language model (LLM) using the Llama library. This application provides a user-friendly interface for loading and using LLM models, engaging in conversations, and even processing uploaded text or PDF files as context for the AI responses.

## Concept and Features

The main concept behind LLModel Chat is to provide an easy-to-use interface for interacting with local LLM models. It combines the power of LLMs with a clean, intuitive GUI, making it accessible for users who may not be comfortable with command-line interfaces.

Key features include:

1. **Model Loading**: Users can load a local GGUF (GPT-Generated Unified Format) model file.
2. **CUDA Support**: Option to set the CUDA folder for GPU acceleration.
3. **File Upload**: Ability to upload and process text or PDF files as additional context for the AI.
4. **Conversation Interface**: A chat-like interface for interacting with the AI model.
5. **Adjustable Parameters**: Users can modify max tokens and temperature settings.
6. **Code Extraction**: Automatically detects Python code in AI responses and allows downloading.
7. **Progress Indication**: Shows the progress of AI response generation.
8. **Conversation Management**: Options to clear the conversation history.

## Libraries and Dependencies

To run this application, you'll need to install the following libraries:

1. PyQt5: For the graphical user interface
2. llama-cpp-python: For interfacing with the Llama models
3. PyMuPDF (fitz): For handling PDF files

You can install these libraries using pip: "pip install PyQt5 llama-cpp-python PyMuPDF"

## How to Run the Code

1. Ensure you have Python 3.6+ installed on your system.
2. Install the required libraries as mentioned above.
3. Download the `LLModel.py` file to your local machine.
4. Open a terminal or command prompt and navigate to the directory containing `LLModel.py`.
5. Run the script using Python:

6. The LLModel Chat window should appear.

## Using the Application

1. **Load CUDA Folder** (Optional): If you have a CUDA-compatible GPU, click the "Load CUDA Folder" button and select your CUDA installation directory.
2. **Load Model**: Click the "Load Model" button and select your GGUF model file.
3. **Upload Context File** (Optional): You can upload a text or PDF file to provide additional context for the AI responses.
4. **Start Chatting**: Type your message in the input field and click "Send" or press Enter.
5. **Adjust Settings**: You can modify the max tokens and temperature using the controls at the bottom of the window.
6. **Download Code**: If the AI generates Python code, you can download it using the "Download Code" button.
7. **Clear Conversation**: Use the "Clear Conversation" button to start a new chat session.

## Note

This application requires a compatible GGUF model file to function. Make sure you have a suitable model before running the application. The model should be compatible with the llama-cpp-python library.

## Contribution

Contributions to improve the application are welcome. Please feel free to submit issues or pull requests to enhance functionality, fix bugs, or improve the user interface.
