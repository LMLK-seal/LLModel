import sys
import os
import re
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QSpinBox, QMessageBox, QComboBox, QFileDialog, QProgressBar, QApplication)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot
import fitz  # PyMuPDF for PDF handling
import pyttsx3  # Import the text-to-speech library

class LlamaThread(QObject):
    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, model, prompt, max_tokens, temperature):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    @pyqtSlot()
    def generate_response(self):
        try:
            print("LlamaThread: Generating response...")
            response = self.model(
                self.prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["\nYou:"],
                stream=True
            )
            full_text = ""
            for i, chunk in enumerate(response):
                full_text += chunk['choices'][0]['text']
                progress = min(100, int((i / self.max_tokens) * 100))
                self.progress_signal.emit(progress)
            
            print("LlamaThread: Response generated successfully.")
            print("Extracted Text:", full_text)
            self.response_signal.emit(full_text)
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished.emit()


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLModel Chat")
        self.setGeometry(100, 100, 800, 600)
        self.system_prompt = "You are a helpful, respectful, and accurate AI assistant. Always provide truthful and appropriate responses."
        self.conversation = [self.system_prompt]
        self.token_count = 0
        self.max_tokens = 2000
        self.temperature = 1  # Default temperature
        self.model = None
        self.current_python_code = None
        self.uploaded_file_content = None
        self.llama_thread = None  # Initialize the thread object
        self.thread = None  # Initialize the QThread object
        self.tts_engine = pyttsx3.init()  # Initialize the text-to-speech engine
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()

        # Model loading button (CUDA-related code removed)
        load_layout = QHBoxLayout()
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        load_layout.addWidget(self.load_model_button)
        layout.addLayout(load_layout)

        # File Upload Button
        self.upload_file_button = QPushButton("Upload File (PDF or TXT)")
        self.upload_file_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_file_button)

        # Chat display
        self.chat_display = ChatDisplayWidget(self.play_tts)  # Pass play_tts function
        layout.addWidget(self.chat_display)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setEnabled(False)  # Disable until model is loaded
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)

        # Control buttons
        control_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Conversation")
        self.clear_button.clicked.connect(self.clear_conversation)
        self.token_label = QLabel("Max Tokens:")
        self.token_spinbox = QSpinBox()
        self.token_spinbox.setRange(100, 4096)
        self.token_spinbox.setValue(self.max_tokens)
        self.token_spinbox.valueChanged.connect(self.update_max_tokens)
        self.download_code_button = QPushButton("Download Code")
        self.download_code_button.clicked.connect(self.download_code)
        self.download_code_button.setEnabled(False)
        control_layout.addWidget(self.clear_button)
        control_layout.addWidget(self.token_label)
        control_layout.addWidget(self.token_spinbox)
        control_layout.addWidget(self.download_code_button)
        layout.addLayout(control_layout)

        # Temperature control
        temp_layout = QHBoxLayout()
        self.temp_label = QLabel("Temperature:")
        self.temp_combo = QComboBox()
        self.temp_combo.addItems(["Precise (0)", "Balanced (1)", "Creative (2)"])
        self.temp_combo.setCurrentIndex(1)  # Default to "Balanced"
        self.temp_combo.currentIndexChanged.connect(self.update_temperature)
        temp_layout.addWidget(self.temp_label)
        temp_layout.addWidget(self.temp_combo)
        layout.addLayout(temp_layout)

        main_widget.setLayout(layout)

        # Set style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTextEdit, QLineEdit {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "GGUF Files (*.gguf)")
        if model_path:
            try:
                from llama_cpp import Llama
                self.model = Llama(model_path=model_path, n_ctx=2048)  # No need for n_gpu_layers
                QMessageBox.information(self, "Success", f"Model loaded successfully!\nPath: {model_path}")
                self.send_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load the model: {str(e)}")

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Text Files (*.txt);;PDF Files (*.pdf)")
        if file_path:
            if file_path.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.uploaded_file_content = file.read()
            elif file_path.endswith(".pdf"):
                doc = fitz.open(file_path)
                self.uploaded_file_content = ""
                for page in doc:
                    self.uploaded_file_content += page.get_text("text")
            else:
                QMessageBox.warning(self, "Error", "Unsupported file type. Please upload a .txt or .pdf file.")
                return

            QMessageBox.information(self, "Success", f"File uploaded successfully!\nPath: {file_path}")

    def send_message(self):
        user_input = self.input_field.text().strip()
        if not user_input or not self.model:
            return

        self.conversation.append(f"You: {user_input}")
        self.update_chat_display()
        self.input_field.clear()

        # Stop any running thread before starting a new one
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
            self.thread = None  # Reset thread reference

        # Prepare the prompt
        prompt = self.prepare_prompt(user_input)

        # Split long inputs into chunks
        chunks = self.split_into_chunks(prompt)

        for chunk in chunks:
            self.process_chunk(chunk)

    def prepare_prompt(self, user_input):
        prompt = "\n".join(self.conversation) + "\nAI:"
        
        if self.uploaded_file_content:
            prompt += f"\n\n**File Content:**\n{self.uploaded_file_content}"
        
        return prompt

    def split_into_chunks(self, text, chunk_size=1000):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def process_chunk(self, chunk):
        self.llama_thread = LlamaThread(self.model, chunk, min(self.max_tokens, 2043), self.temperature)
        self.thread = QThread()
        self.llama_thread.moveToThread(self.thread)
        self.thread.started.connect(self.llama_thread.generate_response)
        self.llama_thread.response_signal.connect(self.handle_response)
        self.llama_thread.error_signal.connect(self.handle_error)
        self.llama_thread.progress_signal.connect(self.update_progress)
        self.llama_thread.finished.connect(self.on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.llama_thread.deleteLater)
        self.thread.start()

        self.send_button.setEnabled(False)
        self.progress_bar.setValue(0)

    @pyqtSlot()
    def on_thread_finished(self):
        self.send_button.setEnabled(True)
        self.progress_bar.setValue(100)

    def handle_response(self, response):
        self.llama_thread.response_signal.disconnect(self.handle_response)  # Disconnect after using it

        if self.validate_response(response):
            self.conversation.append(f"AI: {response}")  # Append the response with "AI: " prefix
            self.token_count += len(response.split())
            self.update_chat_display()
            self.check_for_python_code(response)
        else:
            self.conversation.append("AI: I apologize, but I couldn't generate an appropriate response. Let me try again.")
            self.update_chat_display()
            self.send_message()  # Retry generating a response
        self.send_button.setEnabled(True)
        self.progress_bar.setValue(100)

    def handle_error(self, error_message):
        QMessageBox.warning(self, "Error", f"An error occurred: {error_message}")
        self.send_button.setEnabled(True)
        self.progress_bar.setValue(0)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_chat_display(self):
        # Clear the existing messages
        self.chat_display.clear()
        # Add all messages to the chat display
        for message in self.conversation[1:]:  # Skip the system prompt
            if message.startswith("You: "):
                self.chat_display.add_message("You", message[5:], False)
            elif message.startswith("AI: "):
                self.chat_display.add_message("AI", message[4:], True)
            else:
                self.chat_display.add_message("AI", message, True)

    def clear_conversation(self):
        # Stop any running thread before clearing
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
            self.thread = None  # Reset thread reference

        self.conversation = [self.system_prompt]
        self.token_count = 0
        self.update_chat_display()
        self.current_python_code = None
        self.uploaded_file_content = None
        self.download_code_button.setEnabled(False)

    def update_max_tokens(self, value):
        self.max_tokens = value

    def update_temperature(self, index):
        temp_values = [0, 1, 2]
        self.temperature = temp_values[index]
        print(f"Temperature updated to: {self.temperature}")

    def validate_response(self, response):
        # Simple content filter (can be expanded)
        inappropriate_words = ['offensive', 'rude']
        return not any(word in response.lower() for word in inappropriate_words)

    def check_for_python_code(self, response):
        # Use regex to find Python code blocks
        code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            self.current_python_code = '\n\n'.join(code_blocks)
            self.download_code_button.setEnabled(True)
        else:
            self.current_python_code = None
            self.download_code_button.setEnabled(False)

    def download_code(self):
        if self.current_python_code:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Python Code", "", "Python Files (*.py)")
            if file_path:
                with open(file_path, 'w') as file:
                    file.write(self.current_python_code)
                QMessageBox.information(self, "Success", f"Python code saved to {file_path}")
        else:
            QMessageBox.warning(self, "No Code", "No Python code available to download.")

    def play_tts(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

class ChatDisplayWidget(QWidget):
    def __init__(self, play_tts_function):  # Add play_tts_function as an argument
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.play_tts_function = play_tts_function
        self.clipboard = QApplication.clipboard()  # Get the clipboard

    def add_message(self, sender, text, with_play_button):
        message_widget = QWidget()
        message_layout = QHBoxLayout()
        message_widget.setLayout(message_layout)

        # Add sender label
        sender_label = QLabel(f"{sender}:")
        sender_label.setStyleSheet("font-weight: bold;")
        message_layout.addWidget(sender_label)

        # Add message text
        message_label = QLabel(text)
        message_label.setWordWrap(True)
        message_layout.addWidget(message_label, 1)

        if with_play_button:
            play_button = QPushButton("Play")
            play_button.setFixedSize(60, 35)
            play_button.clicked.connect(lambda: self.play_tts_function(text))  # Call the passed function
            message_layout.addWidget(play_button)

            # Add the "Copy Text" button
            copy_button = QPushButton("Copy Text")
            copy_button.setFixedSize(110, 35)
            copy_button.clicked.connect(lambda: self.copy_text(text))
            message_layout.addWidget(copy_button)

        self.layout.addWidget(message_widget)

    # Function to copy text to clipboard
    def copy_text(self, text):
        self.clipboard.setText(text)
        QMessageBox.information(self, "Copied", "Text copied to clipboard.")

    def clear(self):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
