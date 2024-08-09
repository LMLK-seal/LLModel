import sys
import os
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QSpinBox, QMessageBox, QComboBox, QFileDialog, QProgressBar)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import fitz  # PyMuPDF for PDF handling

class LlamaThread(QThread):
    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, model, prompt, max_tokens, temperature):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self):
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
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()

        # Model and CUDA loading buttons
        load_layout = QHBoxLayout()
        self.load_model_button = QPushButton("(2)-Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_cuda_button = QPushButton("(1)-Load CUDA Folder")
        self.load_cuda_button.clicked.connect(self.load_cuda_folder)
        load_layout.addWidget(self.load_model_button)
        load_layout.addWidget(self.load_cuda_button)
        layout.addLayout(load_layout)

        # File Upload Button
        self.upload_file_button = QPushButton("Upload File (PDF or TXT)")
        self.upload_file_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_file_button)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
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
                self.model = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1)
                QMessageBox.information(self, "Success", f"Model loaded successfully!\nPath: {model_path}")
                self.send_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load the model: {str(e)}")

    def load_cuda_folder(self):
        cuda_folder = QFileDialog.getExistingDirectory(self, "Select CUDA Folder")
        if cuda_folder:
            os.environ['CUDA_PATH'] = cuda_folder
            cuda_bin_path = os.path.join(cuda_folder, "bin")
            if os.path.exists(cuda_bin_path):
                os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ['PATH']
                QMessageBox.information(self, "Success", f"CUDA folder set successfully!\nPath: {cuda_folder}")
            else:
                QMessageBox.warning(self, "Warning", f"CUDA bin directory not found at {cuda_bin_path}")

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

        prompt = "\n".join(self.conversation) + "\nAI:"
        self.token_count += len(prompt.split())

        if self.token_count >= self.max_tokens:
            self.conversation = [self.system_prompt, self.conversation[-1]]
            self.token_count = len("\n".join(self.conversation).split())
            self.update_chat_display()

        # Include uploaded file content in the prompt if available
        if self.uploaded_file_content:
            prompt += f"\n\n**File Content:**\n{self.uploaded_file_content}"

        self.llama_thread = LlamaThread(self.model, prompt, self.max_tokens - self.token_count, self.temperature)
        self.llama_thread.response_signal.connect(self.handle_response)
        self.llama_thread.error_signal.connect(self.handle_error)
        self.llama_thread.progress_signal.connect(self.update_progress)
        self.llama_thread.start()

        self.send_button.setEnabled(False)
        self.progress_bar.setValue(0)

    def handle_response(self, response):
        if self.validate_response(response):
            self.conversation.append(f"AI: {response}")
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
        display_conversation = self.conversation[1:]  # Exclude system prompt from display
        self.chat_display.setPlainText("\n\n".join(display_conversation))
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def clear_conversation(self):
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
