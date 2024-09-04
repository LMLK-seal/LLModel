import sys
import os
import re
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QSpinBox, QMessageBox, QComboBox, QFileDialog, QProgressBar, QApplication,
                             QScrollArea, QMenu, QAction)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot, QTimer
import fitz  # PyMuPDF for PDF handling
import pyttsx3  # Import the text-to-speech library
import weakref  # Import the weakref module
import json
from llama_cpp import Llama  # Import llama_cpp for CPU-based model loading

class LlamaThread(QObject):
    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished = pyqtSignal()

    def __init__(self, model, prompt, max_tokens, temperature):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.token_count = 0

    @pyqtSlot()
    def generate_response(self):
        try:
            print("LlamaThread: Generating response...")
            response = self.model(
                self.prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["\nHuman:", "\nYou:"],  # Add multiple stop sequences
                stream=True
            )
            full_text = ""
            for chunk in response:
                text = chunk['choices'][0]['text']
                full_text += text
                self.token_count += len(text.split())
                progress = min(100, int((self.token_count / self.max_tokens) * 100))
                self.progress_signal.emit(progress, self.token_count)
                
                # Check for natural stopping points
                if text.endswith(('.', '!', '?', '\n')) and len(full_text.split()) >= min(100, self.max_tokens):
                    break
                
                # Stop if we've reached or exceeded max_tokens
                if self.token_count >= self.max_tokens:
                    break

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
        self.auto_mode = False
        self.threads = []  # List to keep track of all threads using weak references
        self.cleanup_timer = QTimer()  # Create a timer for cleanup
        self.cleanup_timer.timeout.connect(self.cleanup_finished_threads)
        self.cleanup_timer.start(1000)  # Start the timer with a 1-second interval
        self.conversation_history_file = "conversation_history.json"  # Filename for conversation history
        # Initialize chat_display here
        self.chat_display = ChatDisplayWidget(self.play_tts)  # Initialize chat_display
        self.load_conversation()  # Load conversation history on startup
        self.setup_ui()


    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()

        # File menu
        self.file_menu = self.menuBar().addMenu("File")
        self.save_conversation_action = QAction("Save Conversation", self)
        self.save_conversation_action.triggered.connect(self.save_conversation)
        self.load_conversation_action = QAction("Load Conversation", self)
        self.load_conversation_action.triggered.connect(self.load_conversation)
        self.file_menu.addAction(self.save_conversation_action)
        self.file_menu.addAction(self.load_conversation_action)

        # Model loading button
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
        # self.chat_display = ChatDisplayWidget(self.play_tts)  # Pass play_tts function
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.chat_display)
        layout.addWidget(scroll_area)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Token count label
        self.token_count_label = QLabel("Tokens Processed: 0")
        layout.addWidget(self.token_count_label)

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
        self.token_label = QLabel("Set Max Tokens:(Manual mode only)")
        self.token_spinbox = QSpinBox()
        self.token_spinbox.setRange(100, 4096)
        self.token_spinbox.setValue(self.max_tokens)
        self.token_spinbox.valueChanged.connect(self.update_max_tokens)
        self.auto_mode_button = QPushButton("Auto Mode")
        self.auto_mode_button.clicked.connect(self.toggle_auto_mode)
        self.download_code_button = QPushButton("Download Code")
        self.download_code_button.clicked.connect(self.download_code)
        self.download_code_button.setEnabled(False)
        control_layout.addWidget(self.clear_button)
        control_layout.addWidget(self.token_label)
        control_layout.addWidget(self.token_spinbox)
        control_layout.addWidget(self.auto_mode_button)
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
                self.model = Llama(model_path=model_path, n_ctx=2048)
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
        # Check for a valid weak reference in the self.threads list
        for thread_ref in self.threads[:]: 
            thread = thread_ref()
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(1000)  # Add a timeout to wait()

        # Now check if self.thread is valid before accessing it
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(1000)
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
        if self.auto_mode:
            max_tokens = self.calculate_auto_tokens(chunk)
        else:
            max_tokens = self.max_tokens

        self.llama_thread = LlamaThread(self.model, chunk, min(max_tokens, 2043), self.temperature)
        self.thread = QThread()
        self.llama_thread.moveToThread(self.thread)
        self.thread.started.connect(self.llama_thread.generate_response)
        self.llama_thread.response_signal.connect(self.handle_response)
        self.llama_thread.error_signal.connect(self.handle_error)
        self.llama_thread.progress_signal.connect(self.update_progress)
        self.llama_thread.finished.connect(self.on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)  # Clean up the thread when it finishes
        self.thread.finished.connect(self.llama_thread.deleteLater)  # Clean up the llama thread when it finishes

        # Add a weak reference to the thread
        thread_ref = weakref.ref(self.thread)
        self.threads.append(thread_ref)

        self.adjust_tokens = False  # Initialize the flag
        self.current_tokens_used = 0  # Initialize the token counter
        self.total_tokens = 0  # Initialize the total tokens used

        self.thread.start()

        self.send_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.token_count_label.setText("Tokens Processed: 0")  # Reset token count label

    @pyqtSlot()
    def on_thread_finished(self):
        self.send_button.setEnabled(True)
        self.progress_bar.setValue(100)

        # Remove the thread reference after handling the response
        for thread in self.threads[:]:  # Iterate over a copy
            if thread() is None:
                self.threads.remove(thread)

        if self.adjust_tokens:
            self.adjust_tokens = False
            # Create a new thread with the adjusted max_tokens
            self.process_chunk(self.conversation[-1])  # Process the last message again

    def handle_response(self, response):
        self.llama_thread.response_signal.disconnect(self.handle_response)  # Disconnect after using it

        if self.validate_response(response):
            # Check if the response is complete or not
            if response.endswith("AI: "):
                # If it doesn't end with "AI: ", it's not complete.
                # So we need to continue processing.
                self.conversation.append(f"AI: {response}")
                self.token_count += len(response.split())
                self.update_chat_display()
                self.check_for_python_code(response)
            else:
                # If the response is complete, append it to conversation
                self.conversation.append(f"AI: {response}") 
                self.token_count += len(response.split())
                self.update_chat_display()
                self.check_for_python_code(response)

            self.total_tokens += len(response.split())  # Update total_tokens

            if self.total_tokens > self.max_tokens:
                self.conversation[-1] = self.conversation[-1][:self.max_tokens]  # Truncate the response
                self.update_chat_display()
                # Add a message to the conversation indicating truncation
                self.conversation.append(
                    f"AI: My response has been truncated due to exceeding the maximum token limit. "
                    f"Please adjust the maximum token limit or provide a shorter query."
                )
                self.update_chat_display()
        else:
            self.conversation.append("AI: I apologize, but I couldn't generate an appropriate response. Let me try again.")
            self.update_chat_display()
            self.send_message()  # Retry generating a response
        self.send_button.setEnabled(True)
        self.progress_bar.setValue(100)

        # Remove the thread reference after handling the response
        for thread in self.threads[:]:  # Iterate over a copy
            if thread() is None:
                self.threads.remove(thread)


    def handle_error(self, error_message):
        # Display a more informative error message
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        self.send_button.setEnabled(True)
        self.progress_bar.setValue(0)

    def update_progress(self, value, token_count):
        self.progress_bar.setValue(value)
        # Update the token count label
        self.token_count_label.setText(f"Tokens Processed: {token_count}") 

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
        # Stop all running threads before clearing
        for thread in self.threads[:]:  # Iterate over a copy
            if thread() is not None and thread().isRunning():
                thread().quit()
                thread().wait(1000)  # Add a timeout to wait()

        # Clear the list after waiting
        self.threads.clear()

        self.conversation = [self.system_prompt]
        self.token_count = 0
        self.update_chat_display()
        self.current_python_code = None
        self.uploaded_file_content = None
        self.download_code_button.setEnabled(False)
        self.thread = None  # Reset the self.thread reference

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

    def toggle_auto_mode(self):
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self.auto_mode_button.setText("Manual Mode")
        else:
            self.auto_mode_button.setText("Auto Mode")

    def calculate_auto_tokens(self, chunk):
        word_count = len(chunk.split())
        sentence_count = len(re.split(r'(?<=\.|\?|\!)\s', chunk))

        # Increase the base estimation
        estimated_tokens = word_count * 2 + sentence_count * 1

        # Ensure a minimum token count that's higher than the current issue
        min_tokens = max(200, self.max_tokens // 10)
        estimated_tokens = max(estimated_tokens, min_tokens)

        # Cap at the user-defined maximum
        estimated_tokens = min(estimated_tokens, self.max_tokens)

        return int(estimated_tokens)

    def closeEvent(self, event):
        # Stop all running threads before closing
        for thread in self.threads[:]:  # Iterate over a copy
            if thread() is not None and thread().isRunning():
                thread().quit()
                thread().wait(1000)  # Add a timeout to wait()

        # Clear the list after waiting
        self.threads.clear()

        # Stop the cleanup timer
        self.cleanup_timer.stop()

        event.accept()

    def cleanup_finished_threads(self):
        # Remove references to finished threads from the list using weak references
        for thread in self.threads[:]:
            if thread() is None:
                self.threads.remove(thread)

    def save_conversation(self):
        with open(self.conversation_history_file, 'w') as f:
            json.dump(self.conversation, f)
        QMessageBox.information(self, "Conversation Saved", "Conversation history saved successfully.")

    def load_conversation(self):
        try:
            with open(self.conversation_history_file, 'r') as f:
                self.conversation = json.load(f)
            self.update_chat_display()  # Update the chat display with loaded conversation
            QMessageBox.information(self, "Conversation Loaded", "Conversation history loaded successfully.")
        except FileNotFoundError:
            QMessageBox.warning(self, "Conversation Not Found", "No conversation history found.")


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
    window = ChatWindow()  # Define ChatWindow correctly
    window.show()
    app.exec_()
