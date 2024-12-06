# Tyrion Lannister AI Agent

This project implements a Tyrion Lannister AI chatbot, designed to simulate the wit and wisdom of the iconic "Game of Thrones" character. It utilizes OpenAI's language models, Pinecone vector database, and LangChain to provide a rich and interactive conversational experience.

View the demo: https://tyrion-lannister.streamlit.app/

---

## Features

- **Enhanced Conversational Memory**: Improved context retention using LangChain's ConversationBufferMemory for more coherent and contextually aware responses
- **Sophisticated Prompt Engineering**: Carefully crafted system prompts that better capture Tyrion's personality, wit, and knowledge
- **Rich Knowledge Base**: Powered by a Pinecone vector index, built from "Game of Thrones" book data
- **User-Friendly Interface**: Clean and intuitive Streamlit UI with example questions and clear instructions
- **Secure API Key Management**: Safe handling of OpenAI API keys with password field protection
- **Customizable Chat Settings**: Advanced settings for model selection and temperature control
- **Persistent Chat History**: Maintains conversation flow throughout the session

---

## Project Structure

```
Tyrion-Lannister-AI/
├── build_pinecone_index.py   # Script to create Pinecone index from book data
├── streamlit_app.py          # Streamlit app with enhanced memory and prompts
├── data/
│   └── got-books/           # Folder containing "Game of Thrones" book text files
├── constants.py             # Configuration constants for models and settings
├── requirements.txt         # Python dependencies
├── Tyrion.jpg              # Tyrion Lannister image for UI
└── README.md               # Project documentation
```

---

## Detailed Setup Instructions

### Prerequisites

1. **Python Environment**:
   - Python 3.11 or higher
   - pip (Python package installer)
   - virtualenv or venv for isolated environment

2. **Required Accounts**:
   - OpenAI account with API access
   - Pinecone account (free tier available)

3. **API Keys**:
   - OpenAI API Key from https://platform.openai.com/api-keys
   - Pinecone API Key from https://app.pinecone.io/

4. **System Requirements**:
   - Minimum 4GB RAM
   - Internet connection for API access
   - Modern web browser

---

### Step-by-Step Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/Tyrion-Lannister-AI.git
    cd Tyrion-Lannister-AI
    ```

2. **Set Up Python Environment**:
    ```bash
    # Create virtual environment
    python -m venv venv

    # Activate virtual environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install Dependencies**:
    ```bash
    # Upgrade pip
    python -m pip install --upgrade pip

    # Install requirements
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables**:
    - Create a `.env` file in the project root:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key_here
    PINECONE_API_KEY=your_pinecone_api_key_here
    PINECONE_ENV=your_pinecone_environment
    LOG_LEVEL=INFO
    ```

5. **Set Up Pinecone Index**:
    ```bash
    # Initialize Pinecone index
    python build_pinecone_index.py
    ```
    This script will:
    - Create a new Pinecone index named "got-bot"
    - Process and embed the Game of Thrones book content
    - Store the embeddings in Pinecone

---

### Running the Application

1. **Start the Streamlit App**:
    ```bash
    streamlit run streamlit_app.py
    ```

2. **Access the Web Interface**:
    - Open your browser to `http://localhost:8501`
    - The app will automatically open in your default browser

3. **First-Time Setup**:
    - Enter your OpenAI API key in the sidebar
    - The key is securely stored in your session
    - You can modify advanced settings in the sidebar

4. **Using the Chatbot**:
    - Start with example questions provided
    - Type your own questions in the input field
    - Click "Ask" or press Enter to submit
    - View Tyrion's responses in the chat interface

---

## Advanced Configuration

### Model Settings

1. **Changing the OpenAI Model**:
   - Access Advanced Settings in the sidebar
   - Available models: gpt-4, gpt-3.5-turbo
   - Default: gpt-4

2. **Adjusting Response Parameters**:
   - Temperature: Controls response creativity (0.0 - 1.0)
   - Default temperature: 0.7

### Memory System Configuration

1. **Conversation Buffer**:
   - Maintains recent conversation history
   - Automatically manages context window
   - Preserves important dialogue elements

2. **Vector Store Settings**:
   - Pinecone index configuration in `constants.py`
   - Customizable number of relevant documents (k=10)
   - Embedding model: text-embedding-ada-002

---

## Troubleshooting

### Common Issues and Solutions

1. **API Key Errors**:
   - Verify API keys in `.env` file
   - Check for proper formatting (no quotes needed)
   - Ensure sufficient API credits

2. **Connection Issues**:
   - Check internet connection
   - Verify Pinecone service status
   - Confirm OpenAI API availability

3. **Memory Issues**:
   - Clear browser cache
   - Restart Streamlit server
   - Check system resources

4. **Rate Limits**:
   - Implement appropriate delays between requests
   - Monitor API usage
   - Consider upgrading API tier if needed

---

## Development and Contribution

### Local Development

1. **Code Style**:
   - Follow PEP 8 guidelines
   - Use meaningful variable names
   - Add comments for complex logic

2. **Testing**:
   - Test new features locally
   - Verify API interactions
   - Check memory management

3. **Contributing**:
   - Fork the repository
   - Create feature branch
   - Submit pull request with descriptions

---

## Technologies Used

- **LangChain**: Advanced LLM chain management and memory systems
- **OpenAI GPT-4**: State-of-the-art language model for responses
- **Pinecone**: Vector database for efficient knowledge retrieval
- **Streamlit**: Modern web interface with session state management
- **Python 3.11+**: Core programming language
- **Environment Management**: python-dotenv for configuration

---

## Future Enhancements

- Voice interaction capabilities
- Multi-character conversation support
- Enhanced emotion detection and response
- Expanded knowledge base with TV show content
- Real-time response streaming
- Multi-language support
- Custom personality adjustments
- Integration with other Game of Thrones APIs

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Resources and References

- [OpenAI Documentation](https://platform.openai.com/docs)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Game of Thrones Books Dataset](https://www.kaggle.com/datasets/khulasasndh/game-of-thrones-books)
- George R.R. Martin's "A Song of Ice and Fire" series
---

## Demo Screenshots
![image](https://github.com/user-attachments/assets/82f79bf9-194a-4163-a6ad-d122e7c7c1b3)

![Snipaste_2024-12-06_18-49-36](https://github.com/user-attachments/assets/0efdf267-8564-4b5a-b675-2e44ac6ba31e)



For more information and updates, follow the project on GitHub.
