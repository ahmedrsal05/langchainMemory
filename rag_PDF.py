import pdfplumber
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

class ChatPDF:
    """
    A class to handle PDF documents and answer questions using text extracted from them,
    with short-term memory using LangChain's ConversationBufferMemory.
    
    Attributes:
        model (ChatOllama): A ChatOllama model for generating responses.
        text_splitter (RecursiveCharacterTextSplitter): A splitter for dividing text into manageable chunks.
        prompt (PromptTemplate): Template for constructing prompts with placeholders for question and context.
        retriever: A retriever component to fetch relevant context from the vector store.
        chain: A processing chain to manage the flow from context retrieval to response generation.
        memory: Memory component to manage conversation history.
    """

    def __init__(self):
        """
        Initializes the ChatPDF with necessary components and configurations.
        """
        self.model = ChatOllama(model="mistral7b")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are a helpful assistant that analyses information from different documents.
            Use the following pieces of retrieved context to answer the question.
            Give specific details when possible. If you don't know the answer,
            just say that you don't know.  [/INST] </s> 
            [INST] Question: {question} 
            Context: {context}
            Answer: [/INST]
            """
        )
        self.retriever = None
        self.chain = None
        self.memory = ConversationBufferMemory(memory_key="context", return_messages=True)  # Initialize conversation memory
        
    def ingest(self, pdf_file_path: str):
        """
        Processes a PDF file, extracts text, splits into chunks, and prepares it for answering queries.
        """
        text = self.extract_text_from_pdf(pdf_file_path)
        chunks = self.text_splitter.create_documents([text])
        chunks = filter_complex_metadata(chunks)
        vector_store = Chroma.from_documents(documents=chunks, embedding=OllamaEmbeddings(model='nomic-embed-text'))
        self.retriever = vector_store.as_retriever()
        # Create a conversation chain with memory
        self.chain = ConversationChain(
            memory=self.memory,
            prompt=self.prompt,
            llm=self.model,
            retriever=self.retriever,
            output_parser=StrOutputParser()
        )

    def extract_text_from_pdf(self, pdf_file_path):
        """
        Extracts all text from a PDF file using pdfplumber.
        """
        text = ""
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + " "
        return text

    def ask(self, query: str):
        """
        Answers a query using the processed PDF text stored in the vector store.
        """
        if not self.chain:
            return "Please, add a PDF document first."
        
        # Get the response from the model using conversation chain
        response = self.chain.run(input=query)
        
        return response

    def clear(self):
        """
        Clears the internal state, including the retriever, processing chain, and conversation history.
        """
        self.retriever = None
        self.chain = None
        self.memory.clear()  # Clear conversation history

