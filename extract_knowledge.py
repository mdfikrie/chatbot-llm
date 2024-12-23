from PyPDF2 import PdfReader
import argparse
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

# Parse input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
args = vars(ap.parse_args())

# Read the PDF file
pdf_reader = PdfReader(args["input"])
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Split text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_text(text)

# embeddings
print("[INFO] embeddings texts...")
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)

# Chat logic
print("[INFO] Starting bot...")
while True:
    chat = input("Anda >> ")
    docs = knowledge_base.similarity_search(chat)

    # Use OpenAI as LLM (replace with another model if needed)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=chat)
    print(f"Bot >> {response}")
