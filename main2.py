import os
import tkinter as tk
from tkinter import ttk
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Set HuggingFaceHub API token
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Specify the path to the PDFs
pdf_folder_path = "./pdfs"

# Create a list of UnstructuredPDFLoader for each PDF in the specified directory
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path) if fn.endswith('.pdf')]

# Create a VectorstoreIndexCreator
index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)).from_loaders(loaders)

# Initialize the HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.1, max_length=512)

# Create a RetrievalQA chain
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=index.vectorstore.as_retriever(),
                                    input_key="question")

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")


def translate(text, src_lang, tgt_lang):
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, max_length=400)
    translation = translator(text)
    return translation[0]['translation_text']

def translate_b(text, src_lang, tgt_lang):
    translation = translate(text, src_lang, tgt_lang)
    return translation

def get_answer():
    src_lang = "arz"
    tgt_lang = "en"

    question = translate_b(entry.get(), src_lang, tgt_lang)
    if question:
        result = chain.run(question)
        src_lang = "en"
        tgt_lang = "arz"
        resultar = translate_b(result, src_lang, tgt_lang)
        answer_label.config(text=resultar)
    else:
        answer_label.config(text="Please enter a question.")



# Create GUI
root = tk.Tk()
root.title("Question Answering System")

# Styling
style = ttk.Style()

# Configure the label style to display text like a block square
style.configure("TLabel", padding=6, background="#2E8B57", foreground="white")

# Adjusted layout
label = ttk.Label(root, text="Enter your question:")
label.grid(row=0, column=0, padx=10, pady=(20, 5))

entry = ttk.Entry(root, width=50)
entry.grid(row=0, column=1, padx=10, pady=(20, 5))

button = ttk.Button(root, text="Get Answer", command=get_answer)
button.grid(row=1, column=0, columnspan=2, pady=5)

# Configure the answer label style to display text like a block square with line breaks
answer_label = ttk.Label(root, text="", style="TLabel", wraplength=400, justify="left")
answer_label.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()