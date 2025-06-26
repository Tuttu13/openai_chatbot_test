import glob
import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# .envからAPIキー読み込み
api_key = os.getenv("OPENAI_API_KEY")

# Markdownファイル読み込み
paths = glob.glob("markdown_samples/*.md")
docs = []
for p in paths:
    loader = UnstructuredMarkdownLoader(p, mode="elements")
    docs.extend(loader.load())

# Markdownをチャンクに分割
splitter = MarkdownHeaderTextSplitter()
chunks = splitter.split_documents(docs)

# EmbeddingとベクトルDB生成
embeddings = OpenAIEmbeddings(api_key=api_key)
vectordb = Chroma.from_documents(chunks, embeddings)

# QAチェーン生成
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=api_key)
qa = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())

# クエリ実行
print(qa.run("VPN パスワードはどのタイミングで変更すべき？"))
