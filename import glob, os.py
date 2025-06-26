import glob
import os

from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# .envから安全にAPIキーを読み込む
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEYが設定されていません。.envファイルまたは環境変数に設定してください。"
    )

# Markdownファイル読み込み（ファイル名を明示的に指定）
markdown_file = "faq.md"
loader = UnstructuredMarkdownLoader(markdown_file, mode="elements")
docs = loader.load()

# Markdownをチャンクに分割（split_textを利用する必要あり）
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
    ]
)

# 各ドキュメントの内容を分割
chunks = []
for doc in docs:
    chunks.extend(splitter.split_text(doc.page_content))

# EmbeddingとベクトルDB生成
embeddings = OpenAIEmbeddings(api_key=api_key)
vectordb = Chroma.from_texts([chunk.page_content for chunk in chunks], embeddings)

# QAチェーン生成
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=api_key)
qa = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())

# クエリ実行
query = "誰の情報ですか"
answer = qa.run(query)
print(f"Q: {query}\nA: {answer}")
