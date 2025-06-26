"""
RAG 用 Markdown → Chroma 取り込み & Retrieval-QA 実行スクリプト
---------------------------------------------------------------
* Markdown はスクリプトと同階層の *.md を対象
* ベクトル DB は ./chroma_db に永続化（初回のみ埋め込み）
* 実行例: python rag_qa.py "誰の情報ですか"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ----------------------------------------------------------------------
# ★ 0. 定数定義
# ----------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).parent
PERSIST_DIR: Path = BASE_DIR / "chroma_db"
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]  # チャンク単位

# ----------------------------------------------------------------------
# ★ 1. API キー読み込み
# ----------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY が未設定です。 .env もしくは環境変数にセットしてください。"
    )


# ----------------------------------------------------------------------
# ★ 2. ドキュメント → チャンク化ユーティリティ
# ----------------------------------------------------------------------
def load_markdown_chunks(files: Iterable[Path]) -> list[Document]:
    """
    Markdown ファイルを読み込み、ヘッダーでチャンク分割して返す。
    """
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS)
    chunks: list[Document] = []

    for md_file in files:
        doc = UnstructuredMarkdownLoader(md_file).load()[0]  # 1 ファイル = 1 Document
        for chunk in splitter.split_text(doc.page_content):
            chunk.metadata = {"source": str(md_file)}  # Chroma 用 (str 型必須)
            chunks.append(chunk)

    return chunks


# ----------------------------------------------------------------------
# ★ 3. ベクトル DB 構築 / 再利用
# ----------------------------------------------------------------------


def get_vectordb(chunks: list[Document]) -> Chroma:
    """
    * 既存フォルダがあっても **ドキュメント 0 件なら再構築**
    * 既存 DB に “新しい source” があれば **差分だけ add_documents()**
      - ここでは metadata["source"] 単位で重複判定
    * 何も追加が無ければそのまま再利用
    """
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # --- ① 既存 DB があるか判定 ---------------------------------
    if PERSIST_DIR.exists():
        vectordb = Chroma(
            persist_directory=str(PERSIST_DIR), embedding_function=embeddings
        )

        total = vectordb._collection.count()
        if total == 0:
            print("⚠️  既存フォルダはあるがレコード 0 件 → 再構築します")
            vectordb.delete_collection()  # フォルダを空に
        else:
            # --- ② 差分チェック（source メタデータで判定） ----------
            existing = {
                meta["source"]
                for meta in vectordb._collection.get(include=["metadatas"])["metadatas"]
            }

            new_chunks = [c for c in chunks if c.metadata["source"] not in existing]
            if new_chunks:
                print(f"🔄  新規 {len(new_chunks)} チャンクを追加")
                vectordb.add_documents(new_chunks)
                vectordb.persist()
            else:
                print("✅  既存 Chroma DB をそのまま再利用します")
            return vectordb

    # --- ③ ここに来たら「フォルダ無し」または「再構築」 ------------
    print("🆕  Chroma DB を新規構築しています…")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    vectordb.persist()
    print(f"✅  Chroma DB を保存しました（{len(chunks)} チャンク）")
    return vectordb


# ----------------------------------------------------------------------
# ★ 4. Retrieval-QA 実行
# ----------------------------------------------------------------------
def run_query(vectordb: Chroma, query: str) -> None:
    llm = ChatOpenAI(
        model_name=MODEL_NAME, temperature=TEMPERATURE, api_key=OPENAI_API_KEY
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
    )

    result = qa(query)
    print(f"\nQ: {query}\nA: {result['result']}\n")
    print("--- 引用元チャンク ---")
    for doc in result["source_documents"]:
        snippet = doc.page_content.replace("\n", " ")[:120]
        print(f"[{doc.metadata['source']}] {snippet} …")


# ----------------------------------------------------------------------
# ★ 5. エントリーポイント
# ----------------------------------------------------------------------
def main() -> None:
    # Markdown ファイル検出
    md_files = list(BASE_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"Markdown (*.md) が {BASE_DIR} に見つかりません")

    # ベクトル DB 用チャンク生成（初回のみ実質コスト）
    chunks = load_markdown_chunks(md_files)

    # ベクトル DB 準備
    vectordb = get_vectordb(chunks)

    # クエリは CLI 引数 or デフォルト
    query = sys.argv[1] if len(sys.argv) > 1 else "誰の情報ですか"
    run_query(vectordb, query)


if __name__ == "__main__":
    main()
