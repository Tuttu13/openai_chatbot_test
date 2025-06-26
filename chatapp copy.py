#!/usr/bin/env python
"""
simple_spell_checker.py
-----------------------
CLI から気軽に使える日本語スペルチェッカー。

使い方:
    # 事前に .env に OPENAI_API_KEY=sk-*** を設定
    # 1) 引数で渡す
    python simple_spell_checker.py "こんんんちわ、真純です。"

    # 2) パイプで渡す
    echo "こんんんちわ、真純です。" | python simple_spell_checker.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------
# 1. 環境変数を読み込む (.env があればそちらも優先)
# ---------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("❌ OPENAI_API_KEY が設定されていません (.env または環境変数を確認)")

# ---------------------------------------------------------
# 2. LangChain コンポーネントを組み立て
# ---------------------------------------------------------
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, api_key=API_KEY)

PROMPT_TEMPLATE = """
次の文章に誤字がないか調べて。誤字があれば訂正してください。

【入力】
{sentences_before_check}

【出力フォーマット】
- 訂正後の全文（誤字を **で囲んで強調**）
- 変更箇所の一覧（「誤り → 修正」の形で）
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", "あなたは優秀な日本語校正者です。"), ("user", PROMPT_TEMPLATE)]
)
parser = StrOutputParser()
chain = prompt | llm | parser


def spell_check(text: str) -> str:
    """OpenAI に校正を依頼し、結果をそのまま返す。"""
    return chain.invoke({"sentences_before_check": text})


# ---------------------------------------------------------
# 3. エントリーポイント
# ---------------------------------------------------------
def main() -> None:

    # 文章を取得
    target_text = "こんんんちわ、真純です。"

    # 校正実行
    print("🔍 校正中...\n")
    result = spell_check(target_text)
    print(result)


if __name__ == "__main__":
    main()
