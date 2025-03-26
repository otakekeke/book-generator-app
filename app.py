# アプリケーションのエントリーポイントを修正
import os
import sys
import gradio as gr
from improved_book_gen_with_cover import BookGenerator, create_ui

# App Runnerはポート8080を使用
port = int(os.environ.get("PORT", 8080))

if __name__ == "__main__":
    # UIを作成
    demo = create_ui()
    
    # 外部からアクセス可能にするための設定
    demo.launch(server_name="0.0.0.0", server_port=port)
