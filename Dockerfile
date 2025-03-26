FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY improved_book_gen_with_cover.py .
COPY app.py .

# 必要なディレクトリを作成
RUN mkdir -p covers illustrations

# ポート8080を公開
EXPOSE 8080

# 起動コマンド
CMD ["python", "app.py"]
