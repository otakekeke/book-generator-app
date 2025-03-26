import os
import sys
import json
import time
import traceback
import markdown
import io
import base64
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# LangChain関連のインポート
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from openai import OpenAI

# Google検索API関連のインポート（オプション）
try:
    from googleapiclient.discovery import build
except ImportError:
    print("Google APIクライアントがインストールされていません。検索機能は無効になります。")
    print("インストールするには: pip install google-api-python-client")

# UI関連のインポート
try:
    import gradio as gr
except ImportError:
    print("Gradioがインストールされていません。UIは使用できません。")
    print("インストールするには: pip install gradio")

# 画像ダウンロード用
try:
    import requests
except ImportError:
    print("requestsがインストールされていません。表紙生成機能は制限されます。")
    print("インストールするには: pip install requests")

# PDF生成用
try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    WEASYPRINT_AVAILABLE = True
except ImportError:
    print("WeasyPrintがインストールされていません。PDF出力機能は無効になります。")
    print("インストールするには: pip install weasyprint")
    WEASYPRINT_AVAILABLE = False

class ImprovedBookGenerator:
    """改善版書籍生成クラス"""
    
    def __init__(self, 
                 openai_api_key: str = None, 
                 google_api_key: str = None,
                 search_engine_id: str = None,
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.7,
                 memory_token_limit: int = 500,
                 writing_style: str = "です・ます調"):
        """
        初期化メソッド
        
        Args:
            openai_api_key: OpenAI APIキー
            google_api_key: Google APIキー（検索機能用、オプション）
            search_engine_id: Google Custom Search Engine ID（検索機能用、オプション）
            model_name: 使用するLLMモデル名
            temperature: 生成時の温度パラメータ
            memory_token_limit: ConversationSummaryMemoryの最大トークン数
            writing_style: 文体スタイル（"です・ます調" または "だ・である調"）
        """
        # APIキーの設定
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.search_engine_id = search_engine_id or os.environ.get("SEARCH_ENGINE_ID", "")
        
        # パラメータの設定
        self.model_name = model_name
        self.temperature = temperature
        self.memory_token_limit = memory_token_limit
        self.writing_style = writing_style
        
        # LLMの初期化
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                temperature=self.temperature,
                model=self.model_name,
                api_key=self.openai_api_key
            )
            
            # ConversationSummaryMemoryの初期化
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                max_token_limit=self.memory_token_limit,
                return_messages=True
            )
            
            # 会話チェーンの初期化
            self.conversation_chain = ConversationChain(
                memory=self.memory,
                llm=self.llm,
                verbose=True
            )
            
            # OpenAI クライアントの初期化（表紙生成用）
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.llm = None
            self.memory = None
            self.conversation_chain = None
            self.openai_client = None
            print("警告: OpenAI APIキーが設定されていません。")
        
        # 出力ディレクトリの作成
        os.makedirs('covers', exist_ok=True)
        os.makedirs('illustrations', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        # 日本語フォントの設定
        # システムにインストールされている日本語フォントを探す
        self.font_path = self.find_japanese_font()
    
    def find_japanese_font(self):
        """システムにインストールされている日本語フォントを探す"""
        # 一般的な日本語フォントのパスリスト
        font_paths = {
            'title': None,
            'subtitle': None,
            'author': None
        }
        
        # 候補となる日本語フォントのパス
        japanese_font_candidates = [
            # Linux
            '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf',  # IPAゴシック
            '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',  # 日本語ゴシック
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc',  # Noto Sans CJK
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf',  # VLゴシック
            '/etc/alternatives/fonts-japanese-gothic.ttf',  # 日本語ゴシック代替
            # Windows
            'C:/Windows/Fonts/msgothic.ttc',  # MSゴシック
            'C:/Windows/Fonts/YuGothB.ttc',   # 游ゴシック
            'C:/Windows/Fonts/meiryo.ttc',    # メイリオ
            # macOS
            '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc',  # ヒラギノ
            '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
            '/Library/Fonts/Osaka.ttf'        # 大阪
        ]
        
        # 候補から存在するフォントを探す
        for font_path in japanese_font_candidates:
            if os.path.exists(font_path):
                # 見つかったフォントをすべての用途に設定
                font_paths['title'] = font_path
                font_paths['subtitle'] = font_path
                font_paths['author'] = font_path
                print(f"日本語フォントを見つけました: {font_path}")
                break
        
        # フォントが見つからなかった場合はデフォルトフォントを使用
        if font_paths['title'] is None:
            print("警告: 日本語フォントが見つかりませんでした。デフォルトフォントを使用します。")
            # デフォルトフォントを設定
            default_font = ImageFont.load_default()
            font_paths['title'] = default_font
            font_paths['subtitle'] = default_font
            font_paths['author'] = default_font
        
        return font_paths
    
    def search_information(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """指定されたクエリで情報を検索する"""
        if not self.google_api_key or not self.search_engine_id:
            print("警告: Google APIキーまたはSearch Engine IDが設定されていません。検索機能は使用できません。")
            return []
        
        try:
            # Google検索APIを使用
            service = build("customsearch", "v1", developerKey=self.google_api_key)
            res = service.cse().list(q=query, cx=self.search_engine_id, num=num_results).execute()
            
            search_results = []
            if "items" in res:
                for item in res["items"]:
                    search_results.append({
                        "title": item["title"],
                        "link": item["link"],
                        "snippet": item.get("snippet", "")
                    })
            
            return search_results
        except Exception as e:
            print("検索中にエラーが発生しました: {}".format(e))
            return []
    
    def generate_book_structure(self, 
                               book_title: str, 
                               genre: str = None, 
                               target_audience: str = None,
                               num_chapters: int = 10,
                               sections_per_chapter: int = 3) -> Dict[str, Any]:
        """
        書籍タイトルから章・節構成を自動生成する
        
        Args:
            book_title: 書籍のタイトル
            genre: 書籍のジャンル（オプション）
            target_audience: 対象読者層（オプション）
            num_chapters: 生成する章の数
            sections_per_chapter: 各章に含める節の数
            
        Returns:
            章・節構成を含む辞書
        """
        if not self.conversation_chain:
            raise ValueError("OpenAI APIキーが設定されていないため、構成を生成できません。")
        
        # プロンプトの作成
        prompt_template = """
        あなたは優れた書籍構成の専門家です。以下の情報に基づいて、書籍の章・節構成を作成してください。

        書籍タイトル: {book_title}
        """
        
        prompt = prompt_template.format(book_title=book_title)
        
        if genre:
            prompt += "\nジャンル: {}".format(genre)
        
        if target_audience:
            prompt += "\n対象読者層: {}".format(target_audience)
        
        prompt_conditions = """
        
        以下の条件に従って構成を作成してください：
        1. 章の数は{num_chapters}章とします。
        2. 各章には{sections_per_chapter}つの節を含めてください。
        3. 章と節には明確で具体的なタイトルをつけてください。タイトルは単調な表現を避け、魅力的で興味を引くものにしてください。
        4. 章と節の構成は論理的で一貫性があり、書籍全体として流れが自然になるようにしてください。
        5. 各章と節の内容を簡潔に説明してください。
        6. 書籍タイトルに含まれる単語を章や節のタイトルで何度も繰り返さないでください。多様な表現を使用してください。
        7. 章や節のタイトルは、一般的な表現ではなく、具体的で独自性のあるものにしてください。
        8. 書籍のタイトルと内容に密接に関連した、読者の興味を引く構成にしてください。
        
        以下のJSON形式で出力してください：
        ```json
        {{
          "title": "書籍タイトル",
          "chapters": [
            {{
              "number": 1,
              "title": "第1章のタイトル",
              "description": "第1章の簡単な説明",
              "sections": [
                {{
                  "number": "1.1",
                  "title": "第1節のタイトル",
                  "description": "第1節の簡単な説明"
                }},
                ...
              ]
            }},
            ...
          ]
        }}
        ```
        
        JSONのみを出力し、他の説明は不要です。
        """
        
        prompt += prompt_conditions.format(
            num_chapters=num_chapters,
            sections_per_chapter=sections_per_chapter
        )
        
        # LLMを使用して構成を生成
        response = self.conversation_chain.predict(input=prompt)
        
        # JSONを抽出
        try:
            # コードブロックから抽出
            if "```json" in response and "```" in response.split("```json")[1]:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response and "```" in response.split("```")[1]:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            # JSONをパース
            structure = json.loads(json_str)
            
            # 構造を検証
            if "title" not in structure or "chapters" not in structure:
                raise ValueError("生成された構造が無効です。")
            
            return structure
        except Exception as e:
            print("構造の解析中にエラーが発生しました: {}".format(e))
            print("生成された応答: {}".format(response))
            raise ValueError("書籍構成の生成に失敗しました。")
    
    def generate_chapter_content(self, book_title: str, chapter_title: str, chapter_number: int, 
                                sections: List[Dict[str, Any]] = None, previous_chapters: List[Dict[str, Any]] = None,
                                use_search: bool = False, search_results_per_chapter: int = 5) -> str:
        """章の内容を生成する"""
        if not self.conversation_chain:
            raise ValueError("OpenAI APIキーが設定されていないため、章を生成できません。")
        
        # 検索結果を取得（オプション）
        search_results = []
        if use_search and self.google_api_key and self.search_engine_id:
            search_query = "{} {}".format(book_title, chapter_title)
            search_results = self.search_information(search_query, search_results_per_chapter)
        
        # 前の章の要約を作成
        previous_chapters_summary = ""
        if previous_chapters and len(previous_chapters) > 0:
            previous_chapters_summary = "前の章の要約:\n"
            for prev_chapter in previous_chapters[-3:]:  # 直近3章のみ
                previous_chapters_summary += "- 第{}章「{}」: ".format(prev_chapter['number'], prev_chapter['title'])
                # 内容の最初の200文字程度を要約として使用
                content_summary = prev_chapter['content'][:200] + "..." if len(prev_chapter['content']) > 200 else prev_chapter['content']
                previous_chapters_summary += "{}\n".format(content_summary)
        
        # 検索結果の情報を整形
        search_info = ""
        if search_results:
            search_info = "参考情報:\n"
            for result in search_results:
                search_info += "- {}: {}\n".format(result['title'], result['snippet'])
        
        # 節の情報を整形
        sections_info = ""
        if sections and len(sections) > 0:
            sections_info = "この章に含める節:\n"
            for section in sections:
                sections_info += "- {} {}: {}\n".format(section['number'], section['title'], section.get('description', ''))
        
        # 文体スタイルの設定
        style_instruction = "です・ます調（ですます調）で書いてください。" if self.writing_style == "です・ます調" else "だ・である調で書いてください。"
        
        # プロンプトの作成
        prompt_template = """
        あなたは「{book_title}」という本の内容を執筆しています。現在は第{chapter_number}章「{chapter_title}」を担当しています。
        
        {previous_chapters_summary}
        
        {sections_info}
        
        {search_info}
        
        以下の指示に従って章の内容を執筆してください：
        
        1. 指定された節の構造に従って内容を構成してください。節のタイトルはマークダウンの見出し（##）で表示してください。
        2. 必要に応じて小節（###）を使用して内容を整理してください。ただし、指定された節構造以外の新しい節番号は作成しないでください。
        3. 専門的な内容でも一般読者が理解できるよう、明確で分かりやすい言葉を使ってください。
        4. 各節は最低でも800〜1000文字以上の充実した内容にしてください。
        5. 具体例や比喩を用いて説明を補強してください。実際の事例、数値データ、歴史的背景なども積極的に取り入れてください。
        6. 各節の内容は、導入（概念の紹介）→本論（詳細な説明）→応用または発展（実践的な使い方や将来展望）という流れで構成してください。
        7. 読者の理解を深めるために、適宜、図表の説明や箇条書きを使用してください。
        8. 重要な概念や用語は太字（**太字**）で強調してください。
        9. 数式が必要な場合は、LaTeX形式（$$ ... $$）で記述してください。
        10. 各節の最後に、その節で学んだ内容の要点をまとめてください。
        11. {style_instruction}
        12. マークダウン形式で出力してください。
        13. 章タイトルは絶対に出力しないでください。章タイトルは別途追加されます。
        14. 「第X章」という表記も絶対に含めないでください。
        15. 章番号や「第X章:」などの表記は一切含めないでください。
        
        節の内容のみを出力し、章タイトルや章番号は絶対に含めないでください。出力の先頭は必ず節のタイトル（## X.X 節タイトル）から始めてください。
        """
        
        # プロンプトをフォーマット
        prompt = prompt_template.format(
            book_title=book_title,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            previous_chapters_summary=previous_chapters_summary,
            sections_info=sections_info,
            search_info=search_info,
            style_instruction=style_instruction
        )
        
        # LLMを使用して内容を生成
        response = self.conversation_chain.predict(input=prompt)
        
        # 章タイトルが含まれている場合は削除
        chapter_title_patterns = [
            r"^#\s+第{}章[:：].*?\n".format(chapter_number),
            r"^#\s+第{}章\s+.*?\n".format(chapter_number),
            r"^#\s+{}[:：].*?\n".format(chapter_title),
            r"^#\s+{}.*?\n".format(chapter_title),
            r"^第{}章[:：].*?\n".format(chapter_number),
            r"^第{}章\s+.*?\n".format(chapter_number)
        ]
        
        for pattern in chapter_title_patterns:
            response = re.sub(pattern, "", response, flags=re.MULTILINE)
        
        # 「第X章」という表記が含まれている場合も削除
        chapter_prefix_patterns = [
            r"第{}章[:：]\s*".format(chapter_number),
            r"第{}章\s+".format(chapter_number)
        ]
        
        for pattern in chapter_prefix_patterns:
            response = re.sub(pattern, "", response)
        
        return response
    
    def generate_cover_image(self, book_title, book_content=None, prompt=None, size="1024x1024", quality="hd"):
        """
        書籍の表紙画像を生成（内容に関連する画像を生成）
        
        Parameters:
        -----------
        book_title : str
            書籍のタイトル
        book_content : dict, optional
            書籍の内容（章や節の情報を含む辞書）
        prompt : str, optional
            画像生成のためのプロンプト。指定しない場合はタイトルと内容から自動生成
        size : str, optional
            画像サイズ。"1024x1024", "1792x1024", "1024x1792"のいずれか
        quality : str, optional
            画像品質。"hd"または"standard"
            
        Returns:
        --------
        str
            生成された画像のパス
        """
        if not self.openai_client:
            raise ValueError("APIキーが設定されていません")
        
        # プロンプトが指定されていない場合、タイトルと内容から自動生成
        if not prompt:
            # タイトルから不要な単語を除去
            clean_title = book_title.replace("入門", "").replace("ガイド", "").replace("講座", "").replace("解説", "")
            
            # 基本プロンプト
            prompt = "「{}」の内容を表現する芸術的なイメージ。".format(clean_title)
            
            # 書籍の内容が提供されている場合、それを活用してプロンプトを強化
            if book_content and "chapters" in book_content:
                # 最初の3章のタイトルから主要キーワードを抽出
                keywords = []
                for i, chapter in enumerate(book_content["chapters"][:3]):
                    # 章タイトルから「第X章:」などの表記を除去
                    clean_chapter_title = re.sub(r"^第\d+章[:：]?\s*", "", chapter["title"])
                    
                    # キーワードを追加（重複を避ける）
                    for word in clean_chapter_title.split():
                        if word not in keywords and word not in clean_title:
                            keywords.append(word)
                
                # キーワードを最大3つまで使用
                if keywords:
                    prompt += " キーワード: {}。".format("、".join(keywords[:3]))
            
            # 表紙らしさを強調せず、内容に関連する画像を生成
            # 虹色や派手な画像を避け、内容に基づいた自然なデザインにする
            prompt += " 落ち着いた色調で自然な構図。内容を象徴的に表現してください。テキストや文字は含めないでください。"
        
        # DALL-E APIを使用して画像を生成
        response = self.openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
        
        # 画像URLを取得
        image_url = response.data[0].url
        
        # 画像をダウンロード
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        
        # 画像を保存
        filename = "covers/{}_cover_raw.png".format(book_title.replace(' ', '_'))
        image.save(filename)
        
        return filename
    
    def add_text_to_cover(self, image_path, title, subtitle=None, author=None, output_path=None):
        """
        表紙画像にテキストを追加（改善版）
        
        Parameters:
        -----------
        image_path : str
            表紙画像のパス
        title : str
            書籍のタイトル
        subtitle : str, optional
            サブタイトル
        author : str, optional
            著者名
        output_path : str, optional
            出力先のパス。指定しない場合は自動生成
            
        Returns:
        --------
        str
            テキストが追加された画像のパス
        """
        try:
            # 画像を開く
            image = Image.open(image_path)
            
            # 画像サイズを取得
            width, height = image.size
            
            # 画像サイズに基づいてフォントサイズを動的に計算
            # 画像の幅に対する比率でフォントサイズを決定（比率を大きくして文字サイズを拡大）
            font_size = {
                'title': int(width * 0.12),  # 画像幅の12%（従来の8%から拡大）
                'subtitle': int(width * 0.08),  # 画像幅の8%（従来の5%から拡大）
                'author': int(width * 0.06)   # 画像幅の6%（従来の4%から拡大）
            }
            
            # 縁取りの太さも画像サイズに応じて調整（より太く）
            outline_thickness = max(4, int(width * 0.008))  # 最小4px、画像幅の0.8%（従来の0.5%から拡大）
            
            # 背景を少し暗くして文字を見やすくする
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            # 上部と下部に暗いグラデーションを追加
            for y in range(height // 3):
                alpha = int(150 * (1 - y / (height // 3)))  # 上部のグラデーション
                overlay_draw.rectangle([(0, y), (width, y)], fill=(0, 0, 0, alpha))
            
            for y in range(height // 3):
                alpha = int(150 * (y / (height // 3)))  # 下部のグラデーション
                overlay_draw.rectangle([(0, height - y - 1), (width, height - y - 1)], fill=(0, 0, 0, alpha))
            
            # オーバーレイを元の画像に合成
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            image = Image.alpha_composite(image, overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # タイトルのフォントを設定
            try:
                if isinstance(self.font_path['title'], str):
                    title_font = ImageFont.truetype(self.font_path['title'], font_size['title'])
                else:
                    # デフォルトフォントの場合
                    title_font = self.font_path['title']
            except Exception as e:
                print(f"フォント読み込みエラー: {e}")
                # フォントが見つからない場合はデフォルトフォントを使用
                print("警告: 指定されたフォントが見つかりません。デフォルトフォントを使用します。")
                title_font = ImageFont.load_default()
            
            # タイトルが長い場合は複数行に分割
            title_lines = []
            
            # タイトルから不要な表記を削除
            clean_title = re.sub(r"第\d+章[:：]?\s*", "", title)
            
            # 単語ごとに分割
            words = clean_title.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                try:
                    # PILのバージョンによって異なる方法でテキストサイズを取得
                    try:
                        bbox = draw.textbbox((0, 0), test_line, font=title_font)
                        text_width = bbox[2] - bbox[0]
                    except AttributeError:
                        # 古いバージョンのPILの場合
                        text_width, _ = draw.textsize(test_line, font=title_font)
                    
                    if text_width < width * 0.9:  # 画像幅の90%以内に収まる場合
                        current_line = test_line
                    else:
                        title_lines.append(current_line)
                        current_line = word
                except Exception as e:
                    print(f"テキストボックス計算エラー: {e}")
                    # エラーが発生した場合は単語ごとに改行
                    if current_line:
                        title_lines.append(current_line)
                    current_line = word
            
            if current_line:
                title_lines.append(current_line)
            
            # タイトルがない場合は元のタイトルをそのまま使用
            if not title_lines:
                title_lines = [clean_title]
            
            # タイトルの描画位置を計算
            try:
                # PILのバージョンによって異なる方法でテキストサイズを取得
                title_height_total = 0
                for line in title_lines:
                    try:
                        bbox = draw.textbbox((0, 0), line, font=title_font)
                        line_height = bbox[3] - bbox[1]
                    except AttributeError:
                        # 古いバージョンのPILの場合
                        _, line_height = draw.textsize(line, font=title_font)
                    title_height_total += line_height
                
                title_spacing = int(font_size['title'] * 0.2)  # フォントサイズの20%を行間に
                title_total_height = title_height_total + (len(title_lines) - 1) * title_spacing
                
                # タイトルの開始Y座標（上部1/4の位置）
                title_start_y = height // 4 - title_total_height // 2
                
                # タイトルを描画（各行ごとに）
                current_y = title_start_y
                for line in title_lines:
                    # テキストボックスを取得
                    try:
                        bbox = draw.textbbox((0, 0), line, font=title_font)
                        line_width = bbox[2] - bbox[0]
                        line_height = bbox[3] - bbox[1]
                    except AttributeError:
                        # 古いバージョンのPILの場合
                        line_width, line_height = draw.textsize(line, font=title_font)
                    
                    # 行を中央に配置
                    line_position = ((width - line_width) // 2, current_y)
                    
                    # 縁取りを強化（より太く）
                    for dx in range(-outline_thickness, outline_thickness + 1):
                        for dy in range(-outline_thickness, outline_thickness + 1):
                            if dx != 0 or dy != 0:  # 中心点以外
                                draw.text((line_position[0] + dx, line_position[1] + dy), line, font=title_font, fill=(0, 0, 0))
                    
                    # メインテキスト
                    draw.text(line_position, line, font=title_font, fill=(255, 255, 255))
                    
                    # 次の行のY座標を更新
                    current_y += line_height + title_spacing
            except Exception as e:
                print(f"タイトル描画エラー: {e}")
                # エラーが発生した場合は単純な方法でタイトルを描画
                try:
                    simple_position = (width // 10, height // 4)
                    draw.text(simple_position, clean_title, font=title_font, fill=(255, 255, 255))
                    current_y = simple_position[1] + font_size['title'] * 1.5
                except Exception as e2:
                    print(f"単純タイトル描画エラー: {e2}")
                    current_y = height // 3
            
            # サブタイトルがある場合
            if subtitle:
                try:
                    if isinstance(self.font_path['subtitle'], str):
                        subtitle_font = ImageFont.truetype(self.font_path['subtitle'], font_size['subtitle'])
                    else:
                        # デフォルトフォントの場合
                        subtitle_font = self.font_path['subtitle']
                except Exception as e:
                    print(f"サブタイトルフォント読み込みエラー: {e}")
                    subtitle_font = ImageFont.load_default()
                
                try:
                    # サブタイトルから不要な表記を削除
                    clean_subtitle = re.sub(r"第\d+章[:：]?\s*", "", subtitle)
                    
                    # サブタイトルが長い場合は複数行に分割
                    subtitle_lines = []
                    current_line = ""
                    words = clean_subtitle.split()
                    
                    for word in words:
                        test_line = current_line + " " + word if current_line else word
                        try:
                            bbox = draw.textbbox((0, 0), test_line, font=subtitle_font)
                            text_width = bbox[2] - bbox[0]
                        except AttributeError:
                            # 古いバージョンのPILの場合
                            text_width, _ = draw.textsize(test_line, font=subtitle_font)
                        
                        if text_width < width * 0.8:  # 画像幅の80%以内に収まる場合
                            current_line = test_line
                        else:
                            subtitle_lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        subtitle_lines.append(current_line)
                    
                    # サブタイトルがない場合は元のサブタイトルをそのまま使用
                    if not subtitle_lines:
                        subtitle_lines = [clean_subtitle]
                    
                    # サブタイトルの描画位置を計算
                    subtitle_height_total = 0
                    for line in subtitle_lines:
                        try:
                            bbox = draw.textbbox((0, 0), line, font=subtitle_font)
                            line_height = bbox[3] - bbox[1]
                        except AttributeError:
                            # 古いバージョンのPILの場合
                            _, line_height = draw.textsize(line, font=subtitle_font)
                        subtitle_height_total += line_height
                    
                    subtitle_spacing = int(font_size['subtitle'] * 0.2)  # フォントサイズの20%を行間に
                    subtitle_total_height = subtitle_height_total + (len(subtitle_lines) - 1) * subtitle_spacing
                    
                    # サブタイトルの開始Y座標（タイトルの下）
                    subtitle_start_y = current_y + int(font_size['title'] * 0.5)  # タイトルの下に適切な間隔
                    
                    # サブタイトルを描画（各行ごとに）
                    current_y = subtitle_start_y
                    for line in subtitle_lines:
                        # テキストボックスを取得
                        try:
                            bbox = draw.textbbox((0, 0), line, font=subtitle_font)
                            line_width = bbox[2] - bbox[0]
                            line_height = bbox[3] - bbox[1]
                        except AttributeError:
                            # 古いバージョンのPILの場合
                            line_width, line_height = draw.textsize(line, font=subtitle_font)
                        
                        # 行を中央に配置
                        line_position = ((width - line_width) // 2, current_y)
                        
                        # 縁取り
                        outline_thickness_subtitle = max(2, outline_thickness - 1)  # サブタイトルは少し細め
                        for dx in range(-outline_thickness_subtitle, outline_thickness_subtitle + 1):
                            for dy in range(-outline_thickness_subtitle, outline_thickness_subtitle + 1):
                                if dx != 0 or dy != 0:  # 中心点以外
                                    draw.text((line_position[0] + dx, line_position[1] + dy), line, font=subtitle_font, fill=(0, 0, 0))
                        
                        # メインテキスト
                        draw.text(line_position, line, font=subtitle_font, fill=(255, 255, 255))
                        
                        # 次の行のY座標を更新
                        current_y += line_height + subtitle_spacing
                except Exception as e:
                    print(f"サブタイトル描画エラー: {e}")
                    # エラーが発生した場合は単純な方法でサブタイトルを描画
                    try:
                        simple_position = (width // 10, current_y)
                        draw.text(simple_position, clean_subtitle, font=subtitle_font, fill=(255, 255, 255))
                        current_y = simple_position[1] + font_size['subtitle'] * 1.5
                    except Exception as e2:
                        print(f"単純サブタイトル描画エラー: {e2}")
            
            # 著者名がある場合
            if author:
                try:
                    if isinstance(self.font_path['author'], str):
                        author_font = ImageFont.truetype(self.font_path['author'], font_size['author'])
                    else:
                        # デフォルトフォントの場合
                        author_font = self.font_path['author']
                except Exception as e:
                    print(f"著者名フォント読み込みエラー: {e}")
                    author_font = ImageFont.load_default()
                
                try:
                    author_text = "著者: {}".format(author)
                    
                    # 著者名のテキストボックスを取得
                    try:
                        bbox = draw.textbbox((0, 0), author_text, font=author_font)
                        author_width = bbox[2] - bbox[0]
                        author_height = bbox[3] - bbox[1]
                    except AttributeError:
                        # 古いバージョンのPILの場合
                        author_width, author_height = draw.textsize(author_text, font=author_font)
                    
                    # 著者名を下部に配置
                    author_position = ((width - author_width) // 2, height - author_height - int(height * 0.05))
                    
                    # 縁取り
                    outline_thickness_author = max(1, outline_thickness - 2)  # 著者名は少し細め
                    for dx in range(-outline_thickness_author, outline_thickness_author + 1):
                        for dy in range(-outline_thickness_author, outline_thickness_author + 1):
                            if dx != 0 or dy != 0:  # 中心点以外
                                draw.text((author_position[0] + dx, author_position[1] + dy), author_text, font=author_font, fill=(0, 0, 0))
                    
                    # メインテキスト
                    draw.text(author_position, author_text, font=author_font, fill=(255, 255, 255))
                except Exception as e:
                    print(f"著者名描画エラー: {e}")
                    # エラーが発生した場合は単純な方法で著者名を描画
                    try:
                        simple_position = (width // 10, height - int(height * 0.1))
                        draw.text(simple_position, author_text, font=author_font, fill=(255, 255, 255))
                    except Exception as e2:
                        print(f"単純著者名描画エラー: {e2}")
            
            # 出力パスが指定されていない場合は自動生成
            if not output_path:
                output_path = "covers/{}_cover_final.png".format(title.replace(' ', '_'))
            
            # 画像を保存
            image.save(output_path)
            
            return output_path
        except Exception as e:
            print(f"表紙テキスト追加中にエラーが発生しました: {e}")
            traceback.print_exc()
            # エラーが発生した場合は元の画像をそのまま返す
            return image_path
    
    def generate_illustration(self, description, chapter_title=None, chapter_number=None, size="1024x1024", quality="hd"):
        """
        挿絵を生成
        
        Parameters:
        -----------
        description : str
            挿絵の説明
        chapter_title : str, optional
            章のタイトル
        chapter_number : int, optional
            章番号
        size : str, optional
            画像サイズ。"1024x1024", "1792x1024", "1024x1792"のいずれか
        quality : str, optional
            画像品質。"hd"または"standard"
            
        Returns:
        --------
        str
            生成された画像のパス
        """
        if not self.openai_client:
            raise ValueError("APIキーが設定されていません")
        
        # 章タイトルから不要な表記を削除
        if chapter_title:
            clean_chapter_title = re.sub(r"第\d+章[:：]?\s*", "", chapter_title)
        else:
            clean_chapter_title = None
        
        # プロンプトを作成
        if clean_chapter_title:
            prompt = "「{}」の内容を表現する芸術的なイメージ。".format(clean_chapter_title)
        else:
            prompt = "「{}」の内容を表現する芸術的なイメージ。".format(description)
        
        # 挿絵らしさを強調せず、内容に関連する画像を生成
        # 虹色や派手な画像を避け、章の内容に基づいた適切なデザインにする
        prompt += " 落ち着いた色調で自然な構図。内容を象徴的に表現してください。テキストや文字は含めないでください。"
        
        # DALL-E APIを使用して画像を生成
        response = self.openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
        
        # 画像URLを取得
        image_url = response.data[0].url
        
        # 画像をダウンロード
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        
        # ファイル名を生成
        if chapter_title and chapter_number:
            filename = "illustrations/chapter_{}_illustration.png".format(chapter_number)
        elif chapter_title:
            filename = "illustrations/{}_illustration.png".format(clean_chapter_title.replace(' ', '_'))
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "illustrations/illustration_{}.png".format(timestamp)
        
        # 画像を保存
        image.save(filename)
        
        return filename
    
    def image_to_base64(self, image_path):
        """
        画像をBase64エンコードされた文字列に変換
        
        Parameters:
        -----------
        image_path : str
            画像のパス
            
        Returns:
        --------
        str
            Base64エンコードされた画像データ
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    def generate_book(self, book_structure: Dict[str, Any], 
                     use_search: bool = False, 
                     search_results_per_chapter: int = 5,
                     generate_cover: bool = True,
                     generate_illustrations: bool = True,
                     author_name: str = None) -> Dict[str, Any]:
        """書籍全体を生成する"""
        if not self.conversation_chain:
            raise ValueError("OpenAI APIキーが設定されていないため、書籍を生成できません。")
        
        book_title = book_structure["title"]
        chapters = book_structure["chapters"]
        
        # 表紙の生成（オプション）
        cover_path = None
        if generate_cover and self.openai_client:
            try:
                print("「{}」の表紙を生成中...".format(book_title))
                raw_cover_path = self.generate_cover_image(book_title, book_structure)
                
                # サブタイトルの生成（オプション）
                subtitle = None
                if len(chapters) > 0:
                    # 最初の章のタイトルをサブタイトルとして使用
                    subtitle = chapters[0]["title"]
                
                cover_path = self.add_text_to_cover(
                    raw_cover_path, 
                    book_title, 
                    subtitle=subtitle, 
                    author=author_name
                )
                print("表紙の生成が完了しました: {}".format(cover_path))
            except Exception as e:
                print("表紙の生成中にエラーが発生しました: {}".format(e))
                cover_path = None
        
        # 生成された章の内容を保存
        generated_chapters = []
        illustrations = {}
        
        print("「{}」の生成を開始します...".format(book_title))
        
        # 各章を生成
        for i, chapter in enumerate(chapters):
            chapter_number = chapter["number"]
            chapter_title = chapter["title"]
            
            # 節の情報を取得（存在する場合）
            sections = chapter.get("sections", [])
            
            print("第{}章「{}」を生成中...".format(chapter_number, chapter_title))
            
            # 章の内容を生成
            chapter_content = self.generate_chapter_content(
                book_title=book_title,
                chapter_title=chapter_title,
                chapter_number=chapter_number,
                sections=sections,
                previous_chapters=generated_chapters,
                use_search=use_search,
                search_results_per_chapter=search_results_per_chapter
            )
            
            # 挿絵の生成（オプション）
            illustration_path = None
            if generate_illustrations and self.openai_client:
                try:
                    print("第{}章「{}」の挿絵を生成中...".format(chapter_number, chapter_title))
                    # 章の内容から挿絵の説明を生成
                    illustration_description = "「{}」の内容を表現する挿絵。".format(chapter_title)
                    if chapter.get("description"):
                        illustration_description += " {}".format(chapter["description"])
                    
                    illustration_path = self.generate_illustration(
                        description=illustration_description,
                        chapter_title=chapter_title,
                        chapter_number=chapter_number
                    )
                    illustrations[chapter_number] = illustration_path
                    print("挿絵の生成が完了しました: {}".format(illustration_path))
                except Exception as e:
                    print("挿絵の生成中にエラーが発生しました: {}".format(e))
            
            # 生成された章を保存
            generated_chapter = {
                "number": chapter_number,
                "title": chapter_title,
                "content": chapter_content,
                "sections": sections,
                "illustration": illustration_path
            }
            generated_chapters.append(generated_chapter)
            
            print("第{}章「{}」の生成が完了しました。".format(chapter_number, chapter_title))
            
            # APIレート制限を考慮して少し待機
            time.sleep(1)
        
        # 生成された書籍の内容を返す
        book_content = {
            "title": book_title,
            "chapters": generated_chapters,
            "cover_path": cover_path,
            "illustrations": illustrations
        }
        
        return book_content
    
    def save_book_content(self, book_content: Dict[str, Any], output_dir: str = "output") -> Tuple[str, str, str]:
        """生成された書籍の内容を保存する（HTML、マークダウン、PDF）"""
        # 出力ディレクトリの作成
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像ディレクトリの作成（HTMLからの相対パス用）
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 表紙と挿絵を画像ディレクトリにコピー
        cover_rel_path = None
        if 'cover_path' in book_content and book_content['cover_path']:
            cover_src = book_content['cover_path']
            cover_dest = images_dir / Path(cover_src).name
            try:
                import shutil
                shutil.copy2(cover_src, cover_dest)
                cover_rel_path = f"images/{Path(cover_src).name}"
            except Exception as e:
                print(f"表紙のコピー中にエラーが発生しました: {e}")
        
        # 挿絵をコピー
        illustration_rel_paths = {}
        if 'illustrations' in book_content:
            for chapter_num, illust_path in book_content['illustrations'].items():
                if illust_path:
                    try:
                        import shutil
                        illust_dest = images_dir / Path(illust_path).name
                        shutil.copy2(illust_path, illust_dest)
                        illustration_rel_paths[chapter_num] = f"images/{Path(illust_path).name}"
                    except Exception as e:
                        print(f"挿絵のコピー中にエラーが発生しました: {e}")
        
        # マークダウンファイルのパス
        md_path = output_dir / "{}.md".format(book_content['title'].replace(' ', '_'))
        
        # マークダウンファイルに書き込み
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# {}\n\n".format(book_content['title']))
            
            # 表紙画像がある場合は追加
            if cover_rel_path:
                f.write("![表紙]({})\n\n".format(cover_rel_path))
            
            # 目次の作成
            f.write("## 目次\n\n")
            for chapter in book_content["chapters"]:
                f.write("- [第{}章: {}](#chapter-{})\n".format(
                    chapter['number'], 
                    chapter['title'], 
                    chapter['number']
                ))
            f.write("\n")
            
            # 各章の内容を書き込み
            for chapter in book_content["chapters"]:
                chapter_id = "chapter-{}".format(chapter['number'])
                f.write('<h2 id="{}">第{}章: {}</h2>\n\n'.format(
                    chapter_id, chapter['number'], chapter['title']
                ))
                
                # 挿絵がある場合は追加
                chapter_num = chapter['number']
                if chapter_num in illustration_rel_paths:
                    f.write("![第{}章の挿絵]({})\n\n".format(chapter_num, illustration_rel_paths[chapter_num]))
                
                f.write("{}\n\n".format(chapter['content']))
        
        # HTMLファイルのパス
        html_path = output_dir / "{}.html".format(book_content['title'].replace(' ', '_'))
        
        # HTMLテンプレート
        html_template = """<!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{ font-family: 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', 'Hiragino Sans', Meiryo, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                h3 {{ color: #2980b9; }}
                .cover-image {{ text-align: center; margin: 20px 0; }}
                .cover-image img {{ max-width: 100%; height: auto; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
                .illustration {{ text-align: center; margin: 20px 0; }}
                .illustration img {{ max-width: 100%; height: auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .toc {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .toc h2 {{ margin-top: 0; }}
                .toc ul {{ list-style-type: none; padding-left: 15px; }}
                .chapter {{ margin-bottom: 40px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                code {{ font-family: Consolas, Monaco, 'Andale Mono', monospace; }}
                blockquote {{ border-left: 4px solid #ddd; padding-left: 15px; color: #777; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
                .math {{ font-style: italic; }}
                @media print {{
                    body {{ font-size: 12pt; }}
                    pre, code {{ font-size: 10pt; }}
                    .no-print {{ display: none; }}
                    a {{ text-decoration: none; color: #000; }}
                    h1, h2, h3, h4, h5, h6 {{ page-break-after: avoid; }}
                    img {{ max-width: 100% !important; }}
                    .cover-image img {{ max-height: 90vh; }}
                    .new-page {{ page-break-before: always; }}
                }}
            </style>
            <!-- MathJax for LaTeX support -->
            <script>
            MathJax = {{
                tex: {{
                    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                    processEscapes: true,
                    processEnvironments: true
                }},
                options: {{
                    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
                }}
            }};
            </script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        </head>
        <body>
            <h1>{title}</h1>
            
            {cover_html}
            
            <div class="toc">
                <h2>目次</h2>
                {toc}
            </div>
            
            <div class="content">
                {content}
            </div>
        </body>
        </html>
        """
        
        # 表紙画像のHTML
        cover_html = ""
        if cover_rel_path:
            cover_html = '<div class="cover-image"><img src="{}" alt="表紙"></div>'.format(cover_rel_path)
        
        # 目次のHTML
        toc_html = "<ul>"
        for chapter in book_content["chapters"]:
            chapter_id = "chapter-{}".format(chapter['number'])
            toc_html += '<li><a href="#{0}">第{1}章: {2}</a></li>'.format(
                chapter_id, chapter["number"], chapter["title"]
            )
        toc_html += "</ul>"
        
        # 章のHTML
        chapters_html = ""
        for chapter in book_content["chapters"]:
            chapter_id = "chapter-{}".format(chapter['number'])
            chapter_html = '<div class="chapter new-page" id="{}">'.format(chapter_id)
            chapter_html += '<h2>第{}章: {}</h2>'.format(chapter["number"], chapter["title"])
            
            # 挿絵がある場合は追加
            chapter_num = chapter['number']
            if chapter_num in illustration_rel_paths:
                chapter_html += '<div class="illustration"><img src="{}" alt="第{}章の挿絵"></div>'.format(
                    illustration_rel_paths[chapter_num], chapter_num
                )
            
            # マークダウンをHTMLに変換
            chapter_content_html = markdown.markdown(
                chapter["content"],
                extensions=['extra', 'codehilite', 'tables']
            )
            
            chapter_html += chapter_content_html
            chapter_html += '</div>'
            
            chapters_html += chapter_html
        
        # HTMLを生成
        html_content = html_template.format(
            title=book_content["title"],
            cover_html=cover_html,
            toc=toc_html,
            content=chapters_html
        )
        
        # HTMLファイルに書き込み
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # PDFファイルのパス
        pdf_path = output_dir / "{}.pdf".format(book_content['title'].replace(' ', '_'))
        
        # PDFを生成（WeasyPrintが利用可能な場合）
        if WEASYPRINT_AVAILABLE:
            try:
                # フォント設定
                font_config = FontConfiguration()
                
                # HTMLからPDFを生成
                html = HTML(filename=str(html_path))
                css = CSS(string='''
                    @page {
                        size: A4;
                        margin: 2cm;
                    }
                    @page:first {
                        margin-top: 0;
                    }
                    h1 { page-break-before: always; }
                    h2 { page-break-before: always; }
                    .cover-image { page-break-after: always; }
                    .cover-image img { max-height: 25cm; }
                ''', font_config=font_config)
                
                # PDFを保存
                html.write_pdf(pdf_path, stylesheets=[css], font_config=font_config)
                print("PDFファイルを生成しました: {}".format(pdf_path))
            except Exception as e:
                print("PDFの生成中にエラーが発生しました: {}".format(e))
                traceback.print_exc()
                pdf_path = None
        else:
            print("WeasyPrintが利用できないため、PDFは生成されませんでした。")
            pdf_path = None
        
        print("マークダウンファイルを生成しました: {}".format(md_path))
        print("HTMLファイルを生成しました: {}".format(html_path))
        
        return str(md_path), str(html_path), str(pdf_path) if pdf_path else None

# Gradio UIの作成
def create_ui():
    """Gradio UIを作成する"""
    
    # 状態の初期化
    initial_state = {
        "openai_api_key": "",
        "google_api_key": "",
        "search_engine_id": "",
        "generator": None,
        "book_structure": None,
        "book_content": None,
        "output_files": None
    }
    
    # CSSスタイル
    css = """
    .container { max-width: 1200px; margin: auto; }
    .output-box { min-height: 300px; }
    .status-box { min-height: 100px; }
    .api-key-input { max-width: 600px; }
    """
    
    # UIの作成
    with gr.Blocks(css=css, title="改善版書籍生成ツール") as app:
        gr.Markdown("# 改善版書籍生成ツール")
        gr.Markdown("LangChainとOpenAI APIを使用して、高品質な書籍を自動生成します。")
        
        # 状態の管理
        state = gr.State(value=initial_state)
        
        # APIキー設定タブ
        with gr.Tab("APIキー設定"):
            gr.Markdown("## APIキーの設定")
            gr.Markdown("各サービスのAPIキーを入力してください。")
            
            with gr.Row():
                with gr.Column():
                    openai_api_key = gr.Textbox(
                        label="OpenAI APIキー（必須）", 
                        placeholder="sk-...", 
                        type="password",
                        container=False,
                        elem_classes=["api-key-input"]
                    )
                    
                    google_api_key = gr.Textbox(
                        label="Google APIキー（検索機能用、オプション）", 
                        placeholder="AIza...", 
                        type="password",
                        container=False,
                        elem_classes=["api-key-input"]
                    )
                    
                    search_engine_id = gr.Textbox(
                        label="Google Custom Search Engine ID（検索機能用、オプション）", 
                        placeholder="1234...", 
                        container=False,
                        elem_classes=["api-key-input"]
                    )
                    
                    save_api_keys_btn = gr.Button("APIキーを保存", variant="primary")
                
                with gr.Column():
                    api_status = gr.Markdown("APIキーは設定されていません。")
        
        # 構成生成タブ
        with gr.Tab("構成生成"):
            gr.Markdown("## 書籍構成の自動生成")
            gr.Markdown("書籍のタイトルから章・節構成を自動生成します。")
            
            with gr.Row():
                with gr.Column():
                    book_title = gr.Textbox(
                        label="書籍タイトル", 
                        placeholder="量子コンピューティング入門"
                    )
                    
                    genre = gr.Textbox(
                        label="ジャンル（オプション）", 
                        placeholder="科学・技術"
                    )
                    
                    target_audience = gr.Textbox(
                        label="対象読者層（オプション）", 
                        placeholder="コンピュータサイエンスに興味のある大学生や技術者"
                    )
                    
                    num_chapters = gr.Slider(
                        label="章の数", 
                        minimum=1, 
                        maximum=20, 
                        value=10, 
                        step=1
                    )
                    
                    sections_per_chapter = gr.Slider(
                        label="各章の節の数", 
                        minimum=1, 
                        maximum=10, 
                        value=3, 
                        step=1
                    )
                    
                    generate_structure_btn = gr.Button("構成を生成", variant="primary")
                
                with gr.Column():
                    structure_output = gr.JSON(label="生成された構成")
                    structure_status = gr.Markdown("構成はまだ生成されていません。")
        
        # 構成編集タブ
        with gr.Tab("構成編集"):
            gr.Markdown("## 書籍構成の編集")
            gr.Markdown("自動生成された構成を編集できます。")
            
            with gr.Row():
                with gr.Column():
                    chapters_json = gr.JSON(
                        label="章・節構成（JSON形式）", 
                        elem_classes=["output-box"]
                    )
                    
                    with gr.Accordion("章の追加", open=False):
                        add_chapter_number = gr.Number(
                            label="章番号", 
                            value=1, 
                            precision=0
                        )
                        
                        add_chapter_title = gr.Textbox(
                            label="章タイトル", 
                            placeholder="新しい章のタイトル"
                        )
                        
                        add_chapter_description = gr.Textbox(
                            label="章の説明", 
                            placeholder="この章の簡単な説明"
                        )
                        
                        add_chapter_btn = gr.Button("章を追加")
                    
                    with gr.Accordion("節の追加", open=False):
                        section_chapter_index = gr.Number(
                            label="対象の章のインデックス（0から始まる）", 
                            value=0, 
                            precision=0
                        )
                        
                        add_section_number = gr.Textbox(
                            label="節番号", 
                            placeholder="1.4"
                        )
                        
                        add_section_title = gr.Textbox(
                            label="節タイトル", 
                            placeholder="新しい節のタイトル"
                        )
                        
                        add_section_description = gr.Textbox(
                            label="節の説明", 
                            placeholder="この節の簡単な説明"
                        )
                        
                        add_section_btn = gr.Button("節を追加")
                
                with gr.Column():
                    structure_preview = gr.Markdown(label="構成プレビュー")
                    edit_status = gr.Markdown("構成を編集できます。")
        
        # 書籍生成タブ
        with gr.Tab("書籍生成"):
            gr.Markdown("## 書籍の生成")
            gr.Markdown("編集した構成に基づいて書籍を生成します。")
            
            with gr.Row():
                with gr.Column():
                    book_title_display = gr.Textbox(
                        label="書籍タイトル", 
                        interactive=False
                    )
                    
                    model_name = gr.Dropdown(
                        label="使用するモデル", 
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], 
                        value="gpt-4o-mini"
                    )
                    
                    temperature = gr.Slider(
                        label="温度（創造性）", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.7, 
                        step=0.1
                    )
                    
                    memory_token_limit = gr.Slider(
                        label="メモリトークン制限", 
                        minimum=100, 
                        maximum=2000, 
                        value=500, 
                        step=100
                    )
                    
                    writing_style = gr.Radio(
                        label="文体スタイル",
                        choices=["です・ます調", "だ・である調"],
                        value="です・ます調"
                    )
                    
                    use_search = gr.Checkbox(
                        label="検索機能を使用", 
                        value=False
                    )
                    
                    search_results_count = gr.Slider(
                        label="各章の検索結果数", 
                        minimum=1, 
                        maximum=10, 
                        value=5, 
                        step=1,
                        visible=False
                    )
                    
                    generate_cover_option = gr.Checkbox(
                        label="表紙を生成", 
                        value=True
                    )
                    
                    generate_illustrations_option = gr.Checkbox(
                        label="挿絵を生成", 
                        value=True
                    )
                    
                    author_name = gr.Textbox(
                        label="著者名（表紙用、オプション）", 
                        placeholder="山田 太郎"
                    )
                    
                    generate_book_btn = gr.Button("書籍を生成", variant="primary")
                
                with gr.Column():
                    generation_status = gr.Markdown("書籍はまだ生成されていません。")
                    output_files = gr.JSON(label="出力ファイル", visible=False)
                    
                    with gr.Row():
                        view_md_btn = gr.Button("マークダウンを表示", visible=False)
                        view_html_btn = gr.Button("HTMLを表示", visible=False)
                        view_pdf_btn = gr.Button("PDFを表示", visible=False)
                        view_cover_btn = gr.Button("表紙を表示", visible=False)
        
        # プレビュータブ
        with gr.Tab("プレビュー"):
            gr.Markdown("## 生成された書籍のプレビュー")
            
            with gr.Row():
                preview_type = gr.Radio(
                    label="プレビュータイプ", 
                    choices=["マークダウン", "HTML"], 
                    value="マークダウン"
                )
            
            with gr.Row():
                preview_content = gr.Markdown(elem_classes=["output-box"])
                preview_html = gr.HTML(visible=False)
            
            with gr.Row():
                cover_preview = gr.Image(label="表紙プレビュー", visible=False)
        
        # イベントハンドラの定義
        def save_api_keys(state_dict, openai_key, google_key, search_id):
            """APIキーを保存する"""
            state_dict["openai_api_key"] = openai_key
            state_dict["google_api_key"] = google_key
            state_dict["search_engine_id"] = search_id
            
            status_text = "APIキーが設定されました。"
            
            # ジェネレーターの初期化
            try:
                generator = ImprovedBookGenerator(
                    openai_api_key=openai_key,
                    google_api_key=google_key,
                    search_engine_id=search_id
                )
                state_dict["generator"] = generator
                status_text += " ジェネレーターの初期化に成功しました。"
            except Exception as e:
                status_text += " エラー: {}".format(str(e))
            
            return state_dict, status_text
        
        def generate_book_structure(state_dict, title, genre_val, audience_val, chapters_num, sections_num):
            """書籍構成を生成する"""
            openai_key = state_dict["openai_api_key"]
            
            if not openai_key:
                return state_dict, {"error": "OpenAI APIキーが設定されていません。APIキー設定タブで設定してください。"}, "OpenAI APIキーが設定されていません。APIキー設定タブで設定してください。", None, ""
            
            try:
                generator = state_dict["generator"]
                if not generator:
                    generator = ImprovedBookGenerator(openai_api_key=openai_key)
                    state_dict["generator"] = generator
                
                structure = generator.generate_book_structure(
                    book_title=title,
                    genre=genre_val,
                    target_audience=audience_val,
                    num_chapters=int(chapters_num),
                    sections_per_chapter=int(sections_num)
                )
                
                state_dict["book_structure"] = structure
                
                return state_dict, structure, "構成の生成が完了しました。", structure, format_structure_preview(structure)
            except Exception as e:
                error_msg = "構成の生成中にエラーが発生しました: {}".format(str(e))
                traceback.print_exc()
                return state_dict, {"error": error_msg}, error_msg, None, ""
        
        def update_structure_preview(structure):
            """構成のプレビューを更新する"""
            if not structure:
                return ""
            
            preview = "# {}\n\n".format(structure.get('title', '無題'))
            preview += "## 目次\n\n"
            
            for chapter in structure.get("chapters", []):
                chapter_num = chapter.get("number", "?")
                chapter_title = chapter.get("title", "無題")
                preview += "- 第{}章: {}\n".format(chapter_num, chapter_title)
                
                for section in chapter.get("sections", []):
                    section_num = section.get("number", "?.?")
                    section_title = section.get("title", "無題")
                    preview += "  - {} {}\n".format(section_num, section_title)
            
            preview += "\n## 詳細\n\n"
            
            for chapter in structure.get("chapters", []):
                chapter_num = chapter.get("number", "?")
                chapter_title = chapter.get("title", "無題")
                chapter_desc = chapter.get("description", "")
                preview += "### 第{}章: {}\n\n".format(chapter_num, chapter_title)
                preview += "{}\n\n".format(chapter_desc)
                
                for section in chapter.get("sections", []):
                    section_num = section.get("number", "?.?")
                    section_title = section.get("title", "無題")
                    section_desc = section.get("description", "")
                    preview += "#### {} {}\n\n".format(section_num, section_title)
                    preview += "{}\n\n".format(section_desc)
            
            return preview
        
        def format_structure_preview(structure):
            """構成のプレビューをフォーマットする"""
            return update_structure_preview(structure)
        
        def add_chapter(state_dict, chapter_num, title, description, chapters_json):
            """章を追加する"""
            if not chapters_json:
                return state_dict, chapters_json, "構成が読み込まれていません。", ""
            
            try:
                new_chapter = {
                    "number": int(chapter_num),
                    "title": title,
                    "description": description,
                    "sections": []
                }
                
                chapters_json["chapters"].append(new_chapter)
                
                # 章番号でソート
                chapters_json["chapters"].sort(key=lambda x: x["number"])
                
                state_dict["book_structure"] = chapters_json
                
                return state_dict, chapters_json, "第{}章「{}」が追加されました。".format(chapter_num, title), format_structure_preview(chapters_json)
            except Exception as e:
                error_msg = "章の追加中にエラーが発生しました: {}".format(str(e))
                return state_dict, chapters_json, error_msg, format_structure_preview(chapters_json)
        
        def add_section(state_dict, chapter_idx, section_num, title, description, chapters_json):
            """節を追加する"""
            if not chapters_json:
                return state_dict, chapters_json, "構成が読み込まれていません。", ""
            
            try:
                chapter_idx = int(chapter_idx)
                if chapter_idx < 0 or chapter_idx >= len(chapters_json["chapters"]):
                    return state_dict, chapters_json, "無効な章インデックス: {}".format(chapter_idx), format_structure_preview(chapters_json)
                
                new_section = {
                    "number": section_num,
                    "title": title,
                    "description": description
                }
                
                chapters_json["chapters"][chapter_idx]["sections"].append(new_section)
                
                # 節番号でソート
                chapters_json["chapters"][chapter_idx]["sections"].sort(key=lambda x: x["number"])
                
                state_dict["book_structure"] = chapters_json
                
                return state_dict, chapters_json, "節{}「{}」が追加されました。".format(section_num, title), format_structure_preview(chapters_json)
            except Exception as e:
                error_msg = "節の追加中にエラーが発生しました: {}".format(str(e))
                return state_dict, chapters_json, error_msg, format_structure_preview(chapters_json)
        
        def update_book_title_display(structure):
            """書籍タイトル表示を更新する"""
            if not structure:
                return ""
            return structure.get("title", "")
        
        def update_search_results_visibility(use_search_flag):
            """検索結果数スライダーの表示/非表示を切り替える"""
            return gr.update(visible=use_search_flag)
        
        def generate_book(state_dict, book_title_val, chapters_json_val,
                         model_val, temp_val, memory_limit_val, writing_style_val,
                         use_search_flag, search_results_count,
                         generate_cover_flag, generate_illustrations_flag, author_name_val):
            """書籍を生成する"""
            openai_key = state_dict["openai_api_key"]
            google_key = state_dict["google_api_key"]
            search_id = state_dict["search_engine_id"]
            
            if not openai_key:
                return (
                    state_dict, 
                    "OpenAI APIキーが設定されていません。APIキー設定タブで設定してください。",
                    None, 
                    gr.update(visible=False), 
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
            if not chapters_json_val:
                return (
                    state_dict, 
                    "書籍構成が設定されていません。構成生成タブで構成を生成してください。",
                    None, 
                    gr.update(visible=False), 
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
            try:
                # ジェネレーターの再初期化（パラメータ更新のため）
                generator = ImprovedBookGenerator(
                    openai_api_key=openai_key,
                    google_api_key=google_key,
                    search_engine_id=search_id,
                    model_name=model_val,
                    temperature=float(temp_val),
                    memory_token_limit=int(memory_limit_val),
                    writing_style=writing_style_val
                )
                state_dict["generator"] = generator
                
                # 書籍の生成
                book_content = generator.generate_book(
                    book_structure=chapters_json_val,
                    use_search=use_search_flag,
                    search_results_per_chapter=int(search_results_count),
                    generate_cover=generate_cover_flag,
                    generate_illustrations=generate_illustrations_flag,
                    author_name=author_name_val
                )
                
                state_dict["book_content"] = book_content
                
                # 書籍の保存
                md_path, html_path, pdf_path = generator.save_book_content(book_content)
                
                output_files = {
                    "markdown": md_path,
                    "html": html_path
                }
                
                if pdf_path:
                    output_files["pdf"] = pdf_path
                
                if 'cover_path' in book_content and book_content['cover_path']:
                    output_files["cover"] = book_content['cover_path']
                
                state_dict["output_files"] = output_files
                
                # 表示ボタンの更新
                view_md_visible = True
                view_html_visible = True
                view_pdf_visible = pdf_path is not None
                view_cover_visible = 'cover_path' in book_content and book_content['cover_path'] is not None
                
                return (
                    state_dict, 
                    "書籍「{}」の生成が完了しました。".format(book_content['title']),
                    output_files, 
                    gr.update(visible=view_md_visible), 
                    gr.update(visible=view_html_visible),
                    gr.update(visible=view_pdf_visible),
                    gr.update(visible=view_cover_visible)
                )
            except Exception as e:
                error_msg = "書籍の生成中にエラーが発生しました: {}".format(str(e))
                traceback.print_exc()
                return (
                    state_dict, 
                    error_msg,
                    None, 
                    gr.update(visible=False), 
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        
        def load_preview_content(state_dict, preview_type):
            """プレビューコンテンツを読み込む"""
            output_files = state_dict.get("output_files")
            if not output_files:
                return (
                    "書籍がまだ生成されていません。",
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
            
            try:
                if preview_type == "マークダウン":
                    with open(output_files["markdown"], "r", encoding="utf-8") as f:
                        content = f.read()
                    return (
                        content,
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=False)
                    )
                else:  # HTML
                    with open(output_files["html"], "r", encoding="utf-8") as f:
                        html_content = f.read()
                    return (
                        "",
                        gr.update(visible=True, value=html_content),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
            except Exception as e:
                error_msg = "プレビューの読み込み中にエラーが発生しました: {}".format(str(e))
                return (
                    error_msg,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
        
        def view_markdown(state_dict):
            """マークダウンを表示する"""
            output_files = state_dict.get("output_files")
            if not output_files or "markdown" not in output_files:
                return "マークダウンファイルが見つかりません。", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            
            try:
                with open(output_files["markdown"], "r", encoding="utf-8") as f:
                    content = f.read()
                return content, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            except Exception as e:
                error_msg = "マークダウンの読み込み中にエラーが発生しました: {}".format(str(e))
                return error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def view_html(state_dict):
            """HTMLを表示する"""
            output_files = state_dict.get("output_files")
            if not output_files or "html" not in output_files:
                return "", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            
            try:
                with open(output_files["html"], "r", encoding="utf-8") as f:
                    html_content = f.read()
                return "", gr.update(visible=True, value=html_content), gr.update(visible=False), gr.update(visible=False)
            except Exception as e:
                error_msg = "HTMLの読み込み中にエラーが発生しました: {}".format(str(e))
                return error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def view_pdf(state_dict):
            """PDFを表示する（ブラウザで開く）"""
            output_files = state_dict.get("output_files")
            if not output_files or "pdf" not in output_files:
                return "PDFファイルが見つかりません。", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            
            try:
                pdf_path = output_files["pdf"]
                # PDFファイルへのリンクを含むHTMLを生成
                html_content = f"""
                <div style="text-align: center; padding: 20px;">
                    <h2>PDFファイルが生成されました</h2>
                    <p>以下のリンクをクリックしてPDFファイルを開いてください：</p>
                    <a href="file={pdf_path}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">PDFを開く</a>
                    <p style="margin-top: 20px;">または、以下のパスからファイルを直接開くこともできます：</p>
                    <code style="background-color: #f5f5f5; padding: 5px 10px; border-radius: 3px;">{pdf_path}</code>
                </div>
                """
                return "", gr.update(visible=True, value=html_content), gr.update(visible=False), gr.update(visible=False)
            except Exception as e:
                error_msg = "PDFの表示中にエラーが発生しました: {}".format(str(e))
                return error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def view_cover(state_dict):
            """表紙を表示する"""
            output_files = state_dict.get("output_files")
            if not output_files or "cover" not in output_files:
                return "", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            
            try:
                cover_path = output_files["cover"]
                return "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=cover_path)
            except Exception as e:
                error_msg = "表紙の読み込み中にエラーが発生しました: {}".format(str(e))
                return error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # イベントの接続
        save_api_keys_btn.click(
            save_api_keys,
            inputs=[state, openai_api_key, google_api_key, search_engine_id],
            outputs=[state, api_status]
        )
        
        generate_structure_btn.click(
            generate_book_structure,
            inputs=[state, book_title, genre, target_audience, num_chapters, sections_per_chapter],
            outputs=[state, structure_output, structure_status, chapters_json, structure_preview]
        )
        
        add_chapter_btn.click(
            add_chapter,
            inputs=[state, add_chapter_number, add_chapter_title, add_chapter_description, chapters_json],
            outputs=[state, chapters_json, edit_status, structure_preview]
        )
        
        add_section_btn.click(
            add_section,
            inputs=[state, section_chapter_index, add_section_number, add_section_title, add_section_description, chapters_json],
            outputs=[state, chapters_json, edit_status, structure_preview]
        )
        
        chapters_json.change(
            update_structure_preview,
            inputs=[chapters_json],
            outputs=[structure_preview]
        )
        
        chapters_json.change(
            update_book_title_display,
            inputs=[chapters_json],
            outputs=[book_title_display]
        )
        
        use_search.change(
            update_search_results_visibility,
            inputs=[use_search],
            outputs=[search_results_count]
        )
        
        generate_book_btn.click(
            generate_book,
            inputs=[
                state, book_title_display, chapters_json, 
                model_name, temperature, memory_token_limit, writing_style,
                use_search, search_results_count,
                generate_cover_option, generate_illustrations_option, author_name
            ],
            outputs=[
                state, generation_status, output_files, 
                view_md_btn, view_html_btn, view_pdf_btn, view_cover_btn
            ]
        )
        
        preview_type.change(
            load_preview_content,
            inputs=[state, preview_type],
            outputs=[preview_content, preview_html, preview_content, cover_preview]
        )
        
        view_md_btn.click(
            view_markdown,
            inputs=[state],
            outputs=[preview_content, preview_html, preview_content, cover_preview]
        )
        
        view_html_btn.click(
            view_html,
            inputs=[state],
            outputs=[preview_content, preview_html, preview_content, cover_preview]
        )
        
        view_pdf_btn.click(
            view_pdf,
            inputs=[state],
            outputs=[preview_content, preview_html, preview_content, cover_preview]
        )
        
        view_cover_btn.click(
            view_cover,
            inputs=[state],
            outputs=[preview_content, preview_html, preview_content, cover_preview]
        )
    
    return app

# メイン関数
def main():
    """メイン関数"""
    app = create_ui()
    app.launch()

if __name__ == "__main__":
    main()
