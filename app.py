import streamlit as st
import anthropic
import base64
from PIL import Image
import io

# ===== Web Search Integration =====
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import BraveSearch
import time
from datetime import datetime
import pytz

# ===== 日本時間取得関数 =====
def get_jst_now():
    """日本時間（JST）の現在時刻を取得"""
    jst = pytz.timezone('Asia/Tokyo')
    return datetime.now(jst)

# ページ設定
st.set_page_config(
    page_title="Claude API チャット",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 認証機能 =====
def check_password():
    """パスワード認証を実装"""
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "パスワードを入力してください",
            type="password",
            on_change=password_entered,
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "パスワードを入力してください",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("😕 パスワードが違います")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ===== Simple Search Function =====
@st.cache_resource  
def init_brave_search(search_results: int = 5):
    """
    Brave Searchツールを初期化
    
    Args:
        search_results: 検索結果の数
        
    Returns:
        BraveSearch: 初期化されたBrave Searchツール
    """
    try:
        brave_search = BraveSearch.from_api_key(
            api_key=st.secrets["BRAVE_SEARCH_API_KEY"],
            search_kwargs={
                "count": search_results,
                "safesearch": "moderate"
            }
        )
        return brave_search
    except Exception as e:
        st.error(f"Brave Search初期化エラー: {str(e)}")
        return None

def perform_search_and_generate_response(model_name: str, brave_search, query: str):
    """
    検索を実行してレスポンスを生成
    
    Args:
        model_name: Claudeモデル名
        brave_search: 検索ツール
        query: 検索クエリ
        
    Returns:
        str: 検索結果を含む回答
    """
    try:
        # 現在の日時情報（日本時間）
        jst_now = get_jst_now()
        current_datetime = jst_now.strftime("%Y年%m月%d日 %H時%M分")
        current_weekday = jst_now.strftime("%A")
        weekday_jp = {
            "Monday": "月曜日", "Tuesday": "火曜日", "Wednesday": "水曜日",
            "Thursday": "木曜日", "Friday": "金曜日", "Saturday": "土曜日", "Sunday": "日曜日"
        }
        current_weekday_jp = weekday_jp.get(current_weekday, current_weekday)
        
        # 日付関連の質問かチェック
        date_keywords = ["今日", "日付", "何日", "いつ", "曜日", "today", "date"]
        is_date_question = any(keyword in query.lower() for keyword in date_keywords)
        
        if is_date_question and not any(keyword in query.lower() for keyword in ["ニュース", "news", "最新", "トレンド"]):
            # 日付関連の基本質問は検索せずに直接回答
            return f"現在の日時は{current_datetime}（{current_weekday_jp}）です。"
        
        # Web検索を実行
        search_results = brave_search.run(query)
        
        # LLMの初期化
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.3,
            max_tokens=4096,
            anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
        )
        
        # 検索結果を使って回答を生成
        enhanced_prompt = f"""現在の日時: {current_datetime} ({current_weekday_jp})

以下の検索結果を参考に、ユーザーの質問「{query}」に対して正確で詳しい回答をしてください。

検索結果:
{search_results}

上記の情報を基に、質問に対する包括的で正確な回答を日本語で提供してください。現在の日時情報も必要に応じて活用してください。"""
        
        # LLMに回答生成を依頼
        response = llm.invoke([{"role": "user", "content": enhanced_prompt}])
        return response.content
        
    except Exception as e:
        return f"検索中にエラーが発生しました: {str(e)}"

def rate_limited_search(func):
    """
    Brave Search APIのレート制限（1秒1リクエスト）に対応するデコレータ
    
    無料プランの制約:
    - 1秒あたり1リクエスト
    - 月間2,000リクエスト
    """
    last_call_time = [0]  # リストで包んで参照を保持
    
    def wrapper(*args, **kwargs):
        current_time = time.time()
        time_since_last_call = current_time - last_call_time[0]
        
        # 前回の呼び出しから1秒未満の場合は待機
        if time_since_last_call < 1.0:
            sleep_time = 1.0 - time_since_last_call
            with st.spinner(f'レート制限のため {sleep_time:.1f}秒待機中...'):
                time.sleep(sleep_time)
        
        result = func(*args, **kwargs)
        last_call_time[0] = time.time()
        
        # 検索カウントの更新
        if "search_count_today" not in st.session_state:
            st.session_state.search_count_today = 0
        st.session_state.search_count_today += 1
        
        return result
    
    return wrapper

# ===== API クライアント初期化 =====
@st.cache_resource
def get_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

client = get_client()

# ===== サイドバー設定 =====
with st.sidebar:
    st.header("⚙️ 設定")
    
    # モデル選択
    model_options = {
        "Claude 4.5 Sonnet（最新・推奨）": "claude-sonnet-4-5-20250929",
        "Claude 4.1 Opus（最高性能）": "claude-opus-4-20250514",
        "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
        "Claude 3.5 Haiku（高速・低コスト）": "claude-3-5-haiku-20241022",
    }
    
    selected_model_name = st.selectbox(
        "🤖 モデルを選択",
        options=list(model_options.keys()),
        help="用途に応じてモデルを選択できます"
    )
    selected_model = model_options[selected_model_name]
    
    # Max tokens設定
    max_tokens = st.slider(
        "最大トークン数",
        min_value=256,
        max_value=8192,
        value=4096,
        step=256,
        help="長い回答が必要な場合は増やしてください"
    )
    
    # Temperature設定
    temperature = st.slider(
        "創造性（Temperature）",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="高いほど創造的、低いほど正確"
    )
    
    st.divider()
    
    # ファイルアップロード
    st.subheader("📎 ファイル添付")
    uploaded_file = st.file_uploader(
        "画像やテキストファイルをアップロード",
        type=["png", "jpg", "jpeg", "webp", "gif", "txt", "md"],
        help="画像やテキストを添付できます"
    )
    
    st.divider()
    
    # チャット履歴管理
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 履歴クリア", use_container_width=True):
            st.session_state.messages = []
            if "uploaded_file_content" in st.session_state:
                del st.session_state["uploaded_file_content"]
            st.rerun()
    
    with col2:
        if st.button("📥 会話保存", use_container_width=True):
            import json
            from datetime import datetime
            
            jst_now = get_jst_now()
            chat_data = {
                "timestamp": jst_now.isoformat(),
                "model": selected_model_name,
                "messages": st.session_state.get('messages', [])
            }
            
            st.download_button(
                label="💾 ダウンロード",
                data=json.dumps(chat_data, ensure_ascii=False, indent=2),
                file_name=f"chat_{jst_now.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # 統計情報
    st.divider()
    message_count = len(st.session_state.get('messages', []))
    st.caption(f"💬 メッセージ数: {message_count}")
    st.caption(f"🤖 使用モデル: {selected_model_name}")
    
    # コスト試算
    if 'total_tokens' in st.session_state:
        st.caption(f"📊 累計トークン: {st.session_state.total_tokens:,}")
    
    # ===== Web Search Settings =====
    st.divider()
    st.subheader("🔍 Web検索設定")
    
    enable_search = st.toggle(
        "Web検索を有効化",
        value=False,
        help="有効にすると、Claudeが必要に応じて最新のWeb情報を検索します"
    )
    
    if enable_search:
        search_count = st.slider(
            "検索結果数",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="検索する結果の数（多いほど詳細ですが時間がかかります）"
        )
        
        if "BRAVE_SEARCH_API_KEY" not in st.secrets:
            st.error("❌ Brave Search APIキーが設定されていません")
            st.caption("📝 .streamlit/secrets.tomlに以下を追加してください:")
            st.code('BRAVE_SEARCH_API_KEY = "your-key"')
            enable_search = False
        else:
            st.success("✅ Brave Search API 設定済み")
            if "search_count_today" in st.session_state:
                st.caption(f"🔍 本日の検索回数: {st.session_state.search_count_today}/2000")
    
    st.divider()

# ===== メインエリア =====
st.title("🤖 Claude API チャット")
st.caption("Claude APIを使った高機能チャットアプリ - 最新モデル対応")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# ファイル処理関数
def process_uploaded_file(file):
    """アップロードされたファイルを処理"""
    file_type = file.type
    
    # 画像ファイルの処理
    if file_type.startswith("image/"):
        image = Image.open(file)
        
        # 画像サイズを適切に調整（大きすぎる場合）
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_base64
            }
        }
    
    # テキストファイルの処理
    elif file_type.startswith("text/"):
        content = file.read().decode("utf-8")
        return {
            "type": "text",
            "text": f"[添付ファイル: {file.name}]\n\n{content}"
        }
    
    return None

# アップロードされたファイルの処理
if uploaded_file and "uploaded_file_content" not in st.session_state:
    file_content = process_uploaded_file(uploaded_file)
    if file_content:
        st.session_state.uploaded_file_content = file_content
        st.success(f"✅ {uploaded_file.name} をアップロードしました")

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        elif isinstance(message["content"], list):
            for content_block in message["content"]:
                if content_block.get("type") == "text":
                    st.markdown(content_block["text"])
                elif content_block.get("type") == "image":
                    st.info("🖼️ 画像が添付されています")

# ユーザー入力
if prompt := st.chat_input("メッセージを入力してください..."):
    # ユーザーメッセージを構築
    user_message_content = []
    
    # テキストを追加
    user_message_content.append({
        "type": "text",
        "text": prompt
    })
    
    # アップロードされたファイルがあれば追加
    if "uploaded_file_content" in st.session_state:
        file_content = st.session_state.uploaded_file_content
        if file_content["type"] == "image":
            user_message_content.append(file_content)
        else:
            user_message_content[0]["text"] = file_content["text"] + "\n\n" + prompt
        
        # ファイルは一度だけ使用
        del st.session_state["uploaded_file_content"]
    
    # メッセージを追加
    st.session_state.messages.append({
        "role": "user",
        "content": user_message_content if len(user_message_content) > 1 else prompt
    })
    
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)
        if len(user_message_content) > 1:
            st.info("🖼️ 画像が添付されています")
    
    full_response = ""
    
    # ===== Web検索モード =====
    if enable_search and "BRAVE_SEARCH_API_KEY" in st.secrets:
        with st.chat_message("assistant"):
            try:
                # 検索ツールの初期化
                brave_search = init_brave_search(search_count if 'search_count' in locals() else 5)
                
                if brave_search is None:
                    st.error("検索ツールの初期化に失敗しました。通常モードで回答します。")
                    enable_search = False
                else:
                    st.info("🔍 Web検索を使用して回答を生成しています...")
                    
                    # 検索と回答生成を実行
                    full_response = perform_search_and_generate_response(
                        selected_model, brave_search, prompt
                    )
                    
                    # レスポンスを表示
                    st.markdown(full_response)
                    
                    # 検索情報の表示
                    with st.expander("📊 検索情報"):
                        st.write("✅ Web検索を使用して最新情報を取得しました")
                        st.write(f"**検索結果数:** {search_count if 'search_count' in locals() else 5}件")
                        st.write(f"**使用モデル:** {selected_model_name}")
                        
            except Exception as e:
                st.error(f"❌ 検索中にエラーが発生しました")
                st.exception(e)
                st.info("💡 通常モードで回答を試みます")
                enable_search = False  # エラー時は通常モードにフォールバック
    
    # ===== 通常のClaude API呼び出し =====
    if not enable_search or "BRAVE_SEARCH_API_KEY" not in st.secrets:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # 現在の日時情報を取得（日本時間）
                jst_now = get_jst_now()
                current_datetime = jst_now.strftime("%Y年%m月%d日 %H時%M分")
                current_weekday = jst_now.strftime("%A")
                
                # 日本語曜日の変換
                weekday_jp = {
                    "Monday": "月曜日", "Tuesday": "火曜日", "Wednesday": "水曜日",
                    "Thursday": "木曜日", "Friday": "金曜日", "Saturday": "土曜日", "Sunday": "日曜日"
                }
                current_weekday_jp = weekday_jp.get(current_weekday, current_weekday)
                
                # システムメッセージに現在日時を含める
                system_message = {
                    "role": "system",
                    "content": f"現在の日時: {current_datetime} ({current_weekday_jp})\n\n" +
                              "日付や時刻について質問された場合は、上記の現在日時情報を使用して正確に回答してください。"
                }
                
                # API呼び出し用のメッセージ形式に変換
                api_messages = [system_message]
                for msg in st.session_state.messages:
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                # ストリーミング応答
                with client.messages.stream(
                    model=selected_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=api_messages
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        message_placeholder.markdown(full_response + "▌")
                    
                    # 最終的なレスポンスを表示
                    message_placeholder.markdown(full_response)
                    
                    # トークン情報を取得
                    final_message = stream.get_final_message()
                    input_tokens = final_message.usage.input_tokens
                    output_tokens = final_message.usage.output_tokens
                    
                    # 累計トークンを更新
                    st.session_state.total_tokens += input_tokens + output_tokens
                    
                    # トークン使用量を表示
                    with st.expander("📊 使用トークン情報"):
                        st.write(f"入力トークン: {input_tokens:,}")
                        st.write(f"出力トークン: {output_tokens:,}")
                        
                        # コスト試算
                        if "opus-4" in selected_model:
                            input_cost = (input_tokens / 1_000_000) * 15.00
                            output_cost = (output_tokens / 1_000_000) * 75.00
                        elif "sonnet-4" in selected_model:
                            input_cost = (input_tokens / 1_000_000) * 3.00
                            output_cost = (output_tokens / 1_000_000) * 15.00
                        elif "haiku" in selected_model:
                            input_cost = (input_tokens / 1_000_000) * 0.25
                            output_cost = (output_tokens / 1_000_000) * 1.25
                        else:
                            input_cost = (input_tokens / 1_000_000) * 3.00
                            output_cost = (output_tokens / 1_000_000) * 15.00
                        
                        total_cost = input_cost + output_cost
                        st.write(f"💰 このメッセージのコスト: ${total_cost:.6f}")
                    
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
                full_response = None
    
    # アシスタントメッセージを履歴に追加
    if full_response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

# フッター
st.divider()
st.caption("💡 ヒント: 左側のサイドバーでモデルやファイルを選択できます")
st.caption("🔍 Web検索を有効にすると、最新情報を取得できます")
st.caption("🔒 このアプリはパスワードで保護されています")