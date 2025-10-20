import streamlit as st
import anthropic
import base64
from PIL import Image
import io
from datetime import datetime
import pytz
import json

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
        "Claude 4.1 Opus（最高性能）": "claude-opus-4-1-20250805",
        "Claude 4 Opus": "claude-opus-4-20250514",
        "Claude 4 Sonnet": "claude-sonnet-4-20250514",
        "Claude 3.5 Sonnet v2": "claude-3-5-sonnet-20241022",
        "Claude 4.5 Haiku（高速・低コスト）": "claude-haiku-4-5-20251001",
        "Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
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
            
            # 会話履歴をJSON形式で保存
            chat_history = {
                "timestamp": datetime.now().isoformat(),
                "model": selected_model_name,
                "messages": st.session_state.messages
            }
            
            # ダウンロードボタンを表示
            st.download_button(
                label="💾 JSONとしてダウンロード",
                data=json.dumps(chat_history, ensure_ascii=False, indent=2),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    st.divider()
    
    # Web検索設定（Claude APIのネイティブ検索）
    st.subheader("🔍 Web検索設定")
    enable_search = st.checkbox(
        "Claude Web検索を有効化",
        value=False,
        help="Claude APIのネイティブWeb検索機能を使用します（$10/1000検索）"
    )
    
    if enable_search:
        search_count = st.slider(
            "最大検索回数",
            min_value=1,
            max_value=10,
            value=5,
            help="1回のレスポンスで実行する最大検索数"
        )
        
        # ドメインフィルタリング設定
        use_domain_filter = st.checkbox("ドメインフィルタリングを使用", value=False)
        
        if use_domain_filter:
            filter_type = st.radio(
                "フィルタータイプ",
                options=["許可リスト", "ブロックリスト"],
                help="特定のドメインのみを許可またはブロック"
            )
            
            domains_input = st.text_area(
                f"{'許可' if filter_type == '許可リスト' else 'ブロック'}するドメイン（1行に1つ）",
                placeholder="example.com\ntrusted-site.org",
                help="HTTPSやサブドメインは含めないでください"
            )
        
        # ローカライゼーション設定
        use_location = st.checkbox("位置情報を使用", value=False)
        
        if use_location:
            col1, col2 = st.columns(2)
            with col1:
                city = st.text_input("都市", value="Tokyo")
                country = st.text_input("国", value="JP")
            with col2:
                region = st.text_input("地域", value="Tokyo")
                timezone = st.text_input("タイムゾーン", value="Asia/Tokyo")
        
        st.info("💡 Claude Web検索は自動的に引用を含めて回答します")
        st.caption("💰 価格: $10/1000検索 + 通常のトークン料金")
    
    st.divider()

# ===== メインエリア =====
st.title("🤖 Claude API チャット")
st.caption("Claude APIを使った高機能チャットアプリ - ネイティブWeb検索対応")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

if "total_searches" not in st.session_state:
    st.session_state.total_searches = 0

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

# 過去のメッセージを表示（Web検索結果を適切に表示）
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
                elif content_block.get("type") == "server_tool_use":
                    # Web検索クエリの表示
                    if content_block.get("name") == "web_search":
                        st.info(f"🔍 検索実行: {content_block.get('input', {}).get('query', '')}")
                elif content_block.get("type") == "web_search_tool_result":
                    # Web検索結果の表示（オプション）
                    with st.expander("検索結果詳細"):
                        for result in content_block.get("content", []):
                            if result.get("type") == "web_search_result":
                                st.write(f"- [{result.get('title')}]({result.get('url')})")

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
    
    # アシスタントの応答
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
            
            # Web検索ツールの設定
            tools = None
            if enable_search:
                # 基本のツール設定
                tool_config = {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": search_count
                }
                
                # ドメインフィルタリングの追加
                if use_domain_filter and domains_input:
                    domains = [d.strip() for d in domains_input.split('\n') if d.strip()]
                    if domains:
                        if filter_type == "許可リスト":
                            tool_config["allowed_domains"] = domains
                        else:
                            tool_config["blocked_domains"] = domains
                
                # ローカライゼーションの追加
                if use_location:
                    tool_config["user_location"] = {
                        "type": "approximate",
                        "city": city,
                        "region": region,
                        "country": country,
                        "timezone": timezone
                    }
                
                tools = [tool_config]
                
                st.info("🔍 Web検索を使用して回答を生成しています...")
            
            # APIリクエストの実行
            if tools:
                # Web検索ありの場合（ストリーミングなし）
                response = client.messages.create(
                    model=selected_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=api_messages,
                    tools=tools
                )
                
                # レスポンスの処理
                response_content = []
                search_queries = []
                citations = []
                
                for content_block in response.content:
                    if content_block.type == "text":
                        full_response += content_block.text
                        # 引用がある場合の処理
                        if hasattr(content_block, 'citations'):
                            for citation in content_block.citations:
                                citations.append({
                                    "url": citation.url,
                                    "title": citation.title,
                                    "text": citation.cited_text[:150]  # 最大150文字
                                })
                    elif content_block.type == "server_tool_use":
                        if content_block.name == "web_search":
                            query = content_block.input.get("query", "")
                            search_queries.append(query)
                            st.info(f"🔍 検索中: {query}")
                
                # 最終的なレスポンスを表示
                message_placeholder.markdown(full_response)
                
                # 検索情報の表示
                if search_queries:
                    with st.expander("📊 Web検索情報"):
                        st.write(f"**実行した検索数:** {len(search_queries)}件")
                        st.write("**検索クエリ:**")
                        for i, query in enumerate(search_queries, 1):
                            st.write(f"{i}. {query}")
                        
                        if citations:
                            st.write("**引用ソース:**")
                            for citation in citations:
                                st.write(f"- [{citation['title']}]({citation['url']})")
                                st.caption(f"  {citation['text']}...")
                
                # 使用状況の更新
                st.session_state.total_searches += len(search_queries)
                
                # トークン使用量の取得
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                web_searches = response.usage.get('server_tool_use', {}).get('web_search_requests', 0)
                
            else:
                # Web検索なしの場合（ストリーミングあり）
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
                    web_searches = 0
            
            # 累計トークンを更新
            st.session_state.total_tokens += input_tokens + output_tokens
            
            # 使用量情報を表示
            with st.expander("📊 使用情報"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**トークン使用量**")
                    st.write(f"入力: {input_tokens:,}")
                    st.write(f"出力: {output_tokens:,}")
                    st.write(f"累計: {st.session_state.total_tokens:,}")
                
                with col2:
                    st.write("**コスト試算**")
                    
                    # モデルごとの料金計算
                    if "opus-4" in selected_model:
                        if "4-1" in selected_model:
                            input_cost = (input_tokens / 1_000_000) * 15.00
                            output_cost = (output_tokens / 1_000_000) * 75.00
                        else:
                            input_cost = (input_tokens / 1_000_000) * 15.00
                            output_cost = (output_tokens / 1_000_000) * 75.00
                    elif "sonnet-4" in selected_model:
                        if "4-5" in selected_model:
                            input_cost = (input_tokens / 1_000_000) * 3.00
                            output_cost = (output_tokens / 1_000_000) * 15.00
                        else:
                            input_cost = (input_tokens / 1_000_000) * 3.00
                            output_cost = (output_tokens / 1_000_000) * 15.00
                    elif "haiku" in selected_model:
                        if "4-5" in selected_model:
                            input_cost = (input_tokens / 1_000_000) * 1.00
                            output_cost = (output_tokens / 1_000_000) * 5.00
                        else:
                            input_cost = (input_tokens / 1_000_000) * 0.25
                            output_cost = (output_tokens / 1_000_000) * 1.25
                    else:
                        # デフォルト（Sonnet 3.5）
                        input_cost = (input_tokens / 1_000_000) * 3.00
                        output_cost = (output_tokens / 1_000_000) * 15.00
                    
                    # Web検索コスト
                    search_cost = web_searches * 0.01  # $10/1000検索 = $0.01/検索
                    
                    total_cost = input_cost + output_cost + search_cost
                    
                    st.write(f"トークン: ${input_cost + output_cost:.6f}")
                    if web_searches > 0:
                        st.write(f"検索: ${search_cost:.6f}")
                    st.write(f"**合計: ${total_cost:.6f}**")
                
                if web_searches > 0:
                    st.write(f"**Web検索実行数:** {web_searches}回")
                    st.write(f"**累計検索数:** {st.session_state.total_searches}回")
                    
        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            
            # エラーの詳細を表示
            if "server_tool_use" in str(e):
                st.info("💡 Web検索機能がまだ有効化されていない可能性があります。")
                st.info("Anthropic Consoleで組織レベルでWeb検索を有効化する必要があります。")
            
            full_response = None
    
    # アシスタントメッセージを履歴に追加
    if full_response:
        # レスポンス全体を保存（Web検索結果も含む）
        if enable_search and 'response' in locals():
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

# フッター
st.divider()

# 使用統計の表示
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("総トークン数", f"{st.session_state.total_tokens:,}")

with col2:
    st.metric("総検索数", f"{st.session_state.total_searches:,}")

with col3:
    # 概算コストの計算（デフォルトでSonnet料金で計算）
    estimated_cost = (st.session_state.total_tokens / 1_000_000) * 10.00  # 平均的な見積もり
    estimated_cost += st.session_state.total_searches * 0.01
    st.metric("概算コスト", f"${estimated_cost:.4f}")

st.caption("💡 ヒント: 左側のサイドバーでモデルやWeb検索設定を調整できます")
st.caption("🔍 Claude Web検索: 自動的に最新情報を取得し、引用付きで回答します")
st.caption("🔒 このアプリはパスワードで保護されています")