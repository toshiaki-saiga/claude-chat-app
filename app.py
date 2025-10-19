import streamlit as st
import anthropic
import base64
from PIL import Image
import io

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
            
            chat_data = {
                "timestamp": datetime.now().isoformat(),
                "model": selected_model_name,
                "messages": st.session_state.get('messages', [])
            }
            
            st.download_button(
                label="💾 ダウンロード",
                data=json.dumps(chat_data, ensure_ascii=False, indent=2),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
    
    # Claude からの応答を取得（ストリーミング）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # API呼び出し用のメッセージ形式に変換
            api_messages = []
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
                        input_cost = (input_tokens / 1_000_000) * 15
                        output_cost = (output_tokens / 1_000_000) * 75
                    elif "sonnet-4" in selected_model:
                        input_cost = (input_tokens / 1_000_000) * 3
                        output_cost = (output_tokens / 1_000_000) * 15
                    elif "sonnet" in selected_model:
                        input_cost = (input_tokens / 1_000_000) * 3
                        output_cost = (output_tokens / 1_000_000) * 15
                    elif "haiku" in selected_model:
                        input_cost = (input_tokens / 1_000_000) * 1
                        output_cost = (output_tokens / 1_000_000) * 5
                    else:
                        input_cost = output_cost = 0
                    
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
st.caption("🔒 このアプリはパスワードで保護されています")