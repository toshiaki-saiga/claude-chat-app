import streamlit as st
import anthropic
import base64
from PIL import Image
import io

# ===== Web Search Integration =====
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import BraveSearch
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import time
from datetime import datetime

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

# ===== LangChain Search Agent Initialization =====
@st.cache_resource
def init_search_agent(model_name: str, search_results: int = 5):
    """
    LangChain ReActエージェントを初期化
    
    Args:
        model_name: 使用するClaudeモデル名
        search_results: 検索結果の数
        
    Returns:
        AgentExecutor: 初期化されたエージェント実行環境
    """
    try:
        # Claude LLMの初期化（LangChain統合版）
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.3,  # 検索時は低温度で正確性を重視
            max_tokens=4096,
            streaming=True,
            anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
        )
        
        # Brave Searchツールの初期化
        brave_search = BraveSearch.from_api_key(
            api_key=st.secrets["BRAVE_SEARCH_API_KEY"],
            search_kwargs={
                "count": search_results,
                "safesearch": "moderate"  # セーフサーチを有効化
            }
        )
        
        tools = [brave_search]
        
        # ReActプロンプトテンプレートの取得
        # このプロンプトは「思考→行動→観察」のサイクルを実装
        prompt = hub.pull("hwchase17/react")
        
        # エージェントの作成
        agent = create_react_agent(llm, tools, prompt)
        
        # エージェント実行環境の作成
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # デバッグ用に詳細ログを出力
            handle_parsing_errors=True,  # パースエラーを自動処理
            max_iterations=5,  # 無限ループ防止
            max_execution_time=120,  # 2分でタイムアウト
            early_stopping_method="generate"  # 早期終了戦略
        )
        
        return agent_executor
        
    except Exception as e:
        st.error(f"エージェント初期化エラー: {str(e)}")
        return None


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


@rate_limited_search
def execute_deep_research(agent_executor, query: str, iterations: int = 3):
    """
    ディープリサーチ: 複数回の検索と分析を実行
    
    Args:
        agent_executor: 初期化されたエージェント
        query: ユーザーのクエリ
        iterations: 検索の繰り返し回数
        
    Returns:
        list: 各検索の結果のリスト
    """
    results = []
    
    with st.status("🔍 ディープリサーチ実行中...", expanded=True) as status:
        # 初回検索
        st.write(f"**検索 1/{iterations}**: 基本情報の収集")
        try:
            initial_result = agent_executor.invoke({"input": query})
            results.append({
                "iteration": 1,
                "type": "initial",
                "result": initial_result
            })
            st.success("✅ 初回検索完了")
        except Exception as e:
            st.error(f"❌ 初回検索エラー: {str(e)}")
            return results
        
        # 追加の深掘り検索
        for i in range(iterations - 1):
            st.write(f"**検索 {i+2}/{iterations}**: 詳細情報の追加収集")
            
            # 前回の結果から追加の質問を生成
            follow_up_query = f"""
            以下のトピックについて、さらに詳しく調査してください：
            {query}
            
            前回の検索で得られた情報を踏まえて、特に以下の観点で情報を補完してください：
            - 最新の動向やトレンド（過去6ヶ月以内）
            - 具体的な数値データや統計
            - 専門家の意見や評価
            - 実際の事例やケーススタディ
            
            前回取得できなかった新しい情報を重点的に調査してください。
            """
            
            try:
                result = agent_executor.invoke({"input": follow_up_query})
                results.append({
                    "iteration": i + 2,
                    "type": "follow_up",
                    "result": result
                })
                st.success(f"✅ 検索 {i+2} 完了")
                
                # レート制限対策: 1秒待機
                time.sleep(1)
                
            except Exception as e:
                st.warning(f"⚠️ 検索 {i+2} でエラー: {str(e)}")
                continue
        
        status.update(label="✅ ディープリサーチ完了", state="complete")
    
    return results

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
        
        deep_research = st.checkbox(
            "🔬 ディープリサーチモード",
            value=False,
            help="3回の検索を実行してより包括的な情報を収集します"
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
                # エージェントの初期化
                agent_executor = init_search_agent(
                    selected_model,
                    search_count if 'search_count' in locals() else 5
                )
                
                if agent_executor is None:
                    st.error("エージェントの初期化に失敗しました。通常モードで回答します。")
                    enable_search = False
                else:
                    # ディープリサーチモードの判定
                    if 'deep_research' in locals() and deep_research:
                        st.info("🔬 ディープリサーチモードで実行します（3回の検索）")
                        
                        # ディープリサーチ実行
                        results = execute_deep_research(
                            agent_executor,
                            prompt,
                            iterations=3
                        )
                        
                        # 結果の統合
                        full_response = f"# 🔬 ディープリサーチ結果\n\n"
                        full_response += f"**検索クエリ:** {prompt}\n\n"
                        full_response += f"**実行時刻:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        full_response += "---\n\n"
                        
                        for idx, res in enumerate(results, 1):
                            full_response += f"## 検索フェーズ {idx}\n\n"
                            full_response += res["result"]["output"] + "\n\n"
                            full_response += "---\n\n"
                        
                        # 統合サマリー
                        full_response += "## 📊 総合まとめ\n\n"
                        full_response += f"上記{len(results)}回の検索から得られた情報を統合し、"
                        full_response += "最新かつ包括的な回答を提供しました。\n"
                        
                    else:
                        # 通常の検索（1回）
                        st.info("🔍 Web検索を使用して回答を生成しています...")
                        
                        with st.expander("🔍 検索プロセス（詳細）", expanded=False):
                            st_callback = StreamlitCallbackHandler(st.container())
                            result = agent_executor.invoke(
                                {"input": prompt},
                                {"callbacks": [st_callback]}
                            )
                        
                        full_response = result["output"]
                    
                    # レスポンスを表示
                    st.markdown(full_response)
                    
                    # 検索情報の表示
                    with st.expander("📊 検索情報"):
                        st.write("✅ Web検索を使用して最新情報を取得しました")
                        st.write(f"**検索結果数:** {search_count if 'search_count' in locals() else 5}件")
                        if 'deep_research' in locals() and deep_research:
                            st.write(f"**モード:** ディープリサーチ（3回検索）")
                        else:
                            st.write(f"**モード:** 通常検索（1回）")
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