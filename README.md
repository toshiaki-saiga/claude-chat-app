## Web検索機能（Brave Search API）

### 新機能
- ✅ リアルタイムWeb検索
- ✅ 最新情報の取得
- ✅ ディープリサーチモード（複数回検索）
- ✅ 検索プロセスの可視化

### 使い方

1. **Brave Search APIキーの取得**
   - https://api-dashboard.search.brave.com/register でアカウント作成
   - Free AIプラン（月2,000検索まで無料）を選択
   - APIキーを取得

2. **Secretsへの追加**
   - Streamlit Cloud > Settings > Secrets
   - `BRAVE_SEARCH_API_KEY = "your-key"` を追加

3. **検索の使用**
   - サイドバーで「Web検索を有効化」をON
   - 検索結果数を調整（1-10件）
   - ディープリサーチモードをONにすると3回の検索を実行

### 使用例

**通常検索:**