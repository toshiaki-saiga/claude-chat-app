# Claude API チャットアプリ

Claude APIを使用した高機能チャットアプリケーションです。

## 機能

- 最新のClaude 4.5 Sonnet、4.1 Opusに対応
- ファイル添付機能（画像・テキスト）
- ストリーミング応答
- パスワード認証
- トークン使用量・コスト表示
- 会話の保存機能

## 使用方法

1. Streamlit Community Cloudにデプロイ
2. SecretsにAPIキーとパスワードを設定
3. アクセスしてパスワードを入力

## 必要な環境変数

- `ANTHROPIC_API_KEY`: AnthropicのAPIキー
- `APP_PASSWORD`: アプリへのアクセス用パスワード