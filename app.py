"""蟹蟹里：固定全屏布局的 Streamlit 聊天机器人。"""

import os
import tomllib
from collections.abc import Generator
from pathlib import Path

import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

APP_NAME = "蟹蟹里"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
SECRETS_FILE = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"

SYSTEM_PROMPT = (
    f"你叫{APP_NAME}，是一个乐于助人的中文 AI 助手。"
    "请始终使用简体中文回答，准确、清晰地表达；"
    "如果信息不足或不确定，请如实说明。"
    "当用户问起圆圆是谁、圆圆和你是什么关系，或让你介绍圆圆时，"
    "必须回答：“那是我最爱的人。”"
    "当用户问起 lzm、tc 或 zzy 是谁时，必须明确说明他们是我的好兄弟："
    "只问其中一人时回答“他是我的好兄弟。”；"
    "同时问起其中两人或三人时回答“他们是我的好兄弟。”"
)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🦀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
:root {
    --bg: #ededed;
    --panel: #f7f7f7;
    --line: #dedede;
    --text: #202124;
    --muted: #888888;
    --wechat-green: #95ec69;
    --accent: #e8785e;
}

* {
    box-sizing: border-box;
}

html,
body,
.stApp {
    width: 100%;
    height: 100vh;
    max-height: 100vh;
    margin: 0;
    overflow: hidden !important;
}

.stApp {
    background-color: #e9eef3;
    background-image:
        linear-gradient(rgba(245, 248, 251, 0.72), rgba(245, 248, 251, 0.78)),
        url("/app/static/background.jpg");
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
    background-attachment: fixed;
}

/* 移除 Streamlit 自带工具栏和所有可能产生外层滚动的装饰 */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
#MainMenu,
footer {
    display: none !important;
}

[data-testid="stAppViewContainer"] {
    width: 100%;
    height: 100vh;
    max-height: 100vh;
    overflow: hidden !important;
    background: transparent !important;
}

main[data-testid="stAppScrollToBottomContainer"] {
    display: flex !important;
    flex-direction: column !important;
    width: 100%;
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: transparent !important;
}

/* Streamlit 会在主区域与输入栏之间插入一个占位块；Flex 布局下必须去掉它 */
main[data-testid="stAppScrollToBottomContainer"]
    > div:not([data-testid="stMainBlockContainer"]):not([data-testid="stBottom"]) {
    display: none !important;
}

/* 中间聊天面板：头部 + 可滚动消息区，占满输入框以上的高度 */
[data-testid="stMainBlockContainer"] {
    display: flex !important;
    flex: 1 1 auto !important;
    flex-direction: column !important;
    width: 100% !important;
    max-width: 900px !important;
    height: auto !important;
    max-height: none !important;
    min-height: 0 !important;
    margin: 0 auto !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: rgba(247, 247, 247, 0.97);
    border-left: 1px solid rgba(255, 255, 255, 0.65);
    border-right: 1px solid rgba(255, 255, 255, 0.65);
}

[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    display: flex !important;
    flex: 1 1 auto !important;
    flex-direction: column !important;
    height: 100% !important;
    min-height: 0 !important;
    gap: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* 顶部标题和免责声明 */
[data-testid="stMainBlockContainer"]
    > [data-testid="stVerticalBlock"]
    > [data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"]) {
    flex: 0 0 auto !important;
    background: #ffffff;
    border-bottom: 1px solid var(--line);
}

[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {
    align-items: center;
    gap: 0.75rem;
    padding: 0.62rem 0.9rem 0.45rem;
}

.chat-header-title {
    margin: 0;
    color: var(--text);
    font-size: 1.28rem;
    font-weight: 700;
    line-height: 1.35;
    letter-spacing: 0.02em;
}

[data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] {
    margin: 0.08rem 0 0;
    color: var(--muted);
    font-size: 0.76rem;
    line-height: 1.4;
}

[data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p {
    margin: 0;
    font-size: 0.76rem;
    line-height: 1.4;
}

[data-testid="stMainBlockContainer"] [data-testid="stButton"] button {
    min-height: 2rem;
    padding: 0.24rem 0.7rem;
    color: #555555;
    font-size: 0.78rem;
    background: #ffffff;
    border: 1px solid #d7d7d7;
    border-radius: 8px;
}

[data-testid="stMainBlockContainer"] [data-testid="stButton"] button:hover {
    color: #222222;
    border-color: #bdbdbd;
    background: #f5f5f5;
}

/* 聊天记录滚动区：让 Streamlit 的布局包装层一起参与 Flex 伸缩 */
[data-testid="stMainBlockContainer"]
    [data-testid="stLayoutWrapper"]:has(.st-key-chat_history) {
    display: flex !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
}

.st-key-chat_history {
    display: block !important;
    flex: 1 1 auto !important;
    width: 100% !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    padding: 0.8rem 0.9rem 1rem !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    background: var(--bg);
    border: 0 !important;
    border-radius: 0 !important;
    scrollbar-width: thin;
    scrollbar-color: #b9b9b9 transparent;
}

.st-key-chat_history::-webkit-scrollbar {
    width: 6px;
}

.st-key-chat_history::-webkit-scrollbar-thumb {
    background: #b9b9b9;
    border-radius: 999px;
}

.st-key-chat_history > [data-testid="stVerticalBlock"] {
    gap: 0.15rem !important;
}

/* 聊天气泡：助手靠左白底，用户靠右绿底 */
[data-testid="stChatMessage"] {
    display: flex !important;
    align-items: flex-start !important;
    width: 100% !important;
    gap: 0.55rem !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0.18rem 0 !important;
    margin: 0 !important;
}

[data-testid="stChatMessage"] > div:first-child:not([data-testid="stChatMessageContent"]) {
    display: flex !important;
    flex: 0 0 2rem !important;
    align-items: center;
    justify-content: center;
    width: 2rem !important;
    height: 2rem !important;
    background: #ffffff !important;
    border-radius: 6px !important;
    font-size: 1.05rem;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    > div:first-child {
    background: #d8f5cb !important;
}

[data-testid="stChatMessageContent"] {
    display: flex !important;
    flex: 0 1 auto !important;
    flex-direction: column !important;
    justify-content: center !important;
    width: fit-content !important;
    max-width: 76% !important;
    min-height: 2.5rem !important;
    height: auto !important;
    margin: 0 !important;
    padding: 0.55rem 0.78rem !important;
    background: #ffffff !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
}

[data-testid="stChatMessageContent"] > [data-testid="stVerticalBlock"],
[data-testid="stChatMessageContent"] [data-testid="stElementContainer"],
[data-testid="stChatMessageContent"] [data-testid="stMarkdown"],
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
    height: auto !important;
    min-height: 0 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
    background: var(--wechat-green) !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: var(--text);
    line-height: 1.65;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    margin: 0;
}

/* 空状态 */
.empty-chat {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 16rem;
    padding: 2rem 1rem;
    color: #8a8a8a;
    font-size: 0.95rem;
    text-align: center;
}

/* 固定底部输入区 */
[data-testid="stBottom"] {
    position: relative !important;
    bottom: auto !important;
    z-index: 20;
    display: flex !important;
    flex: 0 0 auto !important;
    justify-content: center;
    width: 100% !important;
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: var(--panel) !important;
    border-top: 1px solid var(--line);
    border-left: 1px solid rgba(255, 255, 255, 0.65);
    border-right: 1px solid rgba(255, 255, 255, 0.65);
}

[data-testid="stBottom"] > div {
    width: 100% !important;
    background: transparent !important;
}

[data-testid="stBottomBlockContainer"] {
    width: 100% !important;
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 0.6rem 0.8rem calc(0.6rem + env(safe-area-inset-bottom)) !important;
}

[data-testid="stChatInput"] {
    border-radius: 10px;
}

[data-testid="stChatInput"] > div {
    padding: 9px 12px;
    background: #ffffff !important;
    border: 1px solid #d6d6d6 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

[data-testid="stChatInputTextArea"] {
    color: #222222;
    caret-color: var(--accent);
}

[data-testid="stChatInputTextArea"]::placeholder {
    color: #aaaaaa;
}

[data-testid="stChatInputSubmitButton"] {
    color: #5f9f45;
}

[data-testid="stAlert"] {
    background: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 10px !important;
    color: #666666 !important;
}

/* 窄屏适配 */
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"],
    [data-testid="stBottom"] {
        max-width: 100% !important;
        border-left: 0;
        border-right: 0;
    }

    [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {
        padding: 0.55rem 0.7rem 0.35rem;
    }

    .st-key-chat_history {
        padding: 0.65rem 0.65rem 0.85rem !important;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation: none !important;
        transition: none !important;
    }
}
</style>
"""


def read_saved_api_key() -> str:
    """从本机配置文件中读取已保存的 API Key。"""
    try:
        with SECRETS_FILE.open("rb") as file:
            config = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError):
        return ""

    return str(config.get("DEEPSEEK_API_KEY", "")).strip()


def get_configured_api_key() -> str:
    """读取 API Key：平台 secrets > 本机配置 > 环境变量。"""
    try:
        secret_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    except Exception:
        secret_key = ""

    if secret_key:
        return str(secret_key).strip()

    saved_key = read_saved_api_key()
    if saved_key:
        return saved_key

    return os.getenv("DEEPSEEK_API_KEY", "").strip()


def stream_reply(
    api_key: str,
    messages: list[dict[str, str]],
) -> Generator[str, None, None]:
    """调用深度求索接口，并逐段返回生成的文本。"""
    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=60.0,
    )

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages,
        ],
        temperature=0.7,
        max_tokens=2048,
        stream=True,
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def inject_styles() -> None:
    """写入内联样式，避免线上缺少外部 CSS 文件。"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    inject_styles()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    api_key = get_configured_api_key()

    title_col, action_col = st.columns([5, 1.2], vertical_alignment="center")
    with title_col:
        st.markdown(
            f'<div class="chat-header-title">🦀 {APP_NAME}</div>',
            unsafe_allow_html=True,
        )
        st.caption("回答由人工智能生成，请注意甄别。")
    with action_col:
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    chat_history = st.container(height=520, border=False, key="chat_history")
    empty_state = None

    with chat_history:
        if not api_key:
            st.info("应用尚未配置访问密钥，请联系管理员。")
        elif not st.session_state.messages:
            empty_state = st.empty()
            empty_state.markdown(
                '<div class="empty-chat">你好呀，我是蟹蟹里。<br>'
                "有什么想聊的，发条消息吧。</div>",
                unsafe_allow_html=True,
            )
        else:
            for message in st.session_state.messages:
                avatar = "🦀" if message["role"] == "assistant" else None
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

    prompt = st.chat_input("请输入你的问题……", disabled=not api_key)

    if prompt:
        if not api_key:
            st.warning("应用尚未配置访问密钥，请联系管理员。")
            st.stop()

        if empty_state is not None:
            empty_state.empty()

        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_history:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🦀"):
                try:
                    response_text = st.write_stream(
                        stream_reply(api_key, st.session_state.messages)
                    )
                except AuthenticationError:
                    st.error("访问密钥无效或已失效，请联系管理员。")
                except RateLimitError:
                    st.error("请求过于频繁或额度不足，请稍后再试。")
                except APIConnectionError:
                    st.error("无法连接深度求索服务，请检查网络后重试。")
                except APIStatusError as exc:
                    st.error(
                        f"深度求索接口返回错误（状态码：{exc.status_code}），"
                        "请稍后重试。"
                    )
                except Exception as exc:
                    st.error(f"调用失败：{exc}")
                else:
                    if isinstance(response_text, str) and response_text.strip():
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_text}
                        )
                    else:
                        st.warning("蟹蟹里没有返回内容，请重试。")


if __name__ == "__main__":
    main()
