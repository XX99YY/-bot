"""蟹蟹里：基于 Streamlit 和 DeepSeek API 的网页聊天机器人。"""

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
)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🦀",
    layout="centered",
)


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
    """调用 DeepSeek API，并逐段返回生成的文本。"""
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


CUSTOM_CSS = """
<style>
:root {
    --ink: #263a50;
    --muted: #60748a;
    --coral: #e8785e;
    --coral-deep: #cf5d46;
    --sky: #5b93c4;
    --line: rgba(255, 255, 255, 0.82);
}

html,
body,
.stApp,
[data-testid="stMarkdownContainer"],
[data-testid="stChatInputTextArea"],
button,
input,
textarea {
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC",
        "Noto Sans SC", sans-serif;
}

.stApp {
    background-color: #edf6ff;
    background-image:
        linear-gradient(
            180deg,
            rgba(247, 251, 255, 0.22) 0%,
            rgba(255, 249, 247, 0.3) 100%
        ),
        url("/app/static/background.jpg");
    background-position: center center;
    background-repeat: no-repeat;
    background-size: cover;
    background-attachment: fixed;
}

/* 隐藏 Streamlit 自带的英文工具条 */
[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
#MainMenu,
[data-testid="stStatusWidget"],
[data-testid="stToolbarActions"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* 主聊天卡片 */
[data-testid="stMainBlockContainer"] {
    box-sizing: border-box;
    max-width: 780px;
    min-height: calc(100vh - 7.5rem);
    margin-top: 1.1rem;
    margin-bottom: 0;
    padding: 2.2rem 2.5rem 5rem !important;
    background: rgba(255, 255, 255, 0.68);
    border: 1px solid var(--line);
    border-radius: 28px;
    box-shadow: 0 20px 60px rgba(53, 83, 112, 0.17);
    backdrop-filter: blur(26px) saturate(118%);
    -webkit-backdrop-filter: blur(26px) saturate(118%);
}

[data-testid="stMainBlockContainer"] h1 {
    margin: 0 0 0.35rem;
    color: #213d58;
    font-size: clamp(2.35rem, 4.5vw, 3.4rem);
    font-weight: 800;
    letter-spacing: 0.04em;
}

[data-testid="stHeaderActionElements"] {
    display: none !important;
}

[data-testid="stMainBlockContainer"] h1::after {
    content: "";
    display: block;
    width: 4.4rem;
    height: 0.32rem;
    margin-top: 0.8rem;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--coral), #f4bf8f);
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--muted);
}

/* 侧边栏 */
[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(255, 255, 255, 0.76) 0%,
            rgba(239, 247, 255, 0.74) 100%
        );
    border-right: 1px solid rgba(255, 255, 255, 0.78);
    box-shadow: 12px 0 40px rgba(55, 85, 110, 0.08);
    backdrop-filter: blur(20px) saturate(110%);
    -webkit-backdrop-filter: blur(20px) saturate(110%);
}

[data-testid="stSidebarContent"] {
    padding-top: 0.6rem;
}

[data-testid="stSidebar"] h2 {
    color: #294761;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(76, 111, 142, 0.16);
}

/* 按钮 */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"] {
    color: #46647e;
    background: rgba(255, 255, 255, 0.62);
    border: 1px solid rgba(80, 119, 151, 0.24);
    border-radius: 13px;
    transition: all 160ms ease;
}

[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    color: var(--coral-deep);
    border-color: rgba(232, 120, 94, 0.68);
    background: rgba(255, 250, 247, 0.92);
}

[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"] {
    color: #ffffff;
    background: linear-gradient(135deg, #f39a80, var(--coral));
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 13px;
    box-shadow: 0 8px 22px rgba(220, 99, 74, 0.23);
}

[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background: linear-gradient(135deg, #f5a58d, #dd6a52);
    border-color: rgba(255, 255, 255, 0.5);
}

/* 聊天消息 */
[data-testid="stChatMessage"] {
    margin: 0.55rem 0;
    padding: 0.85rem 1rem;
    background: rgba(255, 255, 255, 0.76);
    border: 1px solid rgba(255, 255, 255, 0.86);
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(58, 89, 117, 0.06);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--ink);
    line-height: 1.75;
}

[data-testid="stAlert"] {
    background: rgba(239, 247, 255, 0.82);
    border: 1px solid rgba(126, 174, 210, 0.3);
    border-radius: 16px;
    color: var(--ink);
}

/* 底部聊天输入框 */
[data-testid="stBottom"] {
    background: transparent !important;
}

[data-testid="stBottom"] > div {
    background: transparent !important;
}

[data-testid="stBottomBlockContainer"] {
    max-width: 780px;
    padding-bottom: 1.15rem;
}

[data-testid="stChatInput"] {
    border-radius: 19px;
    box-shadow: 0 12px 32px rgba(47, 77, 103, 0.16);
}

[data-testid="stChatInput"] > div {
    padding: 11px 15px;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.92);
    border-radius: 19px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}

[data-testid="stChatInputTextArea"] {
    color: #314a61;
}

[data-testid="stChatInputTextArea"]::placeholder {
    color: #8193a6;
}

[data-testid="stChatInputSubmitButton"] {
    color: var(--coral);
}

[data-testid="stChatInputSubmitButton"]:disabled {
    color: #b6c3cf;
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] {
        min-height: calc(100vh - 7.5rem);
        margin: 0.7rem 0.55rem 0;
        padding: 1.5rem 1.15rem 5rem !important;
        border-radius: 21px;
    }

    [data-testid="stBottomBlockContainer"] {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        transition: none !important;
        animation: none !important;
    }
}
</style>
"""


def inject_styles() -> None:
    """使用内联样式，避免云端缺少外部 CSS 文件时报错。"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    inject_styles()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    api_key = get_configured_api_key()

    with st.sidebar:
        st.header("聊天设置")
        st.divider()

        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption("对话会发送给深度求索，请勿输入敏感信息。")

    st.title(f"🦀 {APP_NAME}")
    st.caption("你的中文聊天小伙伴，今天想聊点什么？")

    for message in st.session_state.messages:
        avatar = "🦀" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if not api_key:
        st.info("应用尚未配置访问密钥，请联系管理员。")
    elif not st.session_state.messages:
        st.info("你好呀！我是蟹蟹里，很高兴见到你。有什么想聊的？")

    prompt = st.chat_input("请输入你的问题……", disabled=not api_key)

    if prompt:
        if not api_key:
            st.warning("应用尚未配置访问密钥，请联系管理员。")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})

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
                st.error(f"深度求索接口返回错误（状态码：{exc.status_code}），请稍后重试。")
            except Exception as exc:
                st.error(f"调用失败：{exc}")
            else:
                if isinstance(response_text, str) and response_text.strip():
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
                else:
                    st.warning("蟹蟹里没有返回内容，请重试。")

    st.caption("回答由人工智能生成，请注意甄别。")


if __name__ == "__main__":
    main()
