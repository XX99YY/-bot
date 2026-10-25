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
STYLES_FILE = Path(__file__).resolve().parent / "styles.css"

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


def inject_styles() -> None:
    """加载浅色背景和聊天界面的自定义样式。"""
    st.markdown(
        f"<style>{STYLES_FILE.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


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
