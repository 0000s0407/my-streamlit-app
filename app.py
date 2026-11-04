

import os
import re
import requests
import streamlit as st
import xml.etree.ElementTree as ET

from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from html import unescape
from openai import OpenAI

# =====================================================
# AI 개인비서 에이블 - Streamlit 웹앱
# 기능: 음성 챗봇, STT, 대화 기록, 경제뉴스 브리핑
# =====================================================

st.set_page_config(
    page_title="AI 개인비서 에이블",
    page_icon="🤖",
    layout="wide"
)

client = OpenAI()

# AI 개인비서 역할
sys_role = """
당신은 AI 개인비서 '에이블'입니다.
사용자의 업무와 일상생활을 돕는 친절한 비서입니다.
답변은 정확하고 이해하기 쉽게 작성하세요.
"""

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "economy_briefing" not in st.session_state:
    st.session_state.economy_briefing = ""

# 사이드바
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/Jangrae/img/master/ai_robot.png",
        width=150
    )
    st.title("AI 개인비서 에이블")
    menu = st.radio(
        "메뉴 선택",
        ["💬 AI 챗봇", "📰 경제뉴스 브리핑"]
    )

    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# =====================================================
# 1. AI 챗봇
# =====================================================

if menu == "💬 AI 챗봇":

    st.title("💬 AI 개인비서 에이블")
    st.write("텍스트 또는 음성으로 질문해 주세요.")

    uploaded_file = st.file_uploader(
        "참고할 텍스트 파일 업로드 (선택)",
        type=["txt"]
    )

    # 이전 대화 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 음성 입력
    audio_value = st.audio_input("🎤 음성으로 질문하기")

    # 텍스트 입력
    text_input = st.chat_input("에이블에게 질문하세요.")

    user_question = None

    # 음성이 입력된 경우
    if audio_value is not None:

        audio_bytes = audio_value.getvalue()

        # 동일한 녹음을 반복 처리하지 않도록 확인
        import hashlib
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

        if st.session_state.get("last_audio_hash") != audio_hash:

            with st.spinner("음성을 인식하고 있습니다..."):
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", audio_bytes, "audio/wav")
                )

            user_question = transcript.text
            st.session_state.last_audio_hash = audio_hash

            # STT 결과 표시
            st.info(f"🎤 음성 인식 결과: {user_question}")

    # 텍스트 입력이 있으면 사용
    if text_input:
        user_question = text_input

    # 질문 처리
    if user_question:

        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        # 최근 10개 메시지만 유지
        st.session_state.messages = (
            st.session_state.messages[-10:]
        )

        system_content = sys_role

        if uploaded_file is not None:
            try:
                file_text = uploaded_file.getvalue().decode("utf-8")
                system_content += (
                    "\n\n다음 자료를 참고하세요:\n"
                    + file_text[:10000]
                )
            except UnicodeDecodeError:
                st.warning("UTF-8 텍스트 파일을 업로드해 주세요.")

        input_messages = [
            {"role": "system", "content": system_content}
        ] + st.session_state.messages

        with st.spinner("에이블이 답변을 준비하고 있습니다..."):

            response = client.responses.create(
                model="gpt-4o-mini",
                input=input_messages
            )

            answer = response.output_text

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.session_state.messages = (
            st.session_state.messages[-10:]
        )

        with st.chat_message("assistant"):
            st.write(answer)

# =====================================================
# 2. 경제뉴스 수집
# =====================================================

@st.cache_data(ttl=900)
def collect_economy_news():

    keywords = [
        "한국 경제 when:7d",
        "환율 금리 주식 when:7d",
        "부동산 고용 물가 when:7d",
        "세계 경제 미국 중국 when:7d"
    ]

    news = []
    seen = set()

    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    for keyword in keywords:

        url = (
            "https://news.google.com/rss/search?q="
            + quote(keyword)
            + "&hl=ko&gl=KR&ceid=KR:ko"
        )

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            for item in root.findall(".//item"):

                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                pub_date = item.findtext("pubDate", "")

                source_element = item.find("source")
                source = (
                    source_element.text
                    if source_element is not None
                    else "출처 미상"
                )

                description = item.findtext("description", "")
                description = re.sub(
                    r"<[^>]+>", " ", description
                )
                description = unescape(description).strip()

                if not title or not link or not pub_date:
                    continue

                published = parsedate_to_datetime(pub_date)

                if published.tzinfo is None:
                    published = published.replace(
                        tzinfo=timezone.utc
                    )

                if not week_ago <= published <= now:
                    continue

                normalized_title = title.lower()

                if normalized_title in seen:
                    continue

                seen.add(normalized_title)

                news.append({
                    "title": title,
                    "date": published.strftime("%Y-%m-%d"),
                    "source": source,
                    "description": description[:500],
                    "link": link
                })

        except Exception as e:
            print("뉴스 수집 오류:", e)

    news.sort(
        key=lambda x: x["date"],
        reverse=True
    )

    return news[:40]


# =====================================================
# 3. AI 경제뉴스 브리핑 생성
# =====================================================

def generate_economy_briefing(news):

    news_text = "\n\n".join(
        f"제목: {n['title']}\n"
        f"날짜: {n['date']}\n"
        f"언론사: {n['source']}\n"
        f"요약: {n['description']}\n"
        f"링크: {n['link']}"
        for n in news
    )

    prompt = f"""
당신은 AI 개인비서 '에이블'입니다.
경제 초보자를 위한 주간 경제뉴스 브리핑을 작성하세요.

다음은 최근 7일간 수집된 경제뉴스입니다.

{news_text}

조건:
1. 중요한 경제뉴스 5개를 선정하세요.
2. 서로 다른 사건을 우선 선정하세요.
3. 각 뉴스마다 아래 내용을 작성하세요.
   - 제목, 날짜, 언론사
   - 무슨 일이 있었나?
   - 왜 발생했나?
   - 왜 중요한가?
   - 우리 생활에 미치는 영향
   - 어려운 경제용어 풀이
   - 한 줄 요약
   - 원문 링크
4. 마지막에 이번 주 경제 흐름을 3가지로 정리하세요.
5. 경제 초보자도 이해하도록 쉽게 설명하세요.
6. 제공된 기사 정보만 활용하고,
   확인할 수 없는 사실은 추측하지 마세요.
7. Markdown 형식으로 작성하세요.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "당신은 경제뉴스 해설 전문가입니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.output_text


# =====================================================
# 4. 경제뉴스 브리핑 화면
# =====================================================

if menu == "📰 경제뉴스 브리핑":

    st.title("📰 에이블 주간 경제뉴스 브리핑")

    st.write(
        "최근 7일간 주요 경제뉴스 5개를 "
        "AI가 쉽고 자세하게 설명합니다."
    )

    if st.button(
        "이번 주 경제뉴스 브리핑 생성",
        type="primary"
    ):

        with st.spinner("경제뉴스를 수집하고 분석하고 있습니다..."):

            try:
                news = collect_economy_news()

                if not news:
                    st.warning("최근 경제뉴스를 가져오지 못했습니다.")

                else:
                    briefing = generate_economy_briefing(news)
                    st.session_state.economy_briefing = briefing

            except Exception as e:
                st.error(f"브리핑 생성 실패: {e}")

    if st.session_state.economy_briefing:
        st.markdown(st.session_state.economy_briefing)
