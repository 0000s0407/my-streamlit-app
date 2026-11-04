

import csv
import hashlib
import io
import json
import random
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI 개인비서 에이블", page_icon="🤖", layout="wide")
KST = ZoneInfo("Asia/Seoul")
today = datetime.now(KST).date()
client = OpenAI()

st.markdown("""<style>.block-container{max-width:1100px;padding-top:2rem}h1{letter-spacing:-1px}</style>""", unsafe_allow_html=True)

if "chat" not in st.session_state:
    st.session_state.chat = []
if "ledger" not in st.session_state:
    st.session_state.ledger = []
if "goal" not in st.session_state:
    st.session_state.goal = 1000000
if "saved" not in st.session_state:
    st.session_state.saved = 0
if "fortune" not in st.session_state:
    st.session_state.fortune = ""
if "quote" not in st.session_state:
    st.session_state.quote = ""

QUOTES = [
    "천 리 길도 한 걸음부터.", "시작이 반이다.", "오늘의 작은 노력이 내일을 만든다.",
    "느리게 가더라도 멈추지 않는 것이 중요하다.", "어제보다 나은 오늘이면 충분하다.",
    "완벽한 시작보다 꾸준한 실천이 중요하다.", "작은 성취도 축하받을 가치가 있다.",
    "당신의 속도로 걸어가도 괜찮다.", "도전은 새로운 가능성의 시작이다.",
    "하루에 한 걸음씩, 충분히 멋진 변화다."
]


def ask_ai(system, user, max_tokens=1100):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or "답변을 생성하지 못했어요."


def transcribe(audio):
    return client.audio.transcriptions.create(
        model="whisper-1", file=("recording.wav", audio.getvalue(), "audio/wav")
    ).text


@st.cache_data(ttl=900)
def collect_news():
    queries = ["한국 경제 when:7d", "환율 금리 주식 when:7d", "부동산 고용 물가 when:7d", "세계 경제 미국 중국 when:7d"]
    found, seen = [], set()
    now = datetime.now(timezone.utc)
    for query in queries:
        url = "https://news.google.com/rss/search?q=" + quote(query) + "&hl=ko&gl=KR&ceid=KR:ko"
        try:
            r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                raw_date = item.findtext("pubDate", "")
                if not (title and link and raw_date):
                    continue
                pub = parsedate_to_datetime(raw_date)
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                if not now - timedelta(days=7) <= pub <= now or title.lower() in seen:
                    continue
                seen.add(title.lower())
                source_node = item.find("source")
                source = source_node.text if source_node is not None else "출처 미상"
                desc = unescape(re.sub(r"<[^>]*>", " ", item.findtext("description", "")))
                found.append({"title": title, "link": link, "date": pub.strftime("%Y-%m-%d"), "source": source, "description": desc[:350], "sort": pub})
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            print("뉴스 수집 실패:", exc)
    found.sort(key=lambda x: x["sort"], reverse=True)
    return [{k: v for k, v in x.items() if k != "sort"} for x in found[:35]]


def parse_expenses(text):
    system = """한국어 소비 내역을 분석하는 도우미입니다. 반드시 JSON 객체만 반환하세요.
형식: {"items":[{"item":"점심","category":"식비","amount":9000}]}
category는 식비, 카페, 교통, 쇼핑, 생활, 문화, 기타 중 하나입니다.
'9천 원'은 9000, '1만 2천 원'은 12000입니다. 지출이 아닌 저축/수입은 제외하세요.
금액이 불명확한 항목은 제외하세요. 절대로 금액을 추측하지 마세요."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    cleaned = []
    categories = {"식비", "카페", "교통", "쇼핑", "생활", "문화", "기타"}
    for row in payload.get("items", []):
        try:
            amount = int(row["amount"])
            if amount <= 0 or amount > 100000000:
                continue
            category = str(row.get("category", "기타"))
            cleaned.append({"날짜": today.isoformat(), "항목": str(row.get("item", "지출"))[:60], "분류": category if category in categories else "기타", "금액": amount})
        except (KeyError, TypeError, ValueError):
            continue
    return cleaned


with st.sidebar:
    st.title("🤖 에이블")
    st.caption("나만의 AI 생활 비서")
    menu = st.radio("메뉴", ["💬 AI 챗봇", "📰 경제뉴스 브리핑", "🔮 에이블철학관", "💰 에이블 머니가든", "👗 에이블 옷장"])
    st.divider()
    st.caption("OpenAI API 사용료가 발생할 수 있습니다.")

if menu == "💬 AI 챗봇":
    st.title("💬 에이블 AI 챗봇")
    st.write("텍스트 또는 음성으로 질문해 보세요.")
    with st.expander("📎 참고 텍스트 파일 (선택)"):
        uploaded = st.file_uploader("TXT 파일 업로드", type=["txt"], key="chatfile")
    reference = ""
    if uploaded:
        try:
            reference = uploaded.getvalue().decode("utf-8-sig")[:12000]
        except UnicodeDecodeError:
            st.warning("UTF-8 인코딩 TXT 파일만 지원합니다.")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    with st.form("chat_form", clear_on_submit=True):
        typed = st.text_input("질문 입력", placeholder="오늘 할 일을 정리해 줘")
        audio = st.audio_input("🎤 음성 질문")
        submitted = st.form_submit_button("질문 보내기", type="primary")
    if submitted:
        question = typed.strip()
        if audio is not None:
            try:
                with st.spinner("음성을 인식하는 중..."):
                    question = transcribe(audio)
                st.info("🎤 인식된 질문: " + question)
            except Exception as e:
                st.error(f"음성 인식 실패: {e}")
                question = ""
        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            try:
                with st.spinner("에이블이 생각하고 있어요..."):
                    sys = "당신은 친절하고 정확한 한국어 AI 개인비서 '에이블'입니다."
                    if reference:
                        sys += "\n참고 자료:\n" + reference
                    messages = [{"role": "system", "content": sys}] + st.session_state.chat[-10:]
                    result = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
                    answer = result.choices[0].message.content or "답변이 비어 있어요."
                st.session_state.chat.append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"답변 생성 실패: {e}")
        else:
            st.warning("질문을 입력하거나 녹음해 주세요.")
    if st.button("🗑️ 대화 초기화"):
        st.session_state.chat = []
        st.rerun()

elif menu == "📰 경제뉴스 브리핑":
    st.title("📰 주간 경제뉴스 브리핑")
    st.write("최근 7일간의 경제뉴스를 모아 AI가 쉽게 정리합니다.")
    if st.button("📰 이번 주 뉴스 브리핑 생성", type="primary"):
        try:
            with st.spinner("최근 뉴스를 모으고 분석하는 중..."):
                news = collect_news()
                if not news:
                    st.warning("뉴스를 가져오지 못했어요. 잠시 후 다시 시도해 주세요.")
                else:
                    lines = "\n".join(f"{i+1}. [{x['date']}] {x['title']} / {x['source']} / {x['description']} / {x['link']}" for i, x in enumerate(news))
                    prompt = "다음 기사 제목과 RSS 요약만 근거로 가장 중요한 서로 다른 경제 이슈 5개를 선정하세요. 각 이슈마다 제목, 날짜, 무슨 일인지, 왜 중요한지, 일상에 미칠 수 있는 영향, 용어 풀이, 기사 링크를 작성하고 마지막에 주간 흐름 3개를 정리하세요. 기사 본문을 읽지 않았으므로 확인되지 않은 세부 사실은 추측하지 마세요. 마크다운으로 쉽게 설명하세요.\n\n" + lines
                    st.session_state.briefing = ask_ai("당신은 초보자를 위한 경제뉴스 해설가입니다.", prompt, 2400)
        except Exception as e:
            st.error(f"뉴스 브리핑 실패: {e}")
    if st.session_state.get("briefing"):
        st.markdown(st.session_state.briefing)
        st.caption("Google 뉴스 RSS를 기반으로 생성한 요약입니다. 자세한 내용은 원문을 확인하세요.")

elif menu == "🔮 에이블철학관":
    st.title("🔮 에이블철학관")
    st.write("소소하지만 확실한 오늘의 행운 🍀")
    st.caption("재미로 즐기는 AI 운세입니다. 실제 미래를 예측하지 않습니다.")
    nickname = st.text_input("별명", value="에이블 친구", max_chars=30)
    if st.button("✨ 오늘의 운세 보기", type="primary"):
        key = (today.isoformat(), nickname.strip())
        if st.session_state.get("fortune_key") != key:
            try:
                with st.spinner("운세를 준비하고 있어요..."):
                    prompt = f"날짜: {today.isoformat()} / 별명: {nickname}. 오늘의 재미있는 운세를 한국어로 작성하세요. 🌟 종합운, 💗 애정운, 💰 금전운, 📚 학업·업무운, 🍀 행운의 색깔, 🔢 행운의 숫자, 🎁 행운 아이템, 💌 응원 한마디를 각각 작성하세요. 따뜻하고 귀엽게, 운세는 각각 2문장 내외. 사실이나 미래를 예측한다고 주장하지 마세요."
                    st.session_state.fortune = ask_ai("당신은 다정한 가상 철학관 주인입니다. 운세는 오락용 창작 콘텐츠입니다.", prompt, 900)
                    st.session_state.fortune_key = key
            except Exception as e:
                st.error(f"운세 생성 실패: {e}")
        if st.session_state.fortune:
            st.markdown(st.session_state.fortune)
    elif st.session_state.get("fortune_key") == (today.isoformat(), nickname.strip()):
        st.markdown(st.session_state.fortune)
    st.divider()
    st.subheader("💌 마음에 힘이 되는 한 줄")
    if not st.session_state.quote:
        st.session_state.quote = random.choice(QUOTES)
    st.info(st.session_state.quote)
    if st.button("🎲 다른 명언 뽑기"):
        choices = [q for q in QUOTES if q != st.session_state.quote]
        st.session_state.quote = random.choice(choices)
        st.rerun()

elif menu == "💰 에이블 머니가든":
    st.title("💰 에이블 머니가든")
    st.write("🐷 말하는 가계부와 🌱 저축 목표를 한곳에서 관리해요.")
    st.info("기록은 현재 브라우저 세션에 보관돼요. 종료 전에 CSV를 다운로드하고 다음 방문에 다시 업로드해 주세요.")
    tab1, tab2 = st.tabs(["🐷 말하는 가계부", "🌱 돈나무 키우기"])
    with tab1:
        st.subheader("🎤 오늘의 지출 기록")
        with st.form("expense_form", clear_on_submit=True):
            expense_text = st.text_input("지출 입력", placeholder="점심 9천 원, 커피 5천 원 썼어")
            expense_audio = st.audio_input("음성으로 지출 말하기")
            analyze = st.form_submit_button("🤖 지출 분석하기", type="primary")
        if analyze:
            content = expense_text.strip()
            if expense_audio is not None:
                try:
                    with st.spinner("음성을 텍스트로 변환하는 중..."):
                        content = transcribe(expense_audio)
                    st.info("🎤 인식 결과: " + content)
                except Exception as e:
                    st.error(f"음성 인식 실패: {e}")
                    content = ""
            if content:
                try:
                    with st.spinner("지출을 분류하는 중..."):
                        st.session_state.pending_expenses = parse_expenses(content)
                    if not st.session_state.pending_expenses:
                        st.warning("금액이 포함된 지출 내역을 찾지 못했어요.")
                except Exception as e:
                    st.error(f"지출 분석 실패: {e}")
            else:
                st.warning("지출 내용을 입력하거나 녹음해 주세요.")
        if st.session_state.get("pending_expenses"):
            st.write("**분석된 내역을 확인한 후 저장하세요.**")
            st.dataframe(pd.DataFrame(st.session_state.pending_expenses), hide_index=True, use_container_width=True)
            c1, c2 = st.columns(2)
            if c1.button("✅ 가계부에 저장"):
                st.session_state.ledger.extend(st.session_state.pending_expenses)
                st.session_state.pending_expenses = []
                st.rerun()
            if c2.button("❌ 취소"):
                st.session_state.pending_expenses = []
                st.rerun()
        st.divider()
        st.subheader("📒 내 가계부")
        if st.session_state.ledger:
            df = pd.DataFrame(st.session_state.ledger)
            df["금액"] = pd.to_numeric(df["금액"], errors="coerce").fillna(0).astype(int)
            st.metric("누적 기록 지출", f"{df['금액'].sum():,}원")
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.bar_chart(df.groupby("분류")["금액"].sum())
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 가계부 CSV 다운로드", csv_bytes, file_name="aivle_ledger.csv", mime="text/csv")
            if st.button("🗑️ 가계부 전체 삭제"):
                st.session_state.ledger = []
                st.rerun()
        else:
            st.caption("아직 저장된 지출이 없어요.")
        st.divider()
        st.subheader("📤 이전 가계부 불러오기")
        old_csv = st.file_uploader("기존 CSV 업로드", type=["csv"], key="ledger_upload")
        if old_csv and st.button("CSV 기록 불러오기"):
            try:
                incoming = pd.read_csv(io.BytesIO(old_csv.getvalue()), encoding="utf-8-sig")
                required = ["날짜", "항목", "분류", "금액"]
                if not all(c in incoming.columns for c in required):
                    st.error("날짜, 항목, 분류, 금액 열이 필요해요.")
                else:
                    incoming = incoming[required].copy()
                    incoming["금액"] = pd.to_numeric(incoming["금액"], errors="coerce")
                    incoming = incoming.dropna(subset=["금액"])
                    incoming = incoming[(incoming["금액"] > 0) & (incoming["금액"] <= 100000000)]
                    incoming["금액"] = incoming["금액"].astype(int)
                    st.session_state.ledger = incoming.astype({"날짜": str, "항목": str, "분류": str}).to_dict("records")
                    st.success("기록을 불러왔어요! 기존 세션 기록은 업로드한 내용으로 교체했어요.")
                    st.rerun()
            except Exception as e:
                st.error(f"CSV 불러오기 실패: {e}")
    with tab2:
        st.subheader("🌱 나의 돈나무")
        with st.form("goal_form"):
            goal = st.number_input("저축 목표 금액 (원)", min_value=10000, max_value=1000000000, value=int(st.session_state.goal), step=10000)
            saved = st.number_input("현재 저축액 (원)", min_value=0, max_value=1000000000, value=int(st.session_state.saved), step=10000)
            goal_submit = st.form_submit_button("🌱 목표 저장")
        if goal_submit:
            st.session_state.goal = int(goal)
            st.session_state.saved = int(saved)
        ratio = st.session_state.saved / st.session_state.goal
        percent = min(100, ratio * 100)
        if ratio < .25:
            tree = "🌱"
        elif ratio < .5:
            tree = "🪴"
        elif ratio < 1:
            tree = "🌳"
        else:
            tree = "🌳🎉"
        st.markdown(f"<div style='text-align:center;font-size:90px'>{tree}</div>", unsafe_allow_html=True)
        st.progress(min(1.0, ratio))
        st.metric("목표 달성률", f"{percent:.1f}%")
        a, b = st.columns(2)
        a.metric("현재 저축액", f"{st.session_state.saved:,}원")
        b.metric("목표까지 남은 금액", f"{max(0, st.session_state.goal - st.session_state.saved):,}원")
        if ratio >= 1:
            st.success("축하해요! 저축 목표를 달성했어요! 🎉")
        st.divider()
        st.subheader("📈 미래 저축액 계산기")
        monthly = st.slider("매달 저축액 (원)", 10000, 1000000, 100000, 10000)
        years = st.slider("저축 기간 (년)", 1, 20, 3)
        annual_rate = st.slider("가정 연 수익률 (%)", 0.0, 10.0, 3.0, .5)
        months = years * 12
        rate = annual_rate / 100 / 12
        principal = monthly * months
        future = principal if rate == 0 else monthly * ((1 + rate) ** months - 1) / rate
        c1, c2, c3 = st.columns(3)
        c1.metric("납입 원금", f"{principal:,.0f}원")
        c2.metric("예상 수익", f"{future - principal:,.0f}원")
        c3.metric("예상 합계", f"{future:,.0f}원")
        st.caption("매월 말 납입·월복리 가정. 세금, 수수료, 물가 변동은 제외한 예시이며 수익은 보장되지 않습니다.")
        st.caption("저축 목표와 현재 저축액은 세션에만 저장됩니다. 다시 방문하면 재입력해 주세요.")

        
# =====================================================
# 5. 에이블 옷장 - 날씨별 옷차림 추천
# =====================================================

elif menu == "👗 에이블 옷장":

    st.title("👗 에이블 옷장")
    st.write("오늘 날씨에 딱 맞는 옷차림을 추천해 드려요! 🌷")

    city = st.text_input(
        "📍 날씨를 확인할 지역을 입력하세요",
        value="서울",
        placeholder="예: 서울, 부산, 이천"
    )

    if st.button("🌤️ 오늘 뭐 입지?", type="primary"):

        with st.spinner("오늘의 날씨를 확인하고 있어요..."):

            try:
                # 1. 지역 이름을 위도·경도로 변환
                geo_url = "https://geocoding-api.open-meteo.com/v1/search"

                geo_response = requests.get(
                    geo_url,
                    params={
                        "name": city.strip(),
                        "count": 10,
                        "language": "ko",
                        "format": "json"
                    },
                    timeout=15
                )
                geo_response.raise_for_status()

                locations = geo_response.json().get("results", [])

                # 대한민국 지역 우선
                korean_locations = [
                    loc for loc in locations
                    if loc.get("country_code") == "KR"
                ]

                if not korean_locations:
                    st.warning("지역을 찾지 못했어요. 다른 지역명을 입력해 주세요.")

                else:
                    location = korean_locations[0]

                    latitude = location["latitude"]
                    longitude = location["longitude"]

                    # 2. 실시간 날씨 조회
                    weather_url = "https://api.open-meteo.com/v1/forecast"

                    weather_response = requests.get(
                        weather_url,
                        params={
                            "latitude": latitude,
                            "longitude": longitude,
                            "current": (
                                "temperature_2m,"
                                "apparent_temperature,"
                                "relative_humidity_2m,"
                                "precipitation,"
                                "weather_code,"
                                "wind_speed_10m"
                            ),
                            "timezone": "Asia/Seoul"
                        },
                        timeout=15
                    )
                    weather_response.raise_for_status()

                    current = weather_response.json()["current"]

                    temp = current["temperature_2m"]
                    feels = current["apparent_temperature"]
                    humidity = current["relative_humidity_2m"]
                    rain = current["precipitation"]
                    wind = current["wind_speed_10m"]
                    code = current["weather_code"]

                    st.success(
                        f"📍 {location['name']}의 현재 날씨"
                    )

                    # 3. 날씨 정보 표시
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write("🌡️ 현재 기온")
                        st.subheader(f"{temp:.1f}°C")

                    with col2:
                        st.write("🧣 체감온도")
                        st.subheader(f"{feels:.1f}°C")

                    with col3:
                        st.write("💨 풍속")
                        st.subheader(f"{wind:.1f} km/h")

                    st.write(f"💧 습도: {humidity}%")
                    st.write(f"🌧️ 현재 강수량: {rain} mm")

                    st.divider()

                    # 4. 체감온도 기준 옷차림 추천
                    if feels >= 28:
                        outfit = "민소매, 반팔, 반바지, 얇은 원피스"
                        outer = "별도의 아우터는 필요하지 않아요."
                        emoji = "☀️"

                    elif feels >= 23:
                        outfit = "반팔 티셔츠, 얇은 셔츠, 면바지"
                        outer = "실내 냉방 대비 얇은 가디건"
                        emoji = "🌤️"

                    elif feels >= 20:
                        outfit = "긴팔 티셔츠, 얇은 니트, 청바지"
                        outer = "얇은 가디건이나 셔츠"
                        emoji = "🌷"

                    elif feels >= 17:
                        outfit = "맨투맨, 니트, 긴바지"
                        outer = "가디건이나 가벼운 재킷"
                        emoji = "🍂"

                    elif feels >= 12:
                        outfit = "니트, 맨투맨, 긴바지"
                        outer = "트렌치코트, 자켓, 경량패딩"
                        emoji = "🧥"

                    elif feels >= 6:
                        outfit = "기모 맨투맨, 두꺼운 니트, 긴바지"
                        outer = "코트, 두꺼운 재킷, 경량패딩"
                        emoji = "🧣"

                    else:
                        outfit = "히트텍, 기모 상의, 두꺼운 바지"
                        outer = "롱패딩, 두꺼운 코트, 목도리, 장갑"
                        emoji = "❄️"

                    st.subheader(f"{emoji} 오늘의 추천 코디")

                    st.info(f"👕 추천 의상: {outfit}")
                    st.info(f"🧥 아우터: {outer}")

                    # 5. 날씨별 추가 준비물
                    st.subheader("🎒 오늘의 준비물")

                    items = []

                    if code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
                        items.append("☔ 우산과 방수 신발")

                    if code in [71, 73, 75, 77, 85, 86]:
                        items.append("🧤 장갑과 미끄럼 방지 신발")

                    if code in [95, 96, 99]:
                        items.append("⛈️ 천둥번개에 주의하고 실외 활동을 줄이세요.")

                    if wind >= 25:
                        items.append("💨 바람막이 또는 방풍 재킷")

                    if feels <= 5:
                        items.append("🧣 목도리, 장갑, 핫팩")

                    if temp >= 28:
                        items.append("🧴 물병, 모자, 자외선 차단제")

                    if not items:
                        items.append("🍀 편안한 신발과 좋은 기분!")

                    for item in items:
                        st.write(item)

                    st.divider()
                    st.caption(
                        "현재 날씨 기준의 참고용 추천입니다. "
                        "외출 시간의 예보와 개인의 추위·더위 민감도에 따라 조절하세요."
                    )

            except requests.RequestException:
                st.error(
                    "날씨 서버에 연결하지 못했어요. 잠시 후 다시 시도해 주세요."
                )

            except (KeyError, ValueError, TypeError):
                st.error(
                    "날씨 데이터를 읽지 못했어요. 다시 시도해 주세요."
                )
