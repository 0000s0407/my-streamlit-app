
import streamlit as st
from openai import OpenAI

# 페이지 설정
st.set_page_config(page_title="텍스트 음성 변환", page_icon="🔊")
st.title("🔊 텍스트 음성 변환 실습")

# 텍스트 입력
text = st.text_area(
    "읽을 텍스트를 입력하세요",
    "안녕하세요. OpenAI 텍스트 음성 변환 실습입니다."
)

# 버튼
if st.button("음성 변환하기"):
    if text.strip() == "":
        st.warning("텍스트를 입력해주세요.")
    else:
        try:
            # OpenAI 클라이언트 생성
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

            # 음성 생성
            speech = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )

            # 음성 출력
            st.success("음성 변환 완료!")
            st.markdown("### 🔊 음성 재생")
            st.audio(speech.content, format="audio/mp3")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
