
%%writefile app.py

import os
import re
import hashlib
import streamlit as st

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="에이블 백서 | AIVLE School",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"

# ==========================================
# 2. API 키 설정
# ==========================================

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY를 Streamlit Secrets에 등록해 주세요.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ==========================================
# 3. 디자인 CSS
# ==========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [data-testid="stApp"] {
        font-family: 'Noto Sans KR', sans-serif;
    }

    .stApp {
        background: #F7F9FD;
        color: #142A4C;
    }

    [data-testid="stSidebar"] {
        background: #F0F5FC;
        border-right: 1px solid #E0E8F4;
    }

    h1, h2, h3 {
        color: #142A4C;
        letter-spacing: -0.6px;
    }

    div.stButton > button[kind="primary"] {
        background: #193E7A;
        border: none;
        color: white;
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #DAE4F2;
        font-weight: 600;
    }

    div[data-testid="stChatMessage"] {
        background: white;
        border: 1px solid #E6EDF7;
        border-radius: 16px;
        padding: 16px;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 10px;
    }

    .hero {
        background: linear-gradient(120deg, #FFFFFF 60%, #EAF2FF);
        padding: 28px 30px;
        border-radius: 22px;
        border: 1px solid #E1EAF6;
        margin-bottom: 20px;
    }

    .hero-title {
        display: flex;
        align-items: center;
        gap: 16px;
        font-size: 36px;
        font-weight: 800;
        color: #142F5B;
    }

    .hero-desc {
        margin-top: 12px;
        color: #536580;
        font-size: 15px;
    }

    .feature-card {
        background: white;
        border: 1px solid #E2EAF5;
        border-radius: 16px;
        padding: 22px;
        min-height: 125px;
        margin-bottom: 18px;
    }

    .feature-title {
        color: #163B74;
        font-weight: 700;
        font-size: 17px;
        margin-bottom: 8px;
    }

    .feature-desc {
        color: #61718B;
        font-size: 13px;
    }

    .brand {
        font-size: 24px;
        font-weight: 800;
        color: #173C78;
        margin: 10px 0 18px;
    }

    .small-note {
        color: #70809A;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. RAG 파이프라인
# ==========================================

def format_docs(docs):
    return "\n\n".join(
        f"[출처 {i+1}] {doc.metadata.get('source', '백서')} "
        f"/ 페이지 {doc.metadata.get('page', '미상')}\n"
        f"{doc.page_content}"
        for i, doc in enumerate(docs)
    )


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


@st.cache_resource(show_spinner=False)
def get_default_retriever():
    if not os.path.isdir(CHROMA_DIR):
        return None

    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR
    )

    if db._collection.count() == 0:
        return None

    return db.as_retriever(search_kwargs={"k": 3})


# PDF 업로드 시 사용하는 RAG 파이프라인
@st.cache_resource(show_spinner=False)
def rag_pipeline(file_bytes, file_name):

    import tempfile

    # PDF 임시 저장
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp:
        temp.write(file_bytes)
        temp_path = temp.name

    try:
        # 1. PDF 로드
        loader = PyMuPDF4LLMLoader(temp_path)
        docs = loader.load()

        # 2. 문서 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("PDF에서 읽을 수 있는 내용이 없습니다.")

        # 3. 문서 출처 기록
        for chunk in chunks:
            chunk.metadata["source"] = file_name

        # 4. 임베딩 및 Chroma 저장
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        vectorstore = Chroma(
            collection_name=f"pdf_{file_hash[:24]}",
            embedding_function=get_embeddings()
        )

        if vectorstore._collection.count() == 0:
            vectorstore.add_documents(chunks)

        # 5. 검색기 생성
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        return retriever

    finally:
        os.remove(temp_path)


# ==========================================
# 5. AI 시스템 프롬프트
# ==========================================

SYSTEM_PROMPT = """
당신은 KT AIVLE School 학습 도우미 '에이블 백서'입니다.

[역할]
- AIVLE School 교육과정과 프로젝트에 관한 질문에 답변합니다.
- AI 개발자 트랙과 DX 컨설턴트 트랙 학습을 지원합니다.
- 초보자도 이해할 수 있도록 쉽고 친절하게 설명합니다.
- 사용자의 자기주도적 학습을 돕습니다.

[답변 원칙]
1. 반드시 제공된 문서 내용을 근거로 답변합니다.
2. 문서에 없는 사실을 지어내지 않습니다.
3. 확인되지 않는 내용은 "백서에서 확인되지 않습니다"라고 답변합니다.
4. 공식 운영 정보와 일반적인 학습 조언을 구분합니다.
5. 문서에 명시되지 않은 교육 일정, 의무사항, 평가 기준을 단정하지 않습니다.
6. 핵심 내용을 먼저 설명하고 필요하면 구체적인 방법을 제안합니다.
7. 학습 로드맵은 문서에 있는 정보를 우선 활용합니다.
8. 문서에 없는 추천 학습 활동은 일반적인 권장 사항임을 명시합니다.
9. AI 개발자 트랙과 DX 컨설턴트 트랙을 구분합니다.

[참고 문서]
{context}
"""


def get_chain(retriever):

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )

    chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def show_sources(retriever, question):
    try:
        docs = retriever.invoke(question)

        if docs:
            with st.expander("📚 답변에 참고한 백서 내용"):
                for i, doc in enumerate(docs, 1):
                    page = doc.metadata.get("page")
                    source = os.path.basename(
                        doc.metadata.get("source", "백서")
                    )

                    # 로더에 따라 페이지가 0부터 시작할 수 있으므로
                    # 실제 PDF 페이지 번호를 확인해 표시
                    page_label = (
                        f" / 페이지 정보: {page}"
                        if page is not None else ""
                    )

                    st.markdown(
                        f"**출처 {i}. {source}{page_label}**"
                    )
                    st.caption(doc.page_content[:700])

    except Exception:
        st.caption("출처 정보를 표시하지 못했습니다.")


# ==========================================
# 6. 로고 및 상단 소개
# ==========================================

LOGO = """
<svg width="76" height="76" viewBox="0 0 76 76"
xmlns="http://www.w3.org/2000/svg">
<path d="M10 21 Q24 15 37 23 V61 Q24 53 10 58 Z"
fill="none" stroke="#163C77" stroke-width="4"
stroke-linejoin="round"/>
<path d="M37 23 Q50 15 65 21 V58 Q50 53 37 61 Z"
fill="none" stroke="#163C77" stroke-width="4"
stroke-linejoin="round"/>
<path d="M37 23 L49 10 L61 22"
fill="none" stroke="#2775D6" stroke-width="3"/>
<circle cx="49" cy="10" r="5" fill="#2775D6"/>
<circle cx="61" cy="22" r="5" fill="#2775D6"/>
<circle cx="37" cy="23" r="5" fill="#2775D6"/>
</svg>
"""

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">
            {LOGO}
            <div>
                에이블 백서
                <div style="font-size:15px;font-weight:500;
                color:#567093;margin-top:3px">
                    AIVLE School AI Learning Assistant
                </div>
            </div>
        </div>
        <div class="hero-desc">
            AIVLE School 백서를 기반으로 교육과정, 프로젝트,
            학습 전략을 안내하는 AI 학습 도우미입니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 7. 사이드바
# ==========================================

with st.sidebar:

    st.markdown(
        '<div class="brand">📘 에이블 백서</div>',
        unsafe_allow_html=True
    )

    menu = st.radio(
        "메뉴",
        [
            "💬 에이블 백서 챗봇",
            "🎓 학습 로드맵 추천",
            "📄 문서 관리"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 📄 백서 업로드")

    uploaded_file = st.file_uploader(
        "PDF 파일 선택",
        type=["pdf"],
        help="PDF를 업로드하면 해당 문서로 질문할 수 있습니다."
    )

    if uploaded_file:
        st.success(f"업로드됨: {uploaded_file.name}")
    else:
        st.caption("업로드하지 않으면 기존 ChromaDB를 사용합니다.")

    st.divider()

    if st.button("🗑️ 대화 전체 삭제", use_container_width=True):
        st.session_state.messages = []
        st.success("대화 기록을 삭제했습니다.")

    st.markdown(
        '<div class="small-note">AIVLE School 학습 지원 서비스</div>',
        unsafe_allow_html=True
    )

# ==========================================
# 8. 검색기 준비
# ==========================================

try:
    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        with st.spinner("PDF 백서를 분석하고 있습니다..."):
            retriever = rag_pipeline(
                file_bytes,
                uploaded_file.name
            )

        document_id = file_hash

    else:
        retriever = get_default_retriever()
        document_id = "default"

except Exception as e:
    st.error(f"문서 처리 오류: {e}")
    st.stop()


if "document_id" not in st.session_state:
    st.session_state.document_id = document_id

if st.session_state.document_id != document_id:
    st.session_state.messages = []
    st.session_state.document_id = document_id

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ==========================================
# 9. 상단 기능 카드
# ==========================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">💬 문서 기반 Q&A</div>
        <div class="feature-desc">
            백서 내용을 근거로 질문에 답변합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🎓 학습 로드맵 추천</div>
        <div class="feature-desc">
            트랙과 학습 수준에 맞춰 학습 방향을 제안합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">📚 백서 정보 요약</div>
        <div class="feature-desc">
            교육과정과 프로젝트 정보를 정리합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 10. 챗봇 화면
# ==========================================

if menu == "💬 에이블 백서 챗봇":

    st.subheader("💬 에이블 백서 챗봇")

    if retriever is None:
        st.warning("먼저 PDF 백서를 업로드해 주세요.")
        st.stop()

    chain = get_chain(retriever)

    # 기존 대화 표시
    for message in st.session_state.messages:

        role = (
            "user"
            if isinstance(message, HumanMessage)
            else "assistant"
        )

        with st.chat_message(role):
            st.markdown(message.content)

    # 첫 화면 추천 질문
    if not st.session_state.messages:

        st.markdown("#### ✨ 추천 질문")

        questions = [
            "AIVLE School 교육과정은 어떻게 진행되나요?",
            "미니프로젝트에서는 무엇을 하나요?",
            "AI 개발자 트랙과 DX 컨설턴트 트랙의 차이는?",
            "효과적인 학습 방법을 알려주세요."
        ]

        q1, q2 = st.columns(2)

        for i, question in enumerate(questions):

            col = q1 if i % 2 == 0 else q2

            with col:
                if st.button(
                    question,
                    key=f"question_{i}",
                    use_container_width=True
                ):
                    st.session_state.pending_question = question
                    st.rerun()

    user_input = st.chat_input(
        "에이블스쿨에 대해 궁금한 내용을 입력하세요..."
    )

    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_input:

        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append(
            HumanMessage(content=user_input)
        )

        with st.chat_message("assistant"):

            try:
                with st.spinner("백서에서 답변을 찾고 있습니다..."):

                    response = st.write_stream(
                        chain.stream({"input": user_input})
                    )

                show_sources(retriever, user_input)

                st.session_state.messages.append(
                    AIMessage(content=response)
                )

            except Exception as e:
                st.error(f"답변 생성 중 오류: {e}")

# ==========================================
# 11. 학습 로드맵 추천
# ==========================================

elif menu == "🎓 학습 로드맵 추천":

    st.subheader("🎓 나만의 학습 로드맵")

    st.write(
        "트랙과 학습 수준을 선택하면 "
        "백서를 바탕으로 학습 로드맵을 추천합니다."
    )

    if retriever is None:
        st.warning("로드맵 추천을 위해 PDF 백서를 업로드해 주세요.")
        st.stop()

    with st.form("roadmap_form"):

        track = st.radio(
            "트랙 선택",
            [
                "AI 개발자 트랙",
                "DX 컨설턴트 트랙"
            ]
        )

        level = st.radio(
            "현재 학습 수준",
            [
                "초보자",
                "중급자"
            ]
        )

        goal = st.text_area(
            "학습 목표 (선택)",
            placeholder="예: Python 기초부터 공부하고 싶어요."
        )

        submitted = st.form_submit_button(
            "✨ 나의 학습 로드맵 추천받기",
            type="primary",
            use_container_width=True
        )

    if submitted:

        question = f"""
        다음 학습자에게 맞춤형 학습 로드맵을 추천해 주세요.

        트랙: {track}
        학습 수준: {level}
        학습 목표: {goal if goal.strip() else "기본 역량 향상"}

        다음 형식으로 작성해 주세요.

        1. 추천 학습 방향
        2. 단계별 학습 로드맵
        3. 단계별 핵심 학습 내용
        4. 추천 실습 활동
        5. 학습을 위한 조언

        반드시 백서 내용을 우선 근거로 활용하고,
        백서에 없는 추천 활동은 일반적인 학습 조언이라고
        명확히 구분해 주세요.
        """

        chain = get_chain(retriever)

        try:
            with st.spinner("맞춤형 학습 로드맵을 생성하고 있습니다..."):

                result = chain.invoke({"input": question})

            st.session_state.roadmap_result = result
            st.session_state.roadmap_question = question

        except Exception as e:
            st.error(f"로드맵 생성 오류: {e}")

    if "roadmap_result" in st.session_state:

        st.markdown("### 📍 추천 학습 로드맵")

        st.markdown(st.session_state.roadmap_result)

        show_sources(
            retriever,
            st.session_state.roadmap_question
        )

        st.download_button(
            "📥 학습 로드맵 다운로드",
            data=st.session_state.roadmap_result,
            file_name="AIVLE_학습로드맵.txt",
            mime="text/plain"
        )

# ==========================================
# 12. 문서 관리
# ==========================================

elif menu == "📄 문서 관리":

    st.subheader("📄 백서 문서 관리")

    if uploaded_file:

        st.success("현재 업로드한 PDF를 사용하고 있습니다.")

        st.write("**문서명:**", uploaded_file.name)

        st.write(
            "**파일 크기:**",
            f"{uploaded_file.size / 1024:.1f} KB"
        )

        st.info(
            "새로운 PDF를 업로드하면 "
            "해당 문서를 기준으로 RAG 검색이 진행됩니다."
        )

    elif retriever is not None:

        st.success("기존 ChromaDB가 연결되어 있습니다.")

        st.info(
            "사이드바에서 새로운 PDF를 업로드하면 "
            "해당 PDF를 사용하도록 전환됩니다."
        )

    else:

        st.warning("현재 연결된 문서가 없습니다.")

        st.write(
            "사이드바의 PDF 업로드 기능을 이용해 "
            "AIVLE School 백서를 등록해 주세요."
        )
