import os
import re
import hashlib
import tempfile
import streamlit as st

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="에이블 백서 | AIVLE School",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 디자인
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F7F9FC;
    color: #172B4D;
}

[data-testid="stSidebar"] {
    background: #F0F4F9;
    border-right: 1px solid #DFE7F0;
}

.block-container {
    max-width: 1250px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    font-family: 'Noto Sans KR', sans-serif;
}

h1, h2, h3 {
    color: #142B4B;
    letter-spacing: -0.6px;
}

h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 1.5rem !important;
}

div.stButton > button {
    border-radius: 10px;
    border: 1px solid #DCE5EF;
    background: #FFFFFF;
    color: #183B70;
    font-weight: 600;
    min-height: 42px;
}

div.stButton > button:hover {
    border-color: #2C65A7;
    color: #2C65A7;
}

div.stButton > button[kind="primary"] {
    background: #183D72;
    color: #FFFFFF;
    border: none;
}

[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E2E9F1;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border-radius: 12px;
}

[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E1E8F0;
    border-radius: 14px;
    padding: 22px;
}

[data-testid="stChatInput"] {
    background: #FFFFFF;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. API 키
# ==========================================

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("Streamlit Secrets에 OPENAI_API_KEY를 등록해 주세요.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ==========================================
# 4. RAG 파이프라인
# ==========================================

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


@st.cache_resource
def get_default_retriever():
    if not os.path.isdir("chroma_db"):
        return None

    db = Chroma(
        collection_name="aivle_docs",
        embedding_function=get_embeddings(),
        persist_directory="chroma_db"
    )

    if db._collection.count() == 0:
        return None

    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 12,
            "lambda_mult": 0.6
        }
    )


@st.cache_resource(show_spinner=False)
def rag_pipeline(pdf_bytes, filename):

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as f:
        f.write(pdf_bytes)
        path = f.name

    try:
        loader = PyMuPDF4LLMLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError(
                "PDF에서 텍스트를 추출하지 못했습니다."
            )

        for doc in chunks:
            doc.metadata["source"] = filename

        file_hash = hashlib.sha256(
            pdf_bytes
        ).hexdigest()

        db = Chroma(
            collection_name="pdf_" + file_hash[:24],
            embedding_function=get_embeddings()
        )

        db.add_documents(chunks)

        return db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 12,
                "lambda_mult": 0.6
            }
        )

    finally:
        os.remove(path)


# ==========================================
# 5. 대화 맥락 기반 검색
# ==========================================

def rewrite_question(question, history):

    if not history:
        return question

    recent = history[-6:]

    history_text = "\n".join(
        f"{'사용자' if m['role'] == 'user' else 'AI'}: {m['content'][:500]}"
        for m in recent
    )

    prompt = f"""
다음 대화 기록을 참고하여 마지막 질문을
문서 검색에 적합한 독립적인 질문으로 바꾸세요.

규칙:
- 이전 대화의 주제를 유지하세요.
- 원래 질문의 의도를 바꾸지 마세요.
- 새로운 사실을 추가하지 마세요.
- 질문 문장만 출력하세요.

대화 기록:
{history_text}

현재 질문:
{question}
"""

    result = get_llm().invoke([
        ("human", prompt)
    ])

    return result.content.strip()


# ==========================================
# 6. 답변 생성
# ==========================================

SYSTEM_PROMPT = """
당신은 KT AIVLE School의 학습 도우미 '에이블 백서'입니다.

[역할]
AIVLE School 백서를 바탕으로
교육과정, 프로젝트, 학습 전략을 안내합니다.

[답변 원칙]
1. 제공된 백서 내용을 근거로 답변하세요.
2. 백서에 없는 사실은 추측하지 마세요.
3. 확인할 수 없는 정보는
   '백서에서 확인되지 않습니다'라고 설명하세요.
4. 공식 정보와 일반적인 학습 조언을 구분하세요.
5. 초보자가 이해할 수 있도록 간결하고 정확하게 설명하세요.
6. 사용자의 이전 대화 맥락을 고려하세요.
7. AI 개발자 트랙과 DX 컨설턴트 트랙을 구분하세요.
8. 답변의 핵심 내용을 먼저 제시하세요.

[참고 문서]
{context}
"""


def ask_ai(question, retriever, history=None):

    history = history or []

    search_question = rewrite_question(
        question,
        history
    )

    docs = retriever.invoke(search_question)

    context_parts = []

    for i, doc in enumerate(docs, 1):

        source = os.path.basename(
            doc.metadata.get("source", "백서")
        )

        page = doc.metadata.get("page")

        if page is not None:
            page_text = f"페이지 메타데이터: {page}"
        else:
            page_text = "페이지 정보 없음"

        context_parts.append(
            f"[문서 {i}]\n"
            f"출처: {source}\n"
            f"{page_text}\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    recent_history = "\n".join(
        f"{'사용자' if m['role'] == 'user' else 'AI'}: {m['content'][:500]}"
        for m in history[-6:]
    )

    response = get_llm().invoke([
        (
            "system",
            SYSTEM_PROMPT.format(context=context)
        ),
        (
            "human",
            f"이전 대화:\n{recent_history}\n\n"
            f"현재 질문:\n{question}"
        )
    ])

    return response.content, docs


def show_sources(docs):

    with st.expander("답변 근거 문서 확인"):

        for i, doc in enumerate(docs, 1):

            source = os.path.basename(
                doc.metadata.get("source", "백서")
            )

            page = doc.metadata.get("page")

            st.markdown(
                f"**REFERENCE {i:02d} · {source}**"
            )

            if page is not None:
                st.caption(
                    f"페이지 메타데이터: {page}"
                )

            st.write(doc.page_content[:900])

            if i < len(docs):
                st.divider()


# ==========================================
# 7. 사이드바
# ==========================================

with st.sidebar:

    st.caption("AIVLE SCHOOL / KNOWLEDGE")
    st.title("에이블 백서")
    st.caption("AI · DX Learning Knowledge Base")

    st.divider()

    menu = st.radio(
        "서비스",
        [
            "백서 AI 어시스턴트",
            "맞춤 학습 로드맵",
            "문서 라이브러리"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("#### 백서 PDF 등록")

    uploaded = st.file_uploader(
        "PDF 파일 선택",
        type=["pdf"]
    )

    st.caption(
        "PDF를 업로드하거나 저장된 DB를 사용합니다."
    )

    st.divider()

    if st.button(
        "대화 기록 초기화",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "AIVLE WHITEPAPER · LEARNING ASSISTANT"
    )


# ==========================================
# 8. 문서 준비
# ==========================================

try:
    if uploaded:

        pdf_bytes = uploaded.getvalue()

        doc_id = hashlib.sha256(
            pdf_bytes
        ).hexdigest()

        with st.spinner("백서 분석 중..."):
            retriever = rag_pipeline(
                pdf_bytes,
                uploaded.name
            )

    else:
        doc_id = "default"
        retriever = get_default_retriever()

except Exception as e:
    st.error(f"문서 처리 오류: {e}")
    st.stop()


if st.session_state.get("doc_id") != doc_id:
    st.session_state.messages = []
    st.session_state.pop("roadmap", None)
    st.session_state.doc_id = doc_id

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 9. 메인 헤더
# ==========================================

st.caption("AIVLE SCHOOL / DIGITAL WHITEPAPER")

st.title("에이블 백서")

st.write(
    "교육과정의 이해부터 프로젝트와 학습 설계까지, "
    "AIVLE School의 지식을 연결합니다."
)

st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.caption("01 / KNOWLEDGE")
        st.markdown("#### Knowledge Search")
        st.caption("백서 기반 지식 검색 및 답변")

with c2:
    with st.container(border=True):
        st.caption("02 / LEARNING")
        st.markdown("#### Learning Roadmap")
        st.caption("트랙별 맞춤 학습 설계")

with c3:
    with st.container(border=True):
        st.caption("03 / REFERENCE")
        st.markdown("#### Source Transparency")
        st.caption("검색 근거와 문서 출처 확인")


# ==========================================
# 10. 챗봇
# ==========================================

if menu == "백서 AI 어시스턴트":

    st.subheader("Knowledge Assistant")
    st.caption(
        "백서에 수록된 정보를 검색해 "
        "근거 중심으로 답변합니다."
    )

    if retriever is None:

        st.info(
            "왼쪽에서 AIVLE School 백서 PDF를 업로드해 주세요."
        )

    else:

        for msg in st.session_state.messages:

            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if not st.session_state.messages:

            st.markdown("##### Suggested Questions")

            questions = [
                "AIVLE School 교육과정은 어떻게 구성되나요?",
                "미니프로젝트에서는 무엇을 하나요?",
                "AI 개발자 트랙과 DX 컨설턴트 트랙의 차이는?",
                "프로젝트 준비에 필요한 학습 전략은?"
            ]

            cols = st.columns(2)

            for i, question in enumerate(questions):

                with cols[i % 2]:

                    if st.button(
                        question,
                        key=f"q{i}",
                        use_container_width=True
                    ):
                        st.session_state.pending = question
                        st.rerun()

        question = st.chat_input(
            "백서에 대해 질문해 주세요"
        )

        if st.session_state.get("pending"):
            question = st.session_state.pop(
                "pending"
            )

        if question:

            history = st.session_state.messages.copy()

            with st.chat_message("user"):
                st.markdown(question)

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("assistant"):

                try:

                    with st.spinner("백서 검색 및 답변 생성 중..."):

                        answer, docs = ask_ai(
                            question,
                            retriever,
                            history
                        )

                    st.markdown(answer)

                    show_sources(docs)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"답변 생성 오류: {e}")


# ==========================================
# 11. 맞춤 학습 로드맵
# ==========================================

elif menu == "맞춤 학습 로드맵":

    st.subheader("Personalized Learning Roadmap")

    st.write(
        "학습 트랙과 현재 수준을 선택하면 "
        "백서에 기반한 학습 방향을 제안합니다."
    )

    with st.form("roadmap_form"):

        track = st.selectbox(
            "학습 트랙",
            [
                "AI 개발자 트랙",
                "DX 컨설턴트 트랙"
            ]
        )

        level = st.selectbox(
            "현재 학습 수준",
            [
                "초보자",
                "중급자"
            ]
        )

        goal = st.text_area(
            "학습 목표 (선택)",
            placeholder="예: 프로젝트에 필요한 실무 역량을 기르고 싶어요."
        )

        submitted = st.form_submit_button(
            "맞춤 로드맵 생성",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if retriever is None:

            st.warning(
                "먼저 백서 PDF를 등록해 주세요."
            )

        else:

            question = f"""
다음 학습자를 위한 로드맵을 추천해 주세요.

트랙: {track}
학습 수준: {level}
학습 목표: {goal or '기본 역량 강화'}

다음 형식으로 작성하세요.

1. 추천 학습 방향
2. 단계별 학습 로드맵
3. 핵심 학습 내용
4. 추천 실습 활동
5. 학습 조언

백서 내용을 우선 활용하세요.
백서에 없는 내용은 일반적인 권장 사항으로 구분하세요.
"""

            try:

                with st.spinner("학습 로드맵 생성 중..."):

                    answer, docs = ask_ai(
                        question,
                        retriever
                    )

                st.session_state.roadmap = answer
                st.session_state.roadmap_docs = docs

            except Exception as e:
                st.error(f"로드맵 생성 오류: {e}")

    if "roadmap" in st.session_state:

        st.divider()

        st.subheader("Recommended Roadmap")

        st.markdown(st.session_state.roadmap)

        show_sources(
            st.session_state.roadmap_docs
        )

        st.download_button(
            "학습 로드맵 다운로드",
            data=st.session_state.roadmap,
            file_name="AIVLE_학습로드맵.txt",
            mime="text/plain"
        )


# ==========================================
# 12. 문서 라이브러리
# ==========================================

else:

    st.subheader("Document Library")

    if uploaded:

        st.success("PDF 백서가 연결되었습니다.")

        st.write("**문서명:**", uploaded.name)

        st.write(
            "**파일 크기:**",
            f"{uploaded.size / 1024:.1f} KB"
        )

    elif retriever is not None:

        st.success(
            "기존 ChromaDB가 연결되어 있습니다."
        )

    else:

        st.warning(
            "현재 연결된 백서가 없습니다."
        )

    st.info(
        "사이드바에서 PDF를 업로드하면 "
        "해당 문서에 대한 RAG 검색을 수행합니다."
    )
