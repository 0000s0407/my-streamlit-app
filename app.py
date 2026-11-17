import os
import hashlib
import tempfile
import streamlit as st

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

st.set_page_config(
    page_title="에이블 백서",
    page_icon="📘",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #F7F9FC;
}
[data-testid="stSidebar"] {
    background: #EDF3FA;
    border-right: 1px solid #E0E8F3;
}
.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}
h1, h2, h3 {
    color: #142F58;
}
div.stButton > button {
    border-radius: 12px;
    border: 1px solid #D8E3F1;
    background: white;
    color: #183D74;
    font-weight: 600;
}
div.stButton > button:hover {
    border-color: #2563EB;
    color: #2563EB;
}
div.stButton > button[kind="primary"] {
    background: #173D78;
    color: white;
    border: none;
}
[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid #E2EAF4;
    border-radius: 16px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("Streamlit Secrets에 OPENAI_API_KEY를 등록해 주세요.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


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

    return db.as_retriever(search_kwargs={"k": 3})


@st.cache_resource(show_spinner=False)
def rag_pipeline(pdf_bytes, filename):
    with tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False
    ) as f:
        f.write(pdf_bytes)
        path = f.name

    try:
        docs = PyMuPDF4LLMLoader(path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("PDF에서 텍스트를 추출하지 못했습니다.")

        for doc in chunks:
            doc.metadata["source"] = filename

        db = Chroma(
            collection_name="pdf_" + hashlib.sha256(
                pdf_bytes
            ).hexdigest()[:24],
            embedding_function=get_embeddings()
        )

        db.add_documents(chunks)
        return db.as_retriever(search_kwargs={"k": 3})

    finally:
        os.remove(path)


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


def ask_ai(question, retriever):
    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    system = """
당신은 KT AIVLE School의 AI 학습 도우미 '에이블 백서'입니다.

반드시 제공된 백서 내용을 바탕으로 답변하세요.
백서에 없는 사실은 추측하지 마세요.
확인할 수 없는 정보는 백서에서 확인되지 않는다고 말하세요.
공식 정보와 일반적인 학습 조언을 구분하세요.
초보자도 이해하기 쉽게 핵심부터 설명하세요.

참고 문서:
""" + context

    response = get_llm().invoke([
        ("system", system),
        ("human", question)
    ])

    return response.content, docs


def show_sources(docs):
    with st.expander("📚 참고한 백서 내용"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**참고 문서 {i}**")
            st.caption(doc.page_content[:500])


# 사이드바
with st.sidebar:
    st.title("📘 에이블 백서")
    st.caption("AIVLE School Learning Assistant")

    st.divider()

    menu = st.radio(
        "메뉴",
        ["💬 AI 챗봇", "🎓 학습 로드맵", "📄 문서 관리"]
    )

    st.divider()
    st.subheader("백서 업로드")

    uploaded = st.file_uploader(
        "PDF 파일 선택",
        type=["pdf"]
    )

    if st.button("🗑️ 대화 전체 삭제", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# 검색기 준비
if uploaded:
    pdf_bytes = uploaded.getvalue()
    doc_id = hashlib.sha256(pdf_bytes).hexdigest()

    with st.spinner("백서를 분석하고 있습니다..."):
        retriever = rag_pipeline(pdf_bytes, uploaded.name)
else:
    doc_id = "default"
    retriever = get_default_retriever()

if st.session_state.get("doc_id") != doc_id:
    st.session_state.messages = []
    st.session_state.pop("roadmap", None)
    st.session_state.doc_id = doc_id

if "messages" not in st.session_state:
    st.session_state.messages = []


# 상단
st.title("📘 에이블 백서")
st.caption("AIVLE School의 학습 여정을 함께하는 AI 도우미")
st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("#### 💬 문서 기반 Q&A")
        st.caption("백서에 근거한 정확한 답변")

with c2:
    with st.container(border=True):
        st.markdown("#### 🎓 학습 로드맵")
        st.caption("트랙별 맞춤 학습 방향 추천")

with c3:
    with st.container(border=True):
        st.markdown("#### 📚 백서 정보")
        st.caption("교육과정과 프로젝트 정보 확인")


# 챗봇
if menu == "💬 AI 챗봇":
    st.subheader("에이블 백서 AI 챗봇")

    if retriever is None:
        st.info("왼쪽에서 AIVLE School 백서 PDF를 업로드해 주세요.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        questions = [
            "AIVLE School 교육과정은 어떻게 진행되나요?",
            "미니프로젝트는 무엇인가요?",
            "AI 개발자 트랙과 DX 컨설턴트 트랙의 차이는?",
            "효과적인 학습 방법을 알려주세요."
        ]

        if not st.session_state.messages:
            st.markdown("**추천 질문**")
            cols = st.columns(2)

            for i, q in enumerate(questions):
                with cols[i % 2]:
                    if st.button(q, key=f"q{i}", use_container_width=True):
                        st.session_state.pending = q
                        st.rerun()

        question = st.chat_input("궁금한 내용을 입력하세요")

        if st.session_state.get("pending"):
            question = st.session_state.pop("pending")

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("assistant"):
                try:
                    with st.spinner("백서를 검색하고 있습니다..."):
                        answer, docs = ask_ai(question, retriever)

                    st.markdown(answer)
                    show_sources(docs)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                except Exception as e:
                    st.error(f"답변 생성 오류: {e}")


# 학습 로드맵
elif menu == "🎓 학습 로드맵":
    st.subheader("나만의 학습 로드맵")
    st.write("트랙과 수준을 선택해 맞춤형 학습 방향을 확인하세요.")

    with st.form("roadmap_form"):
        track = st.selectbox(
            "트랙 선택",
            ["AI 개발자 트랙", "DX 컨설턴트 트랙"]
        )

        level = st.selectbox(
            "학습 수준",
            ["초보자", "중급자"]
        )

        goal = st.text_input(
            "학습 목표 (선택)",
            placeholder="예: 프로젝트 실력을 높이고 싶어요"
        )

        submitted = st.form_submit_button(
            "✨ 로드맵 추천받기",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if retriever is None:
            st.warning("먼저 백서 PDF를 업로드해 주세요.")
        else:
            question = f"""
{track}의 {level} 학습자를 위한 로드맵을 작성해 주세요.
학습 목표: {goal or '기본 역량 강화'}

다음 항목으로 정리하세요.
1. 추천 학습 방향
2. 단계별 학습 내용
3. 추천 실습
4. 학습 조언

백서에 있는 내용을 우선 활용하고,
백서에 없는 내용은 일반적인 권장 사항이라고 표시하세요.
"""
            try:
                with st.spinner("로드맵을 생성하고 있습니다..."):
                    answer, docs = ask_ai(question, retriever)

                st.session_state.roadmap = answer
                st.session_state.roadmap_docs = docs

            except Exception as e:
                st.error(f"로드맵 생성 오류: {e}")

    if "roadmap" in st.session_state:
        st.markdown("### 추천 학습 로드맵")
        st.markdown(st.session_state.roadmap)
        show_sources(st.session_state.roadmap_docs)

        st.download_button(
            "📥 로드맵 저장",
            data=st.session_state.roadmap,
            file_name="에이블_학습로드맵.txt",
            mime="text/plain"
        )


# 문서 관리
else:
    st.subheader("백서 문서 관리")

    if uploaded:
        st.success(f"현재 문서: {uploaded.name}")
        st.write(f"파일 크기: {uploaded.size / 1024:.1f} KB")
    elif retriever is not None:
        st.success("기존 백서 데이터베이스가 연결되어 있습니다.")
    else:
        st.warning("현재 연결된 백서가 없습니다.")

    st.info("왼쪽 메뉴에서 PDF를 업로드하면 해당 문서를 검색합니다.")
