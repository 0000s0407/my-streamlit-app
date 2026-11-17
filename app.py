import os
import re
import json
import math
import hashlib
import tempfile
from collections import Counter

import streamlit as st
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ==================================================
# 1. 페이지 설정
# ==================================================

st.set_page_config(
    page_title="에이블 백서 | AIVLE School",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. 디자인
# ==================================================

st.markdown("""
<style>
:root {
    --ink: #17283d;
    --muted: #68788a;
    --blue: #2457a6;
    --line: #dce4ec;
}

.stApp {
    background: #f7f9fc;
    color: #17283d;
}

[data-testid="stSidebar"] {
    background: #eef3f8;
    border-right: 1px solid #dce4ec;
}

.block-container {
    max-width: 1160px;
    padding-top: 2.7rem;
    padding-bottom: 5rem;
}

h1, h2, h3 {
    color: #17283d;
    letter-spacing: -.035em;
}

[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e0e7ef;
    border-radius: 14px;
    padding: 1.1rem;
}

[data-testid="stForm"] {
    border-color: #dce4ec;
    border-radius: 14px;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    border: 1px solid #d5e0eb;
    background: #ffffff;
    color: #213b5c;
    font-weight: 550;
}

.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: #1d416d;
    color: #ffffff;
    border-color: #1d416d;
}

.stButton > button:hover {
    border-color: #2457a6;
    color: #2457a6;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 12px;
}

.hero {
    padding: 2.3rem 2.4rem;
    background: #ffffff;
    border: 1px solid #dce4ec;
    border-radius: 16px;
    margin-bottom: 1.6rem;
}

.eyebrow {
    font-size: .72rem;
    letter-spacing: .16em;
    color: #4775aa;
    font-weight: 700;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 2.7rem;
    line-height: 1.2;
    letter-spacing: -.06em;
    font-weight: 760;
    color: #17283d;
}

.hero-desc {
    margin-top: .9rem;
    color: #60738a;
    font-size: 1rem;
    line-height: 1.8;
}

.brand-mark {
    width: 38px;
    height: 38px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: #1d416d;
    color: #ffffff;
    font-size: 1.2rem;
    font-weight: 750;
    margin-right: .55rem;
}

.brand-name {
    font-size: 1.38rem;
    color: #17283d;
    font-weight: 750;
    letter-spacing: -.04em;
}

.feature {
    background: #ffffff;
    border: 1px solid #dce4ec;
    border-radius: 12px;
    padding: 1.3rem;
    min-height: 128px;
}

.feature-num {
    font-size: .7rem;
    color: #5680b2;
    font-weight: 700;
    letter-spacing: .12em;
}

.feature-title {
    font-size: 1.02rem;
    color: #17283d;
    font-weight: 700;
    margin: .7rem 0 .4rem;
}

.feature-desc {
    color: #708096;
    font-size: .83rem;
}

.section-label {
    font-size: .75rem;
    color: #6e8299;
    font-weight: 700;
    letter-spacing: .12em;
    margin: 1.6rem 0 .5rem;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. OpenAI API 키
# ==================================================

try:
    API_KEY = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error(
        "Streamlit Cloud의 Secrets에 "
        "OPENAI_API_KEY를 등록하세요."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = API_KEY

# ==================================================
# 4. 임베딩 및 언어 모델
# ==================================================

@st.cache_resource(show_spinner=False)
def embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


@st.cache_resource(show_spinner=False)
def llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


# ==================================================
# 5. 기본 ChromaDB 연결
# ==================================================

@st.cache_resource(show_spinner=False)
def default_db():

    if not os.path.isdir("chroma_db"):
        return None

    db = Chroma(
        collection_name="aivle_docs",
        embedding_function=embeddings(),
        persist_directory="chroma_db"
    )

    return db if db._collection.count() else None


# ==================================================
# 6. PDF RAG 파이프라인
# ==================================================

@st.cache_resource(show_spinner=False)
def rag_pipeline(file_bytes, filename):

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as temp:
        temp.write(file_bytes)
        path = temp.name

    try:
        # PDF 로드
        docs = PyMuPDF4LLMLoader(path).load()

        # 문서 분할
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=160
        ).split_documents(docs)

        if not chunks:
            raise ValueError(
                "PDF에서 텍스트를 읽을 수 없습니다."
            )

        # 출처 정보
        for chunk in chunks:
            chunk.metadata["source"] = filename

        # 파일별 DB 생성
        digest = hashlib.sha256(
            file_bytes
        ).hexdigest()

        db = Chroma(
            collection_name="pdf_" + digest[:24],
            embedding_function=embeddings()
        )

        db.add_documents(chunks)

        return db

    finally:
        os.remove(path)


# ==================================================
# 7. BM25 키워드 검색
# ==================================================

def terms(text):

    text = str(text).lower()

    words = re.findall(
        r"[가-힣a-z0-9]+",
        text
    )

    grams = []

    for word in words:
        if re.search(r"[가-힣]", word) and len(word) >= 2:
            grams.extend(
                word[i:i+2]
                for i in range(len(word)-1)
            )

    return words + grams


def bm25_ranks(query, docs):

    corpus = [
        Counter(terms(d.page_content))
        for d in docs
    ]

    n = len(corpus)

    if not n:
        return []

    avg_len = (
        sum(sum(c.values()) for c in corpus) / n
    ) or 1

    query_terms = set(terms(query))

    df = Counter(
        term
        for counter in corpus
        for term in counter
    )

    scored = []

    for i, counter in enumerate(corpus):

        length = sum(counter.values())
        score = 0.0

        for term in query_terms:

            frequency = counter[term]

            if frequency:

                idf = math.log(
                    1 + (
                        n - df[term] + 0.5
                    ) / (
                        df[term] + 0.5
                    )
                )

                score += (
                    idf * (frequency * 2.2)
                    / (
                        frequency
                        + 1.2 * (
                            0.25
                            + 0.75 * length / avg_len
                        )
                    )
                )

        scored.append((i, score))

    return [
        i
        for i, score in sorted(
            scored,
            key=lambda x: x[1],
            reverse=True
        )
        if score > 0
    ]


# ==================================================
# 8. 중복 문서 제거
# ==================================================

def unique_docs(docs):

    seen = set()
    result = []

    for doc in docs:

        key = hashlib.sha256(
            doc.page_content.encode("utf-8")
        ).hexdigest()

        if key not in seen:
            seen.add(key)
            result.append(doc)

    return result


# ==================================================
# 9. 대화 맥락 기반 질문 재작성
# ==================================================

def rewrite_question(question, history):

    if not history:
        return question

    recent = "\n".join(
        (
            "사용자"
            if message["role"] == "user"
            else "도우미"
        )
        + ": "
        + message["content"][:350]
        for message in history[-4:]
    )

    try:
        reply = llm().invoke([
            (
                "system",
                "대화 맥락을 이용해 마지막 질문을 "
                "문서 검색용 독립 질문 한 문장으로 바꾸세요. "
                "없는 사실은 추가하지 마세요. "
                "질문만 출력하세요."
            ),
            (
                "human",
                f"대화:\n{recent}\n\n"
                f"마지막 질문: {question}"
            )
        ]).content.strip()

        return reply or question

    except Exception:
        return question


# ==================================================
# 10. 하이브리드 검색: MMR + BM25 + RRF
# ==================================================

def hybrid_retrieve(db, query, top_k=7):

    # 의미 기반 MMR 검색
    vector_docs = db.max_marginal_relevance_search(
        query,
        k=12,
        fetch_k=24,
        lambda_mult=0.65
    )

    count = db._collection.count()

    # 작은 DB는 전체 문서에서 키워드 검색
    if count <= 1800:

        raw = db.get(
            include=["documents", "metadatas"]
        )

        lexical_pool = [
            Document(
                page_content=text,
                metadata=metadata or {}
            )
            for text, metadata in zip(
                raw["documents"],
                raw["metadatas"]
            )
            if text
        ]

    else:
        # 큰 DB는 MMR 후보 내에서 검색
        lexical_pool = vector_docs

    lexical_order = bm25_ranks(
        query,
        lexical_pool
    )[:16]

    lexical_docs = [
        lexical_pool[i]
        for i in lexical_order
    ]

    merged = unique_docs(
        vector_docs + lexical_docs
    )

    # Reciprocal Rank Fusion
    scores = Counter()

    def doc_key(doc):
        return hashlib.sha256(
            doc.page_content.encode("utf-8")
        ).hexdigest()

    for rank, doc in enumerate(vector_docs):
        scores[doc_key(doc)] += 1 / (60 + rank + 1)

    for rank, doc in enumerate(lexical_docs):
        scores[doc_key(doc)] += 1 / (60 + rank + 1)

    merged.sort(
        key=lambda doc: scores[doc_key(doc)],
        reverse=True
    )

    return merged[:top_k]


# ==================================================
# 11. AI 리랭킹 및 관련성 검사
# ==================================================

def rerank(query, candidates):

    if not candidates:
        return []

    excerpts = "\n\n".join(
        f"[{i}] {doc.page_content[:850]}"
        for i, doc in enumerate(candidates)
    )

    try:
        result = llm().invoke([
            (
                "system",
                "당신은 문서 검색 평가자입니다. "
                "질문에 직접 답하는 근거만 선택하세요. "
                "추측하지 마세요. "
                'JSON 객체만 반환하세요: {"ids":[0,2]} 형식. '
                '관련 문서가 없으면 {"ids":[]}를 반환하세요. '
                "최대 4개를 관련성 높은 순서로 선택하세요."
            ),
            (
                "human",
                f"질문: {query}\n\n"
                f"검색 후보:\n{excerpts}"
            )
        ]).content

        match = re.search(
            r"\{[\s\S]*\}",
            result
        )

        parsed = (
            json.loads(match.group(0))
            if match
            else {}
        )

        ids = parsed.get("ids", [])

        if not isinstance(ids, list):
            return candidates[:4]

        indices = []

        for i in ids:

            if (
                isinstance(i, int)
                and 0 <= i < len(candidates)
                and i not in indices
            ):
                indices.append(i)

        return [
            candidates[i]
            for i in indices[:4]
        ]

    except Exception:
        return candidates[:4]


# ==================================================
# 12. 시스템 프롬프트
# ==================================================

SYSTEM = """
당신은 AIVLE School 디지털 백서의
문서 기반 학습 도우미입니다.

[답변 원칙]

1. 근거 문서에 명시된 사실만 백서의 사실로 설명합니다.

2. 근거가 없거나 불충분하면
'업로드된 백서에서 확인되지 않습니다'라고 말합니다.

3. 공식 운영 사실, 교육생 경험,
일반적인 추천 학습 활동을 구분합니다.

4. 교육 일정, 평가 기준, 의무사항,
수료 조건을 만들어내지 않습니다.

5. 답변은 자연스러운 한국어로
간결하고 구조적으로 작성합니다.

6. 답변에 근거 문서를 인용할 때
[자료 1]처럼 번호를 표시합니다.

7. 일반적인 조언은
'일반적인 학습 제안'으로 구분합니다.

8. 트랙은 AI 개발자 트랙과
DX 컨설턴트 트랙만 사용합니다.
"""


# ==================================================
# 13. 최종 RAG 답변 생성
# ==================================================

def answer_question(
    question,
    db,
    history=None,
    roadmap=False
):

    history = history or []

    # 1. 질문 재작성
    search_query = rewrite_question(
        question,
        history
    )

    # 2. 하이브리드 검색
    candidates = hybrid_retrieve(
        db,
        search_query
    )

    # 3. 리랭킹
    selected = rerank(
        search_query,
        candidates
    )

    # 4. 관련 근거가 없으면 답변 제한
    if not selected:
        return (
            "업로드된 백서에서 이 질문에 대한 "
            "근거를 확인하지 못했습니다. "
            "질문을 더 구체적으로 입력해 주세요.",
            [],
            search_query
        )

    # 5. 근거 문서 구성
    context = "\n\n".join(
        f"[자료 {i}]\n"
        f"{doc.page_content[:2300]}"
        for i, doc in enumerate(selected, 1)
    )

    recent = "\n".join(
        (
            "사용자"
            if message["role"] == "user"
            else "도우미"
        )
        + ": "
        + message["content"][:350]
        for message in history[-4:]
    )

    instruction = ""

    if roadmap:
        instruction = (
            "\n로드맵은 백서에서 확인된 학습 내용과 "
            "일반적인 추천 활동을 분리해 "
            "단계별로 작성하세요."
        )

    # 6. 최종 답변
    reply = llm().invoke([
        (
            "system",
            SYSTEM
            + instruction
            + "\n[근거 문서]\n"
            + context
        ),
        (
            "human",
            f"이전 대화:\n{recent}\n\n"
            f"질문: {question}"
        )
    ]).content

    return reply, selected, search_query


# ==================================================
# 14. 출처 표시
# ==================================================

def sources_view(docs):

    with st.expander(
        f"근거 문서 보기 · {len(docs)}개"
    ):

        for i, doc in enumerate(docs, 1):

            source = os.path.basename(
                str(
                    doc.metadata.get(
                        "source",
                        "백서"
                    )
                )
            )

            page = doc.metadata.get("page")

            st.markdown(
                f"**자료 {i:02d} / {source}**"
            )

            if page is not None:
                st.caption(
                    f"PDF 페이지 메타데이터: {page} "
                    "(0부터 시작할 수 있음)"
                )

            st.write(
                doc.page_content[:1200]
            )

            if i != len(docs):
                st.divider()


# ==================================================
# 15. 사이드바
# ==================================================

with st.sidebar:

    st.markdown(
        """
        <div class="eyebrow">
            AIVLE SCHOOL / WHITEPAPER
        </div>
        <div>
            <span class="brand-mark">A</span>
            <span class="brand-name">에이블 백서</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "AI · DX Knowledge & Learning"
    )

    st.divider()

    menu = st.radio(
        "메뉴",
        [
            "지식 어시스턴트",
            "학습 로드맵",
            "백서 탐색"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**백서 연결**")

    uploaded = st.file_uploader(
        "PDF 파일 등록",
        type=["pdf"]
    )

    st.caption(
        "업로드한 PDF를 우선 사용합니다. "
        "미등록 시 저장된 DB를 확인합니다."
    )

    if st.button(
        "대화 초기화",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "AIVLE SCHOOL · DIGITAL REFERENCE"
    )


# ==================================================
# 16. PDF 및 DB 연결
# ==================================================

try:

    if uploaded is not None:

        content = uploaded.getvalue()

        doc_id = hashlib.sha256(
            content
        ).hexdigest()

        with st.spinner(
            "백서 색인 생성 중..."
        ):
            db = rag_pipeline(
                content,
                uploaded.name
            )

        doc_name = uploaded.name

    else:

        db = default_db()
        doc_id = "default"

        doc_name = (
            "기본 백서 DB"
            if db
            else None
        )

except Exception as exc:

    st.error(
        f"PDF 처리에 실패했습니다: {exc}"
    )
    st.stop()


if st.session_state.get("doc_id") != doc_id:

    st.session_state.doc_id = doc_id
    st.session_state.messages = []

    st.session_state.pop(
        "roadmap",
        None
    )

    st.session_state.pop(
        "explore",
        None
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# 17. 메인 헤더
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AIVLE SCHOOL / DIGITAL WHITEPAPER</div>
        <div class="hero-title">에이블 백서</div>
        <div class="hero-desc">
            교육과정과 프로젝트 경험을 연결하는 지식 플랫폼.<br>
            근거 있는 답변과 개인화된 학습 방향을 제공합니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================
# 18. 서비스 카드
# ==================================================

cols = st.columns(3, gap="medium")

features = [
    (
        "01 / SEARCH",
        "근거 기반 지식 검색",
        "하이브리드 검색 · 재순위화"
    ),
    (
        "02 / ROADMAP",
        "맞춤 학습 설계",
        "AI 개발자 · DX 컨설턴트"
    ),
    (
        "03 / EXPLORE",
        "백서 주제 탐색",
        "핵심 주제 · 근거 확인"
    )
]

for col, (num, title, desc) in zip(
    cols,
    features
):

    with col:

        st.markdown(
            f"""
            <div class="feature">
                <div class="feature-num">
                    {num}
                </div>
                <div class="feature-title">
                    {title}
                </div>
                <div class="feature-desc">
                    {desc}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


if db is None:

    st.info(
        "왼쪽에서 AIVLE School 백서 PDF를 "
        "등록하면 서비스를 사용할 수 있습니다."
    )

    st.stop()


# ==================================================
# 19. 지식 어시스턴트
# ==================================================

if menu == "지식 어시스턴트":

    st.markdown(
        '<div class="section-label">'
        'KNOWLEDGE ASSISTANT'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "백서에 질문하기"
    )

    st.caption(
        "검색된 근거를 확인하고, "
        "백서에서 확인되지 않는 내용은 "
        "구분해 안내합니다."
    )

    # 기존 대화 표시
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message.get("docs"):
                sources_view(
                    message["docs"]
                )

    # 추천 질문
    if not st.session_state.messages:

        st.markdown("**추천 질문**")

        suggestions = [
            "미니프로젝트는 어떻게 진행되나요?",
            "빅프로젝트 준비 방법을 알려주세요.",
            "AI 개발자 트랙에서 어떤 역량을 기르나요?",
            "DX 컨설턴트 트랙의 학습 방향은 무엇인가요?"
        ]

        scols = st.columns(2)

        for i, question in enumerate(
            suggestions
        ):

            with scols[i % 2]:

                if st.button(
                    question,
                    key=f"suggest_{i}",
                    use_container_width=True
                ):

                    st.session_state.pending_question = question
                    st.rerun()

    question = st.chat_input(
        "백서에 대해 궁금한 점을 질문하세요"
    )

    if st.session_state.get(
        "pending_question"
    ):

        question = st.session_state.pop(
            "pending_question"
        )

    # 질문 처리
    if question:

        history = (
            st.session_state.messages.copy()
        )

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "근거 검색 및 검증 중..."
                ):

                    response, docs, search_query = (
                        answer_question(
                            question,
                            db,
                            history
                        )
                    )

                st.markdown(response)

                sources_view(docs)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "docs": docs
                })

            except Exception as exc:

                st.error(
                    f"답변 생성 실패: {exc}"
                )


# ==================================================
# 20. 맞춤 학습 로드맵
# ==================================================

elif menu == "학습 로드맵":

    st.markdown(
        '<div class="section-label">'
        'PERSONALIZED LEARNING'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "나만의 학습 로드맵"
    )

    st.caption(
        "백서의 교육 내용과 일반적인 학습 제안을 "
        "구분하여 단계별로 안내합니다."
    )

    with st.form("roadmap_form"):

        track = st.selectbox(
            "학습 트랙",
            [
                "AI 개발자 트랙",
                "DX 컨설턴트 트랙"
            ]
        )

        level = st.select_slider(
            "현재 수준",
            options=[
                "입문",
                "기초",
                "중급"
            ]
        )

        goal = st.text_area(
            "학습 목표",
            placeholder=(
                "예: 프로젝트를 준비하면서 "
                "실무 역량을 쌓고 싶어요."
            ),
            height=95
        )

        submitted = st.form_submit_button(
            "로드맵 생성",
            type="primary",
            use_container_width=True
        )

    if submitted:

        question = (
            f"{track}의 {level} 수준 학습자를 위한 "
            f"단계별 학습 로드맵을 작성해 주세요. "
            f"목표: {goal or '핵심 역량 강화'}. "
            "각 단계의 학습 내용, 실습 제안, "
            "기대 결과를 정리하고 백서에 명시된 사실과 "
            "일반적인 권장 사항을 구분하세요."
        )

        try:

            with st.spinner(
                "학습 계획 구성 중..."
            ):

                response, docs, _ = (
                    answer_question(
                        question,
                        db,
                        roadmap=True
                    )
                )

            st.session_state.roadmap = {
                "text": response,
                "docs": docs
            }

        except Exception as exc:

            st.error(
                f"로드맵 생성 실패: {exc}"
            )

    if st.session_state.get("roadmap"):

        st.divider()

        st.subheader(
            "추천 학습 계획"
        )

        st.markdown(
            st.session_state.roadmap["text"]
        )

        sources_view(
            st.session_state.roadmap["docs"]
        )

        st.download_button(
            "학습 계획 저장 (.txt)",
            st.session_state.roadmap["text"],
            file_name="aivle_roadmap.txt",
            mime="text/plain"
        )


# ==================================================
# 21. 백서 주제 탐색
# ==================================================

else:

    st.markdown(
        '<div class="section-label">'
        'WHITEPAPER EXPLORER'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "백서 주제 탐색"
    )

    st.caption(
        f"현재 연결 문서: {doc_name}"
    )

    st.write(
        "궁금한 주제를 선택하면 백서에서 "
        "해당 내용을 찾아 핵심을 정리합니다."
    )

    topics = [
        "교육과정 개요",
        "미니프로젝트",
        "빅프로젝트",
        "AI 개발자 트랙",
        "DX 컨설턴트 트랙",
        "학습 기록과 성장"
    ]

    selected = st.selectbox(
        "탐색할 주제",
        topics
    )

    if st.button(
        "주제 내용 살펴보기",
        type="primary"
    ):

        prompt = (
            f"백서에서 '{selected}'에 관한 내용을 찾아 "
            "핵심 정보만 항목별로 요약해 주세요. "
            "근거가 부족하면 부족하다고 설명하세요."
        )

        try:

            with st.spinner(
                "백서 주제 탐색 중..."
            ):

                response, docs, _ = (
                    answer_question(
                        prompt,
                        db
                    )
                )

            st.session_state.explore = {
                "topic": selected,
                "text": response,
                "docs": docs,
                "doc_id": doc_id
            }

        except Exception as exc:

            st.error(
                f"탐색 실패: {exc}"
            )

    result = st.session_state.get(
        "explore"
    )

    if (
        result
        and result["doc_id"] == doc_id
    ):

        st.divider()

        st.subheader(
            result["topic"]
        )

        st.markdown(
            result["text"]
        )

        sources_view(
            result["docs"]
        )
