

    st.divider()
    st.markdown("**백서 PDF 업로드**")
    uploaded = st.file_uploader("PDF 파일 선택", type=["pdf"], label_visibility="collapsed")
    st.caption("PDF를 업로드하면 해당 문서를 기준으로 답변합니다.")
    st.divider()
    if st.button("🗑️ 대화 전체 삭제", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("AIVLE School · AI Learning Assistant")

try:
    if uploaded is not None:
        pdf_bytes = uploaded.getvalue()
        doc_id = hashlib.sha256(pdf_bytes).hexdigest()
        with st.spinner("백서를 분석하고 있어요..."):
            retriever = rag_pipeline(pdf_bytes, uploaded.name)
    else:
        doc_id = "default"
        retriever = default_retriever()
except Exception as e:
    st.error(f"PDF 처리 중 오류가 발생했습니다: {e}")
    st.stop()

if st.session_state.get("doc_id") != doc_id:
    st.session_state.doc_id = doc_id
    st.session_state.messages = []
    st.session_state.pop("roadmap", None)
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("""
<div class="hero">
  <div class="hero-kicker">AIVLE SCHOOL · AI LEARNING GUIDE</div>
  <div class="hero-title"><span class="brand-mark">✦</span>에이블 백서</div>
  <div class="hero-sub">백서에 근거한 정확한 답변부터 나에게 맞는 학습 로드맵까지.<br>에이블스쿨의 배움을 더 쉽고 명확하게 안내합니다.</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="medium")
for col, title, desc in [
    (c1, "💬 문서 기반 Q&A", "백서 내용을 찾아 질문에 답변합니다."),
    (c2, "🎓 맞춤 학습 로드맵", "트랙과 수준에 맞는 학습 방향을 제안합니다."),
    (c3, "📚 백서 정보 확인", "검색에 활용된 문서 내용을 확인합니다."),
]:
    with col:
        st.markdown(f'<div class="feature"><div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div>', unsafe_allow_html=True)

if menu == "💬 백서 AI 챗봇":
    st.subheader("에이블 백서 챗봇")
    st.caption("궁금한 내용을 질문하면 업로드한 백서에서 근거를 찾아 답변해요.")
    if retriever is None:
        st.info("왼쪽에서 AIVLE School 백서 PDF를 먼저 업로드해 주세요.")
        st.stop()

    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    suggestions = [
        "AIVLE School 교육과정은 어떻게 진행되나요?",
        "미니프로젝트에서는 무엇을 하나요?",
        "AI 개발자 트랙과 DX 컨설턴트 트랙의 차이는?",
        "효과적으로 학습하는 방법을 알려줘.",
    ]
    picked = None
    if not st.session_state.messages:
        st.markdown("**이런 질문부터 시작해 보세요**")
        for i in range(0, len(suggestions), 2):
            left, right = st.columns(2)
            with left:
                if st.button(suggestions[i], key=f"suggest_{i}", use_container_width=True):
                    picked = suggestions[i]
            with right:
                if st.button(suggestions[i+1], key=f"suggest_{i+1}", use_container_width=True):
                    picked = suggestions[i+1]
    question = st.chat_input("에이블스쿨에 대해 질문해 보세요...") or picked
    if question:
        st.session_state.messages.append(HumanMessage(content=question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("백서에서 근거를 찾고 있어요..."):
                    stream, docs = ask(question, retriever, stream=True)
                    answer = st.write_stream(stream)
                source_details(docs)
                st.session_state.messages.append(AIMessage(content=answer))
            except Exception as e:
                st.error(f"답변 생성에 실패했습니다: {e}")

elif menu == "🎓 학습 로드맵":
    st.subheader("나만의 학습 로드맵")
    st.caption("트랙과 학습 수준을 선택하면 백서에 근거한 학습 방향을 제안해요.")
    if retriever is None:
        st.info("학습 로드맵을 만들려면 먼저 백서 PDF를 업로드해 주세요.")
        st.stop()
    with st.form("roadmap_form"):
        track = st.radio("트랙 선택", ["AI 개발자 트랙", "DX 컨설턴트 트랙"], horizontal=True)
        level = st.radio("현재 학습 수준", ["초보자", "중급자"], horizontal=True)
        goal = st.text_input("학습 목표 (선택)", placeholder="예: Python 기초를 탄탄하게 다지고 싶어요")
        submitted = st.form_submit_button("✨ 학습 로드맵 추천받기", type="primary", use_container_width=True)
    if submitted:
        question = (f"학습자 트랙: {track}\n수준: {level}\n목표: {goal or '기초 역량 강화'}\n"
                    "백서에서 확인할 수 있는 내용을 토대로 단계별 학습 로드맵을 작성해 주세요. "
                    "추천 학습 방향, 단계별 핵심 내용, 실습 제안, 학습 팁을 포함하고 "
                    "백서에 없는 조언은 일반적인 권장사항으로 명시하세요.")
        try:
            with st.spinner("맞춤 로드맵을 작성하고 있어요..."):
                result, docs = ask(question, retriever)
            st.session_state.roadmap = (result, docs)
        except Exception as e:
            st.error(f"로드맵 생성에 실패했습니다: {e}")
    if "roadmap" in st.session_state:
        result, docs = st.session_state.roadmap
        st.markdown("### 추천 학습 로드맵")
        st.markdown(result)
        source_details(docs)
        st.download_button("📥 로드맵 저장", result, file_name="에이블백서_학습로드맵.txt", mime="text/plain")

else:
    st.subheader("백서 문서 관리")
    if uploaded is not None:
        st.success("현재 업로드한 PDF로 검색하고 있습니다.")
        st.write(f"**문서명:** {uploaded.name}")
        st.write(f"**크기:** {uploaded.size / 1024:.1f} KB")
    elif retriever is not None:
        st.success("기존 ChromaDB가 연결되어 있습니다.")
