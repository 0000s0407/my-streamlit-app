import os
import hashlib
import tempfile
import streamlit as st
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

st.set_page_config(page_title='에이블 백서 | AIVLE WHITEPAPER', page_icon='📖', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
:root{--navy:#142943;--blue:#315c8c;--line:#dce3eb}
html,body,[class*="st-"]{font-family:'DM Sans','Noto Sans KR',sans-serif}
.stApp{background:#f8fafc;color:#142943}
[data-testid="stSidebar"]{background:#eef2f7;border-right:1px solid #dce3eb}
[data-testid="stSidebar"] .block-container{padding-top:2rem}
.block-container{max-width:1200px;padding-top:2.7rem;padding-bottom:5rem}
h1,h2,h3{color:#142943!important;letter-spacing:-.045em}
h1{font-size:2.15rem!important;font-weight:800!important}
h2{font-size:1.4rem!important}
h3{font-size:1.13rem!important}
[data-testid="stChatMessage"]{background:#fff;border:1px solid #e0e6ee;border-radius:14px;padding:1rem 1.25rem;margin-bottom:.65rem;box-shadow:0 2px 8px rgba(15,35,60,.025)}
[data-testid="stChatMessageAvatarUser"],[data-testid="stChatMessageAvatarAssistant"]{border-radius:10px}
.stButton button,.stDownloadButton button{border-radius:9px!important;border:1px solid #d8e1eb!important;font-weight:600!important;min-height:2.7rem}
.stButton button[kind="primary"],.stFormSubmitButton button[kind="primary"]{background:#19395e!important;color:#fff!important;border-color:#19395e!important}
.stButton button:hover{border-color:#426d9d!important;color:#315c8c!important}
[data-testid="stChatInput"]{border-radius:12px}
.brand{font-size:12px;letter-spacing:.22em;color:#507297;font-weight:700}
.wordmark{font-size:27px;font-weight:800;letter-spacing:-.07em;color:#142943;margin:4px 0 5px}
.brand-sub{font-size:12px;color:#68798e;line-height:1.6}
.hero{padding:26px 0 28px;border-bottom:1px solid #dbe3ed;margin-bottom:26px}
.hero-eyebrow{font-size:11px;letter-spacing:.23em;font-weight:700;color:#4c7197}
.hero-title{font-size:clamp(32px,3vw,46px);font-weight:800;letter-spacing:-.07em;color:#142943;margin:12px 0 8px}
.hero-description{color:#63758a;font-size:14px;line-height:1.9}
.tile{background:#fff;border:1px solid #dfe6ee;border-radius:13px;padding:20px 20px 18px;min-height:120px}
.tile-num{font-size:11px;color:#527ca5;font-weight:700;letter-spacing:.14em}
.tile-title{font-size:17px;font-weight:700;color:#17314e;margin:10px 0 5px}
.tile-desc{font-size:12px;color:#6e7e90;line-height:1.7}
.section-label{font-size:11px;letter-spacing:.18em;color:#53779a;font-weight:700;margin-top:22px}
.small-rule{height:1px;background:#dce4ec;margin:20px 0}
</style>''', unsafe_allow_html=True)

try:
    api_key = st.secrets.get('OPENAI_API_KEY')
except Exception:
    api_key = None
api_key = api_key or os.environ.get('OPENAI_API_KEY')
if not api_key:
    st.error('Streamlit Cloud → App settings → Secrets에 OPENAI_API_KEY를 등록해 주세요.')
    st.stop()
os.environ['OPENAI_API_KEY'] = api_key

@st.cache_resource(show_spinner=False)
def embeddings():
    return OpenAIEmbeddings(model='text-embedding-3-small')

@st.cache_resource(show_spinner=False)
def llm():
    return ChatOpenAI(model='gpt-4o-mini', temperature=0)

@st.cache_resource(show_spinner=False)
def default_db():
    if not os.path.isdir('chroma_db'):
        return None
    db = Chroma(collection_name='aivle_docs', embedding_function=embeddings(), persist_directory='chroma_db')
    return db if db._collection.count() else None

@st.cache_resource(show_spinner=False, max_entries=4)
def rag_pipeline(pdf_bytes, filename):
    # 01번 노트북과 같은 PDF 로드 → 분할 → 임베딩 → Chroma 흐름
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name
    try:
        docs = PyMuPDF4LLMLoader(path).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=150).split_documents(docs)
        if not chunks:
            raise ValueError('PDF에서 읽을 수 있는 텍스트를 찾지 못했습니다.')
        for d in chunks:
            d.metadata['source'] = filename
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        db = Chroma(collection_name='doc_' + digest[:24], embedding_function=embeddings())
        db.add_documents(chunks)
        return db
    finally:
        os.unlink(path)

def retrieve(db, query):
    # MMR: 유사도와 문서 다양성을 함께 고려
    return db.as_retriever(search_type='mmr', search_kwargs={'k': 4, 'fetch_k': 12, 'lambda_mult': 0.65}).invoke(query)

def contextual_query(question, history):
    if not history:
        return question
    turns = history[-6:]
    transcript = '\n'.join(('사용자' if x['role']=='user' else '도우미') + ': ' + x['content'][:650] for x in turns)
    instruction = ('다음 대화의 마지막 질문을 백서 검색에 적합한 독립적인 검색 질문으로 재작성하세요. '
                   '이전 대화에서 지칭한 대상만 보충하고 새로운 사실은 추가하지 마세요. '
                   '검색 질문 한 문장만 출력하세요.\n\n대화:\n' + transcript + '\n현재 질문: ' + question)
    try:
        return llm().invoke([('human', instruction)]).content.strip() or question
    except Exception:
        return question

def answer_question(question, db, history=None, roadmap=False):
    history = history or []
    search_query = question if roadmap else contextual_query(question, history)
    docs = retrieve(db, search_query)
    context = '\n\n'.join(f'[근거 {i}] {d.page_content}' for i,d in enumerate(docs,1))
    if not context:
        return '관련 내용을 백서에서 찾지 못했습니다.', docs
    rules = '''당신은 AIVLE School의 전문 학습 가이드 '에이블 백서'입니다.
오직 [검색된 백서]의 사실을 근거로 설명하세요. 백서에 없는 일정, 의무, 운영 규정, 수치를 만들어내지 마세요.
검색 근거가 충분하지 않으면 '백서에서 확인되지 않습니다'라고 명시하세요.
백서에서 확인되는 사실과 일반적인 학습 제안을 구분하세요. 일반적인 제안은 '학습 제안'으로 표시하세요.
정확하고 간결한 한국어로 답변하고 필요할 때 소제목과 짧은 목록을 사용하세요.
출처 번호를 인용하려면 제공된 [근거 N]만 사용하세요. 페이지 번호는 임의로 만들지 마세요.
AI 개발자 트랙과 DX 컨설턴트 트랙을 혼동하지 마세요.'''
    response = llm().invoke([('system', rules + '\n\n[검색된 백서]\n' + context), ('human', question)])
    return response.content, docs

def sources(docs):
    if not docs:
        return
    with st.expander('근거 문서 확인 · ' + str(len(docs)) + '개 발췌'):
        for i, d in enumerate(docs, 1):
            source = os.path.basename(str(d.metadata.get('source', '백서')))
            page = d.metadata.get('page')
            page_text = f' · 원본 페이지 메타데이터 {page}' if page is not None else ''
            st.markdown(f'**{i:02d} · {source}{page_text}**')
            st.caption(d.page_content[:850])
            if i < len(docs):
                st.divider()

with st.sidebar:
    st.markdown('<div class="brand">AIVLE SCHOOL / KNOWLEDGE</div><div class="wordmark">에이블 백서</div><div class="brand-sub">AI · DX Learning Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-rule"></div>', unsafe_allow_html=True)
    menu = st.radio('NAVIGATION', ['백서 AI 어시스턴트', '맞춤 학습 로드맵', '문서 라이브러리'], label_visibility='collapsed')
    st.markdown('<div class="small-rule"></div>', unsafe_allow_html=True)
    st.markdown('**백서 PDF 등록**')
    upload = st.file_uploader('PDF 파일', type=['pdf'], label_visibility='collapsed')
    if upload:
        st.caption('현재 문서 · ' + upload.name)
    else:
        st.caption('PDF를 업로드하거나 저장된 DB를 사용합니다.')
    st.markdown('<div class="small-rule"></div>', unsafe_allow_html=True)
    if st.button('대화 기록 초기화', use_container_width=True):
        st.session_state['messages'] = []
        st.rerun()
    st.caption('AIVLE WHITEPAPER · LEARNING ASSISTANT')

try:
    if upload:
        data = upload.getvalue()
        doc_key = hashlib.sha256(data).hexdigest()
        with st.spinner('백서를 분석하고 있습니다...'):
            db = rag_pipeline(data, upload.name)
    else:
        doc_key = 'default'
        db = default_db()
except Exception as exc:
    st.error('문서 처리 중 문제가 발생했습니다: ' + str(exc))
    st.stop()

if st.session_state.get('doc_key') != doc_key:
    st.session_state['messages'] = []
    st.session_state.pop('roadmap_result', None)
    st.session_state['doc_key'] = doc_key
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.markdown('''<div class="hero"><div class="hero-eyebrow">AIVLE SCHOOL · DIGITAL WHITEPAPER</div>
<div class="hero-title">에이블 백서</div><div class="hero-description">교육과정의 이해부터 프로젝트와 학습 설계까지.<br>백서의 근거를 탐색하고, 나에게 필요한 학습 방향을 발견하세요.</div></div>''', unsafe_allow_html=True)
a,b,c = st.columns(3, gap='medium')
for col,num,title,desc in [(a,'01','Knowledge Search','백서에 기반한 신뢰도 높은 문서 검색'),(b,'02','Learning Roadmap','AI 개발자 · DX 컨설턴트 맞춤 가이드'),(c,'03','Source Transparency','검색 근거와 원문 발췌 확인')]:
    with col:
        st.markdown(f'<div class="tile"><div class="tile-num">{num} / SERVICE</div><div class="tile-title">{title}</div><div class="tile-desc">{desc}</div></div>', unsafe_allow_html=True)

if menu == '백서 AI 어시스턴트':
    st.markdown('<div class="section-label">01 / KNOWLEDGE ASSISTANT</div>', unsafe_allow_html=True)
    st.subheader('백서 AI 어시스턴트')
    if db is None:
        st.info('왼쪽 사이드바에서 백서 PDF를 업로드해 주세요.')
    else:
        for msg in st.session_state['messages']:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if msg['role'] == 'assistant' and msg.get('docs'):
                    sources(msg['docs'])
        if not st.session_state['messages']:
            st.caption('추천 질문으로 시작해 보세요.')
            suggestions = ['AIVLE School의 교육과정은 어떻게 구성되나요?', '미니프로젝트에서는 무엇을 배우나요?', 'AI 개발자와 DX 컨설턴트 트랙의 차이는?', '빅프로젝트를 준비할 때 고려할 점은?']
            cols = st.columns(2)
            for i, q in enumerate(suggestions):
                with cols[i%2]:
                    if st.button(q, key=f'suggest_{i}', use_container_width=True):
                        st.session_state['pending'] = q
                        st.rerun()
        question = st.chat_input('백서에 대해 질문하세요')
        question = st.session_state.pop('pending', None) or question
        if question:
            previous = list(st.session_state['messages'])
            st.session_state['messages'].append({'role':'user','content':question})
            with st.chat_message('user'):
                st.markdown(question)
            with st.chat_message('assistant'):
                try:
                    with st.spinner('관련 근거를 검색하고 답변을 작성하고 있습니다...'):
                        answer, docs = answer_question(question, db, previous)
                    st.markdown(answer)
                    sources(docs)
                    st.session_state['messages'].append({'role':'assistant','content':answer,'docs':docs})
                except Exception as exc:
                    st.error('답변을 생성하지 못했습니다: ' + str(exc))

elif menu == '맞춤 학습 로드맵':
    st.markdown('<div class="section-label">02 / PERSONALIZED LEARNING</div>', unsafe_allow_html=True)
    st.subheader('맞춤 학습 로드맵')
    st.caption('트랙과 현재 수준을 선택하면 백서의 근거를 활용해 학습 방향을 제안합니다.')
    with st.form('roadmap'):
        track = st.selectbox('교육 트랙', ['AI 개발자 트랙','DX 컨설턴트 트랙'])
        level = st.selectbox('현재 학습 수준', ['초보자','중급자'])
        goal = st.text_input('학습 목표 (선택)', placeholder='예: 미니프로젝트 수행 역량을 키우고 싶어요')
        submitted = st.form_submit_button('로드맵 생성', type='primary', use_container_width=True)
    if submitted:
        if db is None:
            st.warning('먼저 백서 PDF를 업로드해 주세요.')
        else:
            question = f'''{track}의 {level} 학습자에게 적합한 학습 로드맵을 제안해 주세요. 학습 목표: {goal or '기본 역량 향상'}.
백서에 명시된 트랙별 학습 내용만 사실로 제시하세요. 1. 현재 단계의 학습 방향 2. 단계별 학습 계획 3. 권장 실습 4. 점검 방법 순서로 답하세요.
백서에서 확인되지 않는 구체적 기간·필수 과목은 단정하지 말고, 일반적인 학습 제안은 별도 표시하세요.'''
            try:
                with st.spinner('학습 로드맵을 작성하고 있습니다...'):
                    result, docs = answer_question(question, db, roadmap=True)
                st.session_state['roadmap_result'] = (result, docs)
            except Exception as exc:
                st.error('로드맵 생성 오류: ' + str(exc))
    if st.session_state.get('roadmap_result'):
        result, docs = st.session_state['roadmap_result']
        st.markdown('### 추천 학습 로드맵')
        st.markdown(result)
        sources(docs)
        st.download_button('로드맵 텍스트 저장', result, file_name='aivle_roadmap.txt', mime='text/plain')

else:
    st.markdown('<div class="section-label">03 / DOCUMENT LIBRARY</div>', unsafe_allow_html=True)
    st.subheader('문서 라이브러리')
    if upload:
        st.success('업로드한 PDF가 검색에 연결되었습니다.')
        st.write('**문서명**', upload.name)
        st.write('**파일 크기**', f'{upload.size/1024:.1f} KB')
    elif db is not None:
        st.success('기존 ChromaDB가 연결되어 있습니다.')
    else:
        st.info('연결된 문서가 없습니다. 사이드바에서 PDF를 등록해 주세요.')
    st.caption('새 문서를 업로드하면 해당 문서로 검색 대상이 전환됩니다. 업로드 문서는 서버 재시작 후 다시 등록해야 할 수 있습니다.')

