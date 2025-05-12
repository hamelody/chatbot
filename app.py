import streamlit as st
import os
import io
import fitz  # PyMuPDF
import pandas as pd
import docx
from pptx import Presentation
import faiss
import openai
import numpy as np
import json
import time
from datetime import datetime
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
import tempfile  # 임시 디렉토리 사용
from werkzeug.security import check_password_hash, generate_password_hash
from streamlit_cookies_manager import EncryptedCookieManager
import traceback
import base64

st.set_page_config(
    page_title="유앤생명과학 업무 가이드 봇",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Base64 인코딩 함수 정의 ---
def get_base64_of_bin_file(bin_file_path):
    try:
        with open(bin_file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        print(f"ERROR: 로고 파일을 찾을 수 없습니다: {bin_file_path}")
        return None
    except Exception as e:
        print(f"ERROR: 로고 파일 '{bin_file_path}' 처리 중 오류: {e}")
        return None

# --- 전역 변수 및 경로 설정 ---
# 로컬/리포지토리 기반 파일 (Streamlit Cloud 배포 시 함께 포함됨)
RULES_PATH_REPO = ".streamlit/prompt_rules.txt"
COMPANY_LOGO_PATH_REPO = "company_logo.png"

# Azure Blob Storage 내 객체 이름 (상수화)
INDEX_BLOB_NAME = "vector_db/vector.index" # Blob 내 경로 사용 가능
METADATA_BLOB_NAME = "vector_db/metadata.json"
USERS_BLOB_NAME = "app_data/users.json"
UPLOAD_LOG_BLOB_NAME = "app_logs/upload_log.json"
USAGE_LOG_BLOB_NAME = "app_logs/usage_log.json"

# --- CSS 스타일 ---
st.markdown("""
<style>
    /* CSS 스타일은 이전과 동일하게 유지합니다. */
    .stApp > header ~ div [data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stButton"] > button {
        background-color: #FFFFFF; color: #333F48; border: 1px solid #BCC0C4;
        border-radius: 8px; padding: 8px 12px; font-weight: 500;
        width: auto; min-width: 100px; white-space: nowrap; display: inline-block;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    .stApp > header ~ div [data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stButton"] > button:hover {
        background-color: #F0F2F5; border-color: #A0A4A8;
    }
    .stApp > header ~ div [data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stButton"] > button:active {
        background-color: #E0E2E5;
    }
    .chat-bubble-container { display: flex; flex-direction: column; margin-bottom: 15px; }
    .user-align { align-items: flex-end; }
    .assistant-align { align-items: flex-start; }
    .bubble { padding: 10px 15px; border-radius: 18px; max-width: 75%; word-wrap: break-word; display: inline-block; color: black; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.08); position: relative; }
    .user-bubble { background-color: #90EE90; color: black; border-bottom-right-radius: 5px; }
    .assistant-bubble { background-color: #E9E9EB; border-bottom-left-radius: 5px; }
    .timestamp { font-size: 0.7rem; color: #8E8E93; padding: 2px 5px 0px 5px; }
    .main-title-container { text-align: center; margin-bottom: 24px; }
    .main-title { font-size: 2.1rem; font-weight: bold; display: block; }
    .sub-title { font-size: 0.9rem; color: gray; display: block; margin-top: 4px;}
    .logo-container { display: flex; align-items: center; }
    .logo-image { margin-right: 10px; }
    .version-text { font-size: 0.9rem; color: gray; }
</style>
""", unsafe_allow_html=True)


# --- Azure 클라이언트 초기화 (앱 실행 초기에 한 번만) ---
@st.cache_resource
def get_azure_openai_client_cached():
    print("Attempting to initialize Azure OpenAI client...")
    try:
        client = AzureOpenAI(
            api_key=st.secrets["AZURE_OPENAI_KEY"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            api_version=st.secrets["AZURE_OPENAI_VERSION"]
        )
        print("Azure OpenAI client initialized successfully.")
        return client
    except KeyError as e:
        st.error(f"Azure OpenAI 설정 오류: secrets.toml 파일에 '{e.args[0]}' 키가 없습니다.")
        print(f"ERROR: Missing Azure OpenAI secret: {e.args[0]}")
        return None
    except Exception as e:
        st.error(f"Azure OpenAI 클라이언트 초기화 중 오류 발생: {e}")
        print(f"ERROR: Azure OpenAI client initialization failed: {e}\n{traceback.format_exc()}")
        return None

@st.cache_resource
def get_azure_blob_clients_cached():
    print("Attempting to initialize Azure Blob Service client...")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(st.secrets["AZURE_BLOB_CONN"])
        container_name = st.secrets["BLOB_CONTAINER"]
        container_client = blob_service_client.get_container_client(container_name)
        print(f"Azure Blob Service client and container client for '{container_name}' initialized successfully.")
        return blob_service_client, container_client
    except KeyError as e:
        st.error(f"Azure Blob Storage 설정 오류: secrets.toml 파일에 '{e.args[0]}' 키가 없습니다.")
        print(f"ERROR: Missing Azure Blob Storage secret: {e.args[0]}")
        return None, None
    except Exception as e:
        st.error(f"Azure Blob 클라이언트 초기화 중 오류 발생: {e}")
        print(f"ERROR: Azure Blob client initialization failed: {e}\n{traceback.format_exc()}")
        return None, None

openai_client = get_azure_openai_client_cached()
blob_service, container_client = get_azure_blob_clients_cached()

EMBEDDING_MODEL = None
if openai_client:
    try:
        EMBEDDING_MODEL = st.secrets["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
        print(f"Embedding model set to: {EMBEDDING_MODEL}")
    except KeyError:
        st.error("secrets.toml 파일에 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT' 설정이 없습니다.")
        print("ERROR: Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT secret.")
        openai_client = None
    except Exception as e:
        st.error(f"임베딩 모델 설정 로드 중 오류: {e}")
        print(f"ERROR: Loading AZURE_OPENAI_EMBEDDING_DEPLOYMENT secret: {e}")
        openai_client = None

# --- 사용자 정보 로드/저장 (Azure Blob Storage 기반) ---
@st.cache_resource # 앱 로드 시 한 번만 실행되도록, 단 USERS는 자주 바뀔 수 있으므로 cache 전략 고민 필요
def load_users_from_blob(_container_client):
    if not _container_client:
        print("ERROR: Blob Container client is None for load_users_from_blob.")
        return {"admin": {"name": "관리자(오류)", "department": "오류", "password_hash": generate_password_hash("error"), "approved": True, "role": "admin"}} # 안전 기본값

    users_data = {}
    print(f"Attempting to load users data from Blob: '{USERS_BLOB_NAME}'")
    try:
        users_blob_client = _container_client.get_blob_client(USERS_BLOB_NAME)
        if users_blob_client.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                local_users_path = os.path.join(tmpdir, os.path.basename(USERS_BLOB_NAME))
                with open(local_users_path, "wb") as download_file:
                    download_file.write(users_blob_client.download_blob().readall())
                if os.path.getsize(local_users_path) > 0:
                    with open(local_users_path, "r", encoding="utf-8") as f:
                        users_data = json.load(f)
                    print(f"'{USERS_BLOB_NAME}' loaded successfully from Blob Storage.")
                else: # 파일은 존재하나 비어있는 경우
                    print(f"WARNING: '{USERS_BLOB_NAME}' exists in Blob but is empty.")
        else: # 파일이 아예 없는 경우
            print(f"WARNING: '{USERS_BLOB_NAME}' not found in Blob Storage. Initializing with default admin.")
        
        # users.json이 비어있거나 admin 계정이 없는 경우 기본 관리자 생성
        if "admin" not in users_data:
            users_data["admin"] = {
                "name": "관리자", "department": "품질보증팀",
                "password_hash": generate_password_hash("diteam"), # 실제 운영 시 변경 권장
                "approved": True, "role": "admin"
            }
            # 새 admin 정보를 Blob에 즉시 저장 시도 (최초 실행 시)
            if _container_client: # 재확인
                 # 이 함수는 아래에 정의됨. 순환 호출 피하기 위해 USERS 전역 변수 사용 대신 바로 users_data 전달.
                save_data_to_blob(users_data, USERS_BLOB_NAME, _container_client, "사용자 정보")
            print("Default admin account created/ensured in users_data.")

    except Exception as e:
        print(f"ERROR loading '{USERS_BLOB_NAME}' from Blob: {e}\n{traceback.format_exc()}")
        st.error(f"사용자 정보 로드 중 심각한 오류 발생: {e}")
        # 복구 불가능한 오류 시 안전한 기본값 반환
        users_data = {"admin": {"name": "관리자(오류)", "department": "오류", "password_hash": generate_password_hash("error"), "approved": True, "role": "admin"}}
    return users_data

def save_data_to_blob(data_to_save, blob_name, _container_client, data_description="데이터"):
    if not _container_client:
        st.error(f"Azure Blob 클라이언트가 준비되지 않아 '{data_description}'를 저장할 수 없습니다.")
        print(f"ERROR: Blob Container client is None, cannot save '{blob_name}'.")
        return False
    print(f"Attempting to save '{data_description}' to Blob Storage: '{blob_name}'")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_temp_path = os.path.join(tmpdir, os.path.basename(blob_name))
            with open(local_temp_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            blob_client_instance = _container_client.get_blob_client(blob_name)
            with open(local_temp_path, "rb") as data_stream:
                blob_client_instance.upload_blob(data_stream, overwrite=True)
            print(f"Successfully saved '{data_description}' to Blob: '{blob_name}'")
        return True
    except Exception as e:
        st.error(f"Azure Blob에 '{data_description}' 저장 중 오류: {e}")
        print(f"ERROR saving '{data_description}' to Blob '{blob_name}': {e}\n{traceback.format_exc()}")
        return False

USERS = {} # 전역 변수 선언
if container_client: # Blob 클라이언트가 준비된 후에 사용자 정보 로드
    USERS = load_users_from_blob(container_client)
else:
    st.error("Azure Blob Storage 연결 실패. 사용자 정보를 로드/저장할 수 없습니다.")
    print("CRITICAL: Cannot load USERS due to Blob client initialization failure.")
    # 비상시 로컬 파일 시도 또는 에러 상태 명시
    USERS = {"admin": {"name": "관리자(연결실패)", "department": "시스템", "password_hash": generate_password_hash("fallback"), "approved": True, "role": "admin"}}


# --- 쿠키 매니저 및 세션 상태 초기화 (Azure 클라이언트 초기화 이후 수행) ---
cookies = None
cookie_manager_ready = False
# ... (이전 답변의 쿠키 초기화 로직과 동일, 단 USERS 로드 이후에 배치될 수 있도록 순서 조정 필요 시 가능)
# 이 부분은 USERS 로드와 직접적인 선후관계는 없으나, st.secrets 로드는 이미 위에서 수행됨.
print(f"Attempting to load COOKIE_SECRET from st.secrets (again for context): {st.secrets.get('COOKIE_SECRET')}")
try:
    cookie_secret_key = st.secrets.get("COOKIE_SECRET")
    if not cookie_secret_key:
        st.error("secrets.toml 파일에 'COOKIE_SECRET'이(가) 설정되지 않았거나 비어있습니다. 쿠키 기능을 사용할 수 없습니다.")
        print("ERROR: COOKIE_SECRET is not set or empty in st.secrets (cookie init block).")
    else:
        cookies = EncryptedCookieManager(
            prefix="gmp_chatbot_auth_v2/", # Prefix 변경 시 기존 쿠키와 호환 안됨
            password=cookie_secret_key
        )
        if cookies.ready():
            cookie_manager_ready = True
            print("CookieManager is ready (cookie init block).")
        else:
            print("CookieManager not ready on initial setup (cookie init block).")
except Exception as e:
    st.error(f"쿠키 매니저 초기화 중 알 수 없는 오류 발생: {e}")
    print(f"CRITICAL: CookieManager initialization error (cookie init block): {e}\n{traceback.format_exc()}")

SESSION_TIMEOUT = 1800
try:
    session_timeout_secret = st.secrets.get("SESSION_TIMEOUT")
    if session_timeout_secret: SESSION_TIMEOUT = int(session_timeout_secret)
    print(f"Session timeout set to: {SESSION_TIMEOUT} seconds (cookie init block).")
except: pass # 오류 시 기본값 유지

if "authenticated" not in st.session_state:
    print("Initializing session state: 'authenticated' and 'user' (cookie init block)")
    st.session_state["authenticated"] = False
    st.session_state["user"] = {}
    if cookie_manager_ready:
        auth_cookie_val = cookies.get("authenticated")
        print(f"Cookie 'authenticated' value: {auth_cookie_val} (cookie init block)")
        if auth_cookie_val == "true":
            login_time_str = cookies.get("login_time", "0")
            login_time = float(login_time_str if login_time_str else "0")
            if (time.time() - login_time) < SESSION_TIMEOUT:
                user_json_cookie = cookies.get("user", "{}")
                try:
                    user_data_from_cookie = json.loads(user_json_cookie if user_json_cookie else "{}")
                    if user_data_from_cookie:
                        st.session_state["user"] = user_data_from_cookie
                        st.session_state["authenticated"] = True
                        print(f"User '{user_data_from_cookie.get('name')}' authenticated from cookie (cookie init block).")
                    else: # 쿠키 사용자 정보 비어있음
                        print("User data in cookie is empty. Clearing auth state (cookie init block).")
                        st.session_state["authenticated"] = False
                        if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
                except json.JSONDecodeError: # 쿠키 손상
                    print("ERROR: Failed to decode user JSON from cookie. Clearing auth state (cookie init block).")
                    st.session_state["authenticated"] = False
                    if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
            else: # 세션 타임아웃
                print("Session timeout. Clearing auth state from cookie (cookie init block).")
                st.session_state["authenticated"] = False
                if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
    else:
        print("CookieManager not ready, cannot restore session from cookie (cookie init block).")


# --- 로그인 UI 및 로직 ---
if not st.session_state.get("authenticated", False):
    # ... (이전 답변의 로그인 UI 및 로직과 동일)
    # 단, save_users_local() 호출 부분을 save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "사용자 정보")로 변경
    st.title("🔐 로그인 또는 회원가입")
    if not cookie_manager_ready and st.secrets.get("COOKIE_SECRET"):
        st.warning("쿠키 시스템을 초기화하고 있습니다. 잠시 후 새로고침하거나 다시 시도해주세요.")

    with st.form("auth_form_blob", clear_on_submit=False):
        mode = st.radio("선택", ["로그인", "회원가입"], key="auth_mode_blob")
        uid = st.text_input("ID", key="auth_uid_blob")
        pwd = st.text_input("비밀번호", type="password", key="auth_pwd_blob")
        name, dept = "", ""
        if mode == "회원가입":
            name = st.text_input("이름", key="auth_name_blob")
            dept = st.text_input("부서", key="auth_dept_blob")
        submit_button = st.form_submit_button("확인")

    if submit_button:
        if not uid or not pwd: st.error("ID와 비밀번호를 모두 입력해주세요.")
        elif mode == "회원가입" and (not name or not dept): st.error("이름과 부서를 모두 입력해주세요.")
        else:
            if mode == "로그인":
                user_data_login = USERS.get(uid)
                if not user_data_login: st.error("존재하지 않는 ID입니다.")
                elif not user_data_login.get("approved", False): st.warning("가입 승인이 대기 중입니다.")
                elif check_password_hash(user_data_login["password_hash"], pwd):
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user_data_login
                    if cookie_manager_ready:
                        try:
                            cookies["authenticated"] = "true"; cookies["user"] = json.dumps(user_data_login)
                            cookies["login_time"] = str(time.time()); cookies.save()
                            print(f"Login successful for user '{uid}'. Cookies saved (login block).")
                        except Exception as e_cookie_save: st.warning(f"로그인 쿠키 저장 중 문제 발생: {e_cookie_save}")
                    else: st.warning("쿠키 시스템이 준비되지 않아 로그인 상태를 브라우저에 저장할 수 없습니다.")
                    st.success(f"{user_data_login.get('name', uid)}님, 환영합니다!"); st.rerun()
                else: st.error("비밀번호가 일치하지 않습니다.")
            elif mode == "회원가입":
                if uid in USERS: st.error("이미 존재하는 ID입니다.")
                else:
                    USERS[uid] = {"name": name, "department": dept,
                                  "password_hash": generate_password_hash(pwd),
                                  "approved": False, "role": "user"}
                    if not save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "사용자 정보"):
                        st.error("사용자 정보 저장에 실패했습니다. 관리자에게 문의하세요.")
                        # 실패 시 USERS 딕셔너리 롤백 고려 가능
                    else:
                        st.success("가입 신청 완료! 관리자 승인 후 로그인 가능합니다.")
    st.stop()

# --- 인증 후 메인 애플리케이션 로직 ---
current_user_info = st.session_state.get("user", {})

# --- 헤더 (로고, 버전, 로그아웃 버튼) ---
# ... (이전 답변의 헤더 UI 로직과 동일, COMPANY_LOGO_PATH_REPO 사용)
top_cols = st.columns([0.7, 0.3])
with top_cols[0]:
    if os.path.exists(COMPANY_LOGO_PATH_REPO): # 로컬/리포지토리 경로 사용
        logo_b64 = get_base64_of_bin_file(COMPANY_LOGO_PATH_REPO)
        if logo_b64:
            st.markdown(f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_b64}" class="logo-image" width="150">
                <span class="version-text">ver 1.8 (Full Blob Sync)</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="logo-container"><span class="version-text" style="font-weight:bold;">유앤생명과학</span> <span class="version-text" style="margin-left:10px;">ver 1.8 (Full Blob Sync)</span></div>""", unsafe_allow_html=True)
    else:
        print(f"WARNING: Company logo file not found at {COMPANY_LOGO_PATH_REPO}")
        st.markdown(f"""<div class="logo-container"><span class="version-text" style="font-weight:bold;">유앤생명과학</span> <span class="version-text" style="margin-left:10px;">ver 1.8 (Full Blob Sync)</span></div>""", unsafe_allow_html=True)

with top_cols[1]:
    st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
    if st.button("로그아웃", key="logout_button_blob"):
        st.session_state["authenticated"] = False; st.session_state["user"] = {}
        if cookie_manager_ready:
            try:
                cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
                print("Logout successful. Cookies cleared (logout block).")
            except Exception as e_logout_cookie: print(f"ERROR: Failed to clear cookies on logout: {e_logout_cookie}")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# --- 벡터 DB 로드 (Azure Blob Storage 기반) ---
@st.cache_resource
def load_vector_db_from_blob_cached(_container_client): # 함수명 변경하여 cache 구분
    # ... (이전 답변의 load_vector_db_from_blob 함수 내용과 동일)
    if not _container_client: return faiss.IndexFlatL2(1536), []
    idx, meta = faiss.IndexFlatL2(1536), []
    print(f"Attempting to load vector DB from Blob: '{INDEX_BLOB_NAME}', '{METADATA_BLOB_NAME}'")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # ... (이하 다운로드 및 로드 로직 동일)
            local_index_path = os.path.join(tmpdir, os.path.basename(INDEX_BLOB_NAME)) # 경로 생성 시 basename 사용
            local_metadata_path = os.path.join(tmpdir, os.path.basename(METADATA_BLOB_NAME))

            index_blob_client = _container_client.get_blob_client(INDEX_BLOB_NAME)
            if index_blob_client.exists():
                with open(local_index_path, "wb") as download_file:
                    download_file.write(index_blob_client.download_blob().readall())
                idx = faiss.read_index(local_index_path)
                print(f"'{INDEX_BLOB_NAME}' loaded successfully from Blob Storage.")
            else:
                st.warning(f"Blob Storage에 '{INDEX_BLOB_NAME}' 파일이 없습니다. 새 인덱스가 생성됩니다.")

            metadata_blob_client = _container_client.get_blob_client(METADATA_BLOB_NAME)
            if metadata_blob_client.exists():
                with open(local_metadata_path, "wb") as download_file:
                    download_file.write(metadata_blob_client.download_blob().readall())
                if os.path.getsize(local_metadata_path) > 0 :
                    with open(local_metadata_path, "r", encoding="utf-8") as f: meta = json.load(f)
                else: meta = [] # 파일은 있으나 비어있는 경우
                print(f"'{METADATA_BLOB_NAME}' loaded successfully from Blob Storage.")
            else:
                st.warning(f"Blob Storage에 '{METADATA_BLOB_NAME}' 파일이 없습니다. 빈 메타데이터로 시작합니다.")
                meta = []
    except Exception as e: st.error(f"Azure Blob Storage에서 벡터DB 로드 중 오류: {e}")
    return idx, meta

if container_client:
    index, metadata = load_vector_db_from_blob_cached(container_client)
else:
    index, metadata = faiss.IndexFlatL2(1536), []
    st.error("Azure Blob Storage 연결 실패로 벡터 DB를 로드할 수 없습니다.")


# --- 규칙 파일 로드 ---
@st.cache_data
def load_prompt_rules_cached(): # 함수명 변경하여 cache 구분
    # ... (이전 답변의 load_prompt_rules 함수 내용과 동일, RULES_PATH_REPO 사용)
    if os.path.exists(RULES_PATH_REPO): # 리포지토리 경로 사용
        try:
            with open(RULES_PATH_REPO, "r", encoding="utf-8") as f: rules_content = f.read()
            print(f"Prompt rules loaded successfully from '{RULES_PATH_REPO}'.")
            return rules_content
        except Exception as e: st.warning(f"'{RULES_PATH_REPO}' 파일 로드 중 오류: {e}. 기본 규칙 사용.")
    return "당신은 제약회사 DI/GMP 전문가 챗봇입니다... (기본 규칙)"

PROMPT_RULES_CONTENT = load_prompt_rules_cached()

# --- 텍스트 처리 함수들 (extract_text_from_file, chunk_text_into_pieces, get_text_embedding) ---
# ... (이전 답변의 해당 함수들과 동일하게 유지)
def extract_text_from_file(uploaded_file_obj):
    ext = os.path.splitext(uploaded_file_obj.name)[1].lower(); text_content = ""
    try:
        uploaded_file_obj.seek(0); file_bytes = uploaded_file_obj.read()
        if ext == ".pdf":
            with fitz.open(stream=file_bytes, filetype="pdf") as doc: text_content = "\n".join(p.get_text() for p in doc)
        elif ext == ".docx":
            with io.BytesIO(file_bytes) as doc_io: doc = docx.Document(doc_io); text_content = "\n".join(p.text for p in doc.paragraphs)
        elif ext in (".xlsx", ".xlsm"):
            with io.BytesIO(file_bytes) as excel_io: df = pd.read_excel(excel_io); text_content = df.to_string(index=False)
        elif ext == ".csv":
            with io.BytesIO(file_bytes) as csv_io:
                try: df = pd.read_csv(csv_io)
                except UnicodeDecodeError: df = pd.read_csv(csv_io, encoding='cp949')
                text_content = df.to_string(index=False)
        elif ext == ".pptx":
            with io.BytesIO(file_bytes) as ppt_io: prs = Presentation(ppt_io); text_content = "\n".join(s.text for slide in prs.slides for s in slide.shapes if hasattr(s, "text"))
        else: st.warning(f"지원하지 않는 파일 형식입니다: {ext}"); return ""
    except Exception as e: st.error(f"'{uploaded_file_obj.name}' 파일 내용 추출 중 오류: {e}"); return ""
    return text_content.strip()

def chunk_text_into_pieces(text_to_chunk, chunk_size=500):
    if not text_to_chunk or not text_to_chunk.strip(): return []; chunks_list, current_buffer = [], ""
    for line in text_to_chunk.split("\n"):
        stripped_line = line.strip()
        if not stripped_line and not current_buffer.strip(): continue
        if len(current_buffer) + len(stripped_line) < chunk_size: current_buffer += stripped_line + "\n"
        else:
            if current_buffer.strip(): chunks_list.append(current_buffer.strip())
            current_buffer = stripped_line + "\n"
    if current_buffer.strip(): chunks_list.append(current_buffer.strip())
    return [c for c in chunks_list if c]

def get_text_embedding(text_to_embed):
    if not openai_client or not EMBEDDING_MODEL: return None
    if not text_to_embed or not text_to_embed.strip(): return None
    try:
        response = openai_client.embeddings.create(input=[text_to_embed], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e: st.error(f"텍스트 임베딩 생성 중 오류: {e}"); return None

# --- 유사도 검색 및 문서 추가 (Blob 연동) ---
def search_similar_chunks(query_text, k_results=5):
    # ... (이전 답변의 search_similar_chunks 함수 내용과 동일)
    if index is None or index.ntotal == 0 or not metadata: return []
    query_vector = get_text_embedding(query_text)
    if query_vector is None: return []
    try:
        _, indices_found = index.search(np.array([query_vector]).astype("float32"), k_results)
        return [metadata[i]["content"] for i in indices_found[0] if 0 <= i < len(metadata)]
    except Exception as e: st.error(f"유사도 검색 중 오류: {e}"); return []


def add_document_to_vector_db_and_blob(uploaded_file_obj, text_content, text_chunks, _container_client): # 함수명 변경
    global index, metadata # 전역 index, metadata 직접 수정
    if not text_chunks: st.warning(f"'{uploaded_file_obj.name}' 파일에서 처리할 내용이 없습니다."); return False
    if not _container_client: st.error("Azure Blob 클라이언트가 준비되지 않아 학습 결과를 저장할 수 없습니다."); return False

    # ... (이전 답변의 add_document_to_vector_db 함수 내용과 유사하게 Blob 저장 로직 포함)
    # upload_log.json도 Blob에 저장하도록 수정
    vectors_to_add, new_metadata_entries_for_current_file = [], []
    for chunk in text_chunks:
        embedding = get_text_embedding(chunk)
        if embedding is not None:
            vectors_to_add.append(embedding)
            new_metadata_entries_for_current_file.append({"file_name": uploaded_file_obj.name, "content": chunk})

    if not vectors_to_add: st.warning(f"'{uploaded_file_obj.name}' 파일에서 유효한 임베딩을 생성하지 못했습니다."); return False

    try:
        if vectors_to_add: index.add(np.array(vectors_to_add).astype("float32"))
        metadata.extend(new_metadata_entries_for_current_file)
        print(f"Added {len(vectors_to_add)} new chunks to in-memory DB from '{uploaded_file_obj.name}'.")

        if not save_data_to_blob(index.ntotal > 0 and faiss.serialize_index(index).tobytes(), INDEX_BLOB_NAME, _container_client, "벡터 인덱스", is_binary=True): # 바이너리 저장 함수 필요
             st.error("벡터 인덱스 Blob 저장 실패"); return False # 바이너리 저장을 위한 save_data_to_blob 수정 필요 또는 별도 함수
        # faiss.serialize_index 대신 임시파일 사용
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_index_path = os.path.join(tmpdir, "temp.index")
            faiss.write_index(index, temp_index_path)
            if not save_binary_data_to_blob(temp_index_path, INDEX_BLOB_NAME, _container_client, "벡터 인덱스"):
                 st.error("벡터 인덱스 Blob 저장 실패"); return False

        if not save_data_to_blob(metadata, METADATA_BLOB_NAME, _container_client, "메타데이터"):
            st.error("메타데이터 Blob 저장 실패"); return False

        # 업로드 로그도 Blob에 저장
        user_info = st.session_state.get("user", {}); uploader_name = user_info.get("name", "N/A")
        new_log_entry = {"file": uploaded_file_obj.name, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "chunks_added": len(vectors_to_add), "uploader": uploader_name}
        
        current_upload_logs = load_data_from_blob(UPLOAD_LOG_BLOB_NAME, _container_client, "업로드 로그", default_value=[])
        if not isinstance(current_upload_logs, list): current_upload_logs = [] # 타입 체크
        current_upload_logs.append(new_log_entry)
        if not save_data_to_blob(current_upload_logs, UPLOAD_LOG_BLOB_NAME, _container_client, "업로드 로그"):
            st.warning("업로드 로그를 Blob에 저장하는 데 실패했습니다.")
        return True
    except Exception as e: st.error(f"문서 학습 또는 Azure Blob 업로드 중 오류: {e}"); return False

def save_binary_data_to_blob(local_file_path, blob_name, _container_client, data_description="바이너리 데이터"):
    if not _container_client: return False
    try:
        blob_client_instance = _container_client.get_blob_client(blob_name)
        with open(local_file_path, "rb") as data_stream:
            blob_client_instance.upload_blob(data_stream, overwrite=True)
        print(f"Successfully saved binary '{data_description}' to Blob: '{blob_name}'")
        return True
    except Exception as e: st.error(f"Azure Blob에 '{data_description}' 저장 중 오류: {e}"); return False


def load_data_from_blob(blob_name, _container_client, data_description="데이터", default_value=None, is_binary=False):
    if not _container_client: return default_value
    try:
        blob_client_instance = _container_client.get_blob_client(blob_name)
        if blob_client_instance.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                local_temp_path = os.path.join(tmpdir, os.path.basename(blob_name))
                with open(local_temp_path, "wb") as download_file:
                    download_file.write(blob_client_instance.download_blob().readall())
                if os.path.getsize(local_temp_path) > 0:
                    if is_binary: # 바이너리 파일 처리 (예: Faiss 인덱스) - 이 함수에서는 JSON만 가정
                        # 이 부분은 load_vector_db_from_blob_cached에서 직접 처리
                        print(f"Binary data '{data_description}' should be loaded by a specific function.")
                        return local_temp_path # 임시 경로 반환 또는 실제 데이터 로드
                    else:
                        with open(local_temp_path, "r", encoding="utf-8") as f:
                            loaded_data = json.load(f)
                        print(f"'{data_description}' loaded from Blob: '{blob_name}'")
                        return loaded_data
                else: return default_value if default_value is not None else {} if not is_binary else None # 빈 파일
        else: print(f"'{data_description}' not found in Blob: '{blob_name}'. Returning default.")
    except Exception as e: print(f"Error loading '{data_description}' from Blob '{blob_name}': {e}")
    return default_value if default_value is not None else {} if not is_binary else None


def save_original_file_to_blob(uploaded_file_obj, _container_client): # _container_client 인자 추가
    # ... (이전 답변의 save_original_file_to_blob 함수 내용과 동일, _container_client 사용)
    if not _container_client: st.error("Azure Blob 클라이언트 준비 안됨"); return None
    try:
        uploaded_file_obj.seek(0)
        blob_client_for_original = _container_client.get_blob_client(blob=f"uploaded_originals/{uploaded_file_obj.name}") # 경로 추가
        blob_client_for_original.upload_blob(uploaded_file_obj.getvalue(), overwrite=True)
        file_url = f"Original file '{uploaded_file_obj.name}' saved to Blob." # URL 생성은 복잡하므로 단순 메시지
        print(file_url)
        return file_url
    except Exception as e: st.error(f"'{uploaded_file_obj.name}' 원본 파일 Blob 업로드 오류: {e}"); return None

# --- 사용량 로깅 함수 (Blob 연동) ---
def log_openai_api_usage_to_blob(user_id_str, model_name_str, usage_stats_obj, _container_client): # _container_client 인자 추가
    # ... (이전 답변의 log_openai_api_usage_to_blob 함수 내용과 동일, _container_client 사용)
    if not _container_client: print("ERROR: Blob Container client is None for API usage log."); return

    new_log_entry = {
        "user_id": user_id_str, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": model_name_str, "prompt_tokens": usage_stats_obj.prompt_tokens,
        "completion_tokens": usage_stats_obj.completion_tokens, "total_tokens": usage_stats_obj.total_tokens
    }
    current_usage_logs = load_data_from_blob(USAGE_LOG_BLOB_NAME, _container_client, "API 사용량 로그", default_value=[])
    if not isinstance(current_usage_logs, list): current_usage_logs = [] # 타입 보장
    current_usage_logs.append(new_log_entry)
    if not save_data_to_blob(current_usage_logs, USAGE_LOG_BLOB_NAME, _container_client, "API 사용량 로그"):
        st.warning("API 사용량 로그를 Blob에 저장하는 데 실패했습니다.")


# --- 메인 UI 구성 ---
# ... (이전 답변의 메인 UI 구성과 동일)
# 단, add_document... 호출 시 _container_client 전달
# log_openai_api_usage... 호출 시 _container_client 전달
# 관리자 탭의 사용자 승인/거절 시 save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "사용자 정보") 호출
# 관리자 탭의 사용량 모니터링 시 load_data_from_blob(USAGE_LOG_BLOB_NAME, container_client, ...) 호출

st.markdown("""
<div class="main-title-container">
  <span class="main-title">유앤생명과학 GMP/SOP 업무 가이드 봇</span>
  <span class="sub-title">Made by DI.PART</span>
</div>
""", unsafe_allow_html=True)

tab_labels_list = ["💬 업무 질문"]
if current_user_info.get("role") == "admin":
    tab_labels_list.append("⚙️ 관리자 설정")

main_tabs_list = st.tabs(tab_labels_list)
chat_interface_tab = main_tabs_list[0]
admin_settings_tab = main_tabs_list[1] if len(main_tabs_list) > 1 else None

with chat_interface_tab:
    # ... (챗봇 인터페이스 UI 로직 - 이전과 동일하게 유지)
    # 단, log_openai_api_usage_to_blob 호출 시 container_client 전달
    st.header("업무 질문")
    st.markdown("💡 예시: SOP 백업 주기, PIC/S Annex 11 차이 등")

    if "messages" not in st.session_state: st.session_state["messages"] = []
    for msg_item in st.session_state["messages"]:
        # ... (메시지 표시 로직)
        role, content, time_str = msg_item.get("role"), msg_item.get("content", ""), msg_item.get("time", "")
        align_class = "user-align" if role == "user" else "assistant-align"
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        st.markdown(f"""<div class="chat-bubble-container {align_class}"><div class="bubble {bubble_class}">{content}</div><div class="timestamp">{time_str}</div></div>""", unsafe_allow_html=True)


    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("📂 파일 첨부/숨기기", key="toggle_chat_uploader_button_blob"):
        st.session_state.show_uploader = not st.session_state.get("show_uploader", False)

    chat_file_uploader_key = "chat_file_uploader_widget_blob"
    uploaded_chat_file_runtime = None # form 제출과 무관하게 현재 위젯 상태 반영
    if st.session_state.get("show_uploader", False):
        uploaded_chat_file_runtime = st.file_uploader("질문과 함께 참고할 파일 첨부 (선택 사항)",
                                     type=["pdf","docx","xlsx","xlsm","csv","pptx"],
                                     key=chat_file_uploader_key)
        if uploaded_chat_file_runtime: st.caption(f"첨부됨: {uploaded_chat_file_runtime.name}")

    with st.form("chat_input_form_blob", clear_on_submit=True):
        query_input_col, send_button_col = st.columns([4,1])
        with query_input_col:
            user_query_input = st.text_input("질문 입력:", placeholder="여기에 질문을 입력하세요...",
                                             key="user_query_text_input_blob", label_visibility="collapsed")
        with send_button_col:
            send_query_button = st.form_submit_button("전송")

    if send_query_button and user_query_input.strip() and openai_client:
        timestamp_now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state["messages"].append({"role":"user", "content":user_query_input, "time":timestamp_now_str})
        
        user_id_for_log = current_user_info.get("name", "anonymous_chat_user") # 현재 로그인 사용자 이름 사용

        # 파일 업로더의 현재 값 (form 제출 시점)을 사용
        # uploaded_chat_file_runtime는 위젯의 실시간 상태를 반영하므로, form 제출 시점의 파일은
        # 해당 키로 session_state에서 가져오는 것이 더 명확할 수 있으나, 현재 구조에서는 uploaded_chat_file_runtime 사용도 가능
        
        with st.spinner("답변 생성 중... 잠시만 기다려주세요."):
            try:
                context_chunks_for_prompt = []
                if uploaded_chat_file_runtime: # 위에서 정의된 변수 사용
                    temp_file_text = extract_text_from_file(uploaded_chat_file_runtime)
                    if temp_file_text: # .strip()은 chunk_text_into_pieces 내부에서 처리
                        temp_file_chunks = chunk_text_into_pieces(temp_file_text)
                        if temp_file_chunks:
                            for chunk_piece in temp_file_chunks:
                                context_chunks_for_prompt.extend(search_similar_chunks(chunk_piece, k_results=2))
                            if not context_chunks_for_prompt: context_chunks_for_prompt.append(temp_file_text[:2000])
                        else: st.info(f"'{uploaded_chat_file_runtime.name}' 파일에서 유의미한 텍스트를 추출하지 못했습니다.")
                    else: st.info(f"'{uploaded_chat_file_runtime.name}' 파일이 비어있거나 지원하지 않는 내용입니다.")

                if not context_chunks_for_prompt or len(context_chunks_for_prompt) < 3:
                    needed_k = max(1, 3 - len(context_chunks_for_prompt))
                    context_chunks_for_prompt.extend(search_similar_chunks(user_query_input, k_results=needed_k))

                final_unique_context = list(set(c for c in context_chunks_for_prompt if c and c.strip()))
                if not final_unique_context: st.info("질문과 관련된 참고 정보를 찾지 못했습니다. 일반적인 답변을 시도합니다.")

                context_string_for_llm = "\n\n---\n\n".join(final_unique_context) if final_unique_context else "현재 참고할 문서가 없습니다."
                system_prompt_content = f"{PROMPT_RULES_CONTENT}\n\n위의 규칙을 반드시 준수하여 답변해야 합니다. 다음은 사용자의 질문에 답변하는 데 참고할 수 있는 문서의 내용입니다:\n<문서 시작>\n{context_string_for_llm}\n<문서 끝>"
                
                chat_messages = [{"role":"system", "content": system_prompt_content}, {"role":"user", "content": user_query_input}]
                chat_completion_response = openai_client.chat.completions.create(
                    model=st.secrets["AZURE_OPENAI_DEPLOYMENT"], messages=chat_messages, max_tokens=4000, temperature=0.1)
                assistant_response_content = chat_completion_response.choices[0].message.content.strip()
                st.session_state["messages"].append({"role":"assistant", "content":assistant_response_content, "time":timestamp_now_str})
                
                if chat_completion_response.usage and container_client: # container_client 확인
                    log_openai_api_usage_to_blob(user_id_for_log, st.secrets["AZURE_OPENAI_DEPLOYMENT"], chat_completion_response.usage, container_client)

            except Exception as gen_err: # 더 포괄적인 예외 처리
                st.error(f"답변 생성 중 오류 발생: {gen_err}")
                st.session_state["messages"].append({"role":"assistant", "content":"답변 생성 중 오류가 발생했습니다.", "time":timestamp_now_str})
                print(f"ERROR: General error during response generation: {gen_err}\n{traceback.format_exc()}")
        st.rerun()
    elif send_query_button and not openai_client:
         st.error("OpenAI 서비스가 준비되지 않아 답변을 생성할 수 없습니다. 관리자에게 문의하세요.")


if admin_settings_tab:
    with admin_settings_tab:
        st.header("⚙️ 관리자 설정")
        st.subheader("👥 가입 승인 대기자")
        if not USERS: st.warning("사용자 정보를 로드할 수 없습니다 (USERS 딕셔너리 비어있음).")
        else:
            pending_approval_users = {uid:udata for uid,udata in USERS.items() if not udata.get("approved")}
            if pending_approval_users:
                for pending_uid, pending_user_data in pending_approval_users.items():
                    with st.expander(f"{pending_user_data.get('name','N/A')} ({pending_uid}) - {pending_user_data.get('department','N/A')}"):
                        approve_col, reject_col = st.columns(2)
                        if approve_col.button("승인", key=f"admin_approve_user_blob_{pending_uid}"):
                            USERS[pending_uid]["approved"] = True
                            if save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "사용자 정보"):
                                st.success(f"'{pending_uid}' 사용자를 승인하고 Blob에 저장했습니다."); st.rerun()
                            else: st.error("사용자 승인 정보 Blob 저장 실패.")
                        if reject_col.button("거절", key=f"admin_reject_user_blob_{pending_uid}"):
                            USERS.pop(pending_uid)
                            if save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "사용자 정보"):
                                st.info(f"'{pending_uid}' 사용자의 가입 신청을 거절하고 Blob에 저장했습니다."); st.rerun()
                            else: st.error("사용자 거절 정보 Blob 저장 실패.")
            else: st.info("승인 대기 중인 사용자가 없습니다.")
        st.markdown("---")

        st.subheader("📁 파일 업로드 및 학습 (Azure Blob Storage)")
        admin_file_uploader_key = "admin_file_uploader_widget_blob"
        admin_uploaded_file = st.file_uploader("학습할 파일 업로드", type=["pdf","docx","xlsx","xlsm","csv","pptx"], key=admin_file_uploader_key)

        if admin_uploaded_file and container_client:
            with st.spinner(f"'{admin_uploaded_file.name}' 파일 처리 및 학습 중..."):
                extracted_content = extract_text_from_file(admin_uploaded_file)
                if extracted_content:
                    content_chunks = chunk_text_into_pieces(extracted_content)
                    if content_chunks:
                        original_file_blob_url = save_original_file_to_blob(admin_uploaded_file, container_client)
                        if original_file_blob_url: st.caption(f"원본 파일 Blob 저장됨 (메시지 단순화)")

                        if add_document_to_vector_db_and_blob(admin_uploaded_file, extracted_content, content_chunks, container_client):
                            st.success(f"'{admin_uploaded_file.name}' 파일 학습 및 Azure Blob Storage에 업데이트 완료!")
                        else: st.error(f"'{admin_uploaded_file.name}' 학습 또는 Blob 업데이트 오류.")
                    else: st.warning(f"'{admin_uploaded_file.name}' 파일에서 유의미한 청크를 생성하지 못했습니다.")
                else: st.warning(f"'{admin_uploaded_file.name}' 파일이 비었거나 지원하지 않는 내용입니다.")
            st.rerun()
        elif admin_uploaded_file and not container_client:
            st.error("Azure Blob 클라이언트가 준비되지 않아 파일을 업로드하고 학습할 수 없습니다.")
        st.markdown("---")

        st.subheader("📊 API 사용량 모니터링 (Blob 로그 기반)")
        if container_client:
            usage_data_from_blob = load_data_from_blob(USAGE_LOG_BLOB_NAME, container_client, "API 사용량 로그", default_value=[])
            if usage_data_from_blob and isinstance(usage_data_from_blob, list):
                df_usage_stats=pd.DataFrame(usage_data_from_blob)
                if not df_usage_stats.empty:
                    total_tokens_used = df_usage_stats["total_tokens"].sum() if "total_tokens" in df_usage_stats.columns else 0
                    st.metric("총 API 호출 수", len(df_usage_stats))
                    st.metric("총 사용 토큰 수", f"{int(total_tokens_used):,}") # 정수로 변환
                    token_cost_per_unit = 0.0
                    try: token_cost_per_unit=float(st.secrets.get("TOKEN_COST","0"))
                    except: pass
                    st.metric("예상 비용 (USD)", f"${int(total_tokens_used) * token_cost_per_unit:.4f}")
                    if "timestamp" in df_usage_stats.columns:
                        st.dataframe(df_usage_stats.sort_values(by="timestamp",ascending=False), use_container_width=True)
                    else: st.dataframe(df_usage_stats, use_container_width=True)
                else: st.info("기록된 API 사용량 데이터가 Blob에 없습니다.")
            else: st.info("Blob에서 API 사용량 로그를 가져오지 못했거나 데이터가 없습니다.")
        else: st.warning("Azure Blob 클라이언트가 준비되지 않아 API 사용량 모니터링을 표시할 수 없습니다.")
        st.markdown("---")

        st.subheader("📂 Azure Blob Storage 파일 목록 (최근 100개)")
        # ... (이전 답변의 Blob 파일 목록 UI와 동일하게 유지)
        if container_client:
            try:
                blob_list_display = []
                for blob_item in container_client.list_blobs(results_per_page=100): # name_starts_with="app_data/" 등으로 필터링 가능
                    blob_list_display.append({
                        "파일명": blob_item.name, 
                        "크기 (bytes)": blob_item.size, 
                        "수정일": blob_item.last_modified.strftime('%Y-%m-%d %H:%M:%S') if blob_item.last_modified else 'N/A'
                    })
                if blob_list_display:
                    df_blobs_display = pd.DataFrame(blob_list_display)
                    st.dataframe(df_blobs_display.sort_values(by="수정일", ascending=False), use_container_width=True)
                else: st.info("Azure Blob Storage에 파일이 없습니다 (또는 지정된 경로에).")
            except Exception as e: st.error(f"Azure Blob 파일 목록 조회 중 오류: {e}")
        else: st.warning("Azure Blob 클라이언트가 준비되지 않아 파일 목록을 표시할 수 없습니다.")