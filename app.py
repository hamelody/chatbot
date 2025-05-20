import streamlit as st # 첫 번째 라인 또는 주석/빈 줄 제외 첫 라인
# st.set_page_config 보다 먼저 실행되면 안 되는 import는 아래로 이동

import os
import io
import fitz  # PyMuPDF
import pandas as pd
import docx
from pptx import Presentation
import faiss
import openai # openai 패키지 직접 임포트
import numpy as np
import json
import time
from datetime import datetime
from openai import AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError # 구체적인 예외 타입 임포트
from azure.core.exceptions import AzureError # Azure SDK 공통 예외
from azure.storage.blob import BlobServiceClient
import tempfile
from werkzeug.security import check_password_hash, generate_password_hash
# from streamlit_cookies_manager import EncryptedCookieManager # 아래로 이동
import traceback
import base64 # 이미지 처리를 위해 추가
import tiktoken

# Streamlit 앱의 가장 첫 번째 명령으로 st.set_page_config() 호출
st.set_page_config(
    page_title="유앤생명과학 업무 가이드 봇",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- st.set_page_config() 호출 이후에 관련 라이브러리 import ---
from streamlit_cookies_manager import EncryptedCookieManager
print("Imported streamlit_cookies_manager after set_page_config.")

# --- Tiktoken 인코더 로드 ---
try:
    tokenizer = tiktoken.get_encoding("o200k_base")
    print("Tiktoken 'o200k_base' encoder loaded successfully.")
except Exception as e:
    st.error(f"Tiktoken 인코더 로드 실패: {e}. 토큰 기반 길이 제한이 작동하지 않을 수 있습니다.")
    print(f"ERROR: Failed to load tiktoken encoder: {e}")
    tokenizer = None

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
RULES_PATH_REPO = ".streamlit/prompt_rules.txt"
COMPANY_LOGO_PATH_REPO = "company_logo.png"
INDEX_BLOB_NAME = "vector_db/vector.index"
METADATA_BLOB_NAME = "vector_db/metadata.json"
USERS_BLOB_NAME = "app_data/users.json"
UPLOAD_LOG_BLOB_NAME = "app_logs/upload_log.json"
USAGE_LOG_BLOB_NAME = "app_logs/usage_log.json"
AZURE_OPENAI_TIMEOUT = 60.0
MODEL_MAX_INPUT_TOKENS = 128000
MODEL_MAX_OUTPUT_TOKENS = 16384 # GPT-4o-mini의 경우 16k가 아닐 수 있음, 모델 스펙 확인 필요 (예: 4k 또는 8k)
                                # 여기서는 일단 요청하신대로 유지하나, 실제 모델의 최대 출력 토큰으로 조정 필요
BUFFER_TOKENS = 500
TARGET_INPUT_TOKENS_FOR_PROMPT = MODEL_MAX_INPUT_TOKENS - MODEL_MAX_OUTPUT_TOKENS - BUFFER_TOKENS
IMAGE_DESCRIPTION_MAX_TOKENS = 500 # 이미지 설명 생성 시 최대 토큰


# --- CSS 스타일 ---
st.markdown("""
<style>
    /* (CSS 내용은 이전 답변과 동일) */
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
    .main-app-title-container { text-align: center; margin-bottom: 24px; }
    .main-app-title { font-size: 2.1rem; font-weight: bold; display: block; }
    .main-app-subtitle { font-size: 0.9rem; color: gray; display: block; margin-top: 4px;}
    .logo-container { display: flex; align-items: center; }
    .logo-image { margin-right: 10px; }
    .version-text { font-size: 0.9rem; color: gray; }
    .login-page-header-container { text-align: center; margin-top: 20px; margin-bottom: 10px;}
    .login-page-main-title { font-size: 1.8rem; font-weight: bold; display: block; color: #333F48; }
    .login-page-sub-title { font-size: 0.85rem; color: gray; display: block; margin-top: 2px; margin-bottom: 20px;}
    .login-form-title { font-size: 1.6rem; font-weight: bold; text-align: center; margin-top: 10px; margin-bottom: 25px; }
    @media (max-width: 768px) {
        .main-app-title { font-size: 1.8rem; }
        .main-app-subtitle { font-size: 0.8rem; }
        .login-page-main-title { font-size: 1.5rem; }
        .login-page-sub-title { font-size: 0.8rem; }
        .login-form-title { font-size: 1.3rem; margin-bottom: 20px; }
    }
</style>
""", unsafe_allow_html=True)

# --- Azure 클라이언트 초기화 ---
@st.cache_resource
def get_azure_openai_client_cached():
    print("Attempting to initialize Azure OpenAI client...")
    try:
        api_version_to_use = st.secrets.get("AZURE_OPENAI_VERSION", "2024-02-15-preview") 
        print(f"DEBUG: Initializing AzureOpenAI client with API version: {api_version_to_use}")
        client = AzureOpenAI(
            api_key=st.secrets["AZURE_OPENAI_KEY"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            api_version=api_version_to_use,
            timeout=AZURE_OPENAI_TIMEOUT
        )
        print("Azure OpenAI client initialized successfully.")
        return client
    except KeyError as e:
        st.error(f"Azure OpenAI 설정 오류: secrets에 '{e.args[0]}' 키가 없습니다.") 
        print(f"ERROR: Missing Azure OpenAI secret: {e.args[0]}")
        return None
    except Exception as e:
        st.error(f"Azure OpenAI 클라이언트 초기화 중 오류: {e}.") 
        print(f"ERROR: Azure OpenAI client initialization failed: {e}\n{traceback.format_exc()}")
        return None

@st.cache_resource
def get_azure_blob_clients_cached():
    print("Attempting to initialize Azure Blob Service client...")
    try:
        conn_str = st.secrets["AZURE_BLOB_CONN"]
        blob_service_client = BlobServiceClient.from_connection_string(conn_str, connection_timeout=60, read_timeout=120)
        container_name = st.secrets["BLOB_CONTAINER"]
        container_client = blob_service_client.get_container_client(container_name)
        print(f"Azure Blob Service client and container client for '{container_name}' initialized successfully.")
        return blob_service_client, container_client
    except KeyError as e:
        st.error(f"Azure Blob Storage 설정 오류: secrets에 '{e.args[0]}' 키가 없습니다.") 
        print(f"ERROR: Missing Azure Blob Storage secret: {e.args[0]}")
        return None, None
    except Exception as e:
        st.error(f"Azure Blob 클라이언트 초기화 중 오류: {e}.") 
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
        st.error("secrets에 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT' 설정이 없습니다.") 
        print("ERROR: Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT secret.")
        openai_client = None # 임베딩 모델 없이는 주요 기능 불가
    except Exception as e:
        st.error(f"임베딩 모델 설정 로드 중 오류: {e}")
        print(f"ERROR: Loading AZURE_OPENAI_EMBEDDING_DEPLOYMENT secret: {e}")
        openai_client = None


# --- 데이터 로드/저장 유틸리티 함수 (Blob 연동) ---
# (load_data_from_blob, save_data_to_blob, save_binary_data_to_blob 이전과 동일)
def load_data_from_blob(blob_name, _container_client, data_description="데이터", default_value=None):
    if not _container_client:
        print(f"ERROR: Blob Container client is None for load_data_from_blob ('{data_description}'). Returning default.")
        return default_value if default_value is not None else {}
    print(f"Attempting to load '{data_description}' from Blob: '{blob_name}'")
    try:
        blob_client_instance = _container_client.get_blob_client(blob_name)
        if blob_client_instance.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                local_temp_path = os.path.join(tmpdir, os.path.basename(blob_name))
                print(f"Downloading '{blob_name}' to '{local_temp_path}'...")
                with open(local_temp_path, "wb") as download_file:
                    download_stream = blob_client_instance.download_blob(timeout=60)
                    download_file.write(download_stream.readall())
                if os.path.getsize(local_temp_path) > 0:
                    with open(local_temp_path, "r", encoding="utf-8") as f:
                        loaded_data = json.load(f)
                    print(f"'{data_description}' loaded successfully from Blob: '{blob_name}'")
                    return loaded_data
                else:
                    print(f"WARNING: '{data_description}' file '{blob_name}' exists in Blob but is empty. Returning default.")
                    return default_value if default_value is not None else {}
        else:
            print(f"WARNING: '{data_description}' file '{blob_name}' not found in Blob Storage. Returning default.")
            return default_value if default_value is not None else {}
    except json.JSONDecodeError:
        print(f"ERROR: Failed to decode JSON for '{data_description}' from Blob '{blob_name}'. Returning default.")
        st.warning(f"'{data_description}' 파일({blob_name})이 손상되었거나 올바른 JSON 형식이 아닙니다. 기본값으로 시작합니다.")
        return default_value if default_value is not None else {}
    except AzureError as ae:
        print(f"AZURE ERROR loading '{data_description}' from Blob '{blob_name}': {ae}\n{traceback.format_exc()}")
        st.warning(f"'{data_description}' 로드 중 Azure 서비스 오류 발생: {ae}. 기본값으로 시작합니다.")
        return default_value if default_value is not None else {}
    except Exception as e:
        print(f"GENERAL ERROR loading '{data_description}' from Blob '{blob_name}': {e}\n{traceback.format_exc()}")
        st.warning(f"'{data_description}' 로드 중 알 수 없는 오류 발생: {e}. 기본값으로 시작합니다.")
        return default_value if default_value is not None else {}

def save_data_to_blob(data_to_save, blob_name, _container_client, data_description="데이터"):
    if not _container_client:
        st.error(f"Azure Blob 클라이언트가 준비되지 않아 '{data_description}'를 저장할 수 없습니다.")
        print(f"ERROR: Blob Container client is None, cannot save '{blob_name}'.")
        return False
    print(f"Attempting to save '{data_description}' to Blob Storage: '{blob_name}'")
    try:
        if not isinstance(data_to_save, (dict, list)):
            st.error(f"'{data_description}' 저장 실패: 데이터가 JSON으로 직렬화 가능한 타입(dict 또는 list)이 아닙니다.")
            print(f"ERROR: Data for '{blob_name}' is not JSON serializable (type: {type(data_to_save)}).")
            return False
        with tempfile.TemporaryDirectory() as tmpdir:
            local_temp_path = os.path.join(tmpdir, os.path.basename(blob_name))
            with open(local_temp_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            blob_client_instance = _container_client.get_blob_client(blob_name)
            print(f"Uploading '{local_temp_path}' to Blob '{blob_name}'...")
            with open(local_temp_path, "rb") as data_stream:
                blob_client_instance.upload_blob(data_stream, overwrite=True, timeout=60)
            print(f"Successfully saved '{data_description}' to Blob: '{blob_name}'")
        return True
    except AzureError as ae:
        st.error(f"Azure Blob에 '{data_description}' 저장 중 Azure 서비스 오류: {ae}")
        print(f"AZURE ERROR saving '{data_description}' to Blob '{blob_name}': {ae}\n{traceback.format_exc()}")
        return False
    except Exception as e:
        st.error(f"Azure Blob에 '{data_description}' 저장 중 알 수 없는 오류: {e}")
        print(f"GENERAL ERROR saving '{data_description}' to Blob '{blob_name}': {e}\n{traceback.format_exc()}")
        return False

def save_binary_data_to_blob(local_file_path, blob_name, _container_client, data_description="바이너리 데이터"):
    if not _container_client:
        st.error(f"Azure Blob 클라이언트가 준비되지 않아 '{data_description}' 바이너리 데이터를 저장할 수 없습니다.")
        print(f"ERROR: Blob Container client is None, cannot save binary '{blob_name}'.")
        return False
    if not os.path.exists(local_file_path):
        st.error(f"'{data_description}' 저장을 위한 로컬 파일 '{local_file_path}'를 찾을 수 없습니다.")
        print(f"ERROR: Local file for binary data not found: '{local_file_path}'")
        return False
    print(f"Attempting to save binary '{data_description}' from '{local_file_path}' to Blob: '{blob_name}'")
    try:
        blob_client_instance = _container_client.get_blob_client(blob_name)
        with open(local_file_path, "rb") as data_stream:
            blob_client_instance.upload_blob(data_stream, overwrite=True, timeout=120)
        print(f"Successfully saved binary '{data_description}' to Blob: '{blob_name}'")
        return True
    except AzureError as ae:
        st.error(f"Azure Blob에 바이너리 '{data_description}' 저장 중 Azure 서비스 오류: {ae}")
        print(f"AZURE ERROR saving binary '{data_description}' to Blob '{blob_name}': {ae}\n{traceback.format_exc()}")
        return False
    except Exception as e:
        st.error(f"Azure Blob에 바이너리 '{data_description}' 저장 중 알 수 없는 오류: {e}")
        print(f"GENERAL ERROR saving binary '{data_description}' to Blob '{blob_name}': {e}\n{traceback.format_exc()}")
        return False

# --- 사용자 정보 로드 ---
# (이전과 동일)
USERS = {}
if container_client:
    USERS = load_data_from_blob(USERS_BLOB_NAME, container_client, "사용자 정보", default_value={})
    if not isinstance(USERS, dict) :
        print(f"ERROR: USERS loaded from blob is not a dict ({type(USERS)}). Re-initializing.")
        USERS = {}
    if "admin" not in USERS:
        print(f"'{USERS_BLOB_NAME}' from Blob is empty or admin is missing. Creating default admin.")
        USERS["admin"] = {
            "name": "관리자", "department": "품질보증팀",
            "password_hash": generate_password_hash(st.secrets.get("ADMIN_PASSWORD", "diteam_fallback_secret")), # secrets에서 읽도록 수정
            "approved": True, "role": "admin"
        }
        if not save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "초기 사용자 정보"):
             st.warning("기본 관리자 정보를 Blob에 저장하는데 실패했습니다. 다음 실행 시 다시 시도됩니다.")
else:
    st.error("Azure Blob Storage 연결 실패. 사용자 정보를 초기화할 수 없습니다. 앱이 정상 동작하지 않을 수 있습니다.")
    print("CRITICAL: Cannot initialize USERS due to Blob client failure.")
    USERS = {"admin": {"name": "관리자(연결실패)", "department": "시스템", "password_hash": generate_password_hash("fallback"), "approved": True, "role": "admin"}}


# --- 쿠키 매니저 및 세션 상태 초기화 (안정성 강화 버전) ---
# (이전과 동일)
cookies = None
cookie_manager_ready = False
print(f"Attempting to load COOKIE_SECRET from st.secrets: {st.secrets.get('COOKIE_SECRET')}")
try:
    cookie_secret_key = st.secrets.get("COOKIE_SECRET")
    if not cookie_secret_key:
        st.error("secrets에 'COOKIE_SECRET'이(가) 설정되지 않았거나 비어있습니다.") 
        print("ERROR: COOKIE_SECRET is not set or empty in st.secrets.")
    else:
        cookies = EncryptedCookieManager(
            prefix="gmp_chatbot_auth_v5_1/", # 버전업데이트 (선택사항)
            password=cookie_secret_key
        )
        try:
            if cookies.ready():
                cookie_manager_ready = True
                print("CookieManager is ready on initial setup try.")
            else:
                print("CookieManager not ready on initial setup try (may resolve on first interaction).")
        except Exception as e_ready_init: 
            print(f"WARNING: cookies.ready() call during initial setup failed: {e_ready_init}")
            cookie_manager_ready = False 
except Exception as e:
    st.error(f"쿠키 매니저 객체 생성 중 알 수 없는 오류 발생: {e}")
    print(f"CRITICAL: CookieManager object creation error: {e}\n{traceback.format_exc()}")

SESSION_TIMEOUT = 1800
try:
    session_timeout_secret = st.secrets.get("SESSION_TIMEOUT")
    if session_timeout_secret: SESSION_TIMEOUT = int(session_timeout_secret)
    print(f"Session timeout set to: {SESSION_TIMEOUT} seconds.")
except (ValueError, TypeError):
    print(f"WARNING: SESSION_TIMEOUT in secrets ('{session_timeout_secret}') is not a valid integer. Using default {SESSION_TIMEOUT}s.")
except Exception as e:
     print(f"WARNING: Error reading SESSION_TIMEOUT from secrets: {e}. Using default {SESSION_TIMEOUT}s.")

if "authenticated" not in st.session_state:
    print("Initializing st.session_state: 'authenticated', 'user', and 'messages'")
    st.session_state["authenticated"] = False
    st.session_state["user"] = {}
    st.session_state["messages"] = []
    if cookies:
        is_ready_for_cookie_load = False
        try:
            if cookies.ready():
                is_ready_for_cookie_load = True
                print("CookieManager became ready before attempting to load cookies from browser.")
        except Exception as e_ready_check:
            print(f"WARNING: cookies.ready() check before loading cookies failed: {e_ready_check}")

        if is_ready_for_cookie_load:
            try:
                auth_cookie_val = cookies.get("authenticated")
                print(f"Cookie 'authenticated' value on session init: {auth_cookie_val}")
                if auth_cookie_val == "true":
                    login_time_str = cookies.get("login_time", "0")
                    try:
                        login_time = float(login_time_str if login_time_str and login_time_str.replace('.', '', 1).isdigit() else "0")
                    except ValueError:
                        print(f"WARNING: Invalid login_time_str from cookie: {login_time_str}. Defaulting to 0.")
                        login_time = 0.0
                    if (time.time() - login_time) < SESSION_TIMEOUT:
                        user_json_cookie = cookies.get("user", "{}")
                        try:
                            user_data_from_cookie = json.loads(user_json_cookie if user_json_cookie else "{}")
                            if user_data_from_cookie and isinstance(user_data_from_cookie, dict):
                                st.session_state["user"] = user_data_from_cookie
                                st.session_state["authenticated"] = True
                                print(f"User '{user_data_from_cookie.get('name')}' authenticated from cookie.")
                            else:
                                print("User data in cookie is empty or invalid. Clearing auth state for cookie.")
                                st.session_state["authenticated"] = False
                                if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save(key="cookie_save_on_invalid_user_data")
                        except json.JSONDecodeError:
                            print("ERROR: Failed to decode user JSON from cookie. Clearing auth state for cookie.")
                            st.session_state["authenticated"] = False
                            if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save(key="cookie_save_on_json_decode_error")
                    else:
                        print("Session timeout detected from cookie. Clearing auth state for cookie.")
                        st.session_state["authenticated"] = False
                        st.session_state["messages"] = []
                        if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save(key="cookie_save_on_session_timeout")
                else:
                    print("Authenticated cookie not set to 'true'.")
                    st.session_state["authenticated"] = False
            except Exception as e_cookie_load:
                print(f"ERROR during cookie processing (get/save): {e_cookie_load}\n{traceback.format_exc()}")
                st.session_state["authenticated"] = False
        else:
            print("CookieManager not ready when attempting to load cookies from browser (after initial check).")
            st.session_state["authenticated"] = False
    else:
         print("CookieManager object is None, cannot restore session.")
         st.session_state["authenticated"] = False

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    print("Double check: Initializing messages as it was not in session_state before login UI.")

if cookies and not cookie_manager_ready: 
    try:
        if cookies.ready():
            cookie_manager_ready = True
            print("CookieManager became ready just before login UI check.")
        else:
            print("CookieManager still not ready just before login UI check.")
    except Exception as e_ready_login_ui:
        print(f"WARNING: cookies.ready() call just before login UI check failed: {e_ready_login_ui}")

# --- 로그인 UI 및 로직 ---
# (이전과 동일)
if not st.session_state.get("authenticated", False):
    st.markdown("""
    <div class="login-page-header-container">
      <span class="login-page-main-title">유앤생명과학 GMP/SOP 업무 가이드 봇</span>
      <span class="login-page-sub-title">Made by DI.PART</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="login-form-title">🔐 로그인 또는 회원가입</p>', unsafe_allow_html=True)
    if not cookie_manager_ready and st.secrets.get("COOKIE_SECRET"):
        st.warning("쿠키 시스템을 초기화하고 있습니다. 잠시 후 새로고침하거나 다시 시도해주세요.")

    with st.form("auth_form_final_v5_img_txt", clear_on_submit=False): # 키 변경
        mode = st.radio("선택", ["로그인", "회원가입"], key="auth_mode_final_v5_img_txt")
        uid = st.text_input("ID", key="auth_uid_final_v5_img_txt")
        pwd = st.text_input("비밀번호", type="password", key="auth_pwd_final_v5_img_txt")
        name, dept = "", ""
        if mode == "회원가입":
            name = st.text_input("이름", key="auth_name_final_v5_img_txt")
            dept = st.text_input("부서", key="auth_dept_final_v5_img_txt")
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
                    st.session_state["messages"] = [] # 로그인 시 메시지 초기화
                    print(f"Login successful for user '{uid}'. Chat messages cleared.")
                    if cookies and cookies.ready(): 
                        try:
                            cookies["authenticated"] = "true"; cookies["user"] = json.dumps(user_data_login)
                            cookies["login_time"] = str(time.time()); cookies.save(key="cookie_save_on_login")
                            print(f"Cookies saved for user '{uid}'.")
                        except Exception as e_cookie_save:
                            st.warning(f"로그인 쿠키 저장 중 문제 발생: {e_cookie_save}")
                            print(f"ERROR: Failed to save login cookies: {e_cookie_save}")
                    else:
                        st.warning("쿠키 시스템이 준비되지 않아 로그인 상태를 브라우저에 저장할 수 없습니다.")
                        print("WARNING: CookieManager not ready during login, cannot save cookies.")
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
                        USERS.pop(uid, None) # 저장 실패 시 롤백
                    else:
                        st.success("가입 신청 완료! 관리자 승인 후 로그인 가능합니다.")
    st.stop()

# --- 인증 후 메인 애플리케이션 로직 ---
current_user_info = st.session_state.get("user", {})

# --- 헤더 (로고, 버전, 로그아웃 버튼) ---
# (이전과 동일)
top_cols_main = st.columns([0.7, 0.3])
with top_cols_main[0]:
    if os.path.exists(COMPANY_LOGO_PATH_REPO):
        logo_b64 = get_base64_of_bin_file(COMPANY_LOGO_PATH_REPO)
        if logo_b64:
            st.markdown(f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_b64}" class="logo-image" width="150">
                <span class="version-text">ver 0.9.7 (Image/TXT Support)</span>
            </div>""", unsafe_allow_html=True)
        else: # 로고 파일은 있으나 base64 변환 실패 시
            st.markdown(f"""<div class="logo-container"><span class="version-text" style="font-weight:bold;">유앤생명과학</span> <span class="version-text" style="margin-left:10px;">ver 0.9.7 (Image/TXT Support)</span></div>""", unsafe_allow_html=True)
    else: # 로고 파일 자체가 없을 시
        print(f"WARNING: Company logo file not found at {COMPANY_LOGO_PATH_REPO}")
        st.markdown(f"""<div class="logo-container"><span class="version-text" style="font-weight:bold;">유앤생명과학</span> <span class="version-text" style="margin-left:10px;">ver 0.9.7 (Image/TXT Support)</span></div>""", unsafe_allow_html=True)


with top_cols_main[1]:
    st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
    if st.button("로그아웃", key="logout_button_final_v5_img_txt"): # 키 변경
        st.session_state["authenticated"] = False
        st.session_state["user"] = {}
        st.session_state["messages"] = [] # 로그아웃 시 메시지 초기화
        print("Logout successful. Chat messages cleared.")
        if cookies and cookies.ready(): 
             try:
                 # 쿠키 삭제 시에는 del 보다 빈 값으로 덮어쓰는 것이 일부 환경에서 더 안정적일 수 있음
                 cookies["authenticated"] = "false"
                 cookies["user"] = ""
                 cookies["login_time"] = ""
                 cookies.save(key="cookie_save_on_logout")
                 print("Cookies cleared on logout.")
             except Exception as e_logout_cookie:
                 print(f"ERROR: Failed to clear cookies on logout: {e_logout_cookie}")
        else:
              print("WARNING: CookieManager not ready during logout, cannot clear cookies.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# --- 메인 앱 제목 (로그인 후) ---
# (이전과 동일)
st.markdown("""
<div class="main-app-title-container">
  <span class="main-app-title">유앤생명과학 GMP/SOP 업무 가이드 봇</span>
  <span class="main-app-subtitle">Made by DI.PART</span>
</div>
""", unsafe_allow_html=True)

# --- 벡터 DB 로드 (Azure Blob Storage 기반) ---
# (이전과 동일, current_embedding_dimension 사용 및 차원 검증 로직 포함)
@st.cache_resource
def load_vector_db_from_blob_cached(_container_client):
    if not _container_client:
        print("ERROR: Blob Container client is None for load_vector_db_from_blob_cached.")
        return faiss.IndexFlatL2(1536), [] # 기본 임베딩 차원 (text-embedding-3-small)
    current_embedding_dimension = 1536 
    idx, meta = faiss.IndexFlatL2(current_embedding_dimension), []
    print(f"Attempting to load vector DB from Blob: '{INDEX_BLOB_NAME}', '{METADATA_BLOB_NAME}' with dimension {current_embedding_dimension}")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_index_path = os.path.join(tmpdir, os.path.basename(INDEX_BLOB_NAME))
            local_metadata_path = os.path.join(tmpdir, os.path.basename(METADATA_BLOB_NAME))
            
            index_blob_client = _container_client.get_blob_client(INDEX_BLOB_NAME)
            if index_blob_client.exists():
                print(f"Downloading '{INDEX_BLOB_NAME}'...")
                with open(local_index_path, "wb") as download_file:
                    download_stream = index_blob_client.download_blob(timeout=60)
                    download_file.write(download_stream.readall())
                if os.path.getsize(local_index_path) > 0:
                    try:
                        idx = faiss.read_index(local_index_path)
                        if idx.d != current_embedding_dimension:
                            print(f"WARNING: Loaded FAISS index dimension ({idx.d}) does not match expected dimension ({current_embedding_dimension}). Re-initializing.")
                            idx = faiss.IndexFlatL2(current_embedding_dimension)
                            meta = [] # 메타데이터도 함께 초기화
                        else:
                            print(f"'{INDEX_BLOB_NAME}' loaded successfully from Blob Storage. Dimension: {idx.d}")
                    except Exception as e_faiss_read:
                        print(f"ERROR reading FAISS index: {e_faiss_read}. Re-initializing index.")
                        idx = faiss.IndexFlatL2(current_embedding_dimension)
                        meta = []
                else:
                    print(f"WARNING: '{INDEX_BLOB_NAME}' is empty in Blob. Using new index.")
                    idx = faiss.IndexFlatL2(current_embedding_dimension) # 메타데이터도 비워야 함
                    meta = []
            else:
                print(f"WARNING: '{INDEX_BLOB_NAME}' not found in Blob Storage. New index will be used/created.")
                idx = faiss.IndexFlatL2(current_embedding_dimension)
                meta = [] # 새 인덱스이므로 메타데이터도 비움

            # 메타데이터는 인덱스가 성공적으로 로드되었거나, 새로 생성되었을 때만 로드/초기화
            if idx is not None: # idx가 None일 가능성은 위에서 처리했지만, 방어적으로 체크
                metadata_blob_client = _container_client.get_blob_client(METADATA_BLOB_NAME)
                # 인덱스에 아이템이 있거나, 인덱스 파일이 존재해서 로드 시도했을 때만 메타데이터 로드
                # (완전히 새로 생성된 빈 인덱스면 메타데이터도 비어있어야 함)
                if metadata_blob_client.exists() and (idx.ntotal > 0 or os.path.exists(local_index_path)): 
                    print(f"Downloading '{METADATA_BLOB_NAME}'...")
                    with open(local_metadata_path, "wb") as download_file: # wb로 다운로드 후 r로 읽기
                        download_stream_meta = metadata_blob_client.download_blob(timeout=60)
                        download_file.write(download_stream_meta.readall())
                    if os.path.getsize(local_metadata_path) > 0 :
                        with open(local_metadata_path, "r", encoding="utf-8") as f_meta: meta = json.load(f_meta)
                    else: # 메타데이터 파일은 존재하나 비어있는 경우
                        meta = []
                        print(f"WARNING: '{METADATA_BLOB_NAME}' is empty in Blob.")
                elif idx.ntotal == 0 and not index_blob_client.exists(): # 인덱스 파일도 없고, ntotal도 0이면 (완전 처음)
                     print(f"WARNING: Index is new and empty, starting with empty metadata.")
                     meta = []
                else: # 인덱스 파일은 없지만 (새로 생성), 메타데이터 파일도 없는 경우
                    print(f"WARNING: '{METADATA_BLOB_NAME}' not found in Blob Storage. Starting with empty metadata.")
                    meta = []
            
            # 최종 동기화: 인덱스에 아이템이 없으면 메타데이터도 비어있어야 함
            if idx is not None and idx.ntotal == 0 and len(meta) > 0:
                print(f"WARNING: FAISS index is empty (ntotal=0) but metadata is not. Clearing metadata for consistency.")
                meta = []
            elif idx is not None and idx.ntotal > 0 and not meta:
                print(f"CRITICAL WARNING: FAISS index has data (ntotal={idx.ntotal}) but metadata is empty. This may lead to errors. Consider re-indexing.")
                # st.error("벡터 DB 인덱스와 메타데이터 불일치 발생! 관리자에게 문의하거나 파일 재학습 필요.")
                # idx = faiss.IndexFlatL2(current_embedding_dimension); meta = [] # 안전을 위해 초기화

    except AzureError as ae:
        st.error(f"Azure Blob에서 벡터DB 로드 중 Azure 서비스 오류: {ae}")
        print(f"AZURE ERROR loading vector DB from Blob: {ae}\n{traceback.format_exc()}")
        idx = faiss.IndexFlatL2(current_embedding_dimension); meta = []
    except Exception as e:
        st.error(f"Azure Blob에서 벡터DB 로드 중 알 수 없는 오류: {e}")
        print(f"GENERAL ERROR loading vector DB from Blob: {e}\n{traceback.format_exc()}")
        idx = faiss.IndexFlatL2(current_embedding_dimension); meta = []
    return idx, meta


index, metadata = faiss.IndexFlatL2(1536), [] # 기본값 설정
if container_client:
    index, metadata = load_vector_db_from_blob_cached(container_client)
    print(f"DEBUG: FAISS index loaded after cache. ntotal: {index.ntotal if index else 'Index is None'}, dimension: {index.d if index else 'N/A'}")
    print(f"DEBUG: Metadata loaded after cache. Length: {len(metadata) if metadata is not None else 'Metadata is None'}")
else:
    st.error("Azure Blob Storage 연결 실패로 벡터 DB를 로드할 수 없습니다. 파일 학습 및 검색 기능이 제한될 수 있습니다.")
    print("CRITICAL: Cannot load vector DB due to Blob client initialization failure (main section).")

# --- 규칙 파일 로드 ---
@st.cache_data
def load_prompt_rules_cached():
    default_rules = """1.우선 기준
    1.1. 모든 답변은 컨텍스트로 제공된 참고 문서(첨부 파일, 학습된 SOP/이미지 설명 등)의 내용을 최우선으로 하며, 그 다음은 MFDS 규정, 그리고 사내 SOP 순서를 기준으로 삼습니다.
    1.2. 규정/법령 위반 또는 회색지대의 경우, 관련 문서명, 조항번호, 조항내용과 함께 명확히 경고해야 합니다.
    1.3. 질문에 대한 답변 근거를 제공된 참고 문서나 규정에서 찾을 수 없을 경우, "명확한 규정이나 참고 자료를 찾을 수 없습니다. 내부 QA 검토 필요"임을 고지합니다.
    1.4. 답변을 생성할 때, 컨텍스트로 제공된 문서 내용을 참고했다면 해당 문서의 내용에서 파악되는 주요 식별 정보(예: 문서 제목, 문서 번호, 섹션 제목, 이미지 파일명 등)와 함께 [출처: 파일명.pdf 또는 이미지 설명: 이미지명.png] 형식으로 출처 파일명을 언급해야 합니다. 만약 문서 내용에서 명확한 제목이나 번호를 찾기 어렵다면 [출처: 파일명.ext] 또는 [Image Description for: 이미지명.ext]만 언급합니다.

2.응답 방식
    2.1. 존댓말을 사용하며, 전문적이고 친절한 어조로 답변합니다.
    2.2. 모든 답변은 논리적 구조, 높은 정확성, 실용성을 갖추어야 하며, 필요한 경우 예시 및 설명을 포함하여 전문가 수준을 유지합니다. 답변에는 가능한 한 참고한 근거(문서 내용에서 파악된 SOP 제목/번호, 규정명, 조항, 이미지 내용 등)와 출처(규칙 1.4 참고)를 함께 제시합니다.
    2.3. 번역 시, 일반적인 번역체 대신 **한국 제약 산업 및 GMP 규정/가이드라인(MFDS, PIC/S, ICH 등)에서 통용되는 표준 전문 용어**를 사용하여 번역해야 합니다. (아래 '주요 번역 용어 가이드' 참고)

3.기능 제한 및 파일 처리
    3.1. 다루는 주제: 사내 SOP (Standard Operating Procedure) 내용 질의응답, GMP 가이드라인(FDA, PIC/S, EU-GMP, cGMP, MFDS 등), DI 규정, 외국 규정 번역 등 업무 관련 내용 및 사용자가 첨부한 파일(텍스트 문서 또는 이미지)의 내용 분석 (요약, 설명, 비교 등).
    3.2. 파일 첨부 시(텍스트 또는 이미지) 및 내부 SOP 참고 시 처리:
        - 사용자가 파일을 첨부하여 질문하거나, 질문이 내부적으로 학습된 SOP 문서(또는 이미지 설명)와 관련된 경우, 해당 파일 또는 SOP의 내용을 최우선으로 분석하고 참고하여 답변해야 합니다.
        - 이미지가 첨부된 경우, 이미지의 내용을 이해하고 설명한 내용을 바탕으로 답변을 생성합니다.
        - 사용자가 '전체 번역'을 명시적으로 요청하는 경우 (텍스트 파일에 한함), 다른 모든 규칙(특히 간결성 규칙 및 출처 명시 규칙)에 우선하여 첨부된 문서의 내용을 처음부터 끝까지 순서대로 번역해야 합니다. 번역 결과는 모델의 최대 출력 토큰 내에서 생성되며, 내용이 길 경우 번역이 중간에 완료될 수 있습니다. 이 경우, 번역된 내용의 출처 파일명 [출처: 파일명.pdf] 정도만 언급하거나, 문서 제목이 명확하다면 제목까지 언급할 수 있습니다.
        - 번역 요청이 아니더라도, 파일(텍스트 또는 이미지 설명) 또는 학습된 SOP 내용을 기반으로 질문에 맞춰 요약, 설명, 비교 등의 답변을 생성해야 합니다. 이때, 규칙 1.4 및 2.2에 따라 참고한 출처를 명시합니다.
        - 만약 파일 또는 학습된 SOP 내용이 다른 규정(예: MFDS 규정)과 상충될 가능성이 있다면, 그 점을 명확히 언급하고 사용자에게 확인을 요청해야 합니다.
    3.3. 사용자가 파일을 첨부하고 해당 파일의 내용에 대해 질문하거나, 학습된 SOP 내용에 대해 질문하는 경우는 업무 관련 질문으로 간주합니다. 이 경우를 제외하고, 개인적인 질문, 뉴스, 여가 등 업무와 직접 관련 없는 질문은 “업무 관련 질문만 처리합니다.”로 간결히 응답합니다.

4.챗봇 소개 안내
    4.1. 사용자가 인사하거나 기능을 물을 경우, 본 챗봇의 역할("한국 제약 산업의 DI/GMP 규정 및 사내 SOP 전문가 챗봇")과 처리 가능한 업무 범위(텍스트 및 이미지 파일 내용 분석 포함)를 간단히 소개합니다.

5.표현 및 형식 규칙
    5.1. Markdown 스타일 강조는 사용하지 않습니다.
    5.2. 번호 항목은 동일한 서식(글꼴 크기와 굵기)으로 통일합니다.
    5.3. 답변은 표, 요약, 핵심 정리 중심으로 자세하게 구성합니다. (단, '전체 번역' 요청 시에는 규칙 3.2가 우선하며, 이때는 규칙 1.4의 출처 명시 방식 중 파일명 위주로 간략히 하거나 내용 흐름에 따라 조절)

6. 주요 번역 용어 가이드 (번역 시 최우선 참고)
    - Compliant / Compliance: 규정 준수
    - GxP: Good x Practice (GMP, GLP, GCP 등 우수 관리 기준)
    - Computerized System: 컴퓨터화 시스템
    # ... (이하 용어 목록)
    - Data Integrity (DI): 데이터 완전성
    # (필요에 따라 이 목록에 중요한 제약 용어를 계속 추가해주세요)
"""
    if os.path.exists(RULES_PATH_REPO):
        try:
            with open(RULES_PATH_REPO, "r", encoding="utf-8") as f: rules_content = f.read()
            print(f"Prompt rules loaded successfully from '{RULES_PATH_REPO}'.")
            return rules_content
        except Exception as e:
            st.warning(f"'{RULES_PATH_REPO}' 파일 로드 중 오류: {e}. 위 명시된 기본 규칙을 사용합니다.")
            print(f"WARNING: Error loading prompt rules from '{RULES_PATH_REPO}': {e}. Using default rules defined in code.")
            return default_rules
    else:
        print(f"WARNING: Prompt rules file not found at '{RULES_PATH_REPO}'. Using default rules defined in code.")
        return default_rules
PROMPT_RULES_CONTENT = load_prompt_rules_cached()

# --- 텍스트 처리 함수들 ---
def extract_text_from_file(uploaded_file_obj): # 이미지 처리는 이 함수 밖에서 분기
    ext = os.path.splitext(uploaded_file_obj.name)[1].lower()
    text_content = ""
    
    # 이미지 확장자는 여기서 처리하지 않음 (또는 명시적으로 빈 문자열 반환)
    if ext in [".png", ".jpg", ".jpeg"]:
        # st.info(f"'{uploaded_file_obj.name}'은 이미지 파일입니다. 이 함수는 텍스트 추출 전용입니다.")
        print(f"DEBUG extract_text_from_file: Skipped image file '{uploaded_file_obj.name}'.")
        return "" # 또는 None

    try:
        uploaded_file_obj.seek(0)
        file_bytes = uploaded_file_obj.read()
        
        if ext == ".pdf":
            with fitz.open(stream=file_bytes, filetype="pdf") as doc: text_content = "\n".join(page.get_text() for page in doc)
        elif ext == ".docx":
            with io.BytesIO(file_bytes) as doc_io: doc = docx.Document(doc_io); text_content = "\n".join(para.text for para in doc.paragraphs)
        elif ext in (".xlsx", ".xlsm"):
            with io.BytesIO(file_bytes) as excel_io: df = pd.read_excel(excel_io, sheet_name=None)
            text_content = ""
            for sheet_name, sheet_df in df.items():
                 text_content += f"--- 시트: {sheet_name} ---\n{sheet_df.to_string(index=False)}\n\n"
        elif ext == ".csv":
            with io.BytesIO(file_bytes) as csv_io:
                try: df = pd.read_csv(csv_io)
                except UnicodeDecodeError: df = pd.read_csv(csv_io, encoding='cp949')
                text_content = df.to_string(index=False)
        elif ext == ".pptx":
            with io.BytesIO(file_bytes) as ppt_io: prs = Presentation(ppt_io); text_content = "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
        elif ext == ".txt":
            try:
                text_content = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_content = file_bytes.decode('cp949') # 다른 인코딩 시도
                except Exception as e_txt_decode:
                    st.warning(f"'{uploaded_file_obj.name}' TXT 파일 디코딩 실패: {e_txt_decode}. 내용을 비워둡니다.")
                    text_content = "" 
        else: 
            st.warning(f"지원하지 않는 텍스트 파일 형식입니다: {ext} (파일명: {uploaded_file_obj.name})")
            return "" # 지원하지 않는 텍스트 파일도 빈 내용 반환
    except Exception as e:
        st.error(f"'{uploaded_file_obj.name}' 파일 내용 추출 중 오류: {e}")
        print(f"Error extracting text from '{uploaded_file_obj.name}': {e}\n{traceback.format_exc()}")
        return ""
    return text_content.strip()


def chunk_text_into_pieces(text_to_chunk, chunk_size=500): # 청크 크기는 필요시 조절
    if not text_to_chunk or not text_to_chunk.strip(): return [];
    chunks_list, current_buffer = [], ""
    # 더 긴 줄바꿈이나 문단 구분을 기준으로 청킹하도록 개선 (선택적)
    # 예: sentences = text_to_chunk.split('. ') # 문장 단위로 나누고 합치기
    for line in text_to_chunk.split("\n"): # 현재는 줄바꿈 기준
        stripped_line = line.strip()
        if not stripped_line and not current_buffer.strip(): continue # 연속된 빈 줄 무시
        
        # 현재 버퍼에 추가했을 때 청크 크기를 넘는지 확인
        if len(current_buffer) + len(stripped_line) + 1 < chunk_size: # +1 for newline
            current_buffer += stripped_line + "\n"
        else: # 청크 크기를 넘으면
            if current_buffer.strip(): # 현재 버퍼에 내용이 있으면 청크로 추가
                chunks_list.append(current_buffer.strip())
            current_buffer = stripped_line + "\n" # 새 버퍼 시작
            
    if current_buffer.strip(): # 마지막 버퍼 내용 추가
        chunks_list.append(current_buffer.strip())
        
    return [c for c in chunks_list if c] # 최종적으로 빈 청크 제거

# --- 이미지 설명 생성 함수 ---
def get_image_description(image_bytes, image_filename, client_instance):
    if not client_instance:
        st.error("OpenAI 클라이언트가 준비되지 않아 이미지 설명을 생성할 수 없습니다.")
        print("ERROR get_image_description: OpenAI client not ready.")
        return None
    
    print(f"DEBUG get_image_description: Requesting description for image '{image_filename}'")
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # 이미지 확장자에 따라 "image/jpeg" 또는 "image/png" 등을 사용
        image_ext_desc = os.path.splitext(image_filename)[1].lower()
        mime_type = "image/jpeg" # 기본값
        if image_ext_desc == ".png":
            mime_type = "image/png"
        elif image_ext_desc == ".jpg" or image_ext_desc == ".jpeg":
            mime_type = "image/jpeg"
        # 다른 이미지 타입이 있다면 추가 (예: gif, webp 등)

        vision_model_deployment = st.secrets["AZURE_OPENAI_DEPLOYMENT"] 
        print(f"DEBUG get_image_description: Using vision model deployment: {vision_model_deployment}")

        response = client_instance.chat.completions.create(
            model=vision_model_deployment,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"이 이미지를 업무적 관점에서 자세히 설명해주세요. 이 설명은 나중에 텍스트 검색을 통해 이미지를 찾거나, 이미지 속 상황을 파악하는 데 사용됩니다. 이미지 파일명은 '{image_filename}'입니다. 이미지의 주요 객체, 상태, 가능한 맥락, 그리고 만약 GMP/SOP와 관련된 요소가 있다면 그것도 언급해주세요."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}" 
                            }
                        }
                    ]
                }
            ],
            max_tokens=IMAGE_DESCRIPTION_MAX_TOKENS, 
            temperature=0.2, # 좀 더 사실 기반의 설명을 위해 낮은 값
            timeout=AZURE_OPENAI_TIMEOUT 
        )
        description = response.choices[0].message.content.strip()
        print(f"DEBUG get_image_description: Description for '{image_filename}' (len: {len(description)} chars) generated successfully.")
        return description
    except APIStatusError as ase:
        st.error(f"이미지 설명 생성 중 API 오류 (상태 코드 {ase.status_code}): {ase.message}.")
        print(f"API STATUS ERROR during image description for '{image_filename}' (Status {ase.status_code}): {ase.message}")
        if ase.response and ase.response.content:
            try:
                error_details = json.loads(ase.response.content.decode('utf-8'))
                print(f"DEBUG get_image_description: Azure API error details: {json.dumps(error_details, indent=2, ensure_ascii=False)}")
            except Exception as json_e:
                print(f"DEBUG get_image_description: Could not parse Azure API error content as JSON: {json_e}")
        return None
    except APITimeoutError:
        st.error(f"이미지 '{image_filename}' 설명 생성 중 시간 초과가 발생했습니다.")
        print(f"TIMEOUT ERROR during image description for '{image_filename}'.")
        return None
    except APIConnectionError as ace:
        st.error(f"이미지 '{image_filename}' 설명 생성 중 API 연결 오류가 발생했습니다: {ace}.")
        print(f"API CONNECTION ERROR during image description for '{image_filename}': {ace}")
        return None
    except RateLimitError as rle:
        st.error(f"이미지 '{image_filename}' 설명 생성 중 API 요청량 제한에 도달했습니다: {rle}.")
        print(f"RATE LIMIT ERROR during image description for '{image_filename}': {rle}")
        return None
    except Exception as e:
        st.error(f"이미지 '{image_filename}' 설명 생성 중 예기치 않은 오류: {e}")
        print(f"UNEXPECTED ERROR during image description for '{image_filename}': {e}\n{traceback.format_exc()}")
        return None


# --- 텍스트 임베딩 함수 (디버깅 로그 및 예외 처리 강화) ---
def get_text_embedding(text_to_embed):
    # (이전과 동일 - 변경 없음)
    if not openai_client or not EMBEDDING_MODEL:
        print("ERROR: OpenAI client or embedding model not ready for get_text_embedding (called).")
        return None
    if not text_to_embed or not text_to_embed.strip():
        print("WARNING: Attempted to embed empty or whitespace-only text in get_text_embedding.")
        return None

    print(f"DEBUG get_text_embedding: Requesting embedding for text (first 50 chars): '{text_to_embed[:50]}...'")
    print(f"DEBUG get_text_embedding: Using embedding model deployment name: {EMBEDDING_MODEL}")
    try:
        response = openai_client.embeddings.create(
            input=[text_to_embed],
            model=EMBEDDING_MODEL,
            timeout=AZURE_OPENAI_TIMEOUT / 2 
        )
        print("Embedding received.")
        return response.data[0].embedding
    except APIStatusError as ase:
        st.error(f"텍스트 임베딩 생성 중 API 오류 (상태 코드 {ase.status_code}): {ase.message}. 요청 내용을 확인해주세요.")
        print(f"API STATUS ERROR during embedding (get_text_embedding - Status {ase.status_code}): {ase.message}")
        print(f"DEBUG get_text_embedding: Failing text (first 100 chars): {text_to_embed[:100]}")
        if ase.response and ase.response.content:
            try:
                error_details = json.loads(ase.response.content.decode('utf-8'))
                print(f"DEBUG get_text_embedding: Azure API error details: {json.dumps(error_details, indent=2, ensure_ascii=False)}")
            except Exception as json_e:
                print(f"DEBUG get_text_embedding: Could not parse Azure API error content as JSON: {json_e}")
                print(f"DEBUG get_text_embedding: Raw Azure API error content: {ase.response.content}")
        return None
    except APITimeoutError:
        st.error("텍스트 임베딩 생성 중 시간 초과가 발생했습니다. 잠시 후 다시 시도해주세요.")
        print(f"TIMEOUT ERROR during embedding (get_text_embedding): Request for '{text_to_embed[:50]}...' timed out.")
        return None
    except APIConnectionError as ace:
        st.error(f"텍스트 임베딩 생성 중 API 연결 오류가 발생했습니다: {ace}. 네트워크를 확인하거나 잠시 후 다시 시도해주세요.")
        print(f"API CONNECTION ERROR during embedding (get_text_embedding): {ace}")
        return None
    except RateLimitError as rle:
        st.error(f"텍스트 임베딩 생성 중 API 요청량 제한에 도달했습니다: {rle}. 잠시 후 다시 시도해주세요.")
        print(f"RATE LIMIT ERROR during embedding (get_text_embedding): {rle}")
        return None
    except Exception as e:
        st.error(f"텍스트 임베딩 생성 중 예기치 않은 오류가 발생했습니다: {e}")
        print(f"UNEXPECTED ERROR during embedding (get_text_embedding): {e}\n{traceback.format_exc()}")
        return None

# --- 유사도 검색 함수 (딕셔너리 리스트 반환) ---
def search_similar_chunks(query_text, k_results=3): # k_results는 필요시 조절
    # (이전과 동일 - 변경 없음)
    print(f"DEBUG search_similar_chunks: Called with query '{query_text[:30]}...', k_results={k_results}")
    if index is None:
        print("DEBUG search_similar_chunks: FAISS index is None.")
        return []
    if index.ntotal == 0:
        print("DEBUG search_similar_chunks: FAISS index is empty (ntotal=0).")
        return []
    if not metadata: # metadata가 None이거나 비어있을 때
        print("DEBUG search_similar_chunks: Metadata is empty or None.")
        return []

    print(f"Searching for similar chunks for query: '{query_text[:30]}...'")
    query_vector = get_text_embedding(query_text)
    if query_vector is None:
        print("DEBUG search_similar_chunks: Failed to get query vector.")
        return []
    try:
        actual_k = min(k_results, index.ntotal)
        if actual_k == 0 :
            print("DEBUG search_similar_chunks: No items in index to search (actual_k=0).")
            return []

        distances, indices_found = index.search(np.array([query_vector]).astype("float32"), actual_k)
        print(f"DEBUG search_similar_chunks: FAISS search distances: {distances}")
        print(f"DEBUG search_similar_chunks: FAISS search indices_found: {indices_found}")

        results_with_source = []
        if len(indices_found[0]) > 0:
            for i_val in indices_found[0]:
                if 0 <= i_val < len(metadata): # metadata 인덱스 범위 확인
                    meta_item = metadata[i_val]
                    # 메타데이터가 예상된 딕셔너리 형태인지 확인 (선택적 강화)
                    if isinstance(meta_item, dict):
                        results_with_source.append({
                            "source": meta_item.get("file_name", "출처 불명"), # 원본 파일명
                            "content": meta_item.get("content", ""), # 청크 내용 (텍스트 또는 이미지 설명)
                            "is_image_description": meta_item.get("is_image_description", False), # 이미지 설명 여부 플래그
                            "original_file_extension": meta_item.get("original_file_extension", "") # 원본 확장자
                        })
                    else:
                        print(f"WARNING search_similar_chunks: Metadata item at index {i_val} is not a dict: {meta_item}")
        print(f"Similarity search found {len(results_with_source)} relevant chunks with source.")
        return results_with_source
    except Exception as e:
        st.error(f"유사도 검색 중 오류: {e}")
        print(f"ERROR: Similarity search failed: {e}\n{traceback.format_exc()}")
        return []


# --- 문서 추가, 원본 저장, 사용량 로깅 함수 ---
def add_document_to_vector_db_and_blob(uploaded_file_obj, processed_content, text_chunks, _container_client, is_image_description=False):
    global index, metadata
    if not text_chunks: 
        st.warning(f"'{uploaded_file_obj.name}' 파일에서 처리할 내용(텍스트 또는 이미지 설명)이 없습니다.")
        return False
    if not _container_client: 
        st.error("Azure Blob 클라이언트가 준비되지 않아 학습 결과를 저장할 수 없습니다.")
        return False

    vectors_to_add, new_metadata_entries_for_current_file = [], []
    embedding_failed_for_some_chunks = False
    
    file_type_for_log = "이미지 설명" if is_image_description else "텍스트"
    print(f"Adding '{file_type_for_log}' from '{uploaded_file_obj.name}' to vector DB.")

    for chunk_idx, chunk in enumerate(text_chunks):
        print(f"Processing chunk {chunk_idx+1}/{len(text_chunks)} for embedding from '{uploaded_file_obj.name}' ({file_type_for_log})...")
        embedding = get_text_embedding(chunk) # 청크는 텍스트(원본 또는 이미지 설명)
        if embedding is not None:
            vectors_to_add.append(embedding)
            new_metadata_entries_for_current_file.append({
                "file_name": uploaded_file_obj.name, # 원본 파일명 (이미지 또는 텍스트 문서)
                "content": chunk, # 실제 임베딩된 내용 (텍스트 청크 또는 이미지 설명 청크)
                "is_image_description": is_image_description,
                "original_file_extension": os.path.splitext(uploaded_file_obj.name)[1].lower()
            })
        else:
            embedding_failed_for_some_chunks = True
            print(f"Warning: Failed to get embedding for a chunk in '{uploaded_file_obj.name}'. Skipping chunk.")

    if embedding_failed_for_some_chunks and not vectors_to_add:
        st.error(f"'{uploaded_file_obj.name}' 파일의 모든 내용({file_type_for_log})에 대한 임베딩 생성에 실패했습니다. 학습되지 않았습니다.")
        return False
    elif embedding_failed_for_some_chunks:
         st.warning(f"'{uploaded_file_obj.name}' 파일의 일부 내용({file_type_for_log})에 대한 임베딩 생성에 실패했습니다. 성공한 부분만 학습됩니다.")

    if not vectors_to_add: # 임베딩된 벡터가 하나도 없으면
        st.warning(f"'{uploaded_file_obj.name}' 파일에서 유효한 임베딩을 생성하지 못했습니다. 학습되지 않았습니다.");
        return False

    try:
        current_embedding_dimension = np.array(vectors_to_add[0]).shape[0]
        if index is None or index.d != current_embedding_dimension:
            print(f"WARNING: FAISS index dimension ({index.d if index else 'None'}) mismatch or index is None. Re-initializing with dimension {current_embedding_dimension}.")
            index = faiss.IndexFlatL2(current_embedding_dimension)
            metadata = [] # 인덱스 재생성 시 메타데이터도 초기화

        if vectors_to_add: index.add(np.array(vectors_to_add).astype("float32"))
        metadata.extend(new_metadata_entries_for_current_file)
        print(f"Added {len(vectors_to_add)} new chunks to in-memory DB from '{uploaded_file_obj.name}'. Index total: {index.ntotal}, Index dimension: {index.d}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_index_path = os.path.join(tmpdir, "temp.index")
            if index.ntotal > 0 : # 인덱스에 아이템이 있을 때만 저장
                 faiss.write_index(index, temp_index_path)
                 if not save_binary_data_to_blob(temp_index_path, INDEX_BLOB_NAME, _container_client, "벡터 인덱스"):
                    st.error("벡터 인덱스 Blob 저장 실패"); return False # 저장 실패 시 False 반환
            else: # 빈 인덱스면 저장 건너뛰기 (또는 빈 파일로 덮어쓰도록 할 수도 있음)
                print(f"Skipping saving empty index to Blob: {INDEX_BLOB_NAME}")
                # 만약 Blob에 기존 인덱스 파일이 있다면 삭제하거나 빈 파일로 덮어쓰는 로직 추가 가능

        if not save_data_to_blob(metadata, METADATA_BLOB_NAME, _container_client, "메타데이터"): # 메타데이터는 항상 저장 (빈 리스트일 수도 있음)
            st.error("메타데이터 Blob 저장 실패"); return False

        user_info = st.session_state.get("user", {}); uploader_name = user_info.get("name", "N/A")
        new_log_entry = {"file": uploaded_file_obj.name, 
                         "type": "image" if is_image_description else "text_document",
                         "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "chunks_added": len(vectors_to_add), "uploader": uploader_name}

        current_upload_logs = load_data_from_blob(UPLOAD_LOG_BLOB_NAME, _container_client, "업로드 로그", default_value=[])
        if not isinstance(current_upload_logs, list): current_upload_logs = [] # 로드 실패 시 기본값
        current_upload_logs.append(new_log_entry)
        if not save_data_to_blob(current_upload_logs, UPLOAD_LOG_BLOB_NAME, _container_client, "업로드 로그"):
            st.warning("업로드 로그를 Blob에 저장하는 데 실패했습니다.") # 로그 저장은 실패해도 학습 자체는 성공으로 간주
        return True
    except Exception as e:
        st.error(f"문서 학습 또는 Azure Blob 업로드 중 오류: {e}")
        print(f"ERROR: Failed to add document or upload to Blob: {e}\n{traceback.format_exc()}")
        return False

def save_original_file_to_blob(uploaded_file_obj, _container_client):
    # (이전과 동일 - 변경 없음)
    if not _container_client: st.error("Azure Blob 클라이언트가 준비되지 않아 원본 파일을 저장할 수 없습니다."); return None
    try:
        uploaded_file_obj.seek(0) # 스트림 위치 초기화
        # 파일명에 날짜시간 추가하여 중복 방지 및 추적 용이
        original_blob_name = f"uploaded_originals/{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file_obj.name}"
        blob_client_for_original = _container_client.get_blob_client(blob=original_blob_name)
        # getvalue()를 사용하여 BytesIO 객체의 전체 내용을 전달
        blob_client_for_original.upload_blob(uploaded_file_obj.getvalue(), overwrite=False, timeout=120) 
        print(f"Original file '{uploaded_file_obj.name}' saved to Blob as '{original_blob_name}'.")
        return original_blob_name
    except AzureError as ae:
        st.error(f"'{uploaded_file_obj.name}' 원본 파일 Blob 업로드 중 Azure 서비스 오류: {ae}")
        print(f"AZURE ERROR saving original file to Blob: {ae}\n{traceback.format_exc()}")
        return None
    except Exception as e:
        st.error(f"'{uploaded_file_obj.name}' 원본 파일 Blob 업로드 중 알 수 없는 오류: {e}")
        print(f"GENERAL ERROR saving original file to Blob: {e}\n{traceback.format_exc()}")
        return None

def log_openai_api_usage_to_blob(user_id_str, model_name_str, usage_stats_obj, _container_client, request_type="chat_completion"):
    # (이전과 동일, request_type 파라미터 추가하여 구분 가능)
    if not _container_client:
        print("ERROR: Blob Container client is None for API usage log. Skipping log.")
        return

    prompt_tokens = getattr(usage_stats_obj, 'prompt_tokens', 0)
    completion_tokens = getattr(usage_stats_obj, 'completion_tokens', 0)
    total_tokens = getattr(usage_stats_obj, 'total_tokens', 0)

    new_log_entry = {
        "user_id": user_id_str, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": model_name_str, 
        "request_type": request_type, # 요청 종류 (예: "chat_completion", "embedding", "image_description")
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens, "total_tokens": total_tokens
    }

    current_usage_logs = load_data_from_blob(USAGE_LOG_BLOB_NAME, _container_client, "API 사용량 로그", default_value=[])
    if not isinstance(current_usage_logs, list): current_usage_logs = []
    current_usage_logs.append(new_log_entry)

    if not save_data_to_blob(current_usage_logs, USAGE_LOG_BLOB_NAME, _container_client, "API 사용량 로그"):
        print(f"WARNING: Failed to save API usage log to Blob for user '{user_id_str}'.")

# --- 메인 UI 구성 ---
# (이전과 동일)
tab_labels_list = ["💬 업무 질문"]
if current_user_info.get("role") == "admin":
    tab_labels_list.append("⚙️ 관리자 설정")

main_tabs_list = st.tabs(tab_labels_list)
chat_interface_tab = main_tabs_list[0]
admin_settings_tab = main_tabs_list[1] if len(main_tabs_list) > 1 else None


with chat_interface_tab:
    st.header("업무 질문")
    st.markdown("💡 예시: SOP 백업 주기, PIC/S Annex 11 차이, (파일 첨부 후) 이 사진 속 상황은 어떤 규정에 해당하나요? 등")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        print("Chat messages list re-initialized in chat_tab (should not happen if init is correct).")

    for msg_item in st.session_state["messages"]:
        role, content, time_str = msg_item.get("role"), msg_item.get("content", ""), msg_item.get("time", "")
        align_class = "user-align" if role == "user" else "assistant-align"
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        
        # 이미지 URL이 content에 포함되어 있으면 이미지로 표시 (선택적 기능)
        # if role == "user" and content.startswith("data:image"):
        # st.markdown(f"""<div class="chat-bubble-container {align_class}"><img src="{content}" style="max-width: 300px; border-radius: 10px;"/><div class="timestamp">{time_str}</div></div>""", unsafe_allow_html=True)
        # else:
        st.markdown(f"""<div class="chat-bubble-container {align_class}"><div class="bubble {bubble_class}">{content}</div><div class="timestamp">{time_str}</div></div>""", unsafe_allow_html=True)


    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True) # 간격 추가
    if st.button("📂 파일 첨부/숨기기", key="toggle_chat_uploader_final_v5_button"): # 키 변경
        st.session_state.show_uploader = not st.session_state.get("show_uploader", False)

    chat_file_uploader_key = "chat_file_uploader_final_v5_widget" # 키 변경
    uploaded_chat_file_runtime = None # 위젯 상태 유지를 위해 밖에서 초기화
    if st.session_state.get("show_uploader", False):
        uploaded_chat_file_runtime = st.file_uploader("질문과 함께 참고할 파일 첨부 (선택 사항)",
                                     type=["pdf","docx","xlsx","xlsm","csv","pptx", "txt", "png", "jpg", "jpeg"], # 허용 타입 추가
                                     key=chat_file_uploader_key)
        if uploaded_chat_file_runtime: 
            st.caption(f"첨부됨: {uploaded_chat_file_runtime.name} ({uploaded_chat_file_runtime.type}, {uploaded_chat_file_runtime.size} bytes)")
            # 이미지 파일인 경우 미리보기 (선택적)
            if uploaded_chat_file_runtime.type.startswith("image/"):
                st.image(uploaded_chat_file_runtime, width=200)


    with st.form("chat_input_form_final_v5", clear_on_submit=True): # 키 변경
        query_input_col, send_button_col = st.columns([4,1])
        with query_input_col:
            user_query_input = st.text_input("질문 입력:", placeholder="여기에 질문을 입력하세요...",
                                             key="user_query_text_input_final_v5", label_visibility="collapsed") # 키 변경
        with send_button_col:
            send_query_button = st.form_submit_button("전송")

    if send_query_button and user_query_input.strip():
        if not openai_client:
            st.error("OpenAI 서비스가 준비되지 않아 답변을 생성할 수 없습니다. 관리자에게 문의하세요.")
        elif not tokenizer: # Tiktoken 로더 실패 시
             st.error("Tiktoken 라이브러리 로드 실패. 답변을 생성할 수 없습니다.")
        else:
            timestamp_now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # 사용자 질문 메시지 추가 (이미지 파일명도 함께 표시 가능)
            user_message_content = user_query_input
            if uploaded_chat_file_runtime:
                user_message_content += f"\n(첨부 파일: {uploaded_chat_file_runtime.name})"
            st.session_state["messages"].append({"role":"user", "content":user_message_content, "time":timestamp_now_str})


            user_id_for_log = current_user_info.get("name", "anonymous_chat_user_runtime")
            print(f"User '{user_id_for_log}' submitted query: '{user_query_input[:50]}...' with file: {uploaded_chat_file_runtime.name if uploaded_chat_file_runtime else 'None'}")
            
            with st.spinner("답변 생성 중... 잠시만 기다려주세요."):
                assistant_response_content = "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
                try:
                    print("Step 1: Preparing context and calculating tokens...")
                    context_items_for_prompt = []
                    
                    # --- 채팅 중 파일 처리 로직 ---
                    text_from_chat_file = None
                    is_chat_file_image_description = False
                    chat_file_source_name_for_prompt = None

                    if uploaded_chat_file_runtime:
                        file_ext_chat = os.path.splitext(uploaded_chat_file_runtime.name)[1].lower()
                        is_image_chat = file_ext_chat in [".png", ".jpg", ".jpeg"]
                        
                        if is_image_chat:
                            print(f"DEBUG Chat: Processing uploaded image '{uploaded_chat_file_runtime.name}' for description.")
                            with st.spinner(f"첨부 이미지 '{uploaded_chat_file_runtime.name}' 분석 중..."):
                                image_bytes_chat = uploaded_chat_file_runtime.getvalue()
                                description_chat = get_image_description(image_bytes_chat, uploaded_chat_file_runtime.name, openai_client)
                            if description_chat:
                                text_from_chat_file = description_chat
                                chat_file_source_name_for_prompt = f"사용자 첨부 이미지: {uploaded_chat_file_runtime.name}" # (설명)은 나중에 붙임
                                is_chat_file_image_description = True
                                print(f"DEBUG Chat: Image description generated for '{uploaded_chat_file_runtime.name}'. Length: {len(description_chat)}")
                            else:
                                st.warning(f"채팅 중 첨부된 이미지 '{uploaded_chat_file_runtime.name}'의 설명을 생성하지 못했습니다. 해당 파일은 컨텍스트에서 제외됩니다.")
                        else: # .txt, .pdf 등 텍스트 기반 파일
                            print(f"DEBUG Chat: Extracting text from uploaded file '{uploaded_chat_file_runtime.name}'.")
                            text_from_chat_file = extract_text_from_file(uploaded_chat_file_runtime)
                            if text_from_chat_file:
                                chat_file_source_name_for_prompt = f"사용자 첨부 파일: {uploaded_chat_file_runtime.name}"
                                print(f"DEBUG Chat: Text extracted from '{uploaded_chat_file_runtime.name}'. Length: {len(text_from_chat_file)}")
                            else:
                                st.info(f"채팅 중 첨부된 '{uploaded_chat_file_runtime.name}' 파일이 비어있거나 지원하지 않는 내용입니다. 컨텍스트에서 제외됩니다.")
                        
                        if text_from_chat_file: # 내용이 있는 경우 (텍스트 또는 이미지 설명)
                            context_items_for_prompt.append({
                                "source": chat_file_source_name_for_prompt,
                                "content": text_from_chat_file,
                                "is_image_description": is_chat_file_image_description 
                            })
                    # --- 파일 처리 로직 끝 ---

                    prompt_structure = f"{PROMPT_RULES_CONTENT}\n\n위의 규칙을 반드시 준수하여 답변해야 합니다. 다음은 사용자의 질문에 답변하는 데 참고할 수 있는 문서의 내용입니다:\n<문서 시작>\n{{context}}\n<문서 끝>"
                    base_prompt_text = prompt_structure.replace('{context}', '')
                    try:
                        base_tokens = len(tokenizer.encode(base_prompt_text))
                        query_tokens = len(tokenizer.encode(user_query_input))
                    except Exception as e_tokenize_base:
                        st.error(f"기본 프롬프트 또는 질문 토큰화 중 오류: {e_tokenize_base}")
                        raise # 더 이상 진행 불가

                    print(f"DEBUG: 기본 프롬프트 구조 토큰: {base_tokens}")
                    print(f"DEBUG: 사용자 질문 토큰: {query_tokens}")

                    max_context_tokens = TARGET_INPUT_TOKENS_FOR_PROMPT - base_tokens - query_tokens
                    print(f"DEBUG: 목표 프롬프트 토큰: {TARGET_INPUT_TOKENS_FOR_PROMPT}")
                    print(f"DEBUG: 컨텍스트에 할당 가능한 최대 토큰: {max_context_tokens}")

                    context_string_for_llm = "현재 참고할 문서가 없습니다." # 기본값
                    if max_context_tokens <= 0:
                         st.warning("프롬프트 규칙과 질문만으로도 입력 토큰 제한에 도달했습니다. 추가 컨텍스트(DB 검색 결과, 첨부 파일 내용)를 포함할 수 없습니다.")
                         print("WARNING: No tokens left for context after accounting for rules and query.")
                         context_string_for_llm = "참고할 컨텍스트를 포함할 수 없습니다 (토큰 제한)."
                    else:
                        # DB 검색 결과 추가 (사용자 질문 + 이미지 설명이 있다면 그것도 포함하여 검색)
                        query_for_db_search = user_query_input
                        if is_chat_file_image_description and text_from_chat_file: # 이미지 설명이 있다면 검색 쿼리에 추가
                            query_for_db_search = f"{user_query_input}\n\n이미지 내용: {text_from_chat_file}"
                        
                        print(f"DEBUG: Retrieving context from Vector DB based on query: '{query_for_db_search[:50]}...'")
                        retrieved_items_from_db = search_similar_chunks(query_for_db_search, k_results=3) # 검색 결과 개수 조절 가능
                        if retrieved_items_from_db:
                            context_items_for_prompt.extend(retrieved_items_from_db) # DB 검색 결과 추가
                            print(f"DEBUG: Retrieved {len(retrieved_items_from_db)} items from Vector DB with source info.")
                        else:
                            print(f"DEBUG: No relevant items found in Vector DB for query.")
                        
                        # 컨텍스트 포맷팅 및 토큰 제한 적용
                        if not context_items_for_prompt:
                            print("DEBUG: No context items found (no file attached, no DB results).")
                            # context_string_for_llm는 기본값 "현재 참고할 문서가 없습니다." 유지
                        else:
                            seen_contents_for_final_context = set()
                            formatted_context_chunks = []
                            for item_idx, item in enumerate(context_items_for_prompt):
                                if isinstance(item, dict):
                                    content_value = item.get("content", "")
                                    source_info = item.get('source', f'출처 정보 없음 {item_idx+1}')
                                    is_desc_item = item.get("is_image_description", False)
                                    
                                    content_strip = content_value.strip()
                                    if content_strip and content_strip not in seen_contents_for_final_context:
                                        final_source_display_name = source_info
                                        # 채팅 중 첨부 파일 소스 이름 정리
                                        if source_info.startswith("사용자 첨부 이미지: "):
                                            final_source_display_name = source_info.replace("사용자 첨부 이미지: ", "")
                                        elif source_info.startswith("사용자 첨부 파일: "):
                                            final_source_display_name = source_info.replace("사용자 첨부 파일: ", "")

                                        if is_desc_item:
                                            formatted_context_chunks.append(f"[Image Description for: {final_source_display_name}]\n{content_value}")
                                        else:
                                            formatted_context_chunks.append(f"[출처: {final_source_display_name}]\n{content_value}")
                                        seen_contents_for_final_context.add(content_strip)
                                
                            if not formatted_context_chunks:
                                print("DEBUG: No unique context items after filtering. Using '현재 참고할 문서가 없습니다.'")
                                # context_string_for_llm는 기본값 "현재 참고할 문서가 없습니다." 유지
                            else:
                                full_context_string = "\n\n---\n\n".join(formatted_context_chunks)
                                try:
                                    full_context_tokens = tokenizer.encode(full_context_string)
                                except Exception as e_tokenize_full_ctx:
                                    st.error(f"컨텍스트 문자열 토큰화 중 오류: {e_tokenize_full_ctx}")
                                    raise 

                                print(f"DEBUG: 전체 컨텍스트 문자열 토큰 수 (출처 포함 포맷): {len(full_context_tokens)}")

                                if len(full_context_tokens) > max_context_tokens:
                                    truncated_tokens = full_context_tokens[:max_context_tokens]
                                    try:
                                        context_string_for_llm = tokenizer.decode(truncated_tokens)
                                        if len(full_context_tokens) > len(truncated_tokens) : # 잘렸음을 명시
                                            context_string_for_llm += "\n(...내용 더 있음, 일부 내용이 잘렸을 수 있습니다.)"
                                    except Exception as e_decode_truncated:
                                        st.error(f"잘린 토큰 디코딩 중 오류: {e_decode_truncated}")
                                        context_string_for_llm = "[오류: 컨텍스트 디코딩 실패]"
                                    print(f"WARNING: 컨텍스트 토큰 수가 너무 많아 {max_context_tokens} 토큰으로 잘랐습니다.")
                                    print(f"DEBUG: 잘린 컨텍스트 문자열 (앞 100자): {context_string_for_llm[:100]}")
                                else:
                                    context_string_for_llm = full_context_string
                                    print(f"DEBUG: 전체 컨텍스트를 사용합니다. (앞 100자): {context_string_for_llm[:100]}")
                    
                    # 최종 시스템 프롬프트 구성
                    system_prompt_content = prompt_structure.replace('{context}', context_string_for_llm)
                    try:
                        final_system_tokens = len(tokenizer.encode(system_prompt_content))
                        final_prompt_tokens = final_system_tokens + query_tokens # 시스템 프롬프트 + 사용자 질문
                    except Exception as e_tokenize_final_sys:
                         st.error(f"최종 시스템 프롬프트 토큰화 중 오류: {e_tokenize_final_sys}")
                         raise

                    print(f"DEBUG: 최종 시스템 프롬프트 토큰: {final_system_tokens}")
                    print(f"DEBUG: 최종 API 입력 토큰 (시스템+질문): {final_prompt_tokens}")
                    if final_prompt_tokens > MODEL_MAX_INPUT_TOKENS:
                         print(f"CRITICAL WARNING: 최종 입력 토큰({final_prompt_tokens})이 모델 최대치({MODEL_MAX_INPUT_TOKENS})를 초과했습니다! API 오류 가능성 높음.")
                         # st.warning(f"입력 요청이 너무 깁니다 (현재 {final_prompt_tokens} 토큰, 최대 {MODEL_MAX_INPUT_TOKENS} 토큰). 답변이 생성되지 않거나 오류가 발생할 수 있습니다.")

                    chat_messages_for_api = [{"role":"system", "content": system_prompt_content}, {"role":"user", "content": user_query_input}]

                    print("Step 2: Sending request to Azure OpenAI for chat completion...")
                    chat_completion_response = openai_client.chat.completions.create(
                        model=st.secrets["AZURE_OPENAI_DEPLOYMENT"], # 채팅용 모델
                        messages=chat_messages_for_api,
                        max_tokens=MODEL_MAX_OUTPUT_TOKENS, # 답변 최대 토큰
                        temperature=0.1, # 답변 일관성
                        timeout=AZURE_OPENAI_TIMEOUT
                    )
                    assistant_response_content = chat_completion_response.choices[0].message.content.strip()
                    print("Azure OpenAI response received.")

                    if chat_completion_response.usage and container_client:
                        print("Logging OpenAI API usage for chat completion...")
                        log_openai_api_usage_to_blob(user_id_for_log, st.secrets["AZURE_OPENAI_DEPLOYMENT"], chat_completion_response.usage, container_client, request_type="chat_completion_with_rag")
                
                # ... (이하 예외 처리 로직은 이전과 거의 동일)
                except APITimeoutError:
                    assistant_response_content = "죄송합니다, 답변 생성 시간이 초과되었습니다. 질문을 조금 더 간단하게 해주시거나 잠시 후 다시 시도해주세요."
                    st.error(assistant_response_content)
                    print(f"TIMEOUT ERROR: Chat completion request timed out for user '{user_id_for_log}'.")
                except APIConnectionError as ace:
                    assistant_response_content = f"API 연결 오류가 발생했습니다: {ace}. 네트워크 상태를 확인하거나 잠시 후 다시 시도해주세요."
                    st.error(assistant_response_content)
                    print(f"API CONNECTION ERROR during chat completion: {ace}")
                except RateLimitError as rle:
                    assistant_response_content = f"API 요청 한도를 초과했습니다: {rle}. 잠시 후 다시 시도해주세요."
                    st.error(assistant_response_content)
                    print(f"RATE LIMIT ERROR during chat completion: {rle}")
                except APIStatusError as ase: # OpenAI API가 에러 상태 코드를 반환할 때
                    assistant_response_content = f"API에서 오류 응답을 받았습니다 (상태 코드 {ase.status_code}): {ase.message}. 문제가 지속되면 관리자에게 문의해주세요."
                    st.error(assistant_response_content)
                    print(f"API STATUS ERROR during chat completion (Status {ase.status_code}): {ase.message}")
                    if ase.response and ase.response.content:
                        try:
                            error_details_chat = json.loads(ase.response.content.decode('utf-8'))
                            print(f"DEBUG ChatCompletion: Azure API error details: {json.dumps(error_details_chat, indent=2, ensure_ascii=False)}")
                        except Exception as json_e_chat:
                            print(f"DEBUG ChatCompletion: Could not parse Azure API error content as JSON: {json_e_chat}")
                            print(f"DEBUG ChatCompletion: Raw Azure API error content: {ase.response.content}")
                except Exception as gen_err: # 기타 예외 (토큰화 실패 등 포함)
                    assistant_response_content = f"답변 생성 중 예기치 않은 오류가 발생했습니다: {gen_err}. 관리자에게 문의해주세요."
                    st.error(assistant_response_content)
                    print(f"UNEXPECTED ERROR during response generation: {gen_err}\n{traceback.format_exc()}")

                st.session_state["messages"].append({"role":"assistant", "content":assistant_response_content, "time":timestamp_now_str})
                print("Response processing complete.")
            
            # 위젯 상태 초기화를 위해 uploaded_chat_file_runtime을 None으로 설정 (선택적)
            # st.session_state[chat_file_uploader_key] = None # 이렇게 하면 업로더가 초기화됨
            # 또는 그냥 st.rerun()만 호출
            st.rerun()


if admin_settings_tab:
    with admin_settings_tab:
        st.header("⚙️ 관리자 설정")
        st.subheader("👥 가입 승인 대기자")
        # (이전과 동일 - 변경 없음)
        if not USERS or not isinstance(USERS, dict):
            st.warning("사용자 정보를 로드할 수 없거나 형식이 올바르지 않습니다.")
            print(f"WARNING: USERS data is problematic or empty in admin tab. Type: {type(USERS)}")
        else:
            pending_approval_users = {uid:udata for uid,udata in USERS.items() if isinstance(udata, dict) and not udata.get("approved")}
            if pending_approval_users:
                for pending_uid, pending_user_data in pending_approval_users.items():
                    with st.expander(f"{pending_user_data.get('name','N/A')} ({pending_uid}) - {pending_user_data.get('department','N/A')}"):
                        approve_col, reject_col = st.columns(2)
                        if approve_col.button("승인", key=f"admin_approve_user_final_v5_{pending_uid}"): # 키 변경
                            USERS[pending_uid]["approved"] = True
                            if save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "사용자 정보"):
                                st.success(f"'{pending_uid}' 사용자를 승인하고 Blob에 저장했습니다."); st.rerun()
                            else: st.error("사용자 승인 정보 Blob 저장 실패.")
                        if reject_col.button("거절", key=f"admin_reject_user_final_v5_{pending_uid}"): # 키 변경
                            USERS.pop(pending_uid, None)
                            if save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "사용자 정보"):
                                st.info(f"'{pending_uid}' 사용자의 가입 신청을 거절하고 Blob에 저장했습니다."); st.rerun()
                            else: st.error("사용자 거절 정보 Blob 저장 실패.")
            else: st.info("승인 대기 중인 사용자가 없습니다.")
        st.markdown("---")

        st.subheader("📁 파일 업로드 및 학습 (Azure Blob Storage)")
        if 'processed_admin_file_info' not in st.session_state:
            st.session_state.processed_admin_file_info = None

        def clear_processed_file_info_on_admin_upload_change():
            print(f"DEBUG admin_file_uploader on_change: Clearing processed_admin_file_info (was: {st.session_state.processed_admin_file_info})")
            st.session_state.processed_admin_file_info = None

        admin_file_uploader_key = "admin_file_uploader_v_final_img_txt" # 키 변경
        admin_uploaded_file = st.file_uploader(
            "학습할 파일 업로드 (PDF, DOCX, XLSX, CSV, PPTX, TXT, PNG, JPG, JPEG)",
            type=["pdf","docx","xlsx","xlsm","csv","pptx", "txt", "png", "jpg", "jpeg"], # 허용 타입 추가
            key=admin_file_uploader_key,
            on_change=clear_processed_file_info_on_admin_upload_change,
            accept_multiple_files=False # 한 번에 하나의 파일만 처리
        )

        if admin_uploaded_file and container_client:
            current_file_info = (admin_uploaded_file.name, admin_uploaded_file.size, admin_uploaded_file.type)
            if st.session_state.processed_admin_file_info != current_file_info:
                print(f"DEBUG Admin Upload: New file detected. File Info: {current_file_info}")
                
                file_ext_admin = os.path.splitext(admin_uploaded_file.name)[1].lower()
                is_image_admin_upload = file_ext_admin in [".png", ".jpg", ".jpeg"]
                content_for_learning = None
                is_img_desc_for_learning = False

                if is_image_admin_upload:
                    with st.spinner(f"'{admin_uploaded_file.name}' 이미지 처리 및 설명 생성 중..."):
                        img_bytes_admin = admin_uploaded_file.getvalue()
                        description_admin = get_image_description(img_bytes_admin, admin_uploaded_file.name, openai_client)
                    if description_admin:
                        content_for_learning = description_admin
                        is_img_desc_for_learning = True
                        st.info(f"이미지 '{admin_uploaded_file.name}'에 대한 설명이 생성되었습니다 (길이: {len(description_admin)}). 이 설명을 학습합니다.")
                        st.text_area("생성된 이미지 설명 (학습 대상)", description_admin, height=150, disabled=True)
                    else:
                        st.error(f"'{admin_uploaded_file.name}' 이미지 설명을 생성하지 못했습니다. 학습에서 제외됩니다.")
                else: # 텍스트 기반 파일 (pdf, docx, txt 등)
                    with st.spinner(f"'{admin_uploaded_file.name}' 파일 내용 추출 중..."):
                        content_for_learning = extract_text_from_file(admin_uploaded_file)
                    if content_for_learning:
                        st.info(f"'{admin_uploaded_file.name}' 파일에서 텍스트를 추출했습니다 (길이: {len(content_for_learning)}).")
                    else:
                        st.warning(f"'{admin_uploaded_file.name}' 파일에서 내용을 추출하지 못했거나 비어있습니다. 학습에서 제외됩니다.")
                
                if content_for_learning: # 내용이 있거나 (텍스트) 설명이 생성된 경우
                    with st.spinner(f"'{admin_uploaded_file.name}' 내용 처리 및 학습 진행 중..."):
                        content_chunks_for_learning = chunk_text_into_pieces(content_for_learning)
                        if content_chunks_for_learning:
                            # 원본 파일 저장 (항상)
                            original_file_blob_path = save_original_file_to_blob(admin_uploaded_file, container_client)
                            if original_file_blob_path: 
                                st.caption(f"원본 파일 '{admin_uploaded_file.name}'이 Blob에 '{original_file_blob_path}'로 저장되었습니다.")
                            else: 
                                st.warning(f"원본 파일 '{admin_uploaded_file.name}'을 Blob에 저장하는 데 실패했습니다.")

                            # 벡터 DB에 추가 및 메타데이터 저장
                            if add_document_to_vector_db_and_blob(
                                admin_uploaded_file, 
                                content_for_learning, # 원본 텍스트 또는 이미지 설명 전체 (청킹 전)
                                content_chunks_for_learning, 
                                container_client, 
                                is_image_description=is_img_desc_for_learning
                            ):
                                st.success(f"'{admin_uploaded_file.name}' 파일 학습 및 Azure Blob Storage에 업데이트 완료!")
                                st.session_state.processed_admin_file_info = current_file_info # 성공 시 정보 기록
                                # 성공 후 페이지를 새로고침하여 상태를 명확히 하거나, 업로더를 초기화할 수 있음
                                st.rerun() 
                            else:
                                st.error(f"'{admin_uploaded_file.name}' 학습 또는 Blob 업데이트 중 오류가 발생했습니다.")
                                st.session_state.processed_admin_file_info = None # 실패 시 정보 초기화
                        else: 
                            st.warning(f"'{admin_uploaded_file.name}' 파일에서 유의미한 학습 청크를 생성하지 못했습니다.")
                # (content_for_learning이 없는 경우는 위에서 이미 st.error 또는 st.warning으로 처리됨)
            elif st.session_state.processed_admin_file_info == current_file_info:
                 st.caption(f"'{admin_uploaded_file.name}' 파일은 이전에 성공적으로 학습(또는 처리 시도)되었습니다. 다른 파일을 업로드하거나, 현재 파일을 제거(X 버튼) 후 다시 업로드하여 재학습할 수 있습니다.")
        elif admin_uploaded_file and not container_client:
            st.error("Azure Blob 클라이언트가 준비되지 않아 파일을 업로드하고 학습할 수 없습니다.")
        st.markdown("---")

        st.subheader("📊 API 사용량 모니터링 (Blob 로그 기반)")
        # (이전과 동일 - 변경 없음)
        if container_client:
            usage_data_from_blob = load_data_from_blob(USAGE_LOG_BLOB_NAME, container_client, "API 사용량 로그", default_value=[])
            if usage_data_from_blob and isinstance(usage_data_from_blob, list) and len(usage_data_from_blob) > 0 :
                df_usage_stats=pd.DataFrame(usage_data_from_blob)
                
                # 필수 컬럼 존재 확인 및 없으면 0으로 채우기
                for col in ["total_tokens", "prompt_tokens", "completion_tokens"]:
                     if col not in df_usage_stats.columns:
                         df_usage_stats[col] = 0
                if "request_type" not in df_usage_stats.columns: # 새로 추가된 컬럼
                    df_usage_stats["request_type"] = "unknown"


                token_cols = ["total_tokens", "prompt_tokens", "completion_tokens"]
                for col in token_cols: # 숫자형으로 변환, 변환 불가 시 0
                    df_usage_stats[col] = pd.to_numeric(df_usage_stats[col], errors='coerce').fillna(0)

                total_tokens_used = df_usage_stats["total_tokens"].sum()
                st.metric("총 API 호출 수", len(df_usage_stats))
                st.metric("총 사용 토큰 수", f"{int(total_tokens_used):,}")

                token_cost_per_unit = 0.0
                try: token_cost_per_unit=float(st.secrets.get("TOKEN_COST","0"))
                except (ValueError, TypeError): pass # 변환 실패 시 0.0 유지
                st.metric("예상 비용 (USD)", f"${total_tokens_used * token_cost_per_unit:.4f}") # 소수점 4자리까지

                if "timestamp" in df_usage_stats.columns:
                    try: # timestamp 기준으로 정렬 시도
                         df_usage_stats['timestamp'] = pd.to_datetime(df_usage_stats['timestamp'])
                         st.dataframe(df_usage_stats.sort_values(by="timestamp",ascending=False), use_container_width=True)
                    except Exception as e_sort_ts:
                         print(f"Warning: Could not sort usage log by timestamp due to conversion error: {e_sort_ts}")
                         st.dataframe(df_usage_stats, use_container_width=True) # 정렬 실패 시 원본 표시
                else: # timestamp 컬럼이 없을 경우
                    st.dataframe(df_usage_stats, use_container_width=True)
            else: st.info("기록된 API 사용량 데이터가 Blob에 없거나 비어있습니다.")
        else: st.warning("Azure Blob 클라이언트가 준비되지 않아 API 사용량 모니터링을 표시할 수 없습니다.")
        st.markdown("---")

        st.subheader("📂 Azure Blob Storage 파일 목록 (최근 100개)")
        # (이전과 동일 - 변경 없음)
        if container_client:
            try:
                blob_list_display = []
                count = 0
                max_blobs_to_show = 100 # 표시할 최대 Blob 수
                # last_modified 기준으로 정렬하여 최근 파일부터 가져오기
                blobs_sorted = sorted(container_client.list_blobs(), key=lambda b: b.last_modified, reverse=True)

                for blob_item in blobs_sorted:
                    if count >= max_blobs_to_show:
                        break
                    blob_list_display.append({
                        "파일명": blob_item.name,
                        "크기 (bytes)": blob_item.size,
                        "수정일": blob_item.last_modified.strftime('%Y-%m-%d %H:%M:%S') if blob_item.last_modified else 'N/A'
                    })
                    count += 1
                
                if blob_list_display:
                    df_blobs_display = pd.DataFrame(blob_list_display)
                    st.dataframe(df_blobs_display, use_container_width=True)
                else: st.info("Azure Blob Storage에 파일이 없습니다.")
            except AzureError as ae_list_blob: # Azure 관련 오류 명시적 처리
                 st.error(f"Azure Blob 파일 목록 조회 중 Azure 서비스 오류: {ae_list_blob}")
                 print(f"AZURE ERROR listing blobs: {ae_list_blob}\n{traceback.format_exc()}")
            except Exception as e_list_blob:
                st.error(f"Azure Blob 파일 목록 조회 중 알 수 없는 오류: {e_list_blob}")
                print(f"ERROR listing blobs: {e_list_blob}\n{traceback.format_exc()}")
        else:
            st.warning("Azure Blob 클라이언트가 준비되지 않아 파일 목록을 표시할 수 없습니다.")
