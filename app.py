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
import base64
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
MODEL_MAX_OUTPUT_TOKENS = 16384
BUFFER_TOKENS = 500 # 프롬프트 규칙, 사용자 질문 외 추가 버퍼
TARGET_INPUT_TOKENS_FOR_PROMPT = MODEL_MAX_INPUT_TOKENS - MODEL_MAX_OUTPUT_TOKENS - BUFFER_TOKENS


# --- CSS 스타일 ---
st.markdown("""
<style>
    /* (기존 CSS 스타일 내용 유지) */
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

    /* 메인 앱 제목 및 부제목 (로그인 후) */
    .main-app-title-container { text-align: center; margin-bottom: 24px; }
    .main-app-title { font-size: 2.1rem; font-weight: bold; display: block; }
    .main-app-subtitle { font-size: 0.9rem; color: gray; display: block; margin-top: 4px;}

    /* 로고 및 버전 */
    .logo-container { display: flex; align-items: center; }
    .logo-image { margin-right: 10px; }
    .version-text { font-size: 0.9rem; color: gray; }

    /* 로그인 화면 전용 제목 스타일 */
    .login-page-header-container { text-align: center; margin-top: 20px; margin-bottom: 10px;}
    .login-page-main-title { font-size: 1.8rem; font-weight: bold; display: block; color: #333F48; }
    .login-page-sub-title { font-size: 0.85rem; color: gray; display: block; margin-top: 2px; margin-bottom: 20px;}
    .login-form-title { /* "로그인 또는 회원가입" 제목 */
        font-size: 1.6rem;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
    }

    /* 모바일 화면 대응 */
    @media (max-width: 768px) {
        .main-app-title {
            font-size: 1.8rem; /* 모바일에서 메인 앱 제목 */
        }
        .main-app-subtitle {
            font-size: 0.8rem; /* 모바일에서 메인 앱 부제목 */
        }
        .login-page-main-title {
            font-size: 1.5rem; /* 모바일에서 로그인 페이지의 프로그램 제목 */
        }
        .login-page-sub-title {
            font-size: 0.8rem; /* 모바일에서 로그인 페이지의 프로그램 부제목 */
        }
        .login-form-title { /* "로그인 또는 회원가입" 제목 모바일 크기 */
            font-size: 1.3rem;
            margin-bottom: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Azure 클라이언트 초기화 ---
@st.cache_resource
def get_azure_openai_client_cached():
    print("Attempting to initialize Azure OpenAI client...")
    try:
        client = AzureOpenAI(
            api_key=st.secrets["AZURE_OPENAI_KEY"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            api_version=st.secrets["AZURE_OPENAI_VERSION"],
            timeout=AZURE_OPENAI_TIMEOUT
        )
        print("Azure OpenAI client initialized successfully.")
        return client
    except KeyError as e:
        st.error(f"Azure OpenAI 설정 오류: secrets.toml 파일에 '{e.args[0]}' 키가 없습니다. 앱이 정상 동작하지 않을 수 있습니다.")
        print(f"ERROR: Missing Azure OpenAI secret: {e.args[0]}")
        return None
    except Exception as e:
        st.error(f"Azure OpenAI 클라이언트 초기화 중 심각한 오류 발생: {e}. 앱이 정상 동작하지 않을 수 있습니다.")
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
        st.error(f"Azure Blob Storage 설정 오류: secrets.toml 파일에 '{e.args[0]}' 키가 없습니다. 데이터 기능을 사용할 수 없습니다.")
        print(f"ERROR: Missing Azure Blob Storage secret: {e.args[0]}")
        return None, None
    except Exception as e:
        st.error(f"Azure Blob 클라이언트 초기화 중 심각한 오류 발생: {e}. 데이터 기능을 사용할 수 없습니다.")
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
        st.error("secrets.toml 파일에 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT' 설정이 없습니다. 임베딩 기능을 사용할 수 없습니다.")
        print("ERROR: Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT secret.")
        openai_client = None
    except Exception as e:
        st.error(f"임베딩 모델 설정 로드 중 오류: {e}")
        print(f"ERROR: Loading AZURE_OPENAI_EMBEDDING_DEPLOYMENT secret: {e}")
        openai_client = None


# --- 데이터 로드/저장 유틸리티 함수 (Blob 연동) ---
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
            "password_hash": generate_password_hash("diteam"),
            "approved": True, "role": "admin"
        }
        if not save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "초기 사용자 정보"):
             st.warning("기본 관리자 정보를 Blob에 저장하는데 실패했습니다. 다음 실행 시 다시 시도됩니다.")
else:
    st.error("Azure Blob Storage 연결 실패. 사용자 정보를 초기화할 수 없습니다. 앱이 정상 동작하지 않을 수 있습니다.")
    print("CRITICAL: Cannot initialize USERS due to Blob client failure.")
    USERS = {"admin": {"name": "관리자(연결실패)", "department": "시스템", "password_hash": generate_password_hash("fallback"), "approved": True, "role": "admin"}}


# --- 쿠키 매니저 및 세션 상태 초기화 ---
cookies = None
cookie_manager_ready = False
print(f"Attempting to load COOKIE_SECRET from st.secrets: {st.secrets.get('COOKIE_SECRET')}")
try:
    cookie_secret_key = st.secrets.get("COOKIE_SECRET")
    if not cookie_secret_key:
        st.error("secrets.toml 파일에 'COOKIE_SECRET'이(가) 설정되지 않았거나 비어있습니다. 쿠키 기능을 사용할 수 없습니다.")
        print("ERROR: COOKIE_SECRET is not set or empty in st.secrets.")
    else:
        cookies = EncryptedCookieManager(
            prefix="gmp_chatbot_auth_v5_0/",
            password=cookie_secret_key
        )
        if cookies.ready():
            cookie_manager_ready = True
            print("CookieManager is ready.")
        else:
            print("CookieManager not ready on initial setup (may resolve on first interaction).")

except Exception as e:
    st.error(f"쿠키 매니저 초기화 중 알 수 없는 오류 발생: {e}")
    print(f"CRITICAL: CookieManager initialization error: {e}\n{traceback.format_exc()}")

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

    if 'cookies' in locals() and cookies and cookies.ready():
        cookie_manager_ready = True
        print("CookieManager became ready before cookie check.")
        auth_cookie_val = cookies.get("authenticated")
        print(f"Cookie 'authenticated' value on session init: {auth_cookie_val}")
        if auth_cookie_val == "true":
            login_time_str = cookies.get("login_time", "0")
            login_time = float(login_time_str if login_time_str and login_time_str.replace('.', '', 1).isdigit() else "0")
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
                        if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
                except json.JSONDecodeError:
                    print("ERROR: Failed to decode user JSON from cookie. Clearing auth state for cookie.")
                    st.session_state["authenticated"] = False
                    if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
            else:
                print("Session timeout detected from cookie. Clearing auth state for cookie.")
                st.session_state["authenticated"] = False
                st.session_state["messages"] = []
                if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
        else:
             print("Authenticated cookie not set to 'true'.")
             st.session_state["authenticated"] = False

    elif 'cookies' in locals() and cookies and not cookies.ready():
         print("CookieManager still not ready, cannot restore session from cookie on session init.")
         st.session_state["authenticated"] = False
    else:
         print("CookieManager object is None, cannot restore session.")
         st.session_state["authenticated"] = False


if "messages" not in st.session_state:
    st.session_state["messages"] = []
    print("Redundant check: Initializing messages as it was not in session_state before login UI.")


# --- 로그인 UI 및 로직 ---
if not st.session_state.get("authenticated", False):
    st.markdown("""
    <div class="login-page-header-container">
      <span class="login-page-main-title">유앤생명과학 GMP/SOP 업무 가이드 봇</span>
      <span class="login-page-sub-title">Made by DI.PART</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="login-form-title">🔐 로그인 또는 회원가입</p>', unsafe_allow_html=True)

    if 'cookies' in locals() and cookies and not cookies.ready() and st.secrets.get("COOKIE_SECRET"):
        st.warning("쿠키 시스템을 초기화하고 있습니다. 잠시 후 새로고침하거나 다시 시도해주세요.")
        print("Login UI: CookieManager not ready yet.")


    with st.form("auth_form_final_v4_mobile_ui_fix", clear_on_submit=False):
        mode = st.radio("선택", ["로그인", "회원가입"], key="auth_mode_final_v4_mobile_ui_fix")
        uid = st.text_input("ID", key="auth_uid_final_v4_mobile_ui_fix")
        pwd = st.text_input("비밀번호", type="password", key="auth_pwd_final_v4_mobile_ui_fix")
        name, dept = "", ""
        if mode == "회원가입":
            name = st.text_input("이름", key="auth_name_final_v4_mobile_ui_fix")
            dept = st.text_input("부서", key="auth_dept_final_v4_mobile_ui_fix")
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
                    st.session_state["messages"] = []
                    print(f"Login successful for user '{uid}'. Chat messages cleared.")
                    if 'cookies' in locals() and cookies and cookies.ready():
                        try:
                            cookies["authenticated"] = "true"; cookies["user"] = json.dumps(user_data_login)
                            cookies["login_time"] = str(time.time()); cookies.save()
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
                        USERS.pop(uid, None)
                    else:
                        st.success("가입 신청 완료! 관리자 승인 후 로그인 가능합니다.")
    st.stop()

# --- 인증 후 메인 애플리케이션 로직 ---
current_user_info = st.session_state.get("user", {})

# --- 헤더 (로고, 버전, 로그아웃 버튼) ---
top_cols_main = st.columns([0.7, 0.3])
with top_cols_main[0]:
    if os.path.exists(COMPANY_LOGO_PATH_REPO):
        logo_b64 = get_base64_of_bin_file(COMPANY_LOGO_PATH_REPO)
        if logo_b64:
            st.markdown(f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_b64}" class="logo-image" width="150">
                <span class="version-text">ver 0.9.3 (Vector Search Fix)</span>
            </div>""", unsafe_allow_html=True) # 버전 업데이트
        else:
            st.markdown(f"""<div class="logo-container"><span class="version-text" style="font-weight:bold;">유앤생명과학</span> <span class="version-text" style="margin-left:10px;">ver 0.9.3 (Vector Search Fix)</span></div>""", unsafe_allow_html=True)
    else:
        print(f"WARNING: Company logo file not found at {COMPANY_LOGO_PATH_REPO}")
        st.markdown(f"""<div class="logo-container"><span class="version-text" style="font-weight:bold;">유앤생명과학</span> <span class="version-text" style="margin-left:10px;">ver 0.9.3 (Vector Search Fix)</span></div>""", unsafe_allow_html=True)

with top_cols_main[1]:
    st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
    if st.button("로그아웃", key="logout_button_final_v4_mobile"):
        st.session_state["authenticated"] = False
        st.session_state["user"] = {}
        st.session_state["messages"] = []
        print("Logout successful. Chat messages cleared.")
        if 'cookies' in locals() and cookies and cookies.ready():
             try:
                 cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
                 print("Cookies cleared on logout.")
             except Exception as e_logout_cookie:
                 print(f"ERROR: Failed to clear cookies on logout: {e_logout_cookie}")
        else:
              print("WARNING: CookieManager not ready during logout.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- 메인 앱 제목 (로그인 후) ---
st.markdown("""
<div class="main-app-title-container">
  <span class="main-app-title">유앤생명과학 GMP/SOP 업무 가이드 봇</span>
  <span class="main-app-subtitle">Made by DI.PART</span>
</div>
""", unsafe_allow_html=True)


# --- 벡터 DB 로드 (Azure Blob Storage 기반) ---
@st.cache_resource
def load_vector_db_from_blob_cached(_container_client):
    if not _container_client:
        print("ERROR: Blob Container client is None for load_vector_db_from_blob_cached.")
        return faiss.IndexFlatL2(1536), []
    idx, meta = faiss.IndexFlatL2(1536), []
    print(f"Attempting to load vector DB from Blob: '{INDEX_BLOB_NAME}', '{METADATA_BLOB_NAME}'")
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
                    idx = faiss.read_index(local_index_path)
                    print(f"'{INDEX_BLOB_NAME}' loaded successfully from Blob Storage.")
                else:
                    print(f"WARNING: '{INDEX_BLOB_NAME}' is empty in Blob. Using new index.")
            else:
                print(f"WARNING: '{INDEX_BLOB_NAME}' not found in Blob Storage. New index will be used/created.")

            metadata_blob_client = _container_client.get_blob_client(METADATA_BLOB_NAME)
            if metadata_blob_client.exists():
                print(f"Downloading '{METADATA_BLOB_NAME}'...")
                with open(local_metadata_path, "wb") as download_file:
                    download_stream = metadata_blob_client.download_blob(timeout=60)
                    download_file.write(download_stream.readall())
                if os.path.getsize(local_metadata_path) > 0 :
                    with open(local_metadata_path, "r", encoding="utf-8") as f: meta = json.load(f)
                else:
                    meta = []
                    print(f"WARNING: '{METADATA_BLOB_NAME}' is empty in Blob.")
            else:
                print(f"WARNING: '{METADATA_BLOB_NAME}' not found in Blob Storage. Starting with empty metadata.")
                meta = []
    except AzureError as ae:
        st.error(f"Azure Blob에서 벡터DB 로드 중 Azure 서비스 오류: {ae}")
        print(f"AZURE ERROR loading vector DB from Blob: {ae}\n{traceback.format_exc()}")
    except Exception as e:
        st.error(f"Azure Blob에서 벡터DB 로드 중 알 수 없는 오류: {e}")
        print(f"GENERAL ERROR loading vector DB from Blob: {e}\n{traceback.format_exc()}")
    return idx, meta

index, metadata = faiss.IndexFlatL2(1536), []
if container_client:
    index, metadata = load_vector_db_from_blob_cached(container_client)
    # <<< 추가된 디버깅 로그 시작 >>>
    print(f"DEBUG: FAISS index loaded after cache. ntotal: {index.ntotal if index else 'Index is None'}")
    print(f"DEBUG: Metadata loaded after cache. Length: {len(metadata) if metadata is not None else 'Metadata is None'}")
    # <<< 추가된 디버깅 로그 끝 >>>
else:
    st.error("Azure Blob Storage 연결 실패로 벡터 DB를 로드할 수 없습니다. 파일 학습 및 검색 기능이 제한될 수 있습니다.")
    print("CRITICAL: Cannot load vector DB due to Blob client initialization failure (main section).")


# --- 규칙 파일 로드 ---
@st.cache_data
def load_prompt_rules_cached():
    # 이전에 제공된 프롬프트 내용으로 가정 (사용자가 관리하는 .streamlit/prompt_rules.txt 파일)
    # 실제 배포 시 이 파일이 정확히 반영되어야 합니다.
    # 여기서는 기본값을 가장 최근에 논의된 버전으로 두겠습니다.
    default_rules = """1.우선 기준
    1.1. 모든 답변은 MFDS 규정을 최우선으로 하며, 그 다음은 사내 SOP를 기준으로 삼습니다.
    1.2. 규정/법령 위반 또는 회색지대의 경우, 관련 문서명, 조항번호, 조항내용과 함께 명확히 경고해야 합니다.
    1.3. 명확한 규정이 없을 경우, “내부 QA 검토 필요”임을 고지합니다.
2.응답 방식
    2.1. 존댓말을 사용하며, 전문적이고 친절한 어조로 답변합니다.
    2.2. 모든 답변은 논리적 구조, 높은 정확성, 실용성, 예시 및 설명 포함 등 전문가 수준을 유지합니다.
    2.3. 번역 시, 일반적인 번역체 대신 **한국 제약 산업 및 GMP 규정/가이드라인(MFDS, PIC/S, ICH 등)에서 통용되는 표준 전문 용어**를 사용하여 번역해야 합니다. (아래 '주요 번역 용어 가이드' 참고)
3.기능 제한 및 파일 처리
    3.1. 다루는 주제: SOP, GMP 가이드라인(FDA, PIC/S, EU-GMP, cGMP, MFDS 등), DI 규정, 외국 규정 번역 등 업무 관련 내용 및 사용자가 첨부한 파일의 내용 분석 (요약, 설명, 비교 등).
    3.2. 파일 첨부 시 처리:
        - 사용자가 파일을 첨부하여 질문하는 경우, 해당 파일 내용을 최우선으로 분석하고 참고하여 답변해야 합니다.
        - 사용자가 '전체 번역'을 명시적으로 요청하는 경우, 다른 모든 규칙(특히 간결성 규칙)에 우선하여 첨부된 문서의 내용을 처음부터 끝까지 순서대로 번역해야 합니다. 번역 결과는 모델의 최대 출력 토큰(16384 토큰) 내에서 생성되며, 내용이 길 경우 번역이 중간에 완료될 수 있습니다.
        - 번역 요청이 아니더라도, 파일 내용을 기반으로 질문에 맞춰 요약, 설명, 비교 등의 답변을 생성해야 합니다.
        - 만약 파일 내용이 기존 MFDS 규정이나 사내 SOP와 상충될 가능성이 있다면, 그 점을 명확히 언급하고 사용자에게 확인을 요청해야 합니다. (예: "첨부해주신 문서의 내용은 현재 SOP와 일부 차이가 있을 수 있습니다. 확인이 필요합니다.")
    3.3. 사용자가 파일을 첨부하고 해당 파일의 내용에 대해 질문하는 경우는 업무 관련 질문으로 간주합니다. 이 경우를 제외하고, 개인적인 질문, 뉴스, 여가 등 업무와 직접 관련 없는 질문은 “업무 관련 질문만 처리합니다.”로 간결히 응답합니다.
4.챗봇 소개 안내
    4.1. 사용자가 인사하거나 기능을 물을 경우, 본 챗봇의 역할("한국 제약 산업의 DI/GMP 규정 및 용어에 능통한 전문가 챗봇")과 처리 가능한 업무 범위를 간단히 소개합니다.
5.표현 및 형식 규칙
    5.1. Markdown 스타일 강조는 사용하지 않습니다.
    5.2. 번호 항목은 동일한 서식(글꼴 크기와 굵기)으로 통일합니다.
    5.3. 답변은 표, 요약, 핵심 정리 중심으로 간결하게 구성합니다. (단, '전체 번역' 요청 시에는 규칙 3.2가 우선)
6. 주요 번역 용어 가이드 (번역 시 최우선 참고)
    - Compliant / Compliance: 규정 준수
    - GxP: Good x Practice (GMP, GLP, GCP 등 우수 관리 기준)
    - Computerized System: 컴퓨터화 시스템
    - Risk-Based Approach: 위험 기반 접근법
    - Validation: 밸리데이션
    - Verification: 검증
    - Qualification: 적격성 평가
    - Commissioning: 커미셔닝
    - Specification: 규격 / 기준 / 명세서 (문맥에 맞게)
    - Design: 설계
    - Quality Risk Management (QRM): 품질 위험 관리 (ICH Q9 참고)
    - Quality by Design (QbD): 설계 기반 품질 고도화 (ICH Q8 참고)
    - Pharmaceutical Quality System (PQS): 의약품 품질 시스템 (ICH Q10 참고)
    - Process Validation (PV): 공정 밸리데이션
    - Process Analytical Technology (PAT): 공정 분석 기술
    - Critical Process Parameter (CPP): 중요 공정 변수
    - Standard Operating Procedure (SOP): 표준작업절차서
    - Good Manufacturing Practice (GMP): 의약품 제조 및 품질관리 기준
    - Regulatory: 규제 / 규정의
    - Authority / Agency: 규제 당국 / 기관 (예: FDA, EMA, MFDS)
    - Lifecycle: 수명 주기 / 라이프사이클
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
            return default_rules # 명시적으로 기본 규칙 반환
    else:
        print(f"WARNING: Prompt rules file not found at '{RULES_PATH_REPO}'. Using default rules defined in code.")
        return default_rules # 명시적으로 기본 규칙 반환

PROMPT_RULES_CONTENT = load_prompt_rules_cached()

# --- 텍스트 처리 함수들 ---
def extract_text_from_file(uploaded_file_obj):
    ext = os.path.splitext(uploaded_file_obj.name)[1].lower(); text_content = ""
    try:
        uploaded_file_obj.seek(0); file_bytes = uploaded_file_obj.read()
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
        else: st.warning(f"지원하지 않는 파일 형식입니다: {ext}"); return ""
    except Exception as e:
        st.error(f"'{uploaded_file_obj.name}' 파일 내용 추출 중 오류: {e}")
        print(f"Error extracting text from '{uploaded_file_obj.name}': {e}\n{traceback.format_exc()}")
        return ""
    return text_content.strip()

def chunk_text_into_pieces(text_to_chunk, chunk_size=500):
    if not text_to_chunk or not text_to_chunk.strip(): return [];
    chunks_list, current_buffer = [], ""
    for line in text_to_chunk.split("\n"):
        stripped_line = line.strip()
        if not stripped_line and not current_buffer.strip(): continue
        if len(current_buffer) + len(stripped_line) + 1 < chunk_size:
            current_buffer += stripped_line + "\n"
        else:
            if current_buffer.strip(): chunks_list.append(current_buffer.strip())
            current_buffer = stripped_line + "\n"
    if current_buffer.strip(): chunks_list.append(current_buffer.strip())
    return [c for c in chunks_list if c]

def get_text_embedding(text_to_embed):
    if not openai_client or not EMBEDDING_MODEL:
        print("ERROR: OpenAI client or embedding model not ready for get_text_embedding (called).")
        return None
    if not text_to_embed or not text_to_embed.strip(): return None
    print(f"Requesting embedding for text: '{text_to_embed[:30]}...'")
    try:
        response = openai_client.embeddings.create(input=[text_to_embed], model=EMBEDDING_MODEL, timeout=AZURE_OPENAI_TIMEOUT / 2)
        print("Embedding received.")
        return response.data[0].embedding
    except APITimeoutError:
        st.error("텍스트 임베딩 생성 중 시간 초과가 발생했습니다. 잠시 후 다시 시도해주세요.")
        print(f"TIMEOUT ERROR: Embedding request for '{text_to_embed[:50]}...' timed out.")
        return None
    except APIConnectionError as ace:
        st.error(f"텍스트 임베딩 생성 중 API 연결 오류가 발생했습니다: {ace}. 네트워크를 확인하거나 잠시 후 다시 시도해주세요.")
        print(f"API CONNECTION ERROR during embedding: {ace}")
        return None
    except RateLimitError as rle:
        st.error(f"텍스트 임베딩 생성 중 API 요청량 제한에 도달했습니다: {rle}. 잠시 후 다시 시도해주세요.")
        print(f"RATE LIMIT ERROR during embedding: {rle}")
        return None
    except APIStatusError as ase:
        st.error(f"텍스트 임베딩 생성 중 API 오류 (상태 코드 {ase.status_code}): {ase.message}. 문제가 지속되면 관리자에게 문의해주세요.")
        print(f"API STATUS ERROR during embedding (Status {ase.status_code}): {ase.message}")
        return None
    except Exception as e:
        st.error(f"텍스트 임베딩 생성 중 예기치 않은 오류가 발생했습니다: {e}")
        print(f"UNEXPECTED ERROR during embedding: {e}\n{traceback.format_exc()}")
        return None

def search_similar_chunks(query_text, k_results=3): # <<< k_results 기본값 수정 가능
    # <<< 추가된 디버깅 로그 시작 >>>
    print(f"DEBUG search_similar_chunks: Called with query '{query_text[:30]}...', k_results={k_results}")
    if index is None:
        print("DEBUG search_similar_chunks: FAISS index is None.")
        return []
    if index.ntotal == 0:
        print("DEBUG search_similar_chunks: FAISS index is empty (ntotal=0).")
        return []
    if not metadata:
        print("DEBUG search_similar_chunks: Metadata is empty.")
        return []
    # <<< 추가된 디버깅 로그 끝 >>>

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
        # <<< 추가된 디버깅 로그 시작 >>>
        print(f"DEBUG search_similar_chunks: FAISS search distances: {distances}")
        print(f"DEBUG search_similar_chunks: FAISS search indices_found: {indices_found}")
        # <<< 추가된 디버깅 로그 끝 >>>
        valid_indices = [i for i in indices_found[0] if 0 <= i < len(metadata)]
        results = [metadata[i]["content"] for i in valid_indices]
        print(f"Similarity search found {len(results)} relevant chunks.")
        # <<< 추가된 디버깅 로그 시작 >>>
        # for i, res_chunk in enumerate(results):
        #     print(f"DEBUG search_similar_chunks: Result chunk {i+1} (first 50 chars): {res_chunk[:50]}")
        # <<< 추가된 디버깅 로그 끝 >>>
        return results
    except Exception as e:
        st.error(f"유사도 검색 중 오류: {e}")
        print(f"ERROR: Similarity search failed: {e}\n{traceback.format_exc()}")
        return []


# --- 문서 추가, 원본 저장, 사용량 로깅 함수 ---
def add_document_to_vector_db_and_blob(uploaded_file_obj, text_content, text_chunks, _container_client):
    global index, metadata
    if not text_chunks: st.warning(f"'{uploaded_file_obj.name}' 파일에서 처리할 내용이 없습니다."); return False
    if not _container_client: st.error("Azure Blob 클라이언트가 준비되지 않아 학습 결과를 저장할 수 없습니다."); return False

    vectors_to_add, new_metadata_entries_for_current_file = [], []
    for chunk_idx, chunk in enumerate(text_chunks):
        print(f"Processing chunk {chunk_idx+1}/{len(text_chunks)} for embedding from '{uploaded_file_obj.name}'...")
        embedding = get_text_embedding(chunk)
        if embedding is not None:
            vectors_to_add.append(embedding)
            new_metadata_entries_for_current_file.append({"file_name": uploaded_file_obj.name, "content": chunk})
        else:
            print(f"Warning: Failed to get embedding for a chunk in '{uploaded_file_obj.name}'. Skipping chunk.")

    if not vectors_to_add: st.warning(f"'{uploaded_file_obj.name}' 파일에서 유효한 임베딩을 생성하지 못했습니다."); return False

    try:
        if vectors_to_add: index.add(np.array(vectors_to_add).astype("float32"))
        metadata.extend(new_metadata_entries_for_current_file)
        print(f"Added {len(vectors_to_add)} new chunks to in-memory DB from '{uploaded_file_obj.name}'. Index total: {index.ntotal}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_index_path = os.path.join(tmpdir, "temp.index")
            if index.ntotal > 0 :
                 faiss.write_index(index, temp_index_path)
                 if not save_binary_data_to_blob(temp_index_path, INDEX_BLOB_NAME, _container_client, "벡터 인덱스"):
                    st.error("벡터 인덱스 Blob 저장 실패"); return False
            else:
                print(f"Skipping saving empty index to Blob: {INDEX_BLOB_NAME}")

        if not save_data_to_blob(metadata, METADATA_BLOB_NAME, _container_client, "메타데이터"):
            st.error("메타데이터 Blob 저장 실패"); return False

        user_info = st.session_state.get("user", {}); uploader_name = user_info.get("name", "N/A")
        new_log_entry = {"file": uploaded_file_obj.name, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "chunks_added": len(vectors_to_add), "uploader": uploader_name}

        current_upload_logs = load_data_from_blob(UPLOAD_LOG_BLOB_NAME, _container_client, "업로드 로그", default_value=[])
        if not isinstance(current_upload_logs, list): current_upload_logs = []
        current_upload_logs.append(new_log_entry)
        if not save_data_to_blob(current_upload_logs, UPLOAD_LOG_BLOB_NAME, _container_client, "업로드 로그"):
            st.warning("업로드 로그를 Blob에 저장하는 데 실패했습니다.")
        return True
    except Exception as e:
        st.error(f"문서 학습 또는 Azure Blob 업로드 중 오류: {e}")
        print(f"ERROR: Failed to add document or upload to Blob: {e}\n{traceback.format_exc()}")
        return False

def save_original_file_to_blob(uploaded_file_obj, _container_client):
    if not _container_client: st.error("Azure Blob 클라이언트가 준비되지 않아 원본 파일을 저장할 수 없습니다."); return None
    try:
        uploaded_file_obj.seek(0)
        original_blob_name = f"uploaded_originals/{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file_obj.name}"
        blob_client_for_original = _container_client.get_blob_client(blob=original_blob_name)
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

def log_openai_api_usage_to_blob(user_id_str, model_name_str, usage_stats_obj, _container_client):
    if not _container_client:
        print("ERROR: Blob Container client is None for API usage log. Skipping log.")
        return

    prompt_tokens = getattr(usage_stats_obj, 'prompt_tokens', 0)
    completion_tokens = getattr(usage_stats_obj, 'completion_tokens', 0)
    total_tokens = getattr(usage_stats_obj, 'total_tokens', 0)

    new_log_entry = {
        "user_id": user_id_str, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": model_name_str, "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens, "total_tokens": total_tokens
    }

    current_usage_logs = load_data_from_blob(USAGE_LOG_BLOB_NAME, _container_client, "API 사용량 로그", default_value=[])
    if not isinstance(current_usage_logs, list): current_usage_logs = []
    current_usage_logs.append(new_log_entry)

    if not save_data_to_blob(current_usage_logs, USAGE_LOG_BLOB_NAME, _container_client, "API 사용량 로그"):
        print(f"WARNING: Failed to save API usage log to Blob for user '{user_id_str}'.")


# --- 메인 UI 구성 ---
tab_labels_list = ["💬 업무 질문"]
if current_user_info.get("role") == "admin":
    tab_labels_list.append("⚙️ 관리자 설정")

main_tabs_list = st.tabs(tab_labels_list)
chat_interface_tab = main_tabs_list[0]
admin_settings_tab = main_tabs_list[1] if len(main_tabs_list) > 1 else None

with chat_interface_tab:
    st.header("업무 질문")
    st.markdown("💡 예시: SOP 백업 주기, PIC/S Annex 11 차이 등")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        print("Chat messages list re-initialized in chat_tab (should not happen if init is correct).")

    for msg_item in st.session_state["messages"]:
        role, content, time_str = msg_item.get("role"), msg_item.get("content", ""), msg_item.get("time", "")
        align_class = "user-align" if role == "user" else "assistant-align"
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        st.markdown(f"""<div class="chat-bubble-container {align_class}"><div class="bubble {bubble_class}">{content}</div><div class="timestamp">{time_str}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("📂 파일 첨부/숨기기", key="toggle_chat_uploader_final_v4_button"):
        st.session_state.show_uploader = not st.session_state.get("show_uploader", False)

    chat_file_uploader_key = "chat_file_uploader_final_v4_widget"
    uploaded_chat_file_runtime = None
    if st.session_state.get("show_uploader", False):
        uploaded_chat_file_runtime = st.file_uploader("질문과 함께 참고할 파일 첨부 (선택 사항)",
                                     type=["pdf","docx","xlsx","xlsm","csv","pptx"],
                                     key=chat_file_uploader_key)
        if uploaded_chat_file_runtime: st.caption(f"첨부됨: {uploaded_chat_file_runtime.name}")

    with st.form("chat_input_form_final_v4", clear_on_submit=True):
        query_input_col, send_button_col = st.columns([4,1])
        with query_input_col:
            user_query_input = st.text_input("질문 입력:", placeholder="여기에 질문을 입력하세요...",
                                             key="user_query_text_input_final_v4", label_visibility="collapsed")
        with send_button_col:
            send_query_button = st.form_submit_button("전송")

    if send_query_button and user_query_input.strip():
        if not openai_client:
            st.error("OpenAI 서비스가 준비되지 않아 답변을 생성할 수 없습니다. 관리자에게 문의하세요.")
        elif not tokenizer: 
             st.error("Tiktoken 라이브러리 로드 실패. 답변을 생성할 수 없습니다.")
        else:
            timestamp_now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state["messages"].append({"role":"user", "content":user_query_input, "time":timestamp_now_str})

            user_id_for_log = current_user_info.get("name", "anonymous_chat_user_runtime")

            print(f"User '{user_id_for_log}' submitted query: '{user_query_input[:50]}...'")
            with st.spinner("답변 생성 중... 잠시만 기다려주세요."):
                assistant_response_content = "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
                try:
                    print("Step 1: Preparing context and calculating tokens...")
                    context_chunks_for_prompt = []
                    temp_file_text = ""

                    # --- 기본 프롬프트 및 질문 토큰 계산 ---
                    prompt_structure = f"{PROMPT_RULES_CONTENT}\n\n위의 규칙을 반드시 준수하여 답변해야 합니다. 다음은 사용자의 질문에 답변하는 데 참고할 수 있는 문서의 내용입니다:\n<문서 시작>\n{{context}}\n<문서 끝>"
                    base_prompt_text = prompt_structure.replace('{context}', '')
                    try:
                        base_tokens = len(tokenizer.encode(base_prompt_text))
                        query_tokens = len(tokenizer.encode(user_query_input))
                    except Exception as e:
                        st.error(f"기본 프롬프트 또는 질문 토큰화 중 오류: {e}")
                        raise 

                    print(f"DEBUG: 기본 프롬프트 구조 토큰: {base_tokens}")
                    print(f"DEBUG: 사용자 질문 토큰: {query_tokens}")

                    max_context_tokens = TARGET_INPUT_TOKENS_FOR_PROMPT - base_tokens - query_tokens
                    print(f"DEBUG: 목표 프롬프트 토큰: {TARGET_INPUT_TOKENS_FOR_PROMPT}")
                    print(f"DEBUG: 컨텍스트에 할당 가능한 최대 토큰: {max_context_tokens}")

                    if max_context_tokens <= 0:
                         st.warning("프롬프트 규칙과 질문만으로도 입력 토큰 제한에 도달했습니다. 첨부파일 내용을 포함할 수 없습니다.")
                         print("WARNING: No tokens left for context after accounting for rules and query.")
                         context_string_for_llm = "참고할 컨텍스트를 포함할 수 없습니다 (토큰 제한)."
                    else:
                        # --- <<< 핵심 수정: 컨텍스트 생성 로직 변경 시작 >>> ---
                        # 1. (항상) 사용자 질문을 기반으로 학습된 벡터 DB에서 관련 내용 검색
                        print(f"DEBUG: Retrieving context from Vector DB based on query: '{user_query_input[:50]}...'")
                        retrieved_chunks_from_db = search_similar_chunks(user_query_input, k_results=3) # k_results는 필요에 따라 조절
                        if retrieved_chunks_from_db:
                            context_chunks_for_prompt.extend(retrieved_chunks_from_db)
                            print(f"DEBUG: Retrieved {len(retrieved_chunks_from_db)} chunks from Vector DB.")
                        else:
                            print(f"DEBUG: No relevant chunks found in Vector DB for query.")

                        # 2. (선택적) 사용자가 채팅 중 파일을 첨부했다면, 해당 파일 내용도 컨텍스트에 추가
                        if uploaded_chat_file_runtime:
                            print(f"DEBUG: Processing additionally uploaded file in chat: {uploaded_chat_file_runtime.name}")
                            temp_file_text = extract_text_from_file(uploaded_chat_file_runtime)
                            if temp_file_text:
                                # 여기서는 파일 전체 내용을 추가하지만, 필요시 청킹 또는 요약 가능
                                context_chunks_for_prompt.append(temp_file_text)
                                print(f"DEBUG: Added content from additionally uploaded file (full text).")
                                print(f"DEBUG: Extracted text from uploaded file (first 100 chars): {temp_file_text[:100]}")
                            else:
                                st.info(f"채팅 중 첨부된 '{uploaded_chat_file_runtime.name}' 파일이 비어있거나 지원하지 않는 내용입니다.")
                        # --- <<< 핵심 수정: 컨텍스트 생성 로직 변경 끝 >>> ---

                        # --- 최종 컨텍스트 문자열 생성 및 토큰 기반 자르기 ---
                        final_unique_context = list(dict.fromkeys(c for c in context_chunks_for_prompt if c and c.strip()))
                        if not final_unique_context:
                            st.info("질문과 관련된 참고 정보를 찾지 못했습니다. 일반적인 답변을 시도합니다.")
                            context_string_for_llm = "현재 참고할 문서가 없습니다."
                        else:
                            full_context_string = "\n\n---\n\n".join(final_unique_context)
                            try:
                                full_context_tokens = tokenizer.encode(full_context_string)
                            except Exception as e:
                                st.error(f"컨텍스트 문자열 토큰화 중 오류: {e}")
                                raise

                            print(f"DEBUG: 전체 컨텍스트 문자열 토큰 수: {len(full_context_tokens)}")

                            if len(full_context_tokens) > max_context_tokens:
                                truncated_tokens = full_context_tokens[:max_context_tokens]
                                try:
                                    context_string_for_llm = tokenizer.decode(truncated_tokens)
                                except Exception as e:
                                     st.error(f"잘린 토큰 디코딩 중 오류: {e}")
                                     context_string_for_llm = "[오류: 컨텍스트 디코딩 실패]"
                                print(f"WARNING: 컨텍스트 토큰 수가 너무 많아 {max_context_tokens} 토큰으로 잘랐습니다.")
                                print(f"DEBUG: 잘린 컨텍스트 문자열 (앞 100자): {context_string_for_llm[:100]}")
                            else:
                                context_string_for_llm = full_context_string
                                print(f"DEBUG: 전체 컨텍스트를 사용합니다. (앞 100자): {context_string_for_llm[:100]}")


                    # --- 최종 시스템 프롬프트 구성 ---
                    system_prompt_content = prompt_structure.replace('{context}', context_string_for_llm)
                    try:
                        final_system_tokens = len(tokenizer.encode(system_prompt_content))
                        final_prompt_tokens = final_system_tokens + query_tokens
                    except Exception as e:
                         st.error(f"최종 시스템 프롬프트 토큰화 중 오류: {e}")
                         raise

                    print(f"DEBUG: 최종 시스템 프롬프트 토큰: {final_system_tokens}")
                    print(f"DEBUG: 최종 API 입력 토큰 (시스템+질문): {final_prompt_tokens}")
                    if final_prompt_tokens > MODEL_MAX_INPUT_TOKENS:
                         print(f"CRITICAL WARNING: 최종 입력 토큰({final_prompt_tokens})이 모델 최대치({MODEL_MAX_INPUT_TOKENS})를 초과했습니다! API 오류 가능성 높음.")

                    chat_messages = [{"role":"system", "content": system_prompt_content}, {"role":"user", "content": user_query_input}]

                    print("Step 2: Sending request to Azure OpenAI...")
                    chat_completion_response = openai_client.chat.completions.create(
                        model=st.secrets["AZURE_OPENAI_DEPLOYMENT"],
                        messages=chat_messages,
                        max_tokens=MODEL_MAX_OUTPUT_TOKENS,
                        temperature=0.1,
                        timeout=AZURE_OPENAI_TIMEOUT
                    )
                    assistant_response_content = chat_completion_response.choices[0].message.content.strip()
                    print("Azure OpenAI response received.")

                    if chat_completion_response.usage and container_client:
                        print("Logging OpenAI API usage...")
                        log_openai_api_usage_to_blob(user_id_for_log, st.secrets["AZURE_OPENAI_DEPLOYMENT"], chat_completion_response.usage, container_client)

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
                except APIStatusError as ase:
                    assistant_response_content = f"API에서 오류 응답을 받았습니다 (상태 코드 {ase.status_code}): {ase.message}. 문제가 지속되면 관리자에게 문의해주세요."
                    st.error(assistant_response_content)
                    print(f"API STATUS ERROR during chat completion (Status {ase.status_code}): {ase.message}")
                except Exception as gen_err:
                    assistant_response_content = f"답변 생성 중 예기치 않은 오류가 발생했습니다: {gen_err}. 관리자에게 문의해주세요."
                    st.error(assistant_response_content)
                    print(f"UNEXPECTED ERROR during response generation: {gen_err}\n{traceback.format_exc()}")

                st.session_state["messages"].append({"role":"assistant", "content":assistant_response_content, "time":timestamp_now_str})
                print("Response processing complete.")
            st.rerun()


if admin_settings_tab:
    with admin_settings_tab:
        st.header("⚙️ 관리자 설정")
        st.subheader("👥 가입 승인 대기자")
        if not USERS or not isinstance(USERS, dict):
            st.warning("사용자 정보를 로드할 수 없거나 형식이 올바르지 않습니다.")
            print(f"WARNING: USERS data is problematic or empty in admin tab. Type: {type(USERS)}")
        else:
            pending_approval_users = {uid:udata for uid,udata in USERS.items() if isinstance(udata, dict) and not udata.get("approved")}
            if pending_approval_users:
                for pending_uid, pending_user_data in pending_approval_users.items():
                    with st.expander(f"{pending_user_data.get('name','N/A')} ({pending_uid}) - {pending_user_data.get('department','N/A')}"):
                        approve_col, reject_col = st.columns(2)
                        if approve_col.button("승인", key=f"admin_approve_user_final_v4_{pending_uid}"):
                            USERS[pending_uid]["approved"] = True
                            if save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "사용자 정보"):
                                st.success(f"'{pending_uid}' 사용자를 승인하고 Blob에 저장했습니다."); st.rerun()
                            else: st.error("사용자 승인 정보 Blob 저장 실패.")
                        if reject_col.button("거절", key=f"admin_reject_user_final_v4_{pending_uid}"):
                            USERS.pop(pending_uid, None)
                            if save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "사용자 정보"):
                                st.info(f"'{pending_uid}' 사용자의 가입 신청을 거절하고 Blob에 저장했습니다."); st.rerun()
                            else: st.error("사용자 거절 정보 Blob 저장 실패.")
            else: st.info("승인 대기 중인 사용자가 없습니다.")
        st.markdown("---")

        st.subheader("📁 파일 업로드 및 학습 (Azure Blob Storage)")
        admin_file_uploader_key = "admin_file_uploader_final_v4_widget"
        admin_uploaded_file = st.file_uploader("학습할 파일 업로드", type=["pdf","docx","xlsx","xlsm","csv","pptx"], key=admin_file_uploader_key)

        if admin_uploaded_file and container_client:
            with st.spinner(f"'{admin_uploaded_file.name}' 파일 처리 및 학습 중..."):
                extracted_content = extract_text_from_file(admin_uploaded_file)
                if extracted_content:
                    content_chunks = chunk_text_into_pieces(extracted_content) 
                    if content_chunks:
                        original_file_blob_path = save_original_file_to_blob(admin_uploaded_file, container_client)
                        if original_file_blob_path: st.caption(f"원본 파일이 Blob에 '{original_file_blob_path}'로 저장되었습니다.")
                        else: st.warning("원본 파일을 Blob에 저장하는 데 실패했습니다.")

                        if add_document_to_vector_db_and_blob(admin_uploaded_file, extracted_content, content_chunks, container_client):
                            st.success(f"'{admin_uploaded_file.name}' 파일 학습 및 Azure Blob Storage에 업데이트 완료!")
                        else: st.error(f"'{admin_uploaded_file.name}' 학습 또는 Blob 업데이트 중 오류가 발생했습니다.")
                    else: st.warning(f"'{admin_uploaded_file.name}' 파일에서 유의미한 청크를 생성하지 못했습니다.")
                else: st.warning(f"'{admin_uploaded_file.name}' 파일이 비어있거나 지원하지 않는 내용입니다.")
            st.rerun() 
        elif admin_uploaded_file and not container_client:
            st.error("Azure Blob 클라이언트가 준비되지 않아 파일을 업로드하고 학습할 수 없습니다.")
        st.markdown("---")

        st.subheader("📊 API 사용량 모니터링 (Blob 로그 기반)")
        if container_client:
            usage_data_from_blob = load_data_from_blob(USAGE_LOG_BLOB_NAME, container_client, "API 사용량 로그", default_value=[])
            if usage_data_from_blob and isinstance(usage_data_from_blob, list) and len(usage_data_from_blob) > 0 :
                df_usage_stats=pd.DataFrame(usage_data_from_blob)
                for col in ["total_tokens", "prompt_tokens", "completion_tokens"]:
                     if col not in df_usage_stats.columns:
                         df_usage_stats[col] = 0

                token_cols = ["total_tokens", "prompt_tokens", "completion_tokens"]
                for col in token_cols:
                    df_usage_stats[col] = pd.to_numeric(df_usage_stats[col], errors='coerce').fillna(0)

                total_tokens_used = df_usage_stats["total_tokens"].sum()
                st.metric("총 API 호출 수", len(df_usage_stats))
                st.metric("총 사용 토큰 수", f"{int(total_tokens_used):,}")

                token_cost_per_unit = 0.0
                try: token_cost_per_unit=float(st.secrets.get("TOKEN_COST","0"))
                except (ValueError, TypeError): pass
                st.metric("예상 비용 (USD)", f"${total_tokens_used * token_cost_per_unit:.4f}")

                if "timestamp" in df_usage_stats.columns:
                    try:
                         df_usage_stats['timestamp'] = pd.to_datetime(df_usage_stats['timestamp'])
                         st.dataframe(df_usage_stats.sort_values(by="timestamp",ascending=False), use_container_width=True)
                    except Exception as e:
                         print(f"Warning: Could not sort usage log by timestamp due to conversion error: {e}")
                         st.dataframe(df_usage_stats, use_container_width=True)
                else:
                    st.dataframe(df_usage_stats, use_container_width=True)
            else: st.info("기록된 API 사용량 데이터가 Blob에 없거나 비어있습니다.")
        else: st.warning("Azure Blob 클라이언트가 준비되지 않아 API 사용량 모니터링을 표시할 수 없습니다.")
        st.markdown("---")

        st.subheader("📂 Azure Blob Storage 파일 목록 (최근 100개)")
        if container_client:
            try:
                blob_list_display = []
                count = 0
                max_blobs_to_show = 100
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
            except Exception as e:
                st.error(f"Azure Blob 파일 목록 조회 중 오류: {e}")
                print(f"ERROR listing blobs: {e}\n{traceback.format_exc()}")
        else:
            st.warning("Azure Blob 클라이언트가 준비되지 않아 파일 목록을 표시할 수 없습니다.")
