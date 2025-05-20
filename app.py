import streamlit as st
# st.set_page_config must be the first Streamlit command.
st.set_page_config(
    page_title="유앤생명과학 업무 가이드 봇",
    layout="centered",
    initial_sidebar_state="auto"
)

# 기본 라이브러리 먼저 임포트
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
from openai import AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient
import tempfile
from werkzeug.security import check_password_hash, generate_password_hash
import traceback
import base64
import tiktoken

# streamlit_cookies_manager는 사용하는 곳 근처 또는 여기서 임포트
# CookiesNotReady를 직접 임포트하지 않도록 수정
from streamlit_cookies_manager import EncryptedCookieManager
print("Imported streamlit_cookies_manager (EncryptedCookieManager only).")

try:
    tokenizer = tiktoken.get_encoding("o200k_base")
    print("Tiktoken 'o200k_base' encoder loaded successfully.")
except Exception as e:
    st.error(f"Tiktoken encoder load failed: {e}. Token-based length limit may not work.")
    print(f"ERROR: Failed to load tiktoken encoder: {e}")
    tokenizer = None

def get_base64_of_bin_file(bin_file_path):
    try:
        with open(bin_file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        print(f"ERROR: Logo file not found: {bin_file_path}")
        return None
    except Exception as e:
        print(f"ERROR: Processing logo file '{bin_file_path}': {e}")
        return None

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
BUFFER_TOKENS = 500
TARGET_INPUT_TOKENS_FOR_PROMPT = MODEL_MAX_INPUT_TOKENS - MODEL_MAX_OUTPUT_TOKENS - BUFFER_TOKENS
IMAGE_DESCRIPTION_MAX_TOKENS = 500
EMBEDDING_BATCH_SIZE = 16

st.markdown("""
<style>
    /* CSS 내용 생략 - 이전과 동일 */
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_azure_openai_client_cached():
    # 함수 내용 생략 - 이전과 동일
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
        st.error(f"Azure OpenAI config error: Missing key '{e.args[0]}' in secrets.")
        print(f"ERROR: Missing Azure OpenAI secret: {e.args[0]}")
        return None
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI client: {e}.")
        print(f"ERROR: Azure OpenAI client initialization failed: {e}\n{traceback.format_exc()}")
        return None

@st.cache_resource
def get_azure_blob_clients_cached():
    # 함수 내용 생략 - 이전과 동일
    print("Attempting to initialize Azure Blob Service client...")
    try:
        conn_str = st.secrets["AZURE_BLOB_CONN"]
        blob_service_client = BlobServiceClient.from_connection_string(conn_str, connection_timeout=60, read_timeout=120)
        container_name = st.secrets["BLOB_CONTAINER"]
        container_client = blob_service_client.get_container_client(container_name)
        print(f"Azure Blob Service client and container client for '{container_name}' initialized successfully.")
        return blob_service_client, container_client
    except KeyError as e:
        st.error(f"Azure Blob Storage config error: Missing key '{e.args[0]}' in secrets.")
        print(f"ERROR: Missing Azure Blob Storage secret: {e.args[0]}")
        return None, None
    except Exception as e:
        st.error(f"Error initializing Azure Blob client: {e}.")
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
        st.error("Missing 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT' in secrets.")
        print("ERROR: Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT secret.")
        openai_client = None
    except Exception as e:
        st.error(f"Error loading embedding model config: {e}")
        print(f"ERROR: Loading AZURE_OPENAI_EMBEDDING_DEPLOYMENT secret: {e}")
        openai_client = None

def load_data_from_blob(blob_name, _container_client, data_description="data", default_value=None):
    # 함수 내용 생략 - 이전과 동일
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
        st.warning(f"File '{data_description}' ({blob_name}) is corrupted or not valid JSON. Using default.")
        return default_value if default_value is not None else {}
    except AzureError as ae:
        print(f"AZURE ERROR loading '{data_description}' from Blob '{blob_name}': {ae}\n{traceback.format_exc()}")
        st.warning(f"Azure service error loading '{data_description}': {ae}. Using default.")
        return default_value if default_value is not None else {}
    except Exception as e:
        print(f"GENERAL ERROR loading '{data_description}' from Blob '{blob_name}': {e}\n{traceback.format_exc()}")
        st.warning(f"Unknown error loading '{data_description}': {e}. Using default.")
        return default_value if default_value is not None else {}

def save_data_to_blob(data_to_save, blob_name, _container_client, data_description="data"):
    # 함수 내용 생략 - 이전과 동일
    if not _container_client:
        st.error(f"Cannot save '{data_description}': Azure Blob client not ready.")
        print(f"ERROR: Blob Container client is None, cannot save '{blob_name}'.")
        return False
    print(f"Attempting to save '{data_description}' to Blob Storage: '{blob_name}'")
    try:
        if not isinstance(data_to_save, (dict, list)):
            st.error(f"Save failed for '{data_description}': Data is not JSON serializable (dict or list).")
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
        st.error(f"Azure service error saving '{data_description}' to Blob: {ae}")
        print(f"AZURE ERROR saving '{data_description}' to Blob '{blob_name}': {ae}\n{traceback.format_exc()}")
        return False
    except Exception as e:
        st.error(f"Unknown error saving '{data_description}' to Blob: {e}")
        print(f"GENERAL ERROR saving '{data_description}' to Blob '{blob_name}': {e}\n{traceback.format_exc()}")
        return False

def save_binary_data_to_blob(local_file_path, blob_name, _container_client, data_description="binary data"):
    # 함수 내용 생략 - 이전과 동일
    if not _container_client:
        st.error(f"Cannot save binary '{data_description}': Azure Blob client not ready.")
        print(f"ERROR: Blob Container client is None, cannot save binary '{blob_name}'.")
        return False
    if not os.path.exists(local_file_path):
        st.error(f"Local file for binary '{data_description}' not found: '{local_file_path}'")
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
        st.error(f"Azure service error saving binary '{data_description}' to Blob: {ae}")
        print(f"AZURE ERROR saving binary '{data_description}' to Blob '{blob_name}': {ae}\n{traceback.format_exc()}")
        return False
    except Exception as e:
        st.error(f"Unknown error saving binary '{data_description}' to Blob: {e}")
        print(f"GENERAL ERROR saving binary '{data_description}' to Blob '{blob_name}': {e}\n{traceback.format_exc()}")
        return False

USERS = {}
if container_client:
    USERS = load_data_from_blob(USERS_BLOB_NAME, container_client, "user info", default_value={})
    if not isinstance(USERS, dict) :
        print(f"ERROR: USERS loaded from blob is not a dict ({type(USERS)}). Re-initializing.")
        USERS = {}
    if "admin" not in USERS:
        print(f"'{USERS_BLOB_NAME}' from Blob is empty or admin is missing. Creating default admin.")
        USERS["admin"] = {
            "name": "관리자", "department": "품질보증팀",
            "password_hash": generate_password_hash(st.secrets.get("ADMIN_PASSWORD", "diteam_fallback_secret")),
            "approved": True, "role": "admin"
        }
        if not save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "initial user info"):
             st.warning("Failed to save default admin info to Blob. Will retry on next run.")
else:
    st.error("Azure Blob Storage connection failed. Cannot initialize user info. App may not function correctly.")
    print("CRITICAL: Cannot initialize USERS due to Blob client failure.")
    USERS = {"admin": {"name": "관리자(연결실패)", "department": "시스템", "password_hash": generate_password_hash("fallback"), "approved": True, "role": "admin"}}

cookies = None
cookie_manager_ready = False
print(f"Attempting to load COOKIE_SECRET from st.secrets: {st.secrets.get('COOKIE_SECRET')}")
try:
    cookie_secret_key = st.secrets.get("COOKIE_SECRET")
    if not cookie_secret_key:
        st.error("'COOKIE_SECRET' is not set or empty in st.secrets.")
        print("ERROR: COOKIE_SECRET is not set or empty in st.secrets.")
    else:
        cookies = EncryptedCookieManager(
            prefix="gmp_chatbot_auth_v5_4_import_fix/", # Prefix updated
            password=cookie_secret_key
        )
        try:
            if cookies.ready():
                cookie_manager_ready = True
                print("CookieManager is ready on initial setup try.")
            else:
                print("CookieManager not ready on initial setup try (may resolve on first interaction).")
        except Exception as e_ready_init: # Catches CookiesNotReady or other errors
            print(f"WARNING: cookies.ready() call during initial setup failed: {e_ready_init}")
            cookie_manager_ready = False
except Exception as e: # Errors during EncryptedCookieManager instantiation
    st.error(f"Unknown error creating cookie manager object: {e}")
    print(f"CRITICAL: CookieManager object creation error: {e}\n{traceback.format_exc()}")
    cookies = None
    cookie_manager_ready = False

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

    if cookies is not None and cookie_manager_ready:
        try:
            if cookies.ready():
                print("CookieManager is ready. Attempting to load cookies for session restore.")
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
                                print("User data in cookie is empty or invalid. Clearing auth state.")
                                st.session_state["authenticated"] = False
                                if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save(key="cookie_save_on_invalid_user_data_opt_v4_importfix")
                        except json.JSONDecodeError:
                            print("ERROR: Failed to decode user JSON from cookie. Clearing auth state.")
                            st.session_state["authenticated"] = False
                            if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save(key="cookie_save_on_json_decode_error_opt_v4_importfix")
                    else:
                        print("Session timeout detected from cookie. Clearing auth state.")
                        st.session_state["authenticated"] = False
                        st.session_state["messages"] = []
                        if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save(key="cookie_save_on_session_timeout_opt_v4_importfix")
                else:
                    print("Authenticated cookie not 'true'.")
                    st.session_state["authenticated"] = False
            else:
                print("CookieManager.ready() returned False during session init. Cannot load cookies.")
                st.session_state["authenticated"] = False
        except Exception as e_cookie_load: # Catch CookiesNotReady implicitly if it's raised by .get()
            print(f"Exception (possibly CookiesNotReady) during cookie operations in session init: {e_cookie_load}\n{traceback.format_exc()}")
            st.session_state["authenticated"] = False
    else:
        if cookies is None:
            print("Cookies object is None (COOKIE_SECRET missing or import failed). Cannot restore session.")
        else: 
            print("CookieManager not ready (initial check failed). Cannot restore session from cookies.")
        st.session_state["authenticated"] = False

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    print("Double check: Initializing messages (after auth block).")

if cookies is not None and not cookie_manager_ready:
    print("Attempting to make CookieManager ready once more before login UI check (if object exists)...")
    try:
        if cookies.ready():
            cookie_manager_ready = True
            print("CookieManager became ready just before login UI check (on this second attempt).")
        else:
            print("CookieManager still not ready just before login UI check (on this second attempt).")
    except Exception as e_ready_login_ui:
        print(f"WARNING: cookies.ready() call just before login UI check failed: {e_ready_login_ui}")

if not st.session_state.get("authenticated", False):
    st.markdown("""
    <div class="login-page-header-container">
      <span class="login-page-main-title">유앤생명과학 GMP/SOP 업무 가이드 봇</span>
      <span class="login-page-sub-title">Made by DI.PART</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="login-form-title">🔐 로그인 또는 회원가입</p>', unsafe_allow_html=True)

    if cookies is None or not cookie_manager_ready:
        st.warning("쿠키 시스템 초기화 중이거나 아직 준비되지 않았습니다. 잠시 후 새로고침하거나 다시 시도해주세요. 로그인이 유지되지 않을 수 있습니다.")

    with st.form("auth_form_final_v5_import_fix2", clear_on_submit=False): # Key updated
        mode = st.radio("선택", ["로그인", "회원가입"], key="auth_mode_final_v5_import_fix2")
        uid = st.text_input("ID", key="auth_uid_final_v5_import_fix2")
        pwd = st.text_input("비밀번호", type="password", key="auth_pwd_final_v5_import_fix2")
        name, dept = "", ""
        if mode == "회원가입":
            name = st.text_input("이름", key="auth_name_final_v5_import_fix2")
            dept = st.text_input("부서", key="auth_dept_final_v5_import_fix2")
        submit_button = st.form_submit_button("확인")

    if submit_button:
        if not uid or not pwd: st.error("ID and password are required.")
        elif mode == "회원가입" and (not name or not dept): st.error("Name and department are required for sign-up.")
        else:
            if mode == "로그인":
                user_data_login = USERS.get(uid)
                if not user_data_login: st.error("ID does not exist.")
                elif not user_data_login.get("approved", False): st.warning("Account pending approval.")
                elif check_password_hash(user_data_login["password_hash"], pwd):
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user_data_login
                    st.session_state["messages"] = []
                    print(f"Login successful for user '{uid}'. Chat messages cleared.")

                    if cookies is not None:
                        try:
                            if cookies.ready():
                                cookies["authenticated"] = "true"; cookies["user"] = json.dumps(user_data_login)
                                cookies["login_time"] = str(time.time()); cookies.save(key="cookie_save_on_login_opt_v4_importfix")
                                print(f"Cookies saved for user '{uid}'.")
                            else:
                                st.warning("Cookie system not ready at login, cannot save login state to browser.")
                                print("WARNING: CookieManager not ready during login (after .ready() check), cannot save cookies.")
                        except Exception as e_cookie_save_login:
                            st.warning(f"Problem saving login cookie: {e_cookie_save_login}")
                            print(f"ERROR: Failed to save login cookies: {e_cookie_save_login}")
                    else:
                        st.warning("Cookie system not initialized, cannot save login state.")
                        print("WARNING: CookieManager object is None during login, cannot save cookies.")

                    st.success(f"{user_data_login.get('name', uid)}님, 환영합니다!"); st.rerun()
                else: st.error("Incorrect password.")
            elif mode == "회원가입":
                if uid in USERS: st.error("ID already exists.")
                else:
                    USERS[uid] = {"name": name, "department": dept,
                                  "password_hash": generate_password_hash(pwd),
                                  "approved": False, "role": "user"}
                    if not save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "user info"):
                        st.error("Failed to save user information. Contact admin.")
                        USERS.pop(uid, None)
                    else:
                        st.success("Sign-up request complete! Login possible after admin approval.")
    st.stop()

current_user_info = st.session_state.get("user", {})

top_cols_main = st.columns([0.7, 0.3])
with top_cols_main[0]:
    if os.path.exists(COMPANY_LOGO_PATH_REPO):
        logo_b64 = get_base64_of_bin_file(COMPANY_LOGO_PATH_REPO)
        if logo_b64:
            st.markdown(f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_b64}" class="logo-image" width="150">
                <span class="version-text">ver 1.0.1 (Import Fix Attempt 3)</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="logo-container"><span class="version-text" style="font-weight:bold;">유앤생명과학</span> <span class="version-text" style="margin-left:10px;">ver 1.0.1 (Import Fix Attempt 3)</span></div>""", unsafe_allow_html=True)
    else:
        print(f"WARNING: Company logo file not found at {COMPANY_LOGO_PATH_REPO}")
        st.markdown(f"""<div class="logo-container"><span class="version-text" style="font-weight:bold;">유앤생명과학</span> <span class="version-text" style="margin-left:10px;">ver 1.0.1 (Import Fix Attempt 3)</span></div>""", unsafe_allow_html=True)


with top_cols_main[1]:
    st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
    if st.button("로그아웃", key="logout_button_final_v5_import_fix2"): # Key updated
        st.session_state["authenticated"] = False
        st.session_state["user"] = {}
        st.session_state["messages"] = []
        print("Logout successful. Chat messages cleared.")
        if cookies is not None:
            try:
                if cookies.ready():
                    cookies["authenticated"] = "false"
                    cookies["user"] = ""
                    cookies["login_time"] = ""
                    cookies.save(key="cookie_save_on_logout_opt_v4_importfix")
                    print("Cookies cleared on logout.")
                else:
                    print("WARNING: CookieManager not ready during logout, cannot clear cookies from browser storage.")
            except Exception as e_logout_cookie:
                 print(f"ERROR: Failed to clear cookies on logout: {e_logout_cookie}")
        else:
            print("WARNING: CookieManager object is None during logout, cannot clear cookies.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# 이하 코드는 이전과 동일 (메인 앱 제목, 벡터 DB 로드, 규칙 로드, 파일/이미지 처리, 채팅, 관리자 탭 등)
# ... (이전 답변에서 제공한 나머지 코드 전체를 여기에 붙여넣으세요) ...
# ... (main_tabs_list = st.tabs(...) 부터 코드 끝까지) ...

st.markdown("""
<div class="main-app-title-container">
  <span class="main-app-title">유앤생명과학 GMP/SOP 업무 가이드 봇</span>
  <span class="main-app-subtitle">Made by DI.PART</span>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vector_db_from_blob_cached(_container_client):
    # 함수 내용 생략 - 이전과 동일
    if not _container_client:
        print("ERROR: Blob Container client is None for load_vector_db_from_blob_cached.")
        return faiss.IndexFlatL2(1536), []
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
                            meta = []
                        else:
                            print(f"'{INDEX_BLOB_NAME}' loaded successfully from Blob Storage. Dimension: {idx.d}")
                    except Exception as e_faiss_read:
                        print(f"ERROR reading FAISS index: {e_faiss_read}. Re-initializing index.")
                        idx = faiss.IndexFlatL2(current_embedding_dimension)
                        meta = []
                else:
                    print(f"WARNING: '{INDEX_BLOB_NAME}' is empty in Blob. Using new index.")
                    idx = faiss.IndexFlatL2(current_embedding_dimension)
                    meta = []
            else:
                print(f"WARNING: '{INDEX_BLOB_NAME}' not found in Blob Storage. New index will be used/created.")
                idx = faiss.IndexFlatL2(current_embedding_dimension)
                meta = []

            if idx is not None:
                metadata_blob_client = _container_client.get_blob_client(METADATA_BLOB_NAME)
                if metadata_blob_client.exists() and (idx.ntotal > 0 or (index_blob_client.exists() and os.path.getsize(local_index_path) > 0) ):
                    print(f"Downloading '{METADATA_BLOB_NAME}'...")
                    with open(local_metadata_path, "wb") as download_file_meta:
                        download_stream_meta = metadata_blob_client.download_blob(timeout=60)
                        download_file_meta.write(download_stream_meta.readall())
                    if os.path.getsize(local_metadata_path) > 0 :
                        with open(local_metadata_path, "r", encoding="utf-8") as f_meta: meta = json.load(f_meta)
                    else:
                        meta = []
                        print(f"WARNING: '{METADATA_BLOB_NAME}' is empty in Blob.")
                elif idx.ntotal == 0 and not index_blob_client.exists():
                     print(f"INFO: Index is new and empty, starting with empty metadata.")
                     meta = []
                else:
                    print(f"INFO: Metadata file '{METADATA_BLOB_NAME}' not found or index is empty. Starting with empty metadata.")
                    meta = []

            if idx is not None and idx.ntotal == 0 and len(meta) > 0:
                print(f"INFO: FAISS index is empty (ntotal=0) but metadata is not. Clearing metadata for consistency.")
                meta = []
            elif idx is not None and idx.ntotal > 0 and not meta and index_blob_client.exists() and os.path.getsize(local_index_path) > 0 :
                print(f"CRITICAL WARNING: FAISS index has data (ntotal={idx.ntotal}) but metadata is empty. This may lead to errors.")

    except AzureError as ae:
        st.error(f"Azure service error loading vector DB from Blob: {ae}")
        print(f"AZURE ERROR loading vector DB from Blob: {ae}\n{traceback.format_exc()}")
        idx = faiss.IndexFlatL2(current_embedding_dimension); meta = []
    except Exception as e:
        st.error(f"Unknown error loading vector DB from Blob: {e}")
        print(f"GENERAL ERROR loading vector DB from Blob: {e}\n{traceback.format_exc()}")
        idx = faiss.IndexFlatL2(current_embedding_dimension); meta = []
    return idx, meta

index, metadata = faiss.IndexFlatL2(1536), []
if container_client:
    index, metadata = load_vector_db_from_blob_cached(container_client)
    print(f"DEBUG: FAISS index loaded after cache. ntotal: {index.ntotal if index else 'Index is None'}, dimension: {index.d if index else 'N/A'}")
    print(f"DEBUG: Metadata loaded after cache. Length: {len(metadata) if metadata is not None else 'Metadata is None'}")
else:
    st.error("Azure Blob Storage connection failed. Cannot load vector DB. File learning/search will be limited.")
    print("CRITICAL: Cannot load vector DB due to Blob client initialization failure (main section).")

@st.cache_data
def load_prompt_rules_cached():
    # 함수 내용 생략 - 이전과 동일
    default_rules = """1.Priority Criteria ... (생략) ..."""
    if os.path.exists(RULES_PATH_REPO):
        try:
            with open(RULES_PATH_REPO, "r", encoding="utf-8") as f: rules_content = f.read()
            print(f"Prompt rules loaded successfully from '{RULES_PATH_REPO}'.")
            return rules_content
        except Exception as e:
            st.warning(f"Error loading '{RULES_PATH_REPO}': {e}. Using default rules defined above.")
            print(f"WARNING: Error loading prompt rules from '{RULES_PATH_REPO}': {e}. Using default rules defined in code.")
            return default_rules
    else:
        print(f"WARNING: Prompt rules file not found at '{RULES_PATH_REPO}'. Using default rules defined in code.")
        return default_rules
PROMPT_RULES_CONTENT = load_prompt_rules_cached()

def extract_text_from_file(uploaded_file_obj):
    # 함수 내용 생략 - 이전과 동일 (디버그 프린트 포함)
    ext = os.path.splitext(uploaded_file_obj.name)[1].lower()
    text_content = ""

    if ext in [".png", ".jpg", ".jpeg"]:
        print(f"DEBUG extract_text_from_file: Skipped image file '{uploaded_file_obj.name}' (handled by description generation).")
        return ""

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
                 text_content += f"--- Sheet: {sheet_name} ---\n{sheet_df.to_string(index=False)}\n\n"
        elif ext == ".csv":
            with io.BytesIO(file_bytes) as csv_io:
                try: df = pd.read_csv(csv_io)
                except UnicodeDecodeError: df = pd.read_csv(csv_io, encoding='cp949')
                text_content = df.to_string(index=False)
        elif ext == ".pptx":
            with io.BytesIO(file_bytes) as ppt_io: prs = Presentation(ppt_io); text_content = "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
        elif ext == ".txt":
            try:
                uploaded_file_obj.seek(0)
                file_bytes_content = uploaded_file_obj.read()
                print(f"DEBUG TXT: Original bytes for {uploaded_file_obj.name}: {file_bytes_content!r}")
                text_content = file_bytes_content.decode('utf-8')
                print(f"DEBUG TXT: Decoded as UTF-8: {text_content!r}")
            except UnicodeDecodeError:
                print(f"DEBUG TXT: UTF-8 decode failed for {uploaded_file_obj.name}. Trying CP949.")
                try:
                    uploaded_file_obj.seek(0)
                    file_bytes_content_for_cp949 = uploaded_file_obj.read()
                    text_content = file_bytes_content_for_cp949.decode('cp949')
                    print(f"DEBUG TXT: Decoded as CP949: {text_content!r}")
                except Exception as e_txt_decode_cp949:
                    st.warning(f"TXT file '{uploaded_file_obj.name}' CP949 decode failed: {e_txt_decode_cp949}. Content empty.")
                    print(f"ERROR TXT: CP949 decode failed for {uploaded_file_obj.name}: {e_txt_decode_cp949}")
                    text_content = ""
            except Exception as e_txt_decode_utf8:
                st.warning(f"TXT file '{uploaded_file_obj.name}' UTF-8 decode failed with general error: {e_txt_decode_utf8}. Content empty.")
                print(f"ERROR TXT: UTF-8 decode failed for {uploaded_file_obj.name}: {e_txt_decode_utf8}")
                text_content = ""
        else:
            st.warning(f"Unsupported text file type: {ext} (File: {uploaded_file_obj.name})")
            return ""
    except Exception as e:
        st.error(f"Error extracting text from '{uploaded_file_obj.name}': {e}")
        print(f"Error extracting text from '{uploaded_file_obj.name}': {e}\n{traceback.format_exc()}")
        return ""
    return text_content.strip()

def chunk_text_into_pieces(text_to_chunk, chunk_size=500):
    # 함수 내용 생략 - 이전과 동일
    if not text_to_chunk or not text_to_chunk.strip(): return [];
    chunks_list, current_buffer = [], ""
    for line in text_to_chunk.split("\n"): 
        stripped_line = line.strip()
        if not stripped_line and not current_buffer.strip(): continue 
        
        if len(current_buffer) + len(stripped_line) + 1 < chunk_size: 
            current_buffer += stripped_line + "\n"
        else: 
            if current_buffer.strip(): 
                chunks_list.append(current_buffer.strip())
            current_buffer = stripped_line + "\n" 
            
    if current_buffer.strip(): 
        chunks_list.append(current_buffer.strip())
        
    return [c for c in chunks_list if c]

def get_image_description(image_bytes, image_filename, client_instance):
    # 함수 내용 생략 - 이전과 동일
    if not client_instance:
        st.error("OpenAI client not ready for image description.")
        print("ERROR get_image_description: OpenAI client not ready.")
        return None
    
    print(f"DEBUG get_image_description: Requesting description for image '{image_filename}'")
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        image_ext_desc = os.path.splitext(image_filename)[1].lower()
        mime_type = "image/jpeg" 
        if image_ext_desc == ".png":
            mime_type = "image/png"
        elif image_ext_desc == ".jpg" or image_ext_desc == ".jpeg":
            mime_type = "image/jpeg"

        vision_model_deployment = st.secrets["AZURE_OPENAI_DEPLOYMENT"] 
        print(f"DEBUG get_image_description: Using vision model deployment: {vision_model_deployment}")

        response = client_instance.chat.completions.create(
            model=vision_model_deployment,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Describe this image in detail from a work/professional perspective. This description will be used later for text-based search to find the image or understand the situation depicted. The image filename is '{image_filename}'. Mention key objects, states, possible contexts, and any elements relevant to GMP/SOP if applicable."},
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
            temperature=0.2, 
            timeout=AZURE_OPENAI_TIMEOUT 
        )
        description = response.choices[0].message.content.strip()
        print(f"DEBUG get_image_description: Description for '{image_filename}' (len: {len(description)} chars) generated successfully.")
        return description
    except APIStatusError as ase:
        st.error(f"API error during image description (Status {ase.status_code}): {ase.message}.")
        print(f"API STATUS ERROR during image description for '{image_filename}' (Status {ase.status_code}): {ase.message}")
        if ase.response and ase.response.content:
            try:
                error_details = json.loads(ase.response.content.decode('utf-8'))
                print(f"DEBUG get_image_description: Azure API error details: {json.dumps(error_details, indent=2, ensure_ascii=False)}")
            except Exception as json_e:
                print(f"DEBUG get_image_description: Could not parse Azure API error content as JSON: {json_e}")
        return None
    except APITimeoutError:
        st.error(f"Timeout generating description for image '{image_filename}'.")
        print(f"TIMEOUT ERROR during image description for '{image_filename}'.")
        return None
    except APIConnectionError as ace:
        st.error(f"API connection error generating description for image '{image_filename}': {ace}.")
        print(f"API CONNECTION ERROR during image description for '{image_filename}': {ace}")
        return None
    except RateLimitError as rle:
        st.error(f"API rate limit reached generating description for image '{image_filename}': {rle}.")
        print(f"RATE LIMIT ERROR during image description for '{image_filename}': {rle}")
        return None
    except Exception as e:
        st.error(f"Unexpected error generating description for image '{image_filename}': {e}")
        print(f"UNEXPECTED ERROR during image description for '{image_filename}': {e}\n{traceback.format_exc()}")
        return None

def get_text_embedding(text_to_embed, client=openai_client, model=EMBEDDING_MODEL):
    # 함수 내용 생략 - 이전과 동일
    if not client or not model:
        print("ERROR: OpenAI client or embedding model not ready for get_text_embedding.")
        return None
    if not text_to_embed or not text_to_embed.strip():
        print("WARNING: Attempted to embed empty or whitespace-only text.")
        return None
    try:
        response = client.embeddings.create(
            input=[text_to_embed],
            model=model,
            timeout=AZURE_OPENAI_TIMEOUT / 2
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Unexpected error during text embedding: {e}")
        print(f"UNEXPECTED ERROR during single text embedding: {e}\n{traceback.format_exc()}")
        return None

def get_batch_embeddings(texts_to_embed, client=openai_client, model=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE):
    # 함수 내용 생략 - 이전과 동일
    if not client or not model:
        print("ERROR: OpenAI client or embedding model not ready for get_batch_embeddings.")
        return []
    if not texts_to_embed:
        print("WARNING: No texts provided for batch embedding.")
        return []

    all_embeddings = []
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]
        if not batch: continue

        print(f"DEBUG get_batch_embeddings: Requesting embeddings for batch of {len(batch)} texts...")
        try:
            response = client.embeddings.create(
                input=batch,
                model=model,
                timeout=AZURE_OPENAI_TIMEOUT
            )
            embeddings_data = sorted(response.data, key=lambda e_item: e_item.index) 
            batch_embeddings = [item.embedding for item in embeddings_data]
            all_embeddings.extend(batch_embeddings)
            print(f"DEBUG get_batch_embeddings: Embeddings received for batch {i//batch_size + 1}.")
        except APIStatusError as ase:
            st.error(f"API error during batch text embedding (Status {ase.status_code}): {ase.message}.")
            print(f"API STATUS ERROR during batch embedding (batch starting with: '{batch[0][:30]}...'): {ase.message}")
            all_embeddings.extend([None] * len(batch))
        except Exception as e:
            st.error(f"Unexpected error during batch text embedding: {e}")
            print(f"UNEXPECTED ERROR during batch text embedding (batch starting with: '{batch[0][:30]}...'): {e}\n{traceback.format_exc()}")
            all_embeddings.extend([None] * len(batch))
            
    return all_embeddings

def search_similar_chunks(query_text, k_results=3):
    # 함수 내용 생략 - 이전과 동일
    if index is None or index.ntotal == 0 or not metadata:
        return []
    query_vector = get_text_embedding(query_text)
    if query_vector is None:
        return []
    try:
        actual_k = min(k_results, index.ntotal)
        if actual_k == 0 : return []

        distances, indices_found = index.search(np.array([query_vector]).astype("float32"), actual_k)
        results_with_source = []
        if len(indices_found[0]) > 0:
            for i_val in indices_found[0]:
                if 0 <= i_val < len(metadata):
                    meta_item = metadata[i_val]
                    if isinstance(meta_item, dict):
                        results_with_source.append({
                            "source": meta_item.get("file_name", "Unknown Source"),
                            "content": meta_item.get("content", ""),
                            "is_image_description": meta_item.get("is_image_description", False),
                            "original_file_extension": meta_item.get("original_file_extension", "")
                        })
        return results_with_source
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        print(f"ERROR: Similarity search failed: {e}\n{traceback.format_exc()}")
        return []

def add_document_to_vector_db_and_blob(uploaded_file_obj, processed_content, text_chunks, _container_client, is_image_description=False):
    # 함수 내용 생략 - 이전과 동일
    global index, metadata
    if not text_chunks: 
        st.warning(f"No content (text or image description) to process for '{uploaded_file_obj.name}'.")
        return False
    if not _container_client: 
        st.error("Cannot save learning results: Azure Blob client not ready.")
        return False

    file_type_for_log = "image description" if is_image_description else "text"
    print(f"Adding '{file_type_for_log}' from '{uploaded_file_obj.name}' to vector DB.")

    chunk_embeddings = get_batch_embeddings(text_chunks) 

    vectors_to_add = []
    new_metadata_entries_for_current_file = []
    successful_embeddings_count = 0

    for i, chunk in enumerate(text_chunks):
        embedding = chunk_embeddings[i] if i < len(chunk_embeddings) else None
        if embedding is not None:
            vectors_to_add.append(embedding)
            new_metadata_entries_for_current_file.append({
                "file_name": uploaded_file_obj.name, 
                "content": chunk, 
                "is_image_description": is_image_description,
                "original_file_extension": os.path.splitext(uploaded_file_obj.name)[1].lower()
            })
            successful_embeddings_count += 1
        else:
            print(f"Warning: Failed to get embedding for chunk {i+1} in '{uploaded_file_obj.name}'. Skipping.")

    if successful_embeddings_count == 0:
        st.error(f"Failed to generate any valid embeddings for '{uploaded_file_obj.name}' ({file_type_for_log}). Not learned.")
        return False
    if successful_embeddings_count < len(text_chunks):
         st.warning(f"Some content from '{uploaded_file_obj.name}' ({file_type_for_log}) failed embedding. Only successful parts learned.")
    
    try:
        current_embedding_dimension = np.array(vectors_to_add[0]).shape[0]
        if index is None or index.d != current_embedding_dimension:
            print(f"WARNING: FAISS index dimension mismatch or None. Re-initializing with dimension {current_embedding_dimension}.")
            index = faiss.IndexFlatL2(current_embedding_dimension)
            metadata = [] 

        if vectors_to_add: index.add(np.array(vectors_to_add).astype("float32"))
        metadata.extend(new_metadata_entries_for_current_file)
        print(f"Added {len(vectors_to_add)} new chunks from '{uploaded_file_obj.name}'. Index total: {index.ntotal}, Dim: {index.d}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_index_path = os.path.join(tmpdir, "temp.index")
            if index.ntotal > 0 : 
                 faiss.write_index(index, temp_index_path)
                 if not save_binary_data_to_blob(temp_index_path, INDEX_BLOB_NAME, _container_client, "vector index"):
                    st.error("Failed to save vector index to Blob."); return False 
            else: 
                print(f"Skipping saving empty index to Blob: {INDEX_BLOB_NAME}")
        
        if not save_data_to_blob(metadata, METADATA_BLOB_NAME, _container_client, "metadata"):
            st.error("Failed to save metadata to Blob."); return False

        user_info = st.session_state.get("user", {}); uploader_name = user_info.get("name", "N/A")
        new_log_entry = {"file": uploaded_file_obj.name, 
                         "type": "image" if is_image_description else "text_document",
                         "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "chunks_added": len(vectors_to_add), "uploader": uploader_name}

        current_upload_logs = load_data_from_blob(UPLOAD_LOG_BLOB_NAME, _container_client, "upload log", default_value=[])
        if not isinstance(current_upload_logs, list): current_upload_logs = [] 
        current_upload_logs.append(new_log_entry)
        if not save_data_to_blob(current_upload_logs, UPLOAD_LOG_BLOB_NAME, _container_client, "upload log"):
            st.warning("Failed to save upload log to Blob.") 
        return True
    except Exception as e:
        st.error(f"Error during document learning or Azure Blob upload: {e}")
        print(f"ERROR: Failed to add document or upload to Blob: {e}\n{traceback.format_exc()}")
        return False

def save_original_file_to_blob(uploaded_file_obj, _container_client):
    # 함수 내용 생략 - 이전과 동일
    if not _container_client: st.error("Cannot save original file: Azure Blob client not ready."); return None
    try:
        uploaded_file_obj.seek(0) 
        original_blob_name = f"uploaded_originals/{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file_obj.name}"
        blob_client_for_original = _container_client.get_blob_client(blob=original_blob_name)
        blob_client_for_original.upload_blob(uploaded_file_obj.getvalue(), overwrite=False, timeout=120) 
        print(f"Original file '{uploaded_file_obj.name}' saved to Blob as '{original_blob_name}'.")
        return original_blob_name
    except AzureError as ae:
        st.error(f"Azure service error saving original file '{uploaded_file_obj.name}' to Blob: {ae}")
        print(f"AZURE ERROR saving original file to Blob: {ae}\n{traceback.format_exc()}")
        return None
    except Exception as e:
        st.error(f"Unknown error saving original file '{uploaded_file_obj.name}' to Blob: {e}")
        print(f"GENERAL ERROR saving original file to Blob: {e}\n{traceback.format_exc()}")
        return None

def log_openai_api_usage_to_blob(user_id_str, model_name_str, usage_stats_obj, _container_client, request_type="chat_completion"):
    # 함수 내용 생략 - 이전과 동일
    if not _container_client:
        print("ERROR: Blob Container client is None for API usage log. Skipping log.")
        return

    prompt_tokens = getattr(usage_stats_obj, 'prompt_tokens', 0)
    completion_tokens = getattr(usage_stats_obj, 'completion_tokens', 0)
    total_tokens = getattr(usage_stats_obj, 'total_tokens', 0)

    new_log_entry = {
        "user_id": user_id_str, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": model_name_str, 
        "request_type": request_type, 
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens, "total_tokens": total_tokens
    }

    current_usage_logs = load_data_from_blob(USAGE_LOG_BLOB_NAME, _container_client, "API usage log", default_value=[])
    if not isinstance(current_usage_logs, list): current_usage_logs = []
    current_usage_logs.append(new_log_entry)

    if not save_data_to_blob(current_usage_logs, USAGE_LOG_BLOB_NAME, _container_client, "API usage log"):
        print(f"WARNING: Failed to save API usage log to Blob for user '{user_id_str}'.")

# --- 메인 UI 구성 ---
tab_labels_list = ["💬 업무 질문"]
if current_user_info.get("role") == "admin":
    tab_labels_list.append("⚙️ 관리자 설정")

main_tabs_list = st.tabs(tab_labels_list)
chat_interface_tab = main_tabs_list[0]
admin_settings_tab = main_tabs_list[1] if len(main_tabs_list) > 1 else None

# --- 채팅 인터페이스 탭 ---
with chat_interface_tab:
    # 함수 내용 생략 - 이전과 동일
    st.header("업무 질문")
    st.markdown("💡 예시: SOP 백업 주기, PIC/S Annex 11 차이, (파일 첨부 후) 이 사진 속 상황은 어떤 규정에 해당하나요? 등")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg_item in st.session_state["messages"]:
        role, content, time_str = msg_item.get("role"), msg_item.get("content", ""), msg_item.get("time", "")
        align_class = "user-align" if role == "user" else "assistant-align"
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        st.markdown(f"""<div class="chat-bubble-container {align_class}"><div class="bubble {bubble_class}">{content}</div><div class="timestamp">{time_str}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True) 
    if st.button("📂 파일 첨부/숨기기", key="toggle_chat_uploader_final_v5_import_fix_btn2"): 
        st.session_state.show_uploader = not st.session_state.get("show_uploader", False)

    chat_file_uploader_key = "chat_file_uploader_final_v5_import_fix_widget2" 
    uploaded_chat_file_runtime = None 
    if st.session_state.get("show_uploader", False):
        uploaded_chat_file_runtime = st.file_uploader("질문과 함께 참고할 파일 첨부 (선택 사항)",
                                     type=["pdf","docx","xlsx","xlsm","csv","pptx", "txt", "png", "jpg", "jpeg"], 
                                     key=chat_file_uploader_key)
        if uploaded_chat_file_runtime: 
            st.caption(f"첨부됨: {uploaded_chat_file_runtime.name} ({uploaded_chat_file_runtime.type}, {uploaded_chat_file_runtime.size} bytes)")
            if uploaded_chat_file_runtime.type.startswith("image/"):
                st.image(uploaded_chat_file_runtime, width=200)

    with st.form("chat_input_form_final_v5_import_fix2", clear_on_submit=True): 
        query_input_col, send_button_col = st.columns([4,1])
        with query_input_col:
            user_query_input = st.text_input("질문 입력:", placeholder="여기에 질문을 입력하세요...",
                                             key="user_query_text_input_final_v5_import_fix2", label_visibility="collapsed") 
        with send_button_col:
            send_query_button = st.form_submit_button("전송")

    if send_query_button and user_query_input.strip():
        if not openai_client:
            st.error("OpenAI service not ready. Cannot generate response. Contact admin.")
        elif not tokenizer: 
             st.error("Tiktoken library load failed. Cannot generate response.")
        else:
            timestamp_now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            user_message_content = user_query_input
            if uploaded_chat_file_runtime:
                user_message_content += f"\n(첨부 파일: {uploaded_chat_file_runtime.name})"
            st.session_state["messages"].append({"role":"user", "content":user_message_content, "time":timestamp_now_str})

            user_id_for_log = current_user_info.get("name", "anonymous_chat_user_runtime")
            print(f"User '{user_id_for_log}' submitted query: '{user_query_input[:50]}...' with file: {uploaded_chat_file_runtime.name if uploaded_chat_file_runtime else 'None'}")
            
            with st.spinner("답변 생성 중... 잠시만 기다려주세요."):
                assistant_response_content = "Error generating response. Please try again shortly."
                try: 
                    print("Step 1: Preparing context and calculating tokens...")
                    context_items_for_prompt = []
                    
                    text_from_chat_file = None
                    is_chat_file_image_description = False
                    chat_file_source_name_for_prompt = None

                    if uploaded_chat_file_runtime:
                        file_ext_chat = os.path.splitext(uploaded_chat_file_runtime.name)[1].lower()
                        is_image_chat = file_ext_chat in [".png", ".jpg", ".jpeg"]
                        
                        if is_image_chat:
                            print(f"DEBUG Chat: Processing uploaded image '{uploaded_chat_file_runtime.name}' for description.")
                            with st.spinner(f"Analyzing attached image '{uploaded_chat_file_runtime.name}'..."):
                                image_bytes_chat = uploaded_chat_file_runtime.getvalue()
                                description_chat = get_image_description(image_bytes_chat, uploaded_chat_file_runtime.name, openai_client)
                            if description_chat:
                                text_from_chat_file = description_chat
                                chat_file_source_name_for_prompt = f"User attached image: {uploaded_chat_file_runtime.name}" 
                                is_chat_file_image_description = True
                                print(f"DEBUG Chat: Image description generated. Length: {len(description_chat)}")
                            else:
                                st.warning(f"Failed to generate description for image '{uploaded_chat_file_runtime.name}'. File excluded from context.")
                        else: 
                            print(f"DEBUG Chat: Extracting text from uploaded file '{uploaded_chat_file_runtime.name}'.")
                            text_from_chat_file = extract_text_from_file(uploaded_chat_file_runtime)
                            if text_from_chat_file: 
                                chat_file_source_name_for_prompt = f"User attached file: {uploaded_chat_file_runtime.name}"
                                print(f"DEBUG Chat: Text extracted. Length: {len(text_from_chat_file)}")
                            elif text_from_chat_file == "": 
                                st.info(f"File '{uploaded_chat_file_runtime.name}' is empty or content could not be extracted. Excluded from context.")
                        
                        if text_from_chat_file: 
                            context_items_for_prompt.append({
                                "source": chat_file_source_name_for_prompt,
                                "content": text_from_chat_file,
                                "is_image_description": is_chat_file_image_description 
                            })
                    
                    prompt_structure = f"{PROMPT_RULES_CONTENT}\n\nStrictly adhere to the rules above. The following is document content to help answer the user's question:\n<Document Start>\n{{context}}\n<Document End>"
                    base_prompt_text = prompt_structure.replace('{context}', '')
                    try:
                        base_tokens = len(tokenizer.encode(base_prompt_text))
                        query_tokens = len(tokenizer.encode(user_query_input))
                    except Exception as e_tokenize_base:
                        st.error(f"Error tokenizing base prompt or query: {e_tokenize_base}")
                        raise 
                    
                    max_context_tokens = TARGET_INPUT_TOKENS_FOR_PROMPT - base_tokens - query_tokens
                    context_string_for_llm = "No reference documents currently available." 
                    if max_context_tokens <= 0:
                         st.warning("Input token limit reached by prompt rules and query alone. No additional context can be included.")
                         context_string_for_llm = "Cannot include context (token limit)."
                    else:
                        query_for_db_search = user_query_input
                        if is_chat_file_image_description and text_from_chat_file: 
                            query_for_db_search = f"{user_query_input}\n\nImage content: {text_from_chat_file}"
                        
                        retrieved_items_from_db = search_similar_chunks(query_for_db_search, k_results=3) 
                        if retrieved_items_from_db:
                            context_items_for_prompt.extend(retrieved_items_from_db) 
                        
                        if context_items_for_prompt:
                            seen_contents_for_final_context = set()
                            formatted_context_chunks = []
                            for item_idx, item in enumerate(context_items_for_prompt):
                                if isinstance(item, dict):
                                    content_value = item.get("content", "")
                                    source_info = item.get('source', f'Unknown Source {item_idx+1}')
                                    is_desc_item = item.get("is_image_description", False)
                                    
                                    content_strip = content_value.strip()
                                    if content_strip and content_strip not in seen_contents_for_final_context:
                                        final_source_display_name = source_info.replace("User attached image: ", "").replace("User attached file: ", "")
                                        
                                        if is_desc_item:
                                            formatted_context_chunks.append(f"[Image Description for: {final_source_display_name}]\n{content_value}")
                                        else:
                                            formatted_context_chunks.append(f"[Source: {final_source_display_name}]\n{content_value}")
                                        seen_contents_for_final_context.add(content_strip)
                                
                            if formatted_context_chunks:
                                full_context_string = "\n\n---\n\n".join(formatted_context_chunks)
                                try:
                                    full_context_tokens = tokenizer.encode(full_context_string)
                                except Exception as e_tokenize_full_ctx:
                                    st.error(f"Error tokenizing context string: {e_tokenize_full_ctx}")
                                    raise 

                                if len(full_context_tokens) > max_context_tokens:
                                    truncated_tokens = full_context_tokens[:max_context_tokens]
                                    try:
                                        context_string_for_llm = tokenizer.decode(truncated_tokens)
                                        if len(full_context_tokens) > len(truncated_tokens) : 
                                            context_string_for_llm += "\n(...more content, may be truncated.)"
                                    except Exception as e_decode_truncated:
                                        st.error(f"Error decoding truncated tokens: {e_decode_truncated}")
                                        context_string_for_llm = "[Error: Context decode failed]"
                                else:
                                    context_string_for_llm = full_context_string
                    
                    system_prompt_content = prompt_structure.replace('{context}', context_string_for_llm)
                    try:
                        final_system_tokens = len(tokenizer.encode(system_prompt_content))
                        final_prompt_tokens = final_system_tokens + query_tokens 
                    except Exception as e_tokenize_final_sys:
                         st.error(f"Error tokenizing final system prompt: {e_tokenize_final_sys}")
                         raise

                    if final_prompt_tokens > MODEL_MAX_INPUT_TOKENS:
                         print(f"CRITICAL WARNING: Final input tokens ({final_prompt_tokens}) exceed model max ({MODEL_MAX_INPUT_TOKENS})!")
                    
                    chat_messages_for_api = [{"role":"system", "content": system_prompt_content}, {"role":"user", "content": user_query_input}]

                    print("Step 2: Sending request to Azure OpenAI for chat completion...")
                    chat_completion_response = openai_client.chat.completions.create(
                        model=st.secrets["AZURE_OPENAI_DEPLOYMENT"], 
                        messages=chat_messages_for_api,
                        max_tokens=MODEL_MAX_OUTPUT_TOKENS, 
                        temperature=0.1, 
                        timeout=AZURE_OPENAI_TIMEOUT
                    )
                    assistant_response_content = chat_completion_response.choices[0].message.content.strip()
                    print("Azure OpenAI response received.")

                    if chat_completion_response.usage and container_client:
                        log_openai_api_usage_to_blob(user_id_for_log, st.secrets["AZURE_OPENAI_DEPLOYMENT"], chat_completion_response.usage, container_client, request_type="chat_completion_with_rag")
                
                except Exception as gen_err: 
                    assistant_response_content = f"Unexpected error during response generation: {gen_err}. Contact admin."
                    st.error(assistant_response_content)
                    print(f"UNEXPECTED ERROR during response generation: {gen_err}\n{traceback.format_exc()}")

                st.session_state["messages"].append({"role":"assistant", "content":assistant_response_content, "time":timestamp_now_str})
                print("Response processing complete.")
            st.rerun()

# --- 관리자 설정 탭 ---
if admin_settings_tab:
    with admin_settings_tab:
        # 함수 내용 생략 - 이전과 동일
        st.header("⚙️ 관리자 설정")
        st.subheader("👥 가입 승인 대기자")
        if not USERS or not isinstance(USERS, dict):
            st.warning("Cannot load user info or format is incorrect.")
        else:
            pending_approval_users = {uid:udata for uid,udata in USERS.items() if isinstance(udata, dict) and not udata.get("approved")}
            if pending_approval_users:
                for pending_uid, pending_user_data in pending_approval_users.items():
                    with st.expander(f"{pending_user_data.get('name','N/A')} ({pending_uid}) - {pending_user_data.get('department','N/A')}"):
                        approve_col, reject_col = st.columns(2)
                        if approve_col.button("승인", key=f"admin_approve_user_final_v5_import_fix2_{pending_uid}"): 
                            USERS[pending_uid]["approved"] = True
                            if save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "user info"):
                                st.success(f"User '{pending_uid}' approved and saved to Blob."); st.rerun()
                            else: st.error("Failed to save user approval to Blob.")
                        if reject_col.button("거절", key=f"admin_reject_user_final_v5_import_fix2_{pending_uid}"): 
                            USERS.pop(pending_uid, None)
                            if save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "user info"):
                                st.info(f"User '{pending_uid}' rejected and saved to Blob."); st.rerun()
                            else: st.error("Failed to save user rejection to Blob.")
            else: st.info("No users pending approval.")
        st.markdown("---")

        st.subheader("📁 파일 업로드 및 학습 (Azure Blob Storage)")
        if 'processed_admin_file_info' not in st.session_state:
            st.session_state.processed_admin_file_info = None

        def clear_processed_file_info_on_admin_upload_change():
            st.session_state.processed_admin_file_info = None

        admin_file_uploader_key = "admin_file_uploader_v_final_import_fix2" 
        admin_uploaded_file = st.file_uploader(
            "학습할 파일 업로드 (PDF, DOCX, XLSX, CSV, PPTX, TXT, PNG, JPG, JPEG)",
            type=["pdf","docx","xlsx","xlsm","csv","pptx", "txt", "png", "jpg", "jpeg"], 
            key=admin_file_uploader_key,
            on_change=clear_processed_file_info_on_admin_upload_change,
            accept_multiple_files=False 
        )

        if admin_uploaded_file and container_client:
            current_file_info = (admin_uploaded_file.name, admin_uploaded_file.size, admin_uploaded_file.type)
            if st.session_state.processed_admin_file_info != current_file_info:
                print(f"DEBUG Admin Upload: New file detected. File Info: {current_file_info}")
                try: 
                    file_ext_admin = os.path.splitext(admin_uploaded_file.name)[1].lower()
                    is_image_admin_upload = file_ext_admin in [".png", ".jpg", ".jpeg"]
                    content_for_learning = None
                    is_img_desc_for_learning = False

                    if is_image_admin_upload:
                        with st.spinner(f"Processing image '{admin_uploaded_file.name}' and generating description..."):
                            img_bytes_admin = admin_uploaded_file.getvalue()
                            description_admin = get_image_description(img_bytes_admin, admin_uploaded_file.name, openai_client)
                        if description_admin:
                            content_for_learning = description_admin
                            is_img_desc_for_learning = True
                            st.info(f"Description generated for image '{admin_uploaded_file.name}' (Length: {len(description_admin)}). This description will be learned.")
                            st.text_area("Generated Image Description (for learning)", description_admin, height=150, disabled=True)
                        else:
                            st.error(f"Failed to generate description for image '{admin_uploaded_file.name}'. Excluded from learning.")
                    else: 
                        with st.spinner(f"Extracting text from '{admin_uploaded_file.name}'..."):
                            content_for_learning = extract_text_from_file(admin_uploaded_file)
                        if content_for_learning:
                            st.info(f"Text extracted from '{admin_uploaded_file.name}' (Length: {len(content_for_learning)}).")
                        else: 
                            st.warning(f"Could not extract content from '{admin_uploaded_file.name}' or it is empty. Excluded from learning.")
                    
                    if content_for_learning: 
                        with st.spinner(f"Processing and learning content from '{admin_uploaded_file.name}'..."):
                            content_chunks_for_learning = chunk_text_into_pieces(content_for_learning)
                            if content_chunks_for_learning:
                                original_file_blob_path = save_original_file_to_blob(admin_uploaded_file, container_client)
                                if original_file_blob_path: 
                                    st.caption(f"Original file '{admin_uploaded_file.name}' saved to Blob as '{original_file_blob_path}'.")
                                else: 
                                    st.warning(f"Failed to save original file '{admin_uploaded_file.name}' to Blob.")

                                if add_document_to_vector_db_and_blob(
                                    admin_uploaded_file, 
                                    content_for_learning, 
                                    content_chunks_for_learning, 
                                    container_client, 
                                    is_image_description=is_img_desc_for_learning
                                ):
                                    st.success(f"File '{admin_uploaded_file.name}' learned and updated to Azure Blob Storage successfully!")
                                    st.session_state.processed_admin_file_info = current_file_info 
                                    st.rerun() 
                                else:
                                    st.error(f"Error during learning or Blob update for '{admin_uploaded_file.name}'.")
                                    st.session_state.processed_admin_file_info = None 
                            else: 
                                st.warning(f"No meaningful learning chunks generated for '{admin_uploaded_file.name}'.")
                except Exception as e_admin_file_proc:
                    st.error(f"An unexpected error occurred processing file {admin_uploaded_file.name} in admin upload: {e_admin_file_proc}")
                    print(f"CRITICAL ERROR in admin_upload_processing for {admin_uploaded_file.name}: {e_admin_file_proc}\n{traceback.format_exc()}")
                    st.session_state.processed_admin_file_info = None

            elif st.session_state.processed_admin_file_info == current_file_info:
                 st.caption(f"File '{admin_uploaded_file.name}' was previously processed. Upload a different file or remove and re-upload to re-learn.")
        elif admin_uploaded_file and not container_client:
            st.error("Cannot upload and learn file: Azure Blob client not ready.")
        st.markdown("---")

        st.subheader("📊 API 사용량 모니터링 (Blob 로그 기반)")
        # 함수 내용 생략 - 이전과 동일
        if container_client:
            usage_data_from_blob = load_data_from_blob(USAGE_LOG_BLOB_NAME, container_client, "API usage log", default_value=[])
            if usage_data_from_blob and isinstance(usage_data_from_blob, list) and len(usage_data_from_blob) > 0 :
                df_usage_stats=pd.DataFrame(usage_data_from_blob)
                
                for col in ["total_tokens", "prompt_tokens", "completion_tokens"]:
                     if col not in df_usage_stats.columns: df_usage_stats[col] = 0
                if "request_type" not in df_usage_stats.columns: df_usage_stats["request_type"] = "unknown"

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
                    except Exception as e_sort_ts:
                         print(f"Warning: Could not sort usage log by timestamp: {e_sort_ts}")
                         st.dataframe(df_usage_stats, use_container_width=True) 
                else: 
                    st.dataframe(df_usage_stats, use_container_width=True)
            else: st.info("No API usage data recorded in Blob or data is empty.")
        else: st.warning("Cannot display API usage monitoring: Azure Blob client not ready.")
        st.markdown("---")

        st.subheader("📂 Azure Blob Storage 파일 목록 (최근 100개)")
        # 함수 내용 생략 - 이전과 동일
        if container_client:
            try:
                blob_list_display = []
                count = 0
                max_blobs_to_show = 100 
                blobs_sorted = sorted(container_client.list_blobs(), key=lambda b: b.last_modified, reverse=True)

                for blob_item in blobs_sorted:
                    if count >= max_blobs_to_show: break
                    blob_list_display.append({
                        "파일명": blob_item.name,
                        "크기 (bytes)": blob_item.size,
                        "수정일": blob_item.last_modified.strftime('%Y-%m-%d %H:%M:%S') if blob_item.last_modified else 'N/A'
                    })
                    count += 1
                
                if blob_list_display:
                    df_blobs_display = pd.DataFrame(blob_list_display)
                    st.dataframe(df_blobs_display, use_container_width=True)
                else: st.info("No files in Azure Blob Storage.")
            except AzureError as ae_list_blob: 
                 st.error(f"Azure service error listing Blob files: {ae_list_blob}")
                 print(f"AZURE ERROR listing blobs: {ae_list_blob}\n{traceback.format_exc()}")
            except Exception as e_list_blob:
                st.error(f"Unknown error listing Azure Blob files: {e_list_blob}")
                print(f"ERROR listing blobs: {e_list_blob}\n{traceback.format_exc()}")
        else:
            st.warning("Cannot display file list: Azure Blob client not ready.")
