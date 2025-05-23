import streamlit as st
# st.set_page_config must be the first Streamlit command.
st.set_page_config(
    page_title="유앤생명과학 업무 가이드 봇",
    layout="centered",
    initial_sidebar_state="auto"
)

import os
import io
import fitz # PyMuPDF
import pandas as pd
import docx
from pptx import Presentation
import faiss
import openai
import numpy as np
import json
import time
from datetime import datetime
import uuid # 고유 ID 생성을 위해 추가
from openai import AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient
import tempfile
from werkzeug.security import check_password_hash, generate_password_hash
import traceback
import base64
import tiktoken
import re # For comment removal

from streamlit_cookies_manager import EncryptedCookieManager
print("Imported streamlit_cookies_manager (EncryptedCookieManager only).")


try:
    tokenizer = tiktoken.get_encoding("o200k_base")
    print("Tiktoken 'o200k_base' encoder loaded successfully.")
except Exception as e:
    st.error(f"Tiktoken encoder load failed: {e}. Token-based length limit may not work.")
    print(f"ERROR: Failed to load tiktoken encoder: {e}")
    tokenizer = None

APP_VERSION = "1.0.6 (Chat History Feature)" # 버전 업데이트

RULES_PATH_REPO = ".streamlit/prompt_rules.txt"
COMPANY_LOGO_PATH_REPO = "company_logo.png"
INDEX_BLOB_NAME = "vector_db/vector.index"
METADATA_BLOB_NAME = "vector_db/metadata.json"
USERS_BLOB_NAME = "app_data/users.json"
UPLOAD_LOG_BLOB_NAME = "app_logs/upload_log.json"
USAGE_LOG_BLOB_NAME = "app_logs/usage_log.json"
CHAT_HISTORY_BASE_PATH = "chat_histories/" # 대화 내역 저장 기본 경로

AZURE_OPENAI_TIMEOUT = 60.0
MODEL_MAX_INPUT_TOKENS = 128000
MODEL_MAX_OUTPUT_TOKENS = 16384
BUFFER_TOKENS = 500
TARGET_INPUT_TOKENS_FOR_PROMPT = MODEL_MAX_INPUT_TOKENS - MODEL_MAX_OUTPUT_TOKENS - BUFFER_TOKENS
IMAGE_DESCRIPTION_MAX_TOKENS = 500
EMBEDDING_BATCH_SIZE = 16

# --- 대화 내역 관련 함수 ---
def get_current_user_login_id():
    user_info = st.session_state.get("user", {})
    return user_info.get("uid") # 로그인 시 저장한 uid 사용

def get_user_chat_history_blob_name(user_login_id):
    if not user_login_id:
        return None
    return f"{CHAT_HISTORY_BASE_PATH}{user_login_id}_history.json"

def load_user_conversations_from_blob():
    user_login_id = get_current_user_login_id()
    if not user_login_id or not container_client:
        print(f"Cannot load chat history: User ID ({user_login_id}) or container_client is missing.")
        return []
    blob_name = get_user_chat_history_blob_name(user_login_id)
    history_data = load_data_from_blob(blob_name, container_client, f"chat history for {user_login_id}", default_value={"conversations": []})
    loaded_conversations = history_data.get("conversations", [])
    print(f"Loaded {len(loaded_conversations)} conversations for user '{user_login_id}'.")
    # 날짜/시간 필드를 datetime 객체로 변환하거나, 정렬을 위해 필요시 처리 (현재는 문자열로 유지)
    # 최신순으로 정렬 (last_updated 기준)
    try:
        loaded_conversations.sort(key=lambda x: x.get("last_updated", "1970-01-01T00:00:00"), reverse=True)
    except Exception as e_sort:
        print(f"Error sorting conversations: {e_sort}") # 정렬 실패 시 원본 순서 유지
    return loaded_conversations


def save_user_conversations_to_blob():
    user_login_id = get_current_user_login_id()
    if not user_login_id or not container_client or "all_user_conversations" not in st.session_state:
        print(f"Cannot save chat history: User ID ({user_login_id}), container_client, or all_user_conversations missing.")
        return False
    
    # 저장 전 최신순으로 다시 정렬 (last_updated 기준)
    try:
        st.session_state.all_user_conversations.sort(key=lambda x: x.get("last_updated", "1970-01-01T00:00:00"), reverse=True)
    except Exception as e_sort_save:
        print(f"Error sorting conversations before saving: {e_sort_save}")

    blob_name = get_user_chat_history_blob_name(user_login_id)
    print(f"Saving {len(st.session_state.all_user_conversations)} conversations for user '{user_login_id}' to {blob_name}.")
    return save_data_to_blob({"conversations": st.session_state.all_user_conversations}, blob_name, container_client, f"chat history for {user_login_id}")

def generate_conversation_title(messages_list):
    if not messages_list:
        return "빈 대화"
    for msg in messages_list:
        if msg.get("role") == "user" and msg.get("content","").strip():
            # "(첨부 파일: ...)" 부분 제외
            title_candidate = msg["content"].split("\n(첨부 파일:")[0].strip()
            return title_candidate[:30] + "..." if len(title_candidate) > 30 else title_candidate
    return "대화 시작"


def archive_current_chat_session_if_needed():
    user_login_id = get_current_user_login_id()
    # 현재 메시지가 없거나, 사용자가 없으면 아카이브할 필요 없음
    if not user_login_id or not st.session_state.get("current_chat_messages"):
        print("Archive check: No user ID or no current messages. Skipping archive.")
        return False

    active_id = st.session_state.get("active_conversation_id")
    current_messages_copy = list(st.session_state.current_chat_messages) # 항상 복사본 사용
    
    archived_or_updated = False

    if active_id: # 현재 불러온 대화가 있는 경우 (업데이트 시도)
        found_and_updated = False
        for i, conv in enumerate(st.session_state.all_user_conversations):
            if conv["id"] == active_id:
                # 메시지 내용이 실제로 변경되었는지 간단히 확인 (더 정교한 비교도 가능)
                if conv["messages"] != current_messages_copy:
                    conv["messages"] = current_messages_copy
                    conv["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # 제목은 첫 메시지 기준으로 생성되었으므로, 일반적으로는 업데이트하지 않음.
                    # 필요하다면 여기서 conv["title"] = generate_conversation_title(current_messages_copy) 추가
                    st.session_state.all_user_conversations[i] = conv # 리스트 내 객체 직접 수정 반영
                    print(f"Archived (updated) conversation ID: {active_id}, Title: '{conv['title']}'")
                    archived_or_updated = True
                else:
                    print(f"Conversation ID: {active_id} has no changes to messages. No update to archive.")
                found_and_updated = True
                break
        if not found_and_updated: # active_id가 있었지만 목록에 없는 이상한 경우 (새 대화로 처리)
             print(f"Warning: active_conversation_id '{active_id}' not found in log. Treating as new chat for archiving.")
             active_id = None # 새 대화로 취급하도록 active_id 초기화

    if not active_id: # 새 대화이거나, 위에서 active_id가 None으로 바뀐 경우
        # current_chat_messages가 실제로 내용이 있어야 새 대화로 저장
        if current_messages_copy:
            new_conv_id = str(uuid.uuid4()) # 고유 ID 생성
            title = generate_conversation_title(current_messages_copy)
            timestamp_str = current_messages_copy[0].get("time", datetime.now().strftime("%Y-%m-%d %H:%M"))

            new_conversation = {
                "id": new_conv_id,
                "title": title,
                "timestamp": timestamp_str, # 대화 시작 시점 (첫 메시지 시간)
                "messages": current_messages_copy,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.all_user_conversations.insert(0, new_conversation) # 최신 대화를 맨 앞에 추가
            # st.session_state.active_conversation_id = new_conv_id # 이 함수 호출 후 active_id는 보통 None이나 다른 값으로 바뀜
            print(f"Archived (new) conversation ID: {new_conv_id}, Title: '{title}'")
            archived_or_updated = True
        else: # current_messages_copy가 비어있으면 새 대화로 저장할 내용 없음
             print("Archive check: Current messages empty and no active_id. Skipping archive of new chat.")


    if archived_or_updated:
        save_user_conversations_to_blob()
    
    return archived_or_updated
# --- END 대화 내역 관련 함수 ---

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

def get_logo_and_version_html(app_version_str):
    logo_html_part = ""
    company_name_default = '<span class="version-text" style="font-weight:bold; font-size: 1.5em;">유앤생명과학</span>'
    
    if os.path.exists(COMPANY_LOGO_PATH_REPO):
        logo_b64 = get_base64_of_bin_file(COMPANY_LOGO_PATH_REPO)
        if logo_b64:
            logo_html_part = f'<img src="data:image/png;base64,{logo_b64}" class="logo-image" width="150" style="vertical-align: middle;">'
        else:
            logo_html_part = company_name_default
    else:
        print(f"WARNING: Company logo file not found at {COMPANY_LOGO_PATH_REPO}")
        logo_html_part = company_name_default
        
    return f"""
        {logo_html_part}
        <span class="version-text" style="vertical-align: middle; margin-left: 10px;">{app_version_str}</span>
    """

st.markdown("""
<style>
    /* 기본 CSS 스타일 */
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
    .login-page-header-container { text-align: center; margin-top: 10px; margin-bottom: 10px;}
    .login-page-main-title { font-size: 1.8rem; font-weight: bold; display: block; color: #333F48; } 
    .login-page-sub-title { font-size: 0.85rem; color: gray; display: block; margin-top: 2px; margin-bottom: 20px;}
    .login-form-title { 
        font-size: 1.6rem; 
        font-weight: bold;
        text-align: center;
        margin-top: 10px; 
        margin-bottom: 25px; 
    }

    /* 모바일 화면 대응 */
    @media (max-width: 768px) {
        .main-app-title { font-size: 1.8rem; }
        .main-app-subtitle { font-size: 0.8rem; }
        .login-page-main-title { font-size: 1.5rem; }
        .login-page-sub-title { font-size: 0.8rem; }
        .login-form-title { font-size: 1.3rem; margin-bottom: 20px; }
    }
</style>
""", unsafe_allow_html=True)

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
        st.error(f"Azure OpenAI config error: Missing key '{e.args[0]}' in secrets.")
        print(f"ERROR: Missing Azure OpenAI secret: {e.args[0]}")
        return None
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI client: {e}.")
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
    if not _container_client:
        print(f"ERROR: Blob Container client is None for load_data_from_blob ('{data_description}'). Returning default.")
        return default_value if default_value is not None else {}
    print(f"Attempting to load '{data_description}' from Blob: '{blob_name}'")
    try:
        blob_client_instance = _container_client.get_blob_client(blob_name)
        if blob_client_instance.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                local_temp_path = os.path.join(tmpdir, os.path.basename(blob_name))
                with open(local_temp_path, "wb") as download_file:
                    download_stream = blob_client_instance.download_blob(timeout=60)
                    download_file.write(download_stream.readall())
                if os.path.getsize(local_temp_path) > 0:
                    with open(local_temp_path, "r", encoding="utf-8") as f:
                        loaded_data = json.load(f)
                    print(f"Successfully loaded '{data_description}' from Blob: '{blob_name}'")
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
    if not _container_client:
        st.error(f"Cannot save '{data_description}': Azure Blob client not ready.")
        print(f"ERROR: Blob Container client is None, cannot save '{blob_name}'.")
        return False
    try:
        if not isinstance(data_to_save, (dict, list)): # JSON 직렬화 가능한 타입인지 확인
            st.error(f"Save failed for '{data_description}': Data is not JSON serializable (type: {type(data_to_save)}).")
            print(f"ERROR: Data for '{blob_name}' is not JSON serializable (type: {type(data_to_save)}).")
            return False
        with tempfile.TemporaryDirectory() as tmpdir:
            local_temp_path = os.path.join(tmpdir, os.path.basename(blob_name))
            with open(local_temp_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            blob_client_instance = _container_client.get_blob_client(blob_name)
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
    if not _container_client:
        st.error(f"Cannot save binary '{data_description}': Azure Blob client not ready.")
        print(f"ERROR: Blob Container client is None, cannot save binary '{blob_name}'.")
        return False
    if not os.path.exists(local_file_path):
        st.error(f"Local file for binary '{data_description}' not found: '{local_file_path}'")
        print(f"ERROR: Local file for binary data not found: '{local_file_path}'")
        return False
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
    if "admin" not in USERS: # admin 계정 없으면 생성
        print(f"'{USERS_BLOB_NAME}' from Blob is empty or admin is missing. Creating default admin.")
        USERS["admin"] = {
            "name": "관리자", "department": "품질보증팀", "uid": "admin", # uid 추가
            "password_hash": generate_password_hash(st.secrets.get("ADMIN_PASSWORD", "diteam_fallback_secret")),
            "approved": True, "role": "admin"
        }
        if not save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "initial user info"):
             st.warning("Failed to save default admin info to Blob. Will retry on next run.")
else:
    st.error("Azure Blob Storage connection failed. Cannot initialize user info. App may not function correctly.")
    print("CRITICAL: Cannot initialize USERS due to Blob client failure.")
    USERS = {"admin": {"name": "관리자(연결실패)", "department": "시스템", "uid":"admin", "password_hash": generate_password_hash("fallback"), "approved": True, "role": "admin"}}

cookies = None
cookie_manager_ready = False # 전역 변수로 쿠키 매니저 준비 상태 관리
print(f"Attempting to load COOKIE_SECRET from st.secrets...")
try:
    cookie_secret_key = st.secrets.get("COOKIE_SECRET")
    if not cookie_secret_key:
        st.error("'COOKIE_SECRET' is not set or empty in st.secrets.")
        print("ERROR: COOKIE_SECRET is not set or empty in st.secrets.")
    else:
        cookies = EncryptedCookieManager(
            prefix="gmp_chatbot_auth_v5_6_history/", # 쿠키 prefix 변경 (버전업)
            password=cookie_secret_key
        )
        print("CookieManager object created. Readiness will be checked before use.")
except Exception as e:
    st.error(f"Unknown error creating cookie manager object: {e}")
    print(f"CRITICAL: CookieManager object creation error: {e}\n{traceback.format_exc()}")
    cookies = None

SESSION_TIMEOUT = 1800 # 세션 타임아웃 기본값 (초)
try:
    session_timeout_secret = st.secrets.get("SESSION_TIMEOUT")
    if session_timeout_secret: SESSION_TIMEOUT = int(session_timeout_secret)
    print(f"Session timeout set to: {SESSION_TIMEOUT} seconds.")
except (ValueError, TypeError):
    print(f"WARNING: SESSION_TIMEOUT in secrets ('{session_timeout_secret}') is not a valid integer. Using default {SESSION_TIMEOUT}s.")
except Exception as e:
     print(f"WARNING: Error reading SESSION_TIMEOUT from secrets: {e}. Using default {SESSION_TIMEOUT}s.")

# --- Session State 초기화 ---
# 기존 messages 대신 current_chat_messages 사용, 대화 내역 관련 변수 추가
session_keys_to_initialize = {
    "authenticated": False,
    "user": {},
    "current_chat_messages": [],
    "all_user_conversations": [],
    "active_conversation_id": None,
    "show_uploader": False # 파일 업로더 표시 여부
}
for key, default_value in session_keys_to_initialize.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
        print(f"Initializing st.session_state['{key}'] to {default_value}")

# --- 쿠키를 사용한 세션 복원 시도 (앱 실행 초기) ---
# 이 로직은 st.session_state["authenticated"]가 False일 때만 의미가 있으며,
# 쿠키가 있고 유효하다면 authenticated를 True로 변경하려고 시도합니다.
if not st.session_state.get("authenticated", False) and cookies is not None:
    print("Attempting initial session restore from cookies as user is not authenticated in session_state.")
    try:
        if cookies.ready():
            cookie_manager_ready = True
            print("CookieManager.ready() is True for initial session restore.")
            auth_cookie_val = cookies.get("authenticated")
            print(f"Cookie 'authenticated' value for initial restore: {auth_cookie_val}")

            if auth_cookie_val == "true":
                login_time_str = cookies.get("login_time", "0")
                try:
                    login_time = float(login_time_str if login_time_str and login_time_str.replace('.', '', 1).isdigit() else "0")
                except ValueError:
                    login_time = 0.0
                
                if (time.time() - login_time) < SESSION_TIMEOUT:
                    user_json_cookie = cookies.get("user", "{}")
                    try:
                        user_data_from_cookie = json.loads(user_json_cookie if user_json_cookie else "{}")
                        if user_data_from_cookie and isinstance(user_data_from_cookie, dict) and "uid" in user_data_from_cookie: # uid 존재 확인
                            st.session_state["user"] = user_data_from_cookie
                            st.session_state["authenticated"] = True
                            # 로그인 성공 시 대화 내역 로드
                            st.session_state.all_user_conversations = load_user_conversations_from_blob()
                            st.session_state.current_chat_messages = [] # 새 대화로 시작
                            st.session_state.active_conversation_id = None
                            print(f"User '{user_data_from_cookie.get('name')}' session restored from cookie. Chat history loaded.")
                            # st.rerun() # 여기서 rerun하면 쿠키 관련 오류 발생 가능성 있음. 다음 단계에서 UI가 결정하도록 함.
                        else: # 쿠키에 사용자 정보가 없거나 uid가 없는 경우
                            print("User data in cookie is empty, invalid, or missing uid. Clearing auth state from cookie.")
                            if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
                            st.session_state["authenticated"] = False # 명시적으로 False
                    except json.JSONDecodeError: # 사용자 정보 JSON 파싱 실패
                        print("ERROR: Failed to decode user JSON from cookie. Clearing auth state from cookie.")
                        if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
                        st.session_state["authenticated"] = False # 명시적으로 False
                else: # 세션 타임아웃
                    print("Session timeout detected from cookie. Clearing auth state and cookies.")
                    if cookies.ready(): cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
                    st.session_state["authenticated"] = False # 명시적으로 False
            # else: auth_cookie_val이 "true"가 아니면 st.session_state["authenticated"]는 False로 유지됨
        else: # cookies.ready() is False
            print("CookieManager.ready() is False for initial session restore. Cannot load cookies yet.")
            # cookie_manager_ready는 False로 유지
    except Exception as e_cookie_op_initial:
        print(f"Exception during initial cookie operations: {e_cookie_op_initial}\n{traceback.format_exc()}")
        st.session_state["authenticated"] = False # 안전하게 False
        # cookie_manager_ready는 False로 유지될 수 있음

# 로그인 UI 표시 전 쿠키 매니저 준비 상태 최종 확인 (위에서 ready가 아니었을 수 있으므로)
if cookies is not None and not cookie_manager_ready:
    print("Checking CookieManager readiness again before login UI (if not ready yet)...")
    try:
        if cookies.ready():
            cookie_manager_ready = True
            print("CookieManager became ready before login UI (second check).")
            # 여기서 다시 세션 복원 시도 (위에서 실패한 경우를 위해)
            if not st.session_state.get("authenticated", False): # 아직 인증 안됐으면
                print("Attempting session restore again as CookieManager just became ready.")
                # (위의 쿠키 복원 로직과 유사하게 다시 한번 실행)
                # 이 부분은 복잡성을 증가시킬 수 있으므로, 위의 초기 복원 로직이 충분히 안정적이라면 생략 가능
                # 현재는 위의 초기 복원 시도 후, 그래도 안됐으면 로그인 폼으로 넘어감.
                # 만약 여기서 복원에 성공하면 st.rerun() 필요할 수 있음.
        else:
            print("CookieManager still not ready before login UI (second check).")
    except Exception as e_ready_login_ui:
        print(f"WARNING: cookies.ready() call just before login UI failed: {e_ready_login_ui}")


if not st.session_state.get("authenticated", False):
    st.markdown("""
    <div class="login-page-header-container" style="margin-top: 80px;"> 
      <span class="login-page-main-title">유앤생명과학 GMP/SOP 업무 가이드 봇</span>
      <span class="login-page-sub-title">Made by DI.PART</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="login-form-title">🔐 로그인 또는 회원가입</p>', unsafe_allow_html=True)

    if cookies is None or not cookie_manager_ready:
        st.warning("쿠키 시스템이 아직 준비되지 않았습니다. 로그인이 유지되지 않을 수 있습니다. 잠시 후 새로고침 해보세요.")

    with st.form("auth_form_final_v5_history", clear_on_submit=False):
        mode = st.radio("선택", ["로그인", "회원가입"], key="auth_mode_final_v5_history")
        uid_input = st.text_input("ID", key="auth_uid_final_v5_history") # 변수명 변경 (uid는 내부 사용)
        pwd = st.text_input("비밀번호", type="password", key="auth_pwd_final_v5_history")
        name, dept = "", ""
        if mode == "회원가입":
            name = st.text_input("이름", key="auth_name_final_v5_history")
            dept = st.text_input("부서", key="auth_dept_final_v5_history")
        submit_button = st.form_submit_button("확인")

    if submit_button:
        if not uid_input or not pwd: st.error("ID와 비밀번호를 모두 입력해주세요.")
        elif mode == "회원가입" and (not name or not dept): st.error("회원가입 시 이름과 부서를 모두 입력해주세요.")
        else:
            if mode == "로그인":
                user_data_login = USERS.get(uid_input)
                if not user_data_login: st.error("존재하지 않는 ID입니다.")
                elif not user_data_login.get("approved", False): st.warning("관리자 승인 대기 중인 계정입니다.")
                elif check_password_hash(user_data_login["password_hash"], pwd):
                    
                    # 사용자 정보에 uid 저장 (대화 내역 등에 사용하기 위함)
                    user_data_to_session = user_data_login.copy() # 원본 USERS 딕셔너리 변경 방지
                    user_data_to_session["uid"] = uid_input # 로그인 ID를 uid 키로 저장

                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user_data_to_session
                    
                    # 로그인 성공 시 대화 내역 로드 및 새 대화 준비
                    st.session_state.all_user_conversations = load_user_conversations_from_blob()
                    st.session_state.current_chat_messages = [] # 새 대화로 시작
                    st.session_state.active_conversation_id = None
                    print(f"Login successful for user '{uid_input}'. Chat history loaded. Starting new chat session.")

                    if cookies is not None and cookie_manager_ready:
                        try:
                            cookies["authenticated"] = "true"
                            cookies["user"] = json.dumps(user_data_to_session) # uid 포함된 정보 저장
                            cookies["login_time"] = str(time.time())
                            cookies.save()
                            print(f"Cookies saved for user '{uid_input}'.")
                        except Exception as e_cookie_save_login:
                            st.warning(f"로그인 쿠키 저장 중 문제 발생: {e_cookie_save_login}")
                            print(f"ERROR: Failed to save login cookies: {e_cookie_save_login}")
                    elif cookies is None:
                         st.warning("쿠키 시스템 미초기화로 로그인 상태 저장 불가.")
                    elif not cookie_manager_ready:
                         st.warning("쿠키 시스템 미준비로 로그인 상태 저장 불가.")

                    st.success(f"{user_data_to_session.get('name', uid_input)}님, 환영합니다!"); st.rerun()
                else: st.error("비밀번호가 일치하지 않습니다.")
            elif mode == "회원가입":
                if uid_input in USERS: st.error("이미 존재하는 ID입니다.")
                else:
                    USERS[uid_input] = {"name": name, "department": dept, "uid": uid_input, # uid도 저장
                                  "password_hash": generate_password_hash(pwd),
                                  "approved": False, "role": "user"}
                    if not save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "user info (signup)"):
                        st.error("회원 정보 저장에 실패했습니다. 관리자에게 문의하세요.")
                        USERS.pop(uid_input, None)
                    else:
                        st.success("회원가입 요청이 완료되었습니다. 관리자 승인 후 로그인 가능합니다.")
    st.stop()

# --- 이하 코드는 인증된 사용자에게만 보임 ---

current_user_info = st.session_state.get("user", {}) # uid 포함

# --- 사이드바: 새 대화 버튼 및 대화 내역 ---
with st.sidebar:
    st.markdown(f"**{current_user_info.get('name', '사용자')}님 ({current_user_info.get('uid', '알수없음')})**")
    st.markdown(f"{current_user_info.get('department', '부서정보없음')}")
    st.markdown("---")

    if st.button("➕ 새 대화 시작", use_container_width=True, key="new_chat_button"):
        archive_current_chat_session_if_needed() # 현재 대화가 있다면 저장
        st.session_state.current_chat_messages = []
        st.session_state.active_conversation_id = None
        print("New chat started by user.")
        st.rerun()

    st.markdown("##### 이전 대화 목록")
    if not st.session_state.all_user_conversations:
        st.caption("이전 대화 내역이 없습니다.")
    
    # 대화 목록 표시 (최신 10개 또는 스크롤 가능하게)
    # all_user_conversations는 load 시 last_updated 기준으로 이미 정렬됨
    for i, conv in enumerate(st.session_state.all_user_conversations[:20]): # 최근 20개 표시
        title_display = conv.get('title', f"대화 {conv.get('id', i)}")
        timestamp_display = conv.get('timestamp', conv.get('last_updated','시간없음'))
        # 버튼 레이블에 고유성을 더하기 위해 ID 일부 사용 (제목이 중복될 경우 대비)
        button_label = f"{title_display} ({timestamp_display})"
        button_key = f"conv_btn_{conv['id']}"

        # 현재 활성화된 대화는 다르게 표시 (선택사항)
        if st.session_state.active_conversation_id == conv["id"]:
            st.markdown(f"**➡️ {button_label}**")
        elif st.button(button_label, key=button_key, use_container_width=True):
            print(f"Loading conversation ID: {conv['id']}, Title: '{title_display}'")
            archive_current_chat_session_if_needed() # 현재 활성 대화 저장
            st.session_state.current_chat_messages = list(conv["messages"]) # 대화 내용 불러오기 (복사본)
            st.session_state.active_conversation_id = conv["id"]
            st.rerun()
    
    if len(st.session_state.all_user_conversations) > 20:
        st.caption("더 많은 내역은 전체 보기 기능(추후 구현)을 이용해주세요.")


top_cols_main = st.columns([0.7, 0.3])
with top_cols_main[0]:
    main_logo_html = get_logo_and_version_html(APP_VERSION)
    st.markdown(f"""<div class="logo-container">{main_logo_html}</div>""", unsafe_allow_html=True)


with top_cols_main[1]:
    st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
    if st.button("로그아웃", key="logout_button_final_v5_history"):
        archive_current_chat_session_if_needed() # 로그아웃 전 현재 대화 저장
        
        st.session_state["authenticated"] = False
        st.session_state["user"] = {}
        st.session_state.current_chat_messages = []
        st.session_state.all_user_conversations = []
        st.session_state.active_conversation_id = None
        print("Logout successful. Chat messages and history session state cleared.")
        if cookies is not None and cookie_manager_ready:
            try:
                cookies["authenticated"] = "false"; cookies["user"] = ""; cookies["login_time"] = ""; cookies.save()
                print("Cookies cleared on logout.")
            except Exception as e_logout_cookie:
                 print(f"ERROR: Failed to clear cookies on logout: {e_logout_cookie}")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="main-app-title-container">
  <span class="main-app-title">유앤생명과학 GMP/SOP 업무 가이드 봇</span>
  <span class="main-app-subtitle">Made by DI.PART</span>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vector_db_from_blob_cached(_container_client):
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
                            idx = faiss.IndexFlatL2(current_embedding_dimension); meta = []
                        else:
                            print(f"'{INDEX_BLOB_NAME}' loaded successfully from Blob Storage. Dimension: {idx.d}")
                    except Exception as e_faiss_read:
                        print(f"ERROR reading FAISS index: {e_faiss_read}. Re-initializing index.")
                        idx = faiss.IndexFlatL2(current_embedding_dimension); meta = []
                else:
                    print(f"WARNING: '{INDEX_BLOB_NAME}' is empty in Blob. Using new index."); idx = faiss.IndexFlatL2(current_embedding_dimension); meta = []
            else:
                print(f"WARNING: '{INDEX_BLOB_NAME}' not found in Blob Storage. New index will be used/created."); idx = faiss.IndexFlatL2(current_embedding_dimension); meta = []

            if idx is not None:
                metadata_blob_client = _container_client.get_blob_client(METADATA_BLOB_NAME)
                if metadata_blob_client.exists() and (idx.ntotal > 0 or (index_blob_client.exists() and os.path.exists(local_index_path) and os.path.getsize(local_index_path) > 0) ):
                    print(f"Downloading '{METADATA_BLOB_NAME}'...")
                    with open(local_metadata_path, "wb") as download_file_meta:
                        download_stream_meta = metadata_blob_client.download_blob(timeout=60)
                        download_file_meta.write(download_stream_meta.readall())
                    if os.path.getsize(local_metadata_path) > 0 :
                        with open(local_metadata_path, "r", encoding="utf-8") as f_meta: meta = json.load(f_meta)
                    else: meta = []; print(f"WARNING: '{METADATA_BLOB_NAME}' is empty in Blob.")
                elif idx.ntotal == 0 and not index_blob_client.exists():
                     print(f"INFO: Index is new and empty, starting with empty metadata."); meta = []
                else:
                    print(f"INFO: Metadata file '{METADATA_BLOB_NAME}' not found or index is empty. Starting with empty metadata."); meta = []

            if idx is not None and idx.ntotal == 0 and len(meta) > 0:
                print(f"INFO: FAISS index is empty (ntotal=0) but metadata is not. Clearing metadata for consistency."); meta = []
            elif idx is not None and idx.ntotal > 0 and not meta and index_blob_client.exists() and os.path.exists(local_index_path) and os.path.getsize(local_index_path) > 0 :
                print(f"CRITICAL WARNING: FAISS index has data (ntotal={idx.ntotal}) but metadata is empty. This may lead to errors.")
    except AzureError as ae:
        st.error(f"Azure service error loading vector DB from Blob: {ae}"); print(f"AZURE ERROR loading vector DB: {ae}\n{traceback.format_exc()}"); idx = faiss.IndexFlatL2(current_embedding_dimension); meta = []
    except Exception as e:
        st.error(f"Unknown error loading vector DB from Blob: {e}"); print(f"GENERAL ERROR loading vector DB: {e}\n{traceback.format_exc()}"); idx = faiss.IndexFlatL2(current_embedding_dimension); meta = []
    return idx, meta

index, metadata = faiss.IndexFlatL2(1536), []
if container_client:
    index, metadata = load_vector_db_from_blob_cached(container_client)
    print(f"DEBUG: FAISS index loaded after cache. ntotal: {index.ntotal if index else 'None'}, dimension: {index.d if index else 'N/A'}")
    print(f"DEBUG: Metadata loaded after cache. Length: {len(metadata) if metadata is not None else 'None'}")
else:
    st.error("Azure Blob Storage connection failed. Cannot load vector DB. File learning/search will be limited.")
    print("CRITICAL: Cannot load vector DB due to Blob client failure (main section).")

@st.cache_data
def load_prompt_rules_cached():
    default_rules = """1. 제공된 '문서 내용'을 최우선으로 참고하여 답변합니다. (이하 생략)"""
    if os.path.exists(RULES_PATH_REPO):
        try:
            with open(RULES_PATH_REPO, "r", encoding="utf-8") as f: rules_content = f.read()
            print(f"Prompt rules loaded successfully from '{RULES_PATH_REPO}'.")
            return rules_content
        except Exception as e:
            st.warning(f"Error loading '{RULES_PATH_REPO}': {e}. Using default rules."); print(f"WARNING: Error loading prompt rules: {e}. Using default."); return default_rules
    else:
        print(f"WARNING: Prompt rules file not found at '{RULES_PATH_REPO}'. Using default rules."); return default_rules
PROMPT_RULES_CONTENT = load_prompt_rules_cached()

def extract_text_from_file(uploaded_file_obj):
    ext = os.path.splitext(uploaded_file_obj.name)[1].lower()
    text_content = ""
    if ext in [".png", ".jpg", ".jpeg"]: return ""
    try:
        uploaded_file_obj.seek(0); file_bytes = uploaded_file_obj.read()
        if ext == ".pdf":
            with fitz.open(stream=file_bytes, filetype="pdf") as doc: text_content = "\n".join(page.get_text() for page in doc)
        elif ext == ".docx":
            with io.BytesIO(file_bytes) as doc_io:
                doc = docx.Document(doc_io); full_text = [p.text for p in doc.paragraphs]
                for table_idx, table in enumerate(doc.tables):
                    table_data_text = [f"--- Table {table_idx+1} Start ---"]
                    for row in table.rows: table_data_text.append(" | ".join(cell.text.strip() for cell in row.cells))
                    table_data_text.append(f"--- Table {table_idx+1} End ---"); full_text.append("\n".join(table_data_text))
                text_content = "\n\n".join(full_text)
        elif ext in (".xlsx", ".xlsm"):
            with io.BytesIO(file_bytes) as excel_io: df_dict = pd.read_excel(excel_io, sheet_name=None)
            text_content = "\n\n".join(f"--- Sheet: {name} ---\n{df.to_string(index=False)}" for name, df in df_dict.items())
        elif ext == ".csv":
            with io.BytesIO(file_bytes) as csv_io:
                try: df = pd.read_csv(csv_io)
                except UnicodeDecodeError: df = pd.read_csv(io.BytesIO(file_bytes), encoding='cp949') # seek(0)은 file_bytes 사용 시 불필요
                text_content = df.to_string(index=False)
        elif ext == ".pptx":
            with io.BytesIO(file_bytes) as ppt_io: prs = Presentation(ppt_io); text_content = "\n".join(s.text for sl in prs.slides for s in sl.shapes if hasattr(s, "text"))
        elif ext == ".txt":
            try: text_content = file_bytes.decode('utf-8')
            except UnicodeDecodeError: text_content = file_bytes.decode('cp949')
        else: st.warning(f"Unsupported text file type: {ext}"); return ""
    except Exception as e: st.error(f"Error extracting text from '{uploaded_file_obj.name}': {e}"); print(f"ERROR extracting text: {e}\n{traceback.format_exc()}"); return ""
    return text_content.strip()

def save_original_file_to_blob(uploaded_file_obj, _container_client, base_path="original_files"):
    if not _container_client or not uploaded_file_obj: return None
    try:
        safe_file_name = re.sub(r'[\\/*?:"<>|]', "_", uploaded_file_obj.name)
        blob_name = f"{base_path}/{datetime.now().strftime('%Y%m%d%H%M%S')}_{safe_file_name}"
        blob_client_instance = _container_client.get_blob_client(blob_name)
        uploaded_file_obj.seek(0); file_bytes_for_original = uploaded_file_obj.read()
        with io.BytesIO(file_bytes_for_original) as data_stream:
            blob_client_instance.upload_blob(data_stream, overwrite=True, timeout=120)
        print(f"Successfully saved original file '{safe_file_name}' to Blob as '{blob_name}'"); return blob_name
    except Exception as e: print(f"ERROR saving original file to Blob: {e}"); return None

def log_openai_api_usage_to_blob(user_id, model_name, usage_object, _container_client, request_type="general_api_call"):
    if not _container_client: print(f"ERROR: Blob client None, cannot log API usage."); return False
    log_entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user_id": user_id, "model_name": model_name, "request_type": request_type, "prompt_tokens": getattr(usage_object, 'prompt_tokens', 0), "completion_tokens": getattr(usage_object, 'completion_tokens', 0), "total_tokens": getattr(usage_object, 'total_tokens', 0)}
    try:
        current_logs = load_data_from_blob(USAGE_LOG_BLOB_NAME, _container_client, "API usage log", default_value=[])
        if not isinstance(current_logs, list): current_logs = []
        current_logs.append(log_entry)
        if save_data_to_blob(current_logs, USAGE_LOG_BLOB_NAME, _container_client, "API usage log"): print(f"Successfully logged API usage for '{user_id}'."); return True
        else: print(f"ERROR: Failed to save API usage log after appending."); return False
    except Exception as e: print(f"GENERAL ERROR logging API usage: {e}\n{traceback.format_exc()}"); return False

def chunk_text_into_pieces(text_to_chunk, chunk_size=500):
    if not text_to_chunk or not text_to_chunk.strip(): return [];
    chunks_list, current_buffer = [], ""
    for line in text_to_chunk.split("\n"): 
        stripped_line = line.strip()
        if not stripped_line and not current_buffer.strip(): continue 
        if len(current_buffer) + len(stripped_line) + 1 < chunk_size: current_buffer += stripped_line + "\n"
        else: 
            if current_buffer.strip(): chunks_list.append(current_buffer.strip())
            current_buffer = stripped_line + "\n" 
    if current_buffer.strip(): chunks_list.append(current_buffer.strip())
    return [c for c in chunks_list if c] 

def get_image_description(image_bytes, image_filename, client_instance):
    if not client_instance: print("ERROR: OpenAI client not ready for image description."); return None
    print(f"DEBUG: Requesting description for image '{image_filename}'")
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        ext = os.path.splitext(image_filename)[1].lower(); mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png" if ext == ".png" else "application/octet-stream"
        vision_model = st.secrets["AZURE_OPENAI_DEPLOYMENT"]
        response = client_instance.chat.completions.create(model=vision_model, messages=[{"role": "user", "content": [{"type": "text", "text": f"Describe this image ('{image_filename}') from a work/professional perspective for search and context. Mention key objects, states, GMP/SOP relevance if any."}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}" }}]}], max_tokens=IMAGE_DESCRIPTION_MAX_TOKENS, temperature=0.2, timeout=AZURE_OPENAI_TIMEOUT)
        description = response.choices[0].message.content.strip(); print(f"DEBUG: Image description for '{image_filename}' generated (len: {len(description)})."); return description
    except Exception as e: print(f"ERROR during image description for '{image_filename}': {e}\n{traceback.format_exc()}"); return None

def get_text_embedding(text_to_embed, client=openai_client, model=EMBEDDING_MODEL):
    if not client or not model or not text_to_embed or not text_to_embed.strip(): return None
    try: response = client.embeddings.create(input=[text_to_embed], model=model, timeout=AZURE_OPENAI_TIMEOUT / 2); return response.data[0].embedding
    except Exception as e: print(f"ERROR during single text embedding for '{text_to_embed[:30]}...': {e}"); return None

def get_batch_embeddings(texts_to_embed, client=openai_client, model=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE):
    if not client or not model or not texts_to_embed: return []
    all_embeddings = []
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]; 
        if not batch: continue
        print(f"DEBUG: Requesting embeddings for batch of {len(batch)} texts...")
        try:
            response = client.embeddings.create(input=batch, model=model, timeout=AZURE_OPENAI_TIMEOUT)
            batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda e: e.index)]
            all_embeddings.extend(batch_embeddings); print(f"DEBUG: Embeddings received for batch {i//batch_size + 1}.")
        except Exception as e: print(f"ERROR during batch embedding for batch starting with '{batch[0][:30]}...': {e}"); all_embeddings.extend([None] * len(batch))
    return all_embeddings

def search_similar_chunks(query_text, k_results=3):
    if index is None or index.ntotal == 0 or not metadata: return []
    query_vector = get_text_embedding(query_text)
    if query_vector is None: return []
    try:
        actual_k = min(k_results, index.ntotal); 
        if actual_k == 0 : return []
        distances, indices_found = index.search(np.array([query_vector]).astype("float32"), actual_k)
        results = [{"source": metadata[i].get("file_name", "Unknown"), "content": metadata[i].get("content", ""), "is_image_description": metadata[i].get("is_image_description", False), "original_file_extension": metadata[i].get("original_file_extension", "")} for i in indices_found[0] if 0 <= i < len(metadata) and isinstance(metadata[i], dict)]
        return results
    except Exception as e: print(f"ERROR: Similarity search failed: {e}\n{traceback.format_exc()}"); return []

def add_document_to_vector_db_and_blob(uploaded_file_obj, processed_content, text_chunks, _container_client, is_image_description=False):
    global index, metadata
    if not text_chunks or not _container_client: st.warning(f"No content or Blob client for '{uploaded_file_obj.name}'."); return False
    file_type_log = "image desc" if is_image_description else "text"
    print(f"Adding '{file_type_log}' from '{uploaded_file_obj.name}' to vector DB.")
    chunk_embeddings = get_batch_embeddings(text_chunks)
    vectors_to_add, new_metadata = [], []
    for i, chunk in enumerate(text_chunks):
        embedding = chunk_embeddings[i] if i < len(chunk_embeddings) else None
        if embedding:
            vectors_to_add.append(embedding)
            new_metadata.append({"file_name": uploaded_file_obj.name, "content": chunk, "is_image_description": is_image_description, "original_file_extension": os.path.splitext(uploaded_file_obj.name)[1].lower()})
    if not vectors_to_add: st.error(f"No valid embeddings for '{uploaded_file_obj.name}'. Not learned."); return False
    try:
        dim = np.array(vectors_to_add[0]).shape[0]
        if index is None or index.d != dim: index = faiss.IndexFlatL2(dim); metadata = []
        if vectors_to_add: index.add(np.array(vectors_to_add).astype("float32"))
        metadata.extend(new_metadata)
        print(f"Added {len(vectors_to_add)} chunks from '{uploaded_file_obj.name}'. Index total: {index.ntotal}")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_idx_path = os.path.join(tmpdir, "temp.index")
            if index.ntotal > 0: faiss.write_index(index, tmp_idx_path); save_binary_data_to_blob(tmp_idx_path, INDEX_BLOB_NAME, _container_client, "vector index")
        save_data_to_blob(metadata, METADATA_BLOB_NAME, _container_client, "metadata")
        uploader = st.session_state.user.get("name", "N/A")
        log_entry = {"file": uploaded_file_obj.name, "type": file_type_log, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "chunks": len(vectors_to_add), "uploader": uploader}
        logs = load_data_from_blob(UPLOAD_LOG_BLOB_NAME, _container_client, "upload log", [])
        if not isinstance(logs, list): logs = []
        logs.append(log_entry); save_data_to_blob(logs, UPLOAD_LOG_BLOB_NAME, _container_client, "upload log")
        return True
    except Exception as e: st.error(f"Error learning doc or Blob upload: {e}"); print(f"ERROR adding doc: {e}\n{traceback.format_exc()}"); return False

chat_interface_tab, admin_settings_tab = None, None
if current_user_info.get("role") == "admin":
    chat_interface_tab, admin_settings_tab = st.tabs(["💬 챗봇 질문", "⚙️ 관리자 설정"])
else:
    chat_interface_tab = st.container() 

if chat_interface_tab:
    with chat_interface_tab:
        st.header("업무 질문")
        st.markdown("💡 예시: SOP 백업 주기, PIC/S Annex 11 차이, (파일 첨부 후) 이 사진 속 상황은 어떤 규정에 해당하나요? 등")

        for msg_item in st.session_state.current_chat_messages: # current_chat_messages 사용
            role, content, time_str = msg_item.get("role"), msg_item.get("content", ""), msg_item.get("time", "")
            align_class = "user-align" if role == "user" else "assistant-align"
            bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
            st.markdown(f"""<div class="chat-bubble-container {align_class}"><div class="bubble {bubble_class}">{content}</div><div class="timestamp">{time_str}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True) 
        if st.button("📂 파일 첨부/숨기기", key="toggle_chat_uploader_history_fix"): 
            st.session_state.show_uploader = not st.session_state.get("show_uploader", False)

        chat_file_uploader_key = "chat_file_uploader_history_fix_widget" 
        uploaded_chat_file_runtime = None 
        if st.session_state.get("show_uploader", False):
            uploaded_chat_file_runtime = st.file_uploader("질문과 함께 참고할 파일 첨부 (선택 사항)",
                                     type=["pdf","docx","xlsx","xlsm","csv","pptx", "txt", "png", "jpg", "jpeg"], 
                                     key=chat_file_uploader_key)
            if uploaded_chat_file_runtime: 
                st.caption(f"첨부됨: {uploaded_chat_file_runtime.name} ({uploaded_chat_file_runtime.type}, {uploaded_chat_file_runtime.size} bytes)")
                if uploaded_chat_file_runtime.type.startswith("image/"): st.image(uploaded_chat_file_runtime, width=200)

        with st.form("chat_input_form_history_fix", clear_on_submit=True): 
            query_input_col, send_button_col = st.columns([4,1])
            with query_input_col:
                user_query_input = st.text_input("질문 입력:", placeholder="여기에 질문을 입력하세요...", key="user_query_text_input_history_fix", label_visibility="collapsed") 
            with send_button_col: send_query_button = st.form_submit_button("전송")

        if send_query_button and user_query_input.strip():
            if not openai_client or not tokenizer:
                st.error("OpenAI 서비스 또는 토크나이저가 준비되지 않았습니다. 답변 생성 불가."); st.stop()
            
            timestamp_now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            user_message_content = user_query_input
            if uploaded_chat_file_runtime: user_message_content += f"\n(첨부 파일: {uploaded_chat_file_runtime.name})"
            
            st.session_state.current_chat_messages.append({"role":"user", "content":user_message_content, "time":timestamp_now_str})
            
            # active_conversation_id가 없다면 (새 대화 시작), 이 시점에서 ID를 부여하고 all_user_conversations에 임시 추가 (나중에 archive시 확정)
            # 또는, 답변 후 archive_current_chat_session_if_needed()를 호출하여 저장/업데이트
            
            user_id_for_log = current_user_info.get("name", "anonymous_chat")
            print(f"User '{user_id_for_log}' submitted query: '{user_query_input[:50]}...' (File: {uploaded_chat_file_runtime.name if uploaded_chat_file_runtime else 'None'})")
            
            with st.spinner("답변 생성 중..."):
                assistant_response_content = "Error generating response."
                try: 
                    print("Step 1: Preparing context...")
                    context_items = []
                    text_from_chat_file, is_chat_file_img_desc, chat_file_src_name = None, False, None

                    if uploaded_chat_file_runtime:
                        ext = os.path.splitext(uploaded_chat_file_runtime.name)[1].lower()
                        is_img = ext in [".png", ".jpg", ".jpeg"]
                        if is_img:
                            with st.spinner(f"Analyzing image '{uploaded_chat_file_runtime.name}'..."):
                                desc = get_image_description(uploaded_chat_file_runtime.getvalue(), uploaded_chat_file_runtime.name, openai_client)
                            if desc: text_from_chat_file = desc; chat_file_src_name = f"User image: {uploaded_chat_file_runtime.name}"; is_chat_file_img_desc = True
                            else: st.warning(f"Failed to describe image '{uploaded_chat_file_runtime.name}'.")
                        else: 
                            text_from_chat_file = extract_text_from_file(uploaded_chat_file_runtime)
                            if text_from_chat_file: chat_file_src_name = f"User file: {uploaded_chat_file_runtime.name}"
                            elif text_from_chat_file == "": st.info(f"File '{uploaded_chat_file_runtime.name}' is empty or unreadable.")
                        if text_from_chat_file: context_items.append({"source": chat_file_src_name, "content": text_from_chat_file, "is_image_description": is_chat_file_img_desc})
                    
                    prompt_struct = f"{PROMPT_RULES_CONTENT}\n\nContext:\n<Doc Start>\n{{context}}\n<Doc End>"
                    base_tokens = len(tokenizer.encode(prompt_struct.replace('{context}', '')))
                    query_tokens = len(tokenizer.encode(user_query_input))
                    max_ctx_tokens = TARGET_INPUT_TOKENS_FOR_PROMPT - base_tokens - query_tokens
                    ctx_str = "No relevant documents found."

                    if max_ctx_tokens > 0:
                        query_for_search = user_query_input + (f"\nImage content: {text_from_chat_file}" if is_chat_file_img_desc and text_from_chat_file else "")
                        retrieved_db_items = search_similar_chunks(query_for_search, k_results=3)
                        if retrieved_db_items: context_items.extend(retrieved_db_items)
                        
                        if context_items:
                            seen_ctx = set(); fmt_ctx_chunks = []
                            for item in context_items:
                                content = item.get("content","").strip()
                                if content and content not in seen_ctx:
                                    src = item.get('source','Unknown').replace("User image: ","").replace("User file: ","")
                                    prefix = "[Image Desc: " if item.get("is_image_description") else "[Source: "
                                    fmt_ctx_chunks.append(f"{prefix}{src}]\n{content}")
                                    seen_ctx.add(content)
                            if fmt_ctx_chunks:
                                full_ctx_str = "\n\n---\n\n".join(fmt_ctx_chunks)
                                full_ctx_tokens = tokenizer.encode(full_ctx_str)
                                if len(full_ctx_tokens) > max_ctx_tokens:
                                    truncated_tokens = full_ctx_tokens[:max_ctx_tokens]
                                    ctx_str = tokenizer.decode(truncated_tokens) + "\n(...more content truncated.)"
                                else: ctx_str = full_ctx_str
                    
                    system_prompt = prompt_struct.replace('{context}', ctx_str)
                    final_prompt_tokens = len(tokenizer.encode(system_prompt)) + query_tokens
                    if final_prompt_tokens > MODEL_MAX_INPUT_TOKENS: print(f"CRITICAL WARNING: Final input tokens ({final_prompt_tokens}) > model max ({MODEL_MAX_INPUT_TOKENS})!")
                    
                    api_messages = [{"role":"system", "content": system_prompt}, {"role":"user", "content": user_query_input}]
                    print("Step 2: Sending request to Azure OpenAI...")
                    chat_model = st.secrets.get("AZURE_OPENAI_DEPLOYMENT")
                    if not chat_model: st.error("Chat model not configured in secrets."); raise ValueError("Chat model missing")
                        
                    completion = openai_client.chat.completions.create(model=chat_model, messages=api_messages, max_tokens=MODEL_MAX_OUTPUT_TOKENS, temperature=0.1, timeout=AZURE_OPENAI_TIMEOUT)
                    assistant_response_content = completion.choices[0].message.content.strip()
                    print("Azure OpenAI response received.")
                    if completion.usage and container_client: log_openai_api_usage_to_blob(user_id_for_log, chat_model, completion.usage, container_client, "chat_rag")
                
                except Exception as gen_err: 
                    assistant_response_content = f"Unexpected error: {gen_err}."
                    st.error(assistant_response_content); print(f"ERROR during response generation: {gen_err}\n{traceback.format_exc()}")

            st.session_state.current_chat_messages.append({"role":"assistant", "content":assistant_response_content, "time":timestamp_now_str})
            # 답변 후 현재 대화 상태를 아카이브/업데이트.
            # active_conversation_id가 None이었다면, archive 함수 내에서 새로 생성되고 all_user_conversations에 추가됨.
            # 그 후 active_conversation_id를 업데이트 해줘야 함.
            if st.session_state.active_conversation_id is None and st.session_state.all_user_conversations:
                 # archive_current_chat_session_if_needed가 호출되면 새 ID가 all_user_conversations[0]에 생김
                 # 하지만 archive_current_chat_session_if_needed는 아직 여기서 호출되지 않았으므로,
                 # archive_current_chat_session_if_needed 호출 전에 active_id를 설정해야 한다면,
                 # 또는 archive 함수가 새 ID를 반환하도록 수정해야 함.
                 # 여기서는 그냥 두고, 컨텍스트 전환 시 (새 대화, 다른 대화 로드, 로그아웃) archive가 처리하도록 함.
                 pass # active_id는 컨텍스트 전환 시 결정됨

            print("Response processing complete. Triggering rerun."); st.rerun()

if admin_settings_tab:
    with admin_settings_tab:
        st.header("⚙️ 관리자 설정")
        st.subheader("👥 가입 승인 대기자")
        if not USERS or not isinstance(USERS, dict): st.warning("User info error.")
        else:
            pending = {uid:udata for uid,udata in USERS.items() if isinstance(udata, dict) and not udata.get("approved")}
            if pending:
                for uid, udata in pending.items():
                    with st.expander(f"{udata.get('name','N/A')} ({uid}) - {udata.get('department','N/A')}"):
                        app_col, rej_col = st.columns(2)
                        if app_col.button("승인", key=f"admin_approve_{uid}"): 
                            USERS[uid]["approved"] = True
                            if save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "user approval"): st.success(f"User '{uid}' approved."); st.rerun()
                            else: st.error("Failed to save approval.")
                        if rej_col.button("거절", key=f"admin_reject_{uid}"): 
                            USERS.pop(uid, None)
                            if save_data_to_blob(USERS, USERS_BLOB_NAME, container_client, "user rejection"): st.info(f"User '{uid}' rejected."); st.rerun()
                            else: st.error("Failed to save rejection.")
            else: st.info("No users pending approval.")
        st.markdown("---")

        st.subheader("📁 파일 업로드 및 학습 (Azure Blob Storage)")
        if 'processed_admin_file_info' not in st.session_state: st.session_state.processed_admin_file_info = None
        def clear_admin_file_info(): st.session_state.processed_admin_file_info = None
        admin_file = st.file_uploader("학습 파일 (PDF, DOCX, XLSX, CSV, PPTX, TXT, PNG, JPG, JPEG)", type=["pdf","docx","xlsx","xlsm","csv","pptx", "txt", "png", "jpg", "jpeg"], key="admin_uploader_hist_fix", on_change=clear_admin_file_info)

        if admin_file and container_client:
            file_info = (admin_file.name, admin_file.size, admin_file.type)
            if st.session_state.processed_admin_file_info != file_info:
                print(f"DEBUG Admin Upload: New file {file_info}")
                try: 
                    ext = os.path.splitext(admin_file.name)[1].lower(); is_img = ext in [".png", ".jpg", ".jpeg"]
                    content, is_img_desc = None, False
                    if is_img:
                        with st.spinner(f"Processing image '{admin_file.name}'..."):
                            desc = get_image_description(admin_file.getvalue(), admin_file.name, openai_client)
                        if desc: content = desc; is_img_desc = True; st.info(f"Image '{admin_file.name}' description generated (Len: {len(desc)})."); st.text_area("Desc:", desc, height=150, disabled=True)
                        else: st.error(f"Failed to describe image '{admin_file.name}'.")
                    else: 
                        with st.spinner(f"Extracting text from '{admin_file.name}'..."): content = extract_text_from_file(admin_file)
                        if content: st.info(f"Text extracted from '{admin_file.name}' (Len: {len(content)}).")
                        else: st.warning(f"No content from '{admin_file.name}'.")
                    
                    if content: 
                        with st.spinner(f"Learning content from '{admin_file.name}'..."):
                            chunks = chunk_text_into_pieces(content)
                            if chunks:
                                path = save_original_file_to_blob(admin_file, container_client)
                                if path: st.caption(f"Original '{admin_file.name}' saved to Blob: '{path}'.")
                                else: st.warning(f"Failed to save original '{admin_file.name}' to Blob.")
                                if add_document_to_vector_db_and_blob(admin_file, content, chunks, container_client, is_img_desc):
                                    st.success(f"File '{admin_file.name}' learned and updated to Azure Blob!"); st.session_state.processed_admin_file_info = file_info; st.rerun()
                                else: st.error(f"Error learning or Blob update for '{admin_file.name}'."); st.session_state.processed_admin_file_info = None
                            else: st.warning(f"No learning chunks for '{admin_file.name}'.")
                except Exception as e_admin_proc: st.error(f"Error processing admin file {admin_file.name}: {e_admin_proc}"); print(f"CRITICAL ERROR admin_upload for {admin_file.name}: {e_admin_proc}\n{traceback.format_exc()}"); st.session_state.processed_admin_file_info = None
            elif st.session_state.processed_admin_file_info == file_info: st.caption(f"File '{admin_file.name}' previously processed. Re-upload to re-learn.")
        elif admin_file and not container_client: st.error("Cannot upload: Azure Blob client not ready.")
        st.markdown("---")

        st.subheader("📊 API 사용량 모니터링 (Blob 로그 기반)")
        if container_client:
            usage_data = load_data_from_blob(USAGE_LOG_BLOB_NAME, container_client, "API usage log", [])
            if usage_data and isinstance(usage_data, list) and len(usage_data) > 0 :
                df = pd.DataFrame(usage_data)
                for col in ["total_tokens", "prompt_tokens", "completion_tokens"]: df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
                if "request_type" not in df.columns: df["request_type"] = "unknown"
                total_tokens = df["total_tokens"].sum()
                st.metric("총 API 호출", len(df)); st.metric("총 사용 토큰", f"{int(total_tokens):,}")
                cost_unit = 0.0; 
                try: cost_unit=float(st.secrets.get("TOKEN_COST","0"))
                except: pass
                st.metric("예상 비용 (USD)", f"${total_tokens * cost_unit:.4f}") 
                if "timestamp" in df.columns:
                    try: df['timestamp'] = pd.to_datetime(df['timestamp']); st.dataframe(df.sort_values(by="timestamp",ascending=False), use_container_width=True)
                    except: st.dataframe(df, use_container_width=True) 
                else: st.dataframe(df, use_container_width=True)
            else: st.info("No API usage data recorded.")
        else: st.warning("Cannot display API usage: Azure Blob client not ready.")
        st.markdown("---")

        st.subheader("📂 Azure Blob Storage 파일 목록 (최근 100개)")
        if container_client:
            try:
                blobs_display = []
                blobs_sorted = sorted(container_client.list_blobs(), key=lambda b: b.last_modified, reverse=True)
                for i, blob in enumerate(blobs_sorted):
                    if i >= 100: break
                    blobs_display.append({"파일명": blob.name, "크기 (bytes)": blob.size, "수정일": blob.last_modified.strftime('%Y-%m-%d %H:%M:%S') if blob.last_modified else 'N/A'})
                if blobs_display: st.dataframe(pd.DataFrame(blobs_display), use_container_width=True)
                else: st.info("No files in Azure Blob Storage.")
            except Exception as e_list_blobs: st.error(f"Error listing Blobs: {e_list_blobs}"); print(f"ERROR listing blobs: {e_list_blobs}\n{traceback.format_exc()}")
        else: st.warning("Cannot display file list: Azure Blob client not ready.")
