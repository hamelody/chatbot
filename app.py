import streamlit as st
import traceback # Traceback 출력을 위해 임포트

st.title("Minimal TXT File Upload Test")
st.write("문제가 발생하는 '안녕.txt' 파일과, 영어로만 된 작은 .txt 파일을 각각 업로드 해보세요.")

# 테스트를 위해 TXT 파일만 허용
uploaded_file = st.file_uploader(
    "Upload a TXT file here",
    type=["txt"]
)

if uploaded_file is not None:
    st.subheader("--- File Details ---")
    st.write(f"Name: {uploaded_file.name}")
    st.write(f"Size: {uploaded_file.size} bytes")
    st.write(f"Type (MIME): {uploaded_file.type}")

    try:
        # 파일 내용을 바이트로 읽기 시도
        st.write("Attempting to read file bytes...")
        # getvalue()는 전체 파일을 메모리로 읽으므로 작은 파일에 적합
        file_bytes = uploaded_file.getvalue() 
        st.write(f"Successfully read {len(file_bytes)} bytes from the file object.")
        st.write(f"Raw Bytes (first 20 bytes): {file_bytes[:20]!r}") # 바이트 내용 앞부분을 직접 확인

        # UTF-8로 디코딩 시도 (한글 포함 파일에 일반적)
        st.write("Attempting to decode as UTF-8...")
        try:
            decoded_text_utf8 = file_bytes.decode('utf-8')
            st.write(f"Decoded as UTF-8 (first 50 chars): {decoded_text_utf8[:50]!r}")
            st.success("UTF-8 decoding successful or no error.")
        except UnicodeDecodeError as ude:
            st.error(f"UTF-8 decoding FAILED: {ude}")
            # UTF-8 실패 시 CP949 시도 (Windows 환경 한글 파일)
            st.write("Attempting to decode as CP949...")
            try:
                decoded_text_cp949 = file_bytes.decode('cp949')
                st.write(f"Decoded as CP949 (first 50 chars): {decoded_text_cp949[:50]!r}")
                st.success("CP949 decoding successful.")
            except Exception as ede_cp949:
                st.error(f"CP949 decoding also FAILED: {ede_cp949}")
        except Exception as e_decode: # 기타 디코딩 관련 오류
            st.error(f"General error during UTF-8 decoding step: {e_decode}")
            st.code(traceback.format_exc())


        st.success(f"File '{uploaded_file.name}' seems to have been received and initially processed by Streamlit backend.")
        st.write("If you see this message, the basic upload and read worked.")

    except Exception as e: # 파일 읽기나 디코딩 외의 다른 오류
        st.error(f"An error occurred after file upload but during processing in this minimal Streamlit app: {e}")
        st.error("Full Traceback:")
        st.code(traceback.format_exc())
else:
    st.write("Please upload a file to see details.")
