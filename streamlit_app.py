# streamlit_app.py
import os
import io
import json
import tempfile

import streamlit as st
import importlib

# use a safe fallback import block
try:
    # newer-ish versions expose core objects here
    from llama_index.core import Document
except Exception:
    try:
        # older examples and docs sometimes use top-level export
        from llama_index import Document
    except Exception:
        # last resort: try schema path
        from llama_index.core.schema import Document

# index import (also may move between versions)
try:
    from llama_index import GPTVectorStoreIndex
except Exception:
    try:
        from llama_index.indices import GPTVectorStoreIndex
    except Exception:
        # if this fails, the package version likely doesn't provide it under these names
        GPTVectorStoreIndex = None

from supabase import create_client

# --- Configuration / secrets ---
# In Streamlit Cloud put these into the app Secrets area, keys shown below:
# SUPABASE_URL, SUPABASE_KEY, OPENAI_KEY, SUPABASE_BUCKET (e.g. "uploads")
SUPABASE_URL = st.secrets.get("SUPABASE_URL", None)
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", None)
OPENAI_KEY = st.secrets.get("OPENAI_KEY", None)
SUPABASE_BUCKET = st.secrets.get("SUPABASE_BUCKET", "uploads")

if not (SUPABASE_URL and SUPABASE_KEY and OPENAI_KEY):
    st.warning("Missing secrets. Set SUPABASE_URL, SUPABASE_KEY and OPENAI_KEY in Streamlit Secrets.")
    st.stop()

# Ensure OpenAI env var available to any libs that expect it
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Create supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Helpers ------------------------------------------------------------------
# Robust Supabase helpers

def upload_file_to_supabase(file_bytes: bytes, path: str):
    """
    Upload bytes to Supabase Storage at `path`.
    Tries several call patterns to support different supabase-py versions:
      1. bucket.upload(path, bytes)
      2. bucket.upload(path, BytesIO(bytes))
      3. bucket.update(path, bytes)
      4. bucket.update(path, local_temp_filepath)  (if update expects a path)
    Returns the client response on success or raises RuntimeError with details.
    """
    bucket = supabase.storage.from_(SUPABASE_BUCKET)

    # 1) Try upload with raw bytes (most clients accept this)
    try:
        res = bucket.upload(path, file_bytes)
        return res
    except Exception as e_upload_bytes:
        # If it fails, try upload with BytesIO
        try:
            file_like = io.BytesIO(file_bytes)
            file_like.seek(0)
            res = bucket.upload(path, file_like)
            return res
        except Exception as e_upload_bio:
            # Try update with bytes (replace existing object)
            try:
                res2 = bucket.update(path, file_bytes)
                return res2
            except Exception as e_update_bytes:
                # If update expects a file path, write temp file and pass path
                tmp = None
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False)
                    tmp.write(file_bytes)
                    tmp.flush()
                    tmp.close()
                    try:
                        res3 = bucket.update(path, tmp.name)
                        return res3
                    except Exception as e_update_path:
                        # Give a combined error message so you see all attempts
                        raise RuntimeError(
                            "Upload/update attempts failed:\n"
                            f"upload(bytes) error: {e_upload_bytes}\n"
                            f"upload(BytesIO) error: {e_upload_bio}\n"
                            f"update(bytes) error: {e_update_bytes}\n"
                            f"update(path) error: {e_update_path}"
                        ) from e_update_path
                finally:
                    if tmp is not None:
                        try:
                            os.unlink(tmp.name)
                        except Exception:
                            pass


def download_file_from_supabase(path: str) -> bytes | None:
    """
    Download bytes for a given path. Handles different return shapes.
    Returns bytes or None if not found.
    """
    bucket = supabase.storage.from_(SUPABASE_BUCKET)
    try:
        data = bucket.download(path)
        if data is None:
            return None
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if hasattr(data, "read"):
            return data.read()
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        # attempt conversion
        return bytes(data)
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "404" in err:
            return None
        # re-raise for unexpected failures so the app shows the error
        raise


def list_files_in_bucket(prefix: str = ""):
    """
    List objects in bucket. Accommodates clients that accept list(prefix) or list().
    Returns a list (possibly empty) or raises if listing fails.
    """
    bucket = supabase.storage.from_(SUPABASE_BUCKET)
    try:
        return bucket.list(prefix)
    except TypeError:
        try:
            return bucket.list()
        except Exception as e:
            raise RuntimeError(f"listing bucket failed: {e}") from e
    except Exception:
        # return empty list rather than crash in UI
        return []


# App UI -------------------------------------------------------------------
st.title("Upload → Supabase → Llama-Index (Streamlit)")

tab1, tab2, tab3 = st.tabs(["Upload", "Index / Files", "Query"])

with tab1:
    st.header("Upload a file (goes to Supabase)")
    uploaded = st.file_uploader("Choose a file to upload (txt, md, pdf (text only), etc.)", accept_multiple_files=False)
    if uploaded is not None:
        bytes_data = uploaded.getvalue()
        # Construct a safe path — you can change this
        path = f"{uploaded.name}"
        st.write("Uploading to Supabase bucket:", SUPABASE_BUCKET, "path:", path)
        resp = upload_file_to_supabase(bytes_data, path)
        st.write("Upload response:", resp)
        # Extract text to build index
        text = get_text_from_upload(io.BytesIO(bytes_data))
        if len(text.strip()) == 0:
            st.warning("Uploaded file produced no text with the simple extractor. For PDFs/DOCX add parsing.")
        else:
            st.success(f"Read {len(text)} characters from uploaded file.")
            if st.button("Build index from this file now"):
                doc = Document(text=text, metadata={"source": path})
                st.info("Building index (this may take a few seconds)...")
                index = GPTVectorStoreIndex.from_documents([doc])
                # persist index to a temp file and then upload to supabase as index.json
                idx_file = save_index_to_tempfile(index, filename="index.json")
                with open(idx_file, "rb") as f:
                    upload_file_to_supabase(f.read(), "index.json")
                st.success("Index built and uploaded to Supabase as index.json")
                st.session_state["index"] = index

with tab2:
    st.header("Browse files in Supabase bucket")
    files = list_files_in_bucket("")
    st.write("Files in bucket (first 200):")
    st.write(files)
    if st.button("Load index from Supabase (if index.json exists)"):
        data = download_file_from_supabase("index.json")
        if data is None:
            st.warning("No index.json found in bucket.")
        else:
            # write to temp file and load
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tmp.write(data)
            tmp.flush()
            try:
                idx = load_index_from_disk(tmp.name)
                st.session_state["index"] = idx
                st.success("Index loaded into session.")
            except Exception as e:
                st.error(f"Failed to load index: {e}")

with tab3:
    st.header("Query the index")
    if "index" not in st.session_state:
        st.info("No index in session. Load one from tab 'Index / Files' or build one after uploading.")
    else:
        q = st.text_input("Ask a question about your uploaded data:")
        if q:
            st.write("Querying...")
            try:
                response = st.session_state["index"].as_query_engine().query(q)
                st.write(response.response if hasattr(response, "response") else response)
            except Exception as e:
                st.error(f"Query failed: {e}")
