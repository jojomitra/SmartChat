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
# --- Robust Supabase helpers (replace your current helpers with these) ---
import io

def upload_file_to_supabase(file_bytes: bytes, path: str):
    """
    Upload bytes to Supabase Storage at `path`.
    This function:
      - first attempts a normal upload(...)
      - if that raises because the object exists, it attempts update(...)
      - returns the raw response dict/object for inspection
    """
    bucket = supabase.storage.from_(SUPABASE_BUCKET)
    file_like = io.BytesIO(file_bytes)

    # Try the simplest call first (no upsert kw)
    try:
        res = bucket.upload(path, file_like)
        return res
    except TypeError as te:
        # some versions accept different args; fall back to upload without kwargs
        try:
            file_like.seek(0)
            res = bucket.upload(path, file_like)  # retry
            return res
        except Exception as e:
            # if upload fails because file already exists, try update()
            err_text = str(e).lower()
            if "already exists" in err_text or "object already exists" in err_text or "409" in err_text:
                try:
                    file_like.seek(0)
                    res2 = bucket.update(path, file_like)
                    return res2
                except Exception as e2:
                    raise RuntimeError(f"Upload failed and update fallback failed: {e2}") from e2
            else:
                # unknown error - try update anyway
                try:
                    file_like.seek(0)
                    res2 = bucket.update(path, file_like)
                    return res2
                except Exception as e2:
                    raise RuntimeError(f"Upload failed and update fallback also failed: {e2}") from e2
    except Exception as exc:
        # Generic fallback: try update (replace existing file)
        try:
            file_like.seek(0)
            resu = bucket.update(path, file_like)
            return resu
        except Exception as e2:
            # give a helpful error with both exceptions
            raise RuntimeError(f"upload() error: {exc}; update() fallback error: {e2}") from e2


def download_file_from_supabase(path: str) -> bytes | None:
    """
    Download bytes for a given path. Handles different return formats.
    Returns bytes or None (if not found).
    """
    bucket = supabase.storage.from_(SUPABASE_BUCKET)
    try:
        data = bucket.download(path)
        # older/newer clients may return a bytes-like or a dict {'data': Buffer}
        if data is None:
            return None
        # If the returned object has a .read() (like a HTTP response)
        if hasattr(data, "read"):
            return data.read()
        # If dictionary with 'data' key
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        # If it's already bytes/bytearray
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        # otherwise attempt to convert to bytes
        try:
            return bytes(data)
        except Exception:
            return None
    except Exception as e:
        # Not found or other error
        err = str(e).lower()
        if "not found" in err or "404" in err:
            return None
        # re-raise for unexpected failures so you can see the message in Streamlit
        raise


def list_files_in_bucket(prefix: str = ""):
    """
    List objects in bucket. Many supabase clients accept list(prefix) or list().
    Return a python list (or empty list).
    """
    bucket = supabase.storage.from_(SUPABASE_BUCKET)
    try:
        objs = bucket.list(prefix)
        return objs
    except TypeError:
        # some versions expect no arg
        try:
            objs = bucket.list()
            return objs
        except Exception as e:
            raise RuntimeError(f"listing bucket failed: {e}") from e
    except Exception as e:
        # return empty list rather than crash
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
