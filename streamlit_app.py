# streamlit_app.py
import os
import io
import json
import tempfile
import pickle
import streamlit as st
import importlib

from supabase import create_client

Document = None
GPTVectorStoreIndex = None
GPTSimpleVectorIndex = None

# ---------- Robust llama-index imports & helpers ----------
import tempfile
import pickle
import streamlit as st

# Try several known import locations for Document and index classes
Document = None
GPTVectorStoreIndex = None
GPTSimpleVectorIndex = None

# Document import fallbacks
try:
    from llama_index import Document
except Exception:
    try:
        from llama_index.core.schema import Document
    except Exception:
        try:
            from llama_index.schema import Document
        except Exception:
            Document = None

# GPTVectorStoreIndex fallbacks
try:
    from llama_index import GPTVectorStoreIndex as _gv
    GPTVectorStoreIndex = _gv
except Exception:
    try:
        from llama_index.indices.vector_store import GPTVectorStoreIndex as _gv
        GPTVectorStoreIndex = _gv
    except Exception:
        GPTVectorStoreIndex = None

# GPTSimpleVectorIndex (older name) fallbacks
try:
    from llama_index import GPTSimpleVectorIndex as _gs
    GPTSimpleVectorIndex = _gs
except Exception:
    try:
        from llama_index import SimpleVectorIndex as _gs
        GPTSimpleVectorIndex = _gs
    except Exception:
        try:
            from llama_index.indices.simple import GPTSimpleVectorIndex as _gs
            GPTSimpleVectorIndex = _gs
        except Exception:
            GPTSimpleVectorIndex = None

# Choose whichever Index class is available
def get_index_class():
    for cls in (GPTVectorStoreIndex, GPTSimpleVectorIndex):
        if cls is not None:
            return cls
    return None

# Save index to a temp file robustly
def save_index_to_tempfile(index, filename="index.bin"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".bin")
    path = tmp.name
    tmp.close()
    # Try library-provided save methods first
    try:
        if hasattr(index, "save_to_disk"):
            index.save_to_disk(path)
            return path
    except Exception:
        pass
    try:
        if hasattr(index, "save"):
            index.save(path)
            return path
    except Exception:
        pass
    # Fallback: pickle the index object
    try:
        with open(path, "wb") as f:
            pickle.dump(index, f)
        return path
    except Exception as e:
        raise RuntimeError(f"Failed to persist index to disk (tried save_to_disk/save/pickle): {e}") from e

# Load index from disk robustly
def load_index_from_disk(path):
    # Try class-specific load methods
    cls = get_index_class()
    if cls is not None:
        # try load_from_disk or load
        for method in ("load_from_disk", "load", "from_disk"):
            try:
                fn = getattr(cls, method)
                return fn(path)
            except Exception:
                pass
    # fallback to pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load index from disk (no index class worked, pickle fallback failed): {e}") from e


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

def get_text_from_upload(file_obj) -> str:
    """
    Robust extractor for uploaded content.
    Accepts:
      - raw bytes or bytearray
      - str (already text)
      - file-like objects with .read() (BytesIO, TextIO, uploaded file)
      - path string to a file on disk
    Returns decoded text (possibly empty) and never raises (returns explanatory string on error).
    """
    try:
        # If a path was provided, open the file
        if isinstance(file_obj, (str, os.PathLike)):
            try:
                with open(file_obj, "rb") as fh:
                    raw = fh.read()
            except Exception as e:
                return f"[error opening file path: {e}]"

        # bytes or bytearray -> decode
        elif isinstance(file_obj, (bytes, bytearray)):
            raw = bytes(file_obj)

        # file-like object with read()
        elif hasattr(file_obj, "read"):
            try:
                # some streams are text streams, some are bytes streams
                raw = file_obj.read()
            except Exception as e:
                return f"[error reading file-like object: {e}]"

            # If read() returned a string already, return it
            if isinstance(raw, str):
                return raw

            # else fall through to decode bytes

        # If it's already a Python str
        elif isinstance(file_obj, str):
            return file_obj

        else:
            return f"[unsupported upload object type: {type(file_obj)}]"

        # At this point, raw should be bytes or bytearray
        if raw is None:
            return ""

        if isinstance(raw, (bytes, bytearray)):
            # try utf-8 then latin-1 as fallback
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    return raw.decode("latin-1")
                except Exception:
                    # as last resort, return repr truncated
                    return repr(raw)[:1000]
        elif isinstance(raw, str):
            return raw
        else:
            # Unexpected type (e.g. memoryview)
            try:
                return bytes(raw).decode("utf-8", errors="ignore")
            except Exception:
                return f"[unhandled raw type: {type(raw)}]"
    except Exception as e:
        return f"[extractor error: {e}]"

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
