# streamlit_app.py
import os
import io
import json
import tempfile
import pickle
import traceback
from typing import Optional

import streamlit as st

# ---------------------------
# CONFIG / SECRETS (Streamlit Secrets recommended)
# ---------------------------
# Put these in Streamlit Cloud Secrets (or environment variables if hosted elsewhere)
# SUPABASE_URL, SUPABASE_KEY, OPENAI_KEY, SUPABASE_BUCKET (optional, default "uploads")
SUPABASE_URL = st.secrets.get("SUPABASE_URL", None)
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", None)
OPENAI_KEY = st.secrets.get("OPENAI_KEY", None)
SUPABASE_BUCKET = st.secrets.get("SUPABASE_BUCKET", "uploads")

if not (SUPABASE_URL and SUPABASE_KEY and OPENAI_KEY):
    st.warning("Missing secrets. Set SUPABASE_URL, SUPABASE_KEY and OPENAI_KEY in Streamlit Secrets.")
    st.stop()

# ensure OpenAI env var for libs that expect it
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# ---------------------------
# Import supabase client
# ---------------------------
try:
    from supabase import create_client as create_supabase_client
except Exception:
    st.error("Failed to import supabase.create_client. Install `supabase` package or pin a supported version.")
    st.stop()

supabase = create_supabase_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Robust llama-index imports / fallbacks
# ---------------------------
Document = None
IndexVectorClass = None
IndexSimpleClass = None

# --- DIAGNOSTIC: show where llama_index is coming from and what version (paste output here) ---
import importlib, sys, pkgutil, traceback

try:
    li = importlib.import_module("llama_index")
    st.write("llama_index module object:", li)
    st.write("llama_index __file__:", getattr(li, "__file__", None))
    st.write("llama_index __path__:", getattr(li, "__path__", None))
    st.write("Sample dir(llama_index):", [n for n in dir(li)][:200])
except Exception as e:
    st.write("Importing llama_index raised:", e)
    st.write(traceback.format_exc())

# Try to show pip-installed distribution version for llama-index package
try:
    import pkg_resources
    dist = pkg_resources.get_distribution("llama-index")
    st.write("pip distribution llama-index:", dist.project_name, dist.version, dist.location)
except Exception as e:
    st.write("pkg_resources couldn't find llama-index distribution or failed:", e)


# Document fallbacks
try:
    from llama_index import Document as _Doc  # top-level (some versions)
    Document = _Doc
except Exception:
    try:
        from llama_index.core.schema import Document as _Doc2
        Document = _Doc2
    except Exception:
        try:
            from llama_index.schema import Document as _Doc3
            Document = _Doc3
        except Exception:
            Document = None

# Index class fallbacks (vector store and older simple vector)
try:
    from llama_index import GPTVectorStoreIndex as _IndexVec
    IndexVectorClass = _IndexVec
except Exception:
    try:
        from llama_index.indices.vector_store import GPTVectorStoreIndex as _IndexVec2
        IndexVectorClass = _IndexVec2
    except Exception:
        IndexVectorClass = None

# Older simple index fallback
try:
    from llama_index import GPTSimpleVectorIndex as _IndexSimple
    IndexSimpleClass = _IndexSimple
except Exception:
    try:
        from llama_index.indices.simple import GPTSimpleVectorIndex as _IndexSimple2
        IndexSimpleClass = _IndexSimple2
    except Exception:
        IndexSimpleClass = None

def get_index_class():
    """Return the best available index class (vector preferred, then simple)."""
    if IndexVectorClass is not None:
        return IndexVectorClass
    if IndexSimpleClass is not None:
        return IndexSimpleClass
    return None

# If Document is missing, show an actionable error
if Document is None:
    st.error(
        "Could not import `Document` from installed llama-index. "
        "Pin a working version in requirements.txt (recommended: `llama-index==0.10.17`) "
        "or install a version that exposes Document. See logs for details."
    )
    st.stop()


# ---------------------------
# Robust Supabase helpers (support multiple client APIs)
# ---------------------------
def upload_file_to_supabase(file_bytes: bytes, path: str):
    """
    Upload bytes to Supabase Storage at `path`. Tries multiple patterns to
    support different supabase-py versions (bytes, file-like, update with path).
    Returns the client's response on success or raises RuntimeError with detail.
    """
    bucket = supabase.storage.from_(SUPABASE_BUCKET)

    # Try upload with raw bytes first
    try:
        return bucket.upload(path, file_bytes)
    except Exception as e1:
        # try file-like
        try:
            bio = io.BytesIO(file_bytes)
            bio.seek(0)
            return bucket.upload(path, bio)
        except Exception as e2:
            # try update with bytes
            try:
                return bucket.update(path, file_bytes)
            except Exception as e3:
                # try update with temp file path
                tmp = None
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False)
                    tmp.write(file_bytes)
                    tmp.flush()
                    tmp.close()
                    return bucket.update(path, tmp.name)
                except Exception as e4:
                    # Provide combined error for debugging
                    msg = (
                        "All upload/update attempts failed:\n"
                        f"upload(bytes) error: {repr(e1)}\n"
                        f"upload(BytesIO) error: {repr(e2)}\n"
                        f"update(bytes) error: {repr(e3)}\n"
                        f"update(path) error: {repr(e4)}\n"
                    )
                    raise RuntimeError(msg)
                finally:
                    if tmp is not None:
                        try:
                            os.unlink(tmp.name)
                        except Exception:
                            pass


def download_file_from_supabase(path: str) -> Optional[bytes]:
    """
    Download bytes for a given path. Handles different client return shapes.
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
        # fallback
        return bytes(data)
    except Exception as e:
        estr = str(e).lower()
        if "not found" in estr or "404" in estr:
            return None
        # surface other exceptions to the app
        raise


def list_files_in_bucket(prefix: str = ""):
    """
    List objects in bucket. Supports list(prefix) or list() signatures.
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
        return []


# ---------------------------
# Text extraction for uploaded files
# ---------------------------
def get_text_from_upload(file_obj) -> str:
    """
    Robust extractor for uploaded content.
    Accepts bytes, bytearray, str (path or text), and file-like objects (BytesIO).
    Returns decoded text (possibly empty) and never raises (returns an explanatory string on error).
    """
    try:
        # path string
        if isinstance(file_obj, (str, os.PathLike)):
            try:
                with open(file_obj, "rb") as fh:
                    raw = fh.read()
            except Exception as e:
                return f"[error opening file path: {e}]"

        elif isinstance(file_obj, (bytes, bytearray)):
            raw = bytes(file_obj)

        elif hasattr(file_obj, "read"):
            try:
                raw = file_obj.read()
            except Exception as e:
                return f"[error reading file-like object: {e}]"
            # if read() returned str already
            if isinstance(raw, str):
                return raw
        elif isinstance(file_obj, str):
            return file_obj
        else:
            return f"[unsupported upload object type: {type(file_obj)}]"

        if raw is None:
            return ""

        if isinstance(raw, (bytes, bytearray)):
            try:
                return raw.decode("utf-8")
            except Exception:
                try:
                    return raw.decode("latin-1")
                except Exception:
                    return repr(raw)[:1000]
        elif isinstance(raw, str):
            return raw
        else:
            try:
                return bytes(raw).decode("utf-8", errors="ignore")
            except Exception:
                return f"[unhandled raw type: {type(raw)}]"
    except Exception as e:
        return f"[extractor error: {e}]"


# ---------------------------
# Index persistence helpers (robust)
# ---------------------------
def save_index_to_tempfile(index_obj, filename="index.bin"):
    """
    Try save_to_disk/save or pickle as fallback. Return the filepath of saved artifact.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".bin")
    tmp_path = tmp.name
    tmp.close()

    # try library save methods
    try:
        if hasattr(index_obj, "save_to_disk"):
            index_obj.save_to_disk(tmp_path)
            return tmp_path
    except Exception:
        pass

    try:
        if hasattr(index_obj, "save"):
            index_obj.save(tmp_path)
            return tmp_path
    except Exception:
        pass

    # fallback: pickle
    try:
        with open(tmp_path, "wb") as fh:
            pickle.dump(index_obj, fh)
        return tmp_path
    except Exception as e:
        raise RuntimeError(f"Failed to persist index to disk (tried save_to_disk/save/pickle): {e}") from e


def load_index_from_disk(path):
    """
    Try class-specific load methods (load_from_disk/load) first, then fallback to pickle.
    """
    cls = get_index_class()
    # try class methods if available
    if cls is not None:
        for method in ("load_from_disk", "load", "from_disk", "from_file"):
            if hasattr(cls, method):
                try:
                    fn = getattr(cls, method)
                    return fn(path)
                except Exception:
                    pass
    # fallback to pickle
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        raise RuntimeError(f"Failed to load index from disk (fallback pickle failed): {e}") from e


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Upload → Supabase → Llama-Index", layout="wide")
st.title("Upload → Supabase → Llama-Index (Streamlit)")

tab1, tab2, tab3 = st.tabs(["Upload", "Index / Files", "Query"])

with tab1:
    st.header("Upload a file (goes to Supabase)")
    uploaded = st.file_uploader("Choose a file to upload (txt, md, pdf (text only), etc.)", accept_multiple_files=False)
    if uploaded is not None:
        # show debug info about upload
        try:
            bytes_data = uploaded.getvalue()
            st.write("DEBUG: uploaded type:", type(bytes_data), "length:", len(bytes_data))
        except Exception as e:
            st.error(f"Failed to read uploaded bytes: {e}")
            bytes_data = None

        if bytes_data:
            # safe path name
            path = f"{uploaded.name}"
            st.write("Uploading to Supabase bucket:", SUPABASE_BUCKET, "path:", path)
            try:
                resp = upload_file_to_supabase(bytes_data, path)
                st.write("Upload response (raw):", resp)
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

            # Extract text to build index
            text = get_text_from_upload(io.BytesIO(bytes_data))
            if len(text.strip()) == 0:
                st.warning("Uploaded file produced no text with the extractor. For PDFs/DOCX add parsing libraries.")
            else:
                st.success(f"Read {len(text)} characters from uploaded file.")
                if st.button("Build index from this file now"):
                    try:
                        doc = Document(text=text, metadata={"source": path})
                    except Exception as e:
                        st.error(f"Failed to build Document object: {e}")
                        st.stop()

                    st.info("Building index (this may take a few seconds)...")

                    IndexClass = get_index_class()
                    if IndexClass is None:
                        st.error(
                            "No supported index class found in installed llama-index. "
                            "Recommended: pin `llama-index==0.10.17` in requirements.txt and redeploy."
                        )
                        st.stop()

                    # Try from_documents classmethod first, else attempt constructor
                    try:
                        if hasattr(IndexClass, "from_documents"):
                            index = IndexClass.from_documents([doc])
                        else:
                            # some older classes expect a list in constructor
                            index = IndexClass([doc])
                    except Exception as e:
                        st.error(f"Index construction failed: {e}")
                        st.write(traceback.format_exc())
                        st.stop()

                    # persist index and upload to Supabase
                    try:
                        idx_file = save_index_to_tempfile(index, filename="index.bin")
                        with open(idx_file, "rb") as fh:
                            upload_file_to_supabase(fh.read(), "index.bin")
                        st.success("Index built and uploaded to Supabase as index.bin")
                        st.session_state["index"] = index
                    except Exception as e:
                        st.error(f"Failed to save/upload index: {e}")
                        st.write(traceback.format_exc())

with tab2:
    st.header("Browse files in Supabase bucket")
    try:
        files = list_files_in_bucket("")
        st.write("Files in bucket (first 200):")
        st.write(files)
    except Exception as e:
        st.error(f"Failed to list bucket: {e}")

    if st.button("Load index from Supabase (if index.bin exists)"):
        try:
            data = download_file_from_supabase("index.bin")
            if data is None:
                st.warning("No index.bin found in bucket.")
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
                tmp.write(data)
                tmp.flush()
                tmp.close()
                try:
                    idx = load_index_from_disk(tmp.name)
                    st.session_state["index"] = idx
                    st.success("Index loaded into session.")
                except Exception as e:
                    st.error(f"Failed to load index: {e}")
                    st.write(traceback.format_exc())
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.write(traceback.format_exc())

with tab3:
    st.header("Query the index")
    if "index" not in st.session_state:
        st.info("No index in session. Load one from tab 'Index / Files' or build one after uploading.")
    else:
        q = st.text_input("Ask a question about your uploaded data:")
        if q:
            st.write("Querying...")
            idx = st.session_state["index"]
            # try as_query_engine -> query or .query
            try:
                if hasattr(idx, "as_query_engine"):
                    qe = idx.as_query_engine()
                    # query interfaces vary: .query or .run
                    if hasattr(qe, "query"):
                        resp = qe.query(q)
                    elif hasattr(qe, "run"):
                        resp = qe.run(q)
                    else:
                        resp = qe(q)
                elif hasattr(idx, "query"):
                    resp = idx.query(q)
                elif hasattr(idx, "as_retriever"):
                    retriever = idx.as_retriever()
                    # many retriever objects have get_relevant_documents + then call llm
                    if hasattr(idx, "query"):
                        resp = idx.query(q)
                    else:
                        resp = "Index supports retriever but no standard query; examine index API."
                else:
                    resp = "Index object does not expose a known query method."
                # try to surface textual answer
                if isinstance(resp, str):
                    st.write(resp)
                else:
                    # some response objects have .response or .get_response()
                    if hasattr(resp, "response"):
                        st.write(resp.response)
                    elif hasattr(resp, "answer"):
                        st.write(resp.answer)
                    else:
                        st.write(resp)
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.write(traceback.format_exc())

# End of file
