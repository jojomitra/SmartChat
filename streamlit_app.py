# streamlit_app.py
import os
import io
import tempfile
import pickle
import traceback
import importlib
import pkgutil
from typing import Optional

import streamlit as st

# ---------------------------
# CONFIG / SECRETS (Streamlit Secrets recommended)
# ---------------------------
# Add to Streamlit Cloud Secrets (or set env vars):
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
    st.error("Failed to import supabase.create_client. Install `supabase` package or pin a supported version in requirements.txt.")
    st.stop()

supabase = create_supabase_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Dynamic discovery for llama-index objects (Document and an Index class)
# ---------------------------
DetectedDocument = None
DetectedIndexClass = None
DetectedIndexName = None
DetectedIndexModule = None

def find_in_package(pkg, name_substrings):
    """Scan top-level attrs and one-level submodules in pkg for attributes containing any substring."""
    # top-level
    for attr in dir(pkg):
        for sub in name_substrings:
            if sub.lower() in attr.lower():
                try:
                    obj = getattr(pkg, attr)
                    if isinstance(obj, type) or callable(obj):
                        return obj, getattr(pkg, "__name__", "<pkg>"), attr
                except Exception:
                    pass
    # one-level submodules
    if getattr(pkg, "__path__", None) is None:
        return None, None, None
    for finder, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        fullname = f"{pkg.__name__}.{modname}"
        try:
            subpkg = importlib.import_module(fullname)
        except Exception:
            continue
        for attr in dir(subpkg):
            for sub in name_substrings:
                if sub.lower() in attr.lower():
                    try:
                        obj = getattr(subpkg, attr)
                        if isinstance(obj, type) or callable(obj):
                            return obj, fullname, attr
                    except Exception:
                        pass
    return None, None, None

# Try to import llama_index and print diagnostics in UI
try:
    llama_pkg = importlib.import_module("llama_index")
    st.write("Detected llama_index package path:", getattr(llama_pkg, "__path__", None))
    st.write("Top-level names (slice):", [n for n in dir(llama_pkg)][:200])
except Exception as e:
    st.write("Failed to import llama_index package:", e)
    llama_pkg = None

# Find Document
if llama_pkg is not None:
    cand_doc_names = ["Document", "Node", "TextNode"]
    for nm in cand_doc_names:
        if hasattr(llama_pkg, nm):
            try:
                DetectedDocument = getattr(llama_pkg, nm)
                st.write(f"Found Document-like object at llama_index.{nm}")
                break
            except Exception:
                DetectedDocument = None
    if DetectedDocument is None:
        doc_obj, doc_mod, doc_attr = find_in_package(llama_pkg, ["Document"])
        if doc_obj:
            DetectedDocument = doc_obj
            st.write(f"Found Document in {doc_mod}.{doc_attr}")

# Find an index class
index_search_terms = [
    "GPTVectorStoreIndex", "GPTSimpleVectorIndex", "SimpleVectorIndex",
    "VectorStoreIndex", "VectorIndex", "GPTVectorIndex", "SimpleIndex", "GPTSimpleIndex"
]
if llama_pkg is not None:
    for nm in index_search_terms:
        if hasattr(llama_pkg, nm):
            try:
                DetectedIndexClass = getattr(llama_pkg, nm)
                DetectedIndexName = nm
                DetectedIndexModule = "llama_index"
                st.write(f"Found Index class on top-level: llama_index.{nm}")
                break
            except Exception:
                DetectedIndexClass = None
    if DetectedIndexClass is None:
        idx_obj, idx_mod, idx_attr = find_in_package(llama_pkg, index_search_terms)
        if idx_obj:
            DetectedIndexClass = idx_obj
            DetectedIndexName = idx_attr
            DetectedIndexModule = idx_mod
            st.write(f"Found Index class in {idx_mod}.{idx_attr}")

# Report status
if DetectedDocument is None:
    st.warning("Document class NOT found inside llama-index. Some features will use OpenAI fallback instead.")
else:
    st.write(f"Using detected Document: {DetectedDocument}")

if DetectedIndexClass is None:
    st.warning("No Index class detected in llama-index. The app will offer an OpenAI fallback for queries.")
else:
    st.write(f"Using detected Index class `{DetectedIndexName}` from module `{DetectedIndexModule}`")

# ---------------------------
# Robust Supabase helpers (support multiple client APIs)
# ---------------------------
def upload_file_to_supabase(file_bytes: bytes, path: str):
    """
    Upload bytes to Supabase Storage at `path`. Tries multiple patterns to
    support different supabase-py versions.
    """
    bucket = supabase.storage.from_(SUPABASE_BUCKET)
    # 1) upload bytes
    try:
        return bucket.upload(path, file_bytes)
    except Exception as e1:
        # 2) upload BytesIO
        try:
            bio = io.BytesIO(file_bytes)
            bio.seek(0)
            return bucket.upload(path, bio)
        except Exception as e2:
            # 3) update bytes
            try:
                return bucket.update(path, file_bytes)
            except Exception as e3:
                # 4) update path
                tmp = None
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False)
                    tmp.write(file_bytes)
                    tmp.flush()
                    tmp.close()
                    return bucket.update(path, tmp.name)
                except Exception as e4:
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
        return bytes(data)
    except Exception as e:
        estr = str(e).lower()
        if "not found" in estr or "404" in estr:
            return None
        raise

def list_files_in_bucket(prefix: str = ""):
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
    try:
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
# Index persistence helpers
# ---------------------------
def save_index_to_tempfile(index_obj, filename="index.bin"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".bin")
    tmp_path = tmp.name
    tmp.close()

    # library save_to_disk
    try:
        if hasattr(index_obj, "save_to_disk"):
            index_obj.save_to_disk(tmp_path)
            return tmp_path
    except Exception:
        pass
    # library save
    try:
        if hasattr(index_obj, "save"):
            index_obj.save(tmp_path)
            return tmp_path
    except Exception:
        pass
    # fallback pickle
    try:
        with open(tmp_path, "wb") as fh:
            pickle.dump(index_obj, fh)
        return tmp_path
    except Exception as e:
        raise RuntimeError(f"Failed to persist index (save/pickle attempts failed): {e}") from e

def load_index_from_disk(path):
    # try detected class load methods if available
    if DetectedIndexClass is not None:
        for method in ("load_from_disk", "load", "from_disk", "from_file"):
            if hasattr(DetectedIndexClass, method):
                try:
                    fn = getattr(DetectedIndexClass, method)
                    return fn(path)
                except Exception:
                    pass
    # fallback pickle
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        raise RuntimeError(f"Failed to load index from disk (pickle fallback failed): {e}") from e

# ---------------------------
# Simple OpenAI fallback (uses uploaded text directly)
# ---------------------------
def simple_llm_answer(context_text: str, question: str) -> str:
    try:
        import openai
        openai.api_key = OPENAI_KEY
        # Trim context if too long
        max_context_chars = 14000
        if len(context_text) > max_context_chars:
            context_text = context_text[-max_context_chars:]
        prompt = (
            "You are a helpful assistant. Use the following context to answer the user's question.\n\n"
            f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}\n\nAnswer concisely."
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # replace if unavailable
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OpenAI error: {e}]"

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Upload → Supabase → Llama-Index", layout="wide")
st.title("Upload → Supabase → Llama-Index (Streamlit)")

tab1, tab2, tab3 = st.tabs(["Upload", "Index / Files", "Query"])

with tab1:
    st.header("Upload a file (goes to Supabase)")
    uploaded = st.file_uploader("Choose a file (txt, md, small pdf text-only, docx).", accept_multiple_files=False)
    if uploaded is not None:
        try:
            bytes_data = uploaded.getvalue()
            st.write("DEBUG: uploaded type:", type(bytes_data), "length:", len(bytes_data))
        except Exception as e:
            st.error(f"Failed to read uploaded bytes: {e}")
            bytes_data = None

        if bytes_data:
            path = f"{uploaded.name}"
            st.write("Uploading to Supabase bucket:", SUPABASE_BUCKET, "path:", path)
            try:
                resp = upload_file_to_supabase(bytes_data, path)
                st.write("Upload response (raw):", resp)
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

            # extract text and store for fallback
            text = get_text_from_upload(io.BytesIO(bytes_data))
            st.session_state["uploaded_text"] = text
            if len(text.strip()) == 0:
                st.warning("Uploaded file produced no text with the extractor. For PDFs/DOCX add parsing libs (pdfplumber, python-docx).")
            else:
                st.success(f"Read {len(text)} characters from uploaded file.")
                if st.button("Build index from this file now"):
                    if DetectedDocument is None or DetectedIndexClass is None:
                        st.warning("Index class or Document class not detected; cannot build llama-index index. Use OpenAI fallback or pin a compatible llama-index version.")
                    else:
                        try:
                            doc = DetectedDocument(text=text, metadata={"source": path})
                        except Exception:
                            # older Document constructors may accept (text, metadata) differently; try fallback
                            try:
                                doc = DetectedDocument(text, {"source": path})
                            except Exception as e:
                                st.error(f"Failed to instantiate Document object: {e}")
                                st.write(traceback.format_exc())
                                st.stop()

                        st.info("Building index (this may take a few seconds)...")
                        try:
                            if hasattr(DetectedIndexClass, "from_documents"):
                                index = DetectedIndexClass.from_documents([doc])
                            else:
                                index = DetectedIndexClass([doc])
                        except Exception as e:
                            st.error(f"Index construction failed with detected Index class ({DetectedIndexName}): {e}")
                            st.write(traceback.format_exc())
                            st.stop()
                        # persist and upload
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
    st.header("Query the index / uploaded document")
    # prefer vector index if present in session
    if "index" in st.session_state:
        q = st.text_input("Ask a question about your uploaded data (vector index):")
        if q:
            st.write("Querying vector index...")
            idx = st.session_state["index"]
            try:
                if hasattr(idx, "as_query_engine"):
                    qe = idx.as_query_engine()
                    if hasattr(qe, "query"):
                        resp = qe.query(q)
                    elif hasattr(qe, "run"):
                        resp = qe.run(q)
                    else:
                        resp = qe(q)
                elif hasattr(idx, "query"):
                    resp = idx.query(q)
                else:
                    resp = "Index object does not expose a known query method."
                if isinstance(resp, str):
                    st.write(resp)
                else:
                    if hasattr(resp, "response"):
                        st.write(resp.response)
                    elif hasattr(resp, "answer"):
                        st.write(resp.answer)
                    else:
                        st.write(resp)
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.write(traceback.format_exc())
    else:
        # fallback to OpenAI using uploaded_text
        if st.session_state.get("uploaded_text"):
            st.info("No vector index loaded. Using OpenAI fallback on uploaded text.")
            q = st.text_input("Ask a question (using uploaded file as context):")
            if q:
                with st.spinner("Calling OpenAI..."):
                    ans = simple_llm_answer(st.session_state["uploaded_text"], q)
                    st.write(ans)
        else:
            st.info("No index loaded and no uploaded text available. Upload a file in 'Upload' tab first.")

# end of file
