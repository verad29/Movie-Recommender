import streamlit as st
import pandas as pd
from pathlib import Path

DATA_FILENAME = "omdb.pkl"

@st.cache_data
def load_data(path: Path):
    if path.exists():
        return pd.read_pickle(path)
    return None


# ---------------------- Recommendation utilities (from notebook) ----------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import difflib
import re
import numpy as np
import unicodedata

@st.cache_resource
def prepare_recommender_nbf(df: pd.DataFrame, max_features: int = 5000, metric: str = "cosine", algorithm: str = "brute"):
    """Prepare TF-IDF matrix and NearestNeighbors model following `week9.ipynb` steps.

    Parameters are exposed so the model can be re-built with different TF-IDF sizes or NN metrics.
    """
    columns_select = [
        "Title", "Year", "Genre", "Director", "Actors", "Plot",
        "Language", "Country", "Type", "imdbRating"
    ]
    dfc = df[columns_select].copy()

    text_cols = ["Genre", "Director", "Actors", "Plot", "Language", "Country"]
    dfc[text_cols] = dfc[text_cols].fillna("")

    # build combined text from all selected text columns
    dfc["text"] = dfc[text_cols].apply(lambda row: " ".join([str(v) for v in row.values]), axis=1)

    def _preprocess_text(s: str):
        # Normalize unicode (remove accents), lowercase, remove non-letters, collapse whitespace
        text = str(s)
        text = unicodedata.normalize("NFKD", text)
        text = "".join([c for c in text if not unicodedata.combining(c)])
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    dfc["text_processed"] = dfc["text"].apply(_preprocess_text)

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(dfc["text_processed"])

    nn = NearestNeighbors(metric=metric, algorithm=algorithm)
    nn.fit(tfidf_matrix)

    # map titles to positional indices (0..n-1) to be used with knn matrix rows
    title_to_idx = pd.Series(range(len(dfc)), index=dfc["Title"].astype(str).str.lower()).drop_duplicates()

    titles_list = dfc["Title"].astype(str).tolist()

    return vectorizer, tfidf_matrix, nn, title_to_idx, dfc, titles_list

def get_recommendations_nbf(title: str, dfc: pd.DataFrame, tfidf_matrix, nn: NearestNeighbors, title_to_idx, top_n: int = 5):
    """Return top-n recommendations (nearest neighbors) following notebook approach."""
    if not isinstance(title, str) or title.strip() == "":
        return pd.DataFrame()

    t = title.strip().lower()
    if t not in title_to_idx:
        suggestions = difflib.get_close_matches(title, title_to_idx.index.tolist(), n=5, cutoff=0.5)
        return pd.DataFrame({"suggestions": suggestions})

    # Defensive mapping: title_to_idx should give a positional int, but it may contain labels
    val = title_to_idx[t]
    try:
        idx = int(val)
    except Exception:
        # try to resolve if val is an index label in dfc
        try:
            idx = int(dfc.index.get_loc(val))
        except Exception:
            st.error("Internal error mapping title to dataset row; try a different title.")
            return pd.DataFrame()

    distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=top_n + 1)
    rec_indices = indices.flatten()[1:]
    rec_distances = distances.flatten()[1:]

    # use positional indexing since rec_indices are positional row numbers from NN
    out = dfc.iloc[rec_indices][["Title", "Year", "Genre", "imdbRating", "Plot", "Director", "Actors"]].copy()
    out["similarity"] = (1 - rec_distances).round(3)
    return out.reset_index(drop=True)


# ---------------------- Display helpers ----------------------
def display_recommendations_grid(recs: pd.DataFrame, poster_map: dict, cols: int = 3, img_width: int = 140):
    """Display recommendations in a simple grid with posters under each movie (fixed cols)."""
    if recs is None or recs.empty:
        return

    recs = recs.copy()
    n = len(recs)
    cols = 3  # fixed to 3 columns
    rows = (n + cols - 1) // cols
    idx = 0
    for r in range(rows):
        columns = st.columns(cols)
        for c in range(cols):
            if idx >= n:
                # fill empty column for alignment
                with columns[c]:
                    st.write("")
                idx += 1
                continue
            row = recs.iloc[idx]
            title = str(row.get("Title", ""))
            poster = poster_map.get(title)
            with columns[c]:
                # ensure a consistent visual block for image + title + meta so the expander aligns across columns
                poster_url = poster if (poster and isinstance(poster, str) and poster.strip().lower().startswith("http")) else "https://via.placeholder.com/150x225?text=No+Image"
                # build meta list
                meta = []
                if pd.notna(row.get("Year", None)):
                    meta.append(str(row.get("Year")))
                if pd.notna(row.get("Genre", None)):
                    meta.append(str(row.get("Genre")))
                if pd.notna(row.get("imdbRating", None)):
                    meta.append(f"IMDB: {row.get('imdbRating')}")
                if pd.notna(row.get("similarity", None)):
                    meta.append(f"score: {row.get('similarity')}")

                # escape minimal HTML-sensitive chars
                def _esc(s):
                    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                title_html = _esc(title)
                meta_html = _esc(" • ".join(meta)) if meta else ""

                # fixed-height container (300px) to align expanders across columns
                html_block = f"""
                <div style='min-height:300px; display:flex; flex-direction:column; align-items:center; justify-content:flex-start;'>
                    <img src='{poster_url}' style='max-width:{img_width}px; height:auto; border-radius:6px;'/>
                    <div style='margin-top:8px; font-weight:700; color:#C84630;'>{title_html}</div>
                    <div style='color:#898989; font-size:12px; margin-top:6px;'>{meta_html}</div>
                </div>
                """
                st.markdown(html_block, unsafe_allow_html=True)

                # details expander (plot, director, actors)
                with st.expander("Details", expanded=False):
                    plot = row.get("Plot", "")
                    if pd.notna(plot) and plot:
                        st.write(plot)
                    director = row.get("Director", "")
                    if pd.notna(director) and director:
                        st.markdown(f"**Director:** {director}")
                    actors = row.get("Actors", "")
                    if pd.notna(actors) and actors:
                        st.markdown(f"**Actors:** {actors}")
            idx += 1


def main():
    st.title("Film-aanbeveler")
    st.write("Streamlit app van `week9.ipynb` — content-based recommender using TF-IDF + NearestNeighbors.")

    base_path = Path(__file__).parent
    data_path = base_path / DATA_FILENAME

    df = load_data(data_path)

    if df is None:
        st.warning(f"Data file {DATA_FILENAME} not found in {base_path}. You can upload a file.")
        uploaded = st.file_uploader("Upload omdb.pkl (pickle file)", type=["pkl"])
        if uploaded is not None:
            try:
                df = pd.read_pickle(uploaded)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")

    if df is None:
        st.stop()

    # keep theme CSS (palette)
    st.markdown(
        """
        <style>
        button[data-testid="stButton"] {
            background-color: #C84630 !important;
            color: white !important;
            border-radius: 6px !important;
        }
        h3 { color: #C84630; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Model settings fixed per user preference (hidden UI)
    max_features = 5000
    metric = "cosine"
    algorithm = "brute"

    # Prepare recommender following notebook processing
    vectorizer, tfidf_matrix, nn, title_to_idx, dfc, titles_list = prepare_recommender_nbf(df, max_features=max_features, metric=metric, algorithm=algorithm)

    # Build a poster lookup (first occurrence wins)
    poster_map = {}
    for _, r in df.iterrows():
        t = r.get("Title")
        purl = r.get("Poster")
        if pd.notna(t) and t not in poster_map:
            poster_map[t] = purl

    st.header("Movie recommender")
    st.write("Type (part of) a movie title below and select the best match.")

    # Autocomplete: substring matching, fallback to fuzzy matches
    # use session_state key for the query so we can watch changes and support Enter
    MIN_CHARS_AUTO = 3

    def _on_enter():
        # set a flag that Enter was pressed on the input
        st.session_state.enter_pressed = True

    if "query" not in st.session_state:
        st.session_state.query = ""
    if "selected_title" not in st.session_state:
        st.session_state.selected_title = None
    if "recs" not in st.session_state:
        st.session_state.recs = None
    if "enter_pressed" not in st.session_state:
        st.session_state.enter_pressed = False

    # allow user to choose number of columns for the results grid
    cols_grid = 3  # fixed to 3 columns

    query = st.text_input("Type a movie title", value=st.session_state.query, key="query", on_change=_on_enter)
    top_n = st.slider("Number of recommendations", 1, 20, 5)

    # If user edits the query after selecting a title, clear previous selection/results
    if st.session_state.selected_title:
        if st.session_state.query.strip().lower() != st.session_state.selected_title.strip().lower():
            st.session_state.selected_title = None
            st.session_state.recs = None

    matches = []
    if st.session_state.query:
        ql = st.session_state.query.strip().lower()
        # substring matches first
        matches = [t for t in titles_list if ql in t.lower()]
        # fallback to fuzzy matches
        if not matches:
            matches = difflib.get_close_matches(st.session_state.query, titles_list, n=10, cutoff=0.4)

    # Helper to choose a robust top match from candidates
    def _choose_top_match(q, candidates):
        ql = q.strip().lower()
        # prefer exact
        for t in candidates:
            if t.strip().lower() == ql:
                return t
        # prefer startswith
        for t in candidates:
            if t.strip().lower().startswith(ql):
                return t
        # prefer longest match (more specific)
        return sorted(candidates, key=lambda s: len(s), reverse=True)[0]

    # Auto-select top match when typing (after MIN_CHARS) and show recommendations
    if st.session_state.query and len(st.session_state.query.strip()) >= MIN_CHARS_AUTO and matches:
        top = _choose_top_match(st.session_state.query, matches)
        if st.session_state.selected_title != top:
            st.session_state.selected_title = top
            st.session_state.recs = get_recommendations_nbf(top, dfc, tfidf_matrix, nn, title_to_idx, top_n=top_n)

    # If Enter was pressed, pick top match (if any) and show results
    if st.session_state.enter_pressed:
        st.session_state.enter_pressed = False
        if matches:
            top = _choose_top_match(st.session_state.query, matches)
            st.session_state.selected_title = top
            st.session_state.recs = get_recommendations_nbf(top, dfc, tfidf_matrix, nn, title_to_idx, top_n=top_n)

    # Live typeahead: show top matches as clickable buttons directly under the input
    if matches:
        st.write("### Suggestions (click a title to see recommendations)")
        show = matches[:20]
        ncols = min(5, len(show))
        cols = st.columns(ncols)
        for i, title in enumerate(show):
            with cols[i % ncols]:
                if st.button(title, key=f"sugg_{i}"):
                    # immediately compute recommendations and store in session state
                    st.session_state.selected_title = title
                    st.session_state.recs = get_recommendations_nbf(title, dfc, tfidf_matrix, nn, title_to_idx, top_n=top_n)

    # Display recommendations immediately if available in session state
    if st.session_state.recs is not None and not st.session_state.recs.empty:
        st.subheader(f"Recommended movies for {st.session_state.get('selected_title','')}")
        display_recommendations_grid(st.session_state.recs, poster_map, cols=cols_grid, img_width=140)

    # Inform user when no matches
    if not matches and st.session_state.query:
        st.info("No substring matches — try different words or check spelling.")

    st.markdown("---")
    if st.checkbox("Show dataset preview (for debugging)"):
        st.write(dfc.head(10))

    if st.checkbox("Show dataframe summary"):
        st.write(df.describe(include='all'))

    if st.checkbox("Show full dataframe"):
        st.dataframe(df)


    st.info("To run: `pip install streamlit` then `streamlit run week9_streamlit.py`")


if __name__ == "__main__":
    main()
