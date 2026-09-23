import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_emotion
from src.config import MAX_UPLOAD_SIZE_MB


# ============================================================
# MARKDOWN SAFETY PATCH
# ============================================================
# Streamlit's st.markdown runs content through a Markdown parser
# before honoring unsafe_allow_html. Markdown treats any line
# indented 4+ spaces as a preformatted code block, so indented
# HTML can be printed as literal text instead of being rendered.
#
# This patch strips leading whitespace from every line whenever
# unsafe_allow_html is used, ensuring the HTML renders correctly.

_original_markdown = st.markdown


def _safe_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html"):
        body = "\n".join(
            line.lstrip() for line in str(body).split("\n")
        )

    return _original_markdown(body, *args, **kwargs)


st.markdown = _safe_markdown


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EmoLens",
    page_icon="😊",
    layout="wide",
)


# ============================================================
# THEME
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode


if dark:
    BG = "#171518"
    CARD = "#242124"
    CARD_ALT = "#2B272B"
    TEXT = "#F7F1EE"
    MUTED = "#BDB4B8"
    BORDER = "#3D373C"
    ACCENT = "#FF9B85"
    ACCENT_DARK = "#FFB09E"
    BAR_BG = "#40393D"
    WARNING_BG = "#3A3027"
    WARNING_BORDER = "#B98A58"

else:
    BG = "#FFF9F5"
    CARD = "#FFFFFF"
    CARD_ALT = "#FFF4EE"
    TEXT = "#2D2A2E"
    MUTED = "#746D72"
    BORDER = "#F0DDD5"
    ACCENT = "#FF9B85"
    ACCENT_DARK = "#E9826D"
    BAR_BG = "#F4E8E3"
    WARNING_BG = "#FFF5E8"
    WARNING_BORDER = "#E7B66E"


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
<style>

#MainMenu {{
    visibility: hidden;
}}

header {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

.stApp {{
    background: {BG};
}}

.block-container {{
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}}

.stButton > button {{
    background: {ACCENT};
    color: #2D2A2E;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.65rem 1.4rem;
}}

.stButton > button:hover {{
    background: {ACCENT_DARK};
    color: #2D2A2E;
}}

.stFileUploader {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 0.5rem;
}}

.stFileUploader section {{
    background: {CARD_ALT};
    border-radius: 12px;
    border: 1px dashed {BORDER};
}}

[data-testid="stImage"] img {{
    border-radius: 14px;
}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

col_title, col_theme = st.columns([5, 1])


with col_title:

    st.markdown(
        f"""
        <div style="
            color:{TEXT};
            font-size:2.5rem;
            font-weight:800;
            letter-spacing:-1px;
        ">
            😊 EmoLens
        </div>

        <div style="
            color:{MUTED};
            font-size:1.05rem;
            margin-top:4px;
        ">
            Facial Emotion Recognition using Deep Learning
        </div>
        """,
        unsafe_allow_html=True,
    )


with col_theme:

    if st.button(
        "☀️ Light" if dark else "🌙 Dark",
        use_container_width=True,
    ):
        st.session_state.dark_mode = not dark
        st.rerun()


st.write("")


# ============================================================
# INTRODUCTION CARD
# ============================================================

st.markdown(
    f"""
    <div style="
        background:{CARD};
        border:1px solid {BORDER};
        border-radius:18px;
        padding:22px;
        margin-bottom:20px;
    ">

        <div style="
            color:{TEXT};
            font-size:1.15rem;
            font-weight:750;
            margin-bottom:7px;
        ">
            Understand the emotion in a face
        </div>

        <div style="
            color:{MUTED};
            line-height:1.6;
        ">
            Upload a clear image containing exactly one face.
            EmoLens detects the face and uses a CNN model to classify
            it as Happy, Sad, Angry, Surprise, or Neutral.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    f"""
    <div style="
        color:{TEXT};
        font-size:1.15rem;
        font-weight:750;
        margin-bottom:5px;
    ">
        📤 Upload an image
    </div>

    <div style="
        color:{MUTED};
        font-size:0.88rem;
        margin-bottom:8px;
    ">
        JPG, JPEG or PNG • Maximum size: {MAX_UPLOAD_SIZE_MB} MB
    </div>
    """,
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader(
    "Upload",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    file_size_mb = uploaded_file.size / (1024 * 1024)

    # ========================================================
    # FILE SIZE VALIDATION
    # ========================================================

    if file_size_mb > MAX_UPLOAD_SIZE_MB:

        st.markdown(
            f"""
            <div style="
                background:{WARNING_BG};
                border:1px solid {WARNING_BORDER};
                border-radius:16px;
                padding:18px;
                margin-top:15px;
            ">

                <div style="
                    color:{TEXT};
                    font-weight:750;
                    font-size:1.1rem;
                ">
                    ⚠️ Image too large
                </div>

                <div style="
                    color:{MUTED};
                    margin-top:5px;
                ">
                    Please upload an image smaller than
                    {MAX_UPLOAD_SIZE_MB} MB.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        image = Image.open(uploaded_file).convert("RGB")

        st.write("")


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "✨ Analyze Emotion",
            use_container_width=True,
        ):

            with st.spinner("Analyzing facial expression..."):
                result = predict_emotion(image)


            # =================================================
            # NO FACE
            # =================================================

            if result["status"] == "no_face":

                st.markdown(
                    f"""
                    <div style="
                        background:{WARNING_BG};
                        border:1px solid {WARNING_BORDER};
                        border-radius:18px;
                        padding:22px;
                        margin-top:20px;
                    ">

                        <div style="
                            color:{TEXT};
                            font-size:1.3rem;
                            font-weight:800;
                        ">
                            ⚠️ No face detected
                        </div>

                        <div style="
                            color:{MUTED};
                            margin-top:7px;
                            line-height:1.5;
                        ">
                            EmoLens could not find a clear face in this
                            image. Please upload a clearer facial image.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # =================================================
            # MULTIPLE FACES
            # =================================================

            elif result["status"] == "multiple_faces":

                st.markdown(
                    f"""
                    <div style="
                        background:{WARNING_BG};
                        border:1px solid {WARNING_BORDER};
                        border-radius:18px;
                        padding:22px;
                        margin-top:20px;
                    ">

                        <div style="
                            color:{TEXT};
                            font-size:1.3rem;
                            font-weight:800;
                        ">
                            ⚠️ Multiple faces detected
                        </div>

                        <div style="
                            color:{MUTED};
                            margin-top:7px;
                            line-height:1.5;
                        ">
                            EmoLens analyzes one face at a time.
                            Please upload an image containing exactly
                            one face.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # =================================================
            # SUCCESS
            # =================================================

            elif result["status"] == "success":

                face_box = result["face_box"]


                # =============================================
                # DRAW FACE BOX
                # =============================================

                marked_image = image.copy()

                draw = ImageDraw.Draw(marked_image)

                x, y, w, h = face_box

                draw.rectangle(
                    [x, y, x + w, y + h],
                    outline="#FF8A75",
                    width=5,
                )


                # =============================================
                # ANALYSIS TITLE
                # =============================================

                st.markdown(
                    f"""
                    <div style="
                        color:{TEXT};
                        font-size:1.2rem;
                        font-weight:800;
                        margin-top:25px;
                        margin-bottom:12px;
                    ">
                        🔍 Analysis
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                # =============================================
                # IMAGE CARDS
                # =============================================

                left, right = st.columns(2)


                # ---------------------------------------------
                # UPLOADED IMAGE
                # ---------------------------------------------

                with left:

                    st.markdown(
                        f"""
                        <div style="
                            background:{CARD};
                            border:1px solid {BORDER};
                            border-radius:18px;
                            padding:14px 14px 8px 14px;
                        ">

                            <div style="
                                color:{TEXT};
                                font-weight:700;
                                margin-bottom:10px;
                            ">
                                Uploaded Image
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.image(
                        marked_image,
                        use_container_width=True,
                    )


                # ---------------------------------------------
                # DETECTED FACE
                # ---------------------------------------------

                with right:

                    st.markdown(
                        f"""
                        <div style="
                            background:{CARD};
                            border:1px solid {BORDER};
                            border-radius:18px;
                            padding:14px 14px 8px 14px;
                        ">

                            <div style="
                                color:{TEXT};
                                font-weight:700;
                                margin-bottom:10px;
                            ">
                                Detected Face
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.image(
                        result["face"],
                        use_container_width=True,
                    )


                # =============================================
                # EMOTION RESULT
                # =============================================

                emotion = result["emotion"]
                emoji = result["emoji"]
                confidence = result["confidence"] * 100


                st.markdown(
                    f"""
                    <div style="
                        background:{CARD_ALT};
                        border:1px solid {BORDER};
                        border-radius:20px;
                        padding:25px;
                        margin-top:22px;
                        text-align:center;
                    ">

                        <div style="
                            color:{MUTED};
                            font-size:0.85rem;
                            font-weight:700;
                            letter-spacing:1px;
                            text-transform:uppercase;
                        ">
                            Detected Emotion
                        </div>

                        <div style="
                            font-size:3rem;
                            margin-top:7px;
                        ">
                            {emoji}
                        </div>

                        <div style="
                            color:{TEXT};
                            font-size:2rem;
                            font-weight:800;
                        ">
                            {emotion.capitalize()}
                        </div>

                        <div style="
                            color:{MUTED};
                            margin-top:8px;
                        ">
                            Prediction confidence
                        </div>

                        <div style="
                            color:{ACCENT_DARK};
                            font-size:1.5rem;
                            font-weight:800;
                            margin-top:3px;
                        ">
                            {confidence:.3f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                # =============================================
                # PROBABILITIES
                # =============================================

                st.markdown(
                    f"""
                    <div style="
                        background:{CARD};
                        border:1px solid {BORDER};
                        border-radius:20px;
                        padding:22px;
                        margin-top:18px;
                    ">

                        <div style="
                            color:{TEXT};
                            font-size:1.15rem;
                            font-weight:800;
                            margin-bottom:18px;
                        ">
                            Emotion Probabilities
                        </div>
                    """,
                    unsafe_allow_html=True,
                )


                probabilities = result["probabilities"]


                emotion_info = {
                    "angry": ("😠", "#E98B7A"),
                    "happy": ("😊", "#E8B85C"),
                    "neutral": ("😐", "#9BA7B4"),
                    "sad": ("😢", "#7FA7D8"),
                    "surprise": ("😮", "#B39AD8"),
                }


                sorted_probs = sorted(
                    probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )


                # =============================================
                # PROBABILITY BARS
                # =============================================

                for label, probability in sorted_probs:

                    bar_emoji, bar_color = emotion_info[label]

                    percentage = probability * 100


                    # Prevent microscopic floating-point values
                    # from creating awkward CSS widths.

                    if percentage < 0.05:
                        display_percentage = "0.000"
                        bar_width = 0

                    else:
                        display_percentage = f"{percentage:.3f}"
                        bar_width = min(percentage, 100)


                    st.markdown(
                        f"""
                        <div style="margin-bottom:15px;">

                            <div style="
                                display:flex;
                                justify-content:space-between;
                                margin-bottom:5px;
                            ">

                                <span style="
                                    color:{TEXT};
                                    font-weight:650;
                                ">
                                    {bar_emoji} {label.capitalize()}
                                </span>

                                <span style="
                                    color:{MUTED};
                                    font-size:0.85rem;
                                    font-weight:600;
                                ">
                                    {display_percentage}%
                                </span>

                            </div>


                            <div style="
                                width:100%;
                                height:9px;
                                background:{BAR_BG};
                                border-radius:20px;
                                overflow:hidden;
                            ">

                                <div style="
                                    width:{bar_width}%;
                                    height:100%;
                                    background:{bar_color};
                                    border-radius:20px;
                                "></div>

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


                # =============================================
                # CLOSE PROBABILITY CARD
                # =============================================

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div style="
        color:{MUTED};
        text-align:center;
        font-size:0.82rem;
        margin-top:40px;
        padding-top:15px;
        border-top:1px solid {BORDER};
    ">
        EmoLens • CNN-based Facial Emotion Recognition •
        5 Emotion Classes
    </div>
    """,
    unsafe_allow_html=True,
)