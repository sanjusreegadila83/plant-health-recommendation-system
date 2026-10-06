"""Streamlit interface for the Plant Health Recommendation System."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from geopy.exc import GeopyError
from geopy.geocoders import Nominatim
from PIL import Image, UnidentifiedImageError

from model import ModelNotFoundError, predict_plant_health
from recommendation import (
    get_ai_recommendation,
    get_environment_advice,
    get_recommendations,
    get_weather,
)

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:
    folium = None
    st_folium = None

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "plant_model.pkl"
BACKGROUND_IMAGE_PATH = PROJECT_DIR / "assets" / "forest_background.jpg"
ALLOWED_FORMATS = {"JPEG", "PNG"}

st.set_page_config(
    page_title="Plant Health Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

background_image = base64.b64encode(BACKGROUND_IMAGE_PATH.read_bytes()).decode("ascii")
st.markdown(
    f"""
    <style>
      .stApp {{
        background-image: linear-gradient(rgba(8, 12, 10, .74), rgba(8, 12, 10, .82)),
          url("data:image/jpeg;base64,{background_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
      }}
      .main .block-container {{max-width: 1180px; padding-top: 1.6rem;}}
      [data-testid="stSidebar"] {{background: rgba(13, 17, 15, .96);}}
      [data-testid="stHeader"] {{background: rgba(12, 16, 14, .76);}}
      .hero {{padding: 1.6rem; border-radius: 18px;
             background: rgba(16, 20, 18, .82);
             border: 1px solid rgba(220, 230, 220, .25);}}
      .hero h1 {{color: #f4f5f2; margin-bottom: .5rem;}}
      .hero p {{color: #e0e4df; font-size: 1.1rem; margin-bottom: 0;}}
      .workflow {{padding: .8rem 1rem; border-left: 4px solid #a8b7a1;
                 background: rgba(23, 28, 25, .84); border-radius: 6px; margin: .35rem 0;}}
      [data-testid="stMetric"] {{background: rgba(24, 29, 26, .88);
                               padding: .7rem; border-radius: 12px;}}
    </style>
    """,
    unsafe_allow_html=True,
)

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "location" not in st.session_state:
    st.session_state.location = None
if "weather" not in st.session_state:
    st.session_state.weather = None
if st.session_state.pop("navigate_to_analysis", False):
    st.session_state.page = "📷 Plant Analysis"


def show_analysis_notice() -> None:
    """Show a consistent prompt when a page needs a previous leaf analysis."""
    st.info("Analyze a leaf image on the **📷 Plant Analysis** page to see these results.")


def current_analysis() -> dict | None:
    analysis = st.session_state.analysis
    return analysis if isinstance(analysis, dict) else None


def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🌱 Plant Health Recommendation System</h1>
          <p>AI-powered plant health analysis and personalized recommendations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.write(
        "Upload a clear leaf photo to get a model-based health category, an "
        "application-generated health assessment, and practical care suggestions."
    )
    first, second = st.columns([1, 1])
    with first:
        st.subheader("Main features")
        st.markdown(
            "- Local image classification with scikit-learn\n"
            "- Confidence, risk level, and a clearly labeled health score\n"
            "- Built-in care guidance, with optional Gemini explanations\n"
            "- City-based geocoding, map, and public weather information\n"
            "- Downloadable plant analysis report"
        )
    with second:
        st.subheader("How the system works")
        for item in (
            "Upload Leaf Image",
            "Image Analysis",
            "Disease/Health Prediction",
            "Health Assessment",
            "Treatment Recommendation",
            "Prevention Tips",
        ):
            st.markdown(f'<div class="workflow">{item}</div>', unsafe_allow_html=True)
    if st.button("📷 Upload and analyze a leaf", type="primary"):
        st.session_state.navigate_to_analysis = True
        st.rerun()
    st.caption(
        "This educational demonstration is not a guaranteed diagnosis. "
        "Image predictions depend on the training dataset and photo quality."
    )


def render_analysis() -> None:
    st.header("📷 Plant Analysis")
    st.write("Upload a clear, well-lit image of a single plant leaf.")
    upload = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, and PNG.",
    )
    if upload is None:
        st.info("No image uploaded yet. Choose a JPG, JPEG, or PNG image to begin.")
        return

    image_bytes = upload.getvalue()
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
        if image.format not in ALLOWED_FORMATS:
            st.error("The uploaded file is not a valid JPG or PNG image.")
            return
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        st.error("This image could not be opened. Please select a valid JPG or PNG file.")
        return

    image_column, details_column = st.columns([1, 1])
    with image_column:
        st.image(image, caption="Uploaded plant leaf", use_container_width=True)
    with details_column:
        st.subheader("Image details")
        st.write(f"**File name:** {Path(upload.name).name}")
        st.write(f"**Image size:** {image.width} × {image.height} pixels")
        st.write(f"**File size:** {len(image_bytes) / 1024:.1f} KB")
        st.caption("Images are analyzed locally by the saved scikit-learn model.")

    if st.button("Analyze plant health", type="primary"):
        try:
            result = predict_plant_health(image, model_path=MODEL_PATH)
        except ModelNotFoundError:
            st.error(
                "The plant model has not been trained yet. From the project folder, "
                "run `python train_model.py`, then restart or retry the analysis."
            )
            return
        except (ValueError, OSError) as error:
            st.error(f"Could not analyze this image: {error}")
            return
        st.session_state.analysis = {
            **result,
            "image_bytes": image_bytes,
            "file_name": Path(upload.name).name,
            "image_size": f"{image.width} × {image.height} pixels",
        }
        st.success("Analysis complete.")
        st.rerun()

    analysis = current_analysis()
    if analysis:
        st.divider()
        st.subheader("Analysis result")
        one, two, three = st.columns(3)
        one.metric("Prediction", analysis["predicted_class"])
        two.metric("Health status", analysis["health_status"])
        three.metric("Confidence", f'{analysis["confidence"]:.1%}')
        st.metric("Application-generated health score", f'{analysis["health_score"]}/100')
        st.progress(analysis["health_score"] / 100)
        st.caption("The score is an application-generated assessment, not a validated measurement.")
        st.info(f'**Risk level:** {analysis["risk_level"]}')


def render_recommendations() -> None:
    st.header("💊 Recommendations")
    analysis = current_analysis()
    if not analysis:
        show_analysis_notice()
        return

    st.subheader(f'Guidance for: {analysis["health_status"]}')
    recommendations = get_recommendations(analysis["health_status"])
    for heading, items in recommendations.items():
        with st.expander(heading, expanded=True):
            for item in items:
                st.markdown(f"- {item}")

    weather = st.session_state.weather
    if weather:
        st.subheader("Environment-aware care")
        for item in get_environment_advice(weather):
            st.markdown(f"- {item}")

    st.subheader("Optional AI explanation")
    st.warning(
        "AI output is a recommendation, not a guaranteed agricultural diagnosis. "
        "Confirm treatment choices with a local agriculture expert."
    )
    if st.button("Generate Gemini explanation"):
        with st.spinner("Requesting an explanation..."):
            ai_text, status = get_ai_recommendation(
                analysis["predicted_class"],
                analysis["health_status"],
                analysis["confidence"],
            )
        if ai_text:
            st.session_state.ai_recommendation = ai_text
            st.session_state.ai_status = status
        else:
            st.session_state.ai_recommendation = None
            st.session_state.ai_status = status
    if st.session_state.get("ai_recommendation"):
        st.markdown(st.session_state.ai_recommendation)
    elif st.session_state.get("ai_status"):
        st.info(st.session_state.ai_status)
    else:
        st.info(
            "Built-in recommendations work without an API key. Add `GEMINI_API_KEY` "
            "to `.env` to request an optional AI explanation."
        )


def geocode_location(city: str, state: str, country: str) -> tuple[dict | None, str | None]:
    query = ", ".join(part.strip() for part in (city, state, country) if part.strip())
    if not city.strip() or not country.strip():
        return None, "Please provide at least a city and a country."
    try:
        geocoder = Nominatim(user_agent="plant_health_recommendation_system")
        place = geocoder.geocode(query, timeout=10)
    except (GeopyError, OSError) as error:
        return None, f"Location lookup failed. Please try again later. ({error})"
    if place is None:
        return None, "Location not found. Check the spelling and try a nearby city."
    return {
        "query": query,
        "address": place.address,
        "latitude": float(place.latitude),
        "longitude": float(place.longitude),
    }, None


def render_location_weather() -> None:
    st.header("🌦️ Location & Weather")
    st.write("Enter a place name; precise GPS access is not needed.")
    with st.form("location_form"):
        city = st.text_input("City", placeholder="e.g. Pune")
        state = st.text_input("State / region", placeholder="e.g. Maharashtra")
        country = st.text_input("Country", placeholder="e.g. India")
        submitted = st.form_submit_button("Find location and weather", type="primary")

    if submitted:
        location, error = geocode_location(city, state, country)
        if error:
            st.session_state.location = None
            st.session_state.weather = None
            st.error(error)
        else:
            st.session_state.location = location
            weather, weather_error = get_weather(location["latitude"], location["longitude"])
            st.session_state.weather = weather
            st.session_state.weather_error = weather_error

    location = st.session_state.location
    if not location:
        st.info("Submit a city, state/region, and country to display its map and weather.")
        return

    st.success(f'Location found: {location["address"]}')
    lat, lon = st.columns(2)
    lat.metric("Latitude", f'{location["latitude"]:.5f}')
    lon.metric("Longitude", f'{location["longitude"]:.5f}')
    if folium is None or st_folium is None:
        st.warning(
            "The map is unavailable because Folium or Streamlit-Folium is missing "
            "from the Python environment running this app. Location coordinates "
            "and weather are still available."
        )
    else:
        location_map = folium.Map(
            location=[location["latitude"], location["longitude"]],
            zoom_start=11,
            control_scale=True,
        )
        folium.Marker(
            [location["latitude"], location["longitude"]],
            tooltip=location["query"],
            popup=location["address"],
        ).add_to(location_map)
        st_folium(location_map, height=380, use_container_width=True)

    weather = st.session_state.weather
    if not weather:
        st.warning(st.session_state.get("weather_error") or "Weather information is unavailable.")
        return
    st.subheader("Current weather")
    a, b, c, d = st.columns(4)
    a.metric("Temperature", f'{weather["temperature"]:.1f} °C')
    b.metric("Humidity", f'{weather["humidity"]:.0f}%')
    c.metric("Rain", f'{weather["rain"]:.1f} mm')
    d.metric("Conditions", weather["description"])
    if st.session_state.get("weather_error"):
        st.caption(st.session_state.weather_error)


def render_report() -> None:
    st.header("📊 Health Report")
    analysis = current_analysis()
    if not analysis:
        show_analysis_notice()
        return

    recommendations = get_recommendations(analysis["health_status"])
    location = st.session_state.location
    weather = st.session_state.weather
    cause = recommendations["Possible causes"][0]
    treatment = "; ".join(recommendations["Treatment"])
    prevention = "; ".join(recommendations["Prevention"])
    location_text = location["address"] if location else "Not provided"
    weather_text = (
        f'{weather["temperature"]:.1f} °C, humidity {weather["humidity"]:.0f}%, '
        f'rain {weather["rain"]:.1f} mm, {weather["description"]}'
        if weather
        else "Not available"
    )

    st.subheader("Plant Analysis Report")
    report_fields = [
        ("Uploaded Image", analysis["file_name"]),
        ("Prediction", analysis["predicted_class"]),
        ("Health status", analysis["health_status"]),
        ("Confidence", f'{analysis["confidence"]:.1%}'),
        ("Health Score", f'{analysis["health_score"]}/100 (application-generated)'),
        ("Risk Level", analysis["risk_level"]),
        ("Possible Cause", cause),
        ("Treatment", treatment),
        ("Prevention", prevention),
        ("Location", location_text),
        ("Weather", weather_text),
    ]
    for label, value in report_fields:
        st.markdown(f"**{label}:** {value}")

    st.subheader("Assessment overview")
    figure, axis = plt.subplots(figsize=(8, 2.6))
    sns.barplot(
        x=[analysis["health_score"], analysis["confidence"] * 100],
        y=["Health score", "Prediction confidence"],
        hue=["Health score", "Prediction confidence"],
        palette=["#43a047", "#1e88e5"],
        legend=False,
        ax=axis,
    )
    axis.set_xlim(0, 100)
    axis.set_xlabel("Percent / score (0–100)")
    axis.set_ylabel("")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    report_text = "Plant Analysis Report\n" + "\n".join(
        f"{label}: {value}" for label, value in report_fields
    )
    st.download_button(
        "Download report as text",
        data=report_text,
        file_name="plant_health_report.txt",
        mime="text/plain",
    )
    st.caption(
        "Scores and predictions are generated by this application and are not "
        "scientifically validated agricultural measurements."
    )


def render_about() -> None:
    st.header("ℹ️ About")
    st.write(
        "This college-project demonstration combines traditional image features "
        "and a scikit-learn classifier with understandable plant-care guidance."
    )
    st.info(
        "The model is only as useful as its labeled training images. It does not "
        "replace laboratory testing or advice from an agricultural professional."
    )
    with st.expander("Model and privacy"):
        st.write(
            "Leaf image features are calculated locally. Images are not sent to a "
            "hosted classification service. If Gemini is configured and requested, "
            "the predicted label and confidence are sent to the Gemini API to create "
            "an explanation; the image itself is not sent."
        )


with st.sidebar:
    st.title("🌿 Plant Health")
    st.caption("Plant care, made easier")
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📷 Plant Analysis",
            "💊 Recommendations",
            "🌦️ Location & Weather",
            "📊 Health Report",
            "ℹ️ About",
        ],
        key="page",
        label_visibility="collapsed",
    )

if page == "🏠 Home":
    render_home()
elif page == "📷 Plant Analysis":
    render_analysis()
elif page == "💊 Recommendations":
    render_recommendations()
elif page == "🌦️ Location & Weather":
    render_location_weather()
elif page == "📊 Health Report":
    render_report()
else:
    render_about()
