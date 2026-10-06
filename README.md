# Plant Health Recommendation System Using Python

A beginner-friendly Streamlit college-project demonstration. It extracts color,
spatial, and simple texture features from leaf photos and trains a local
scikit-learn Gaussian Naive Bayes classifier. It does not require TensorFlow, PyTorch,
Hugging Face, or an online image-classification model.

> **Important:** This project is an educational demonstration, not a guaranteed
> agricultural diagnosis. Its predictions depend on the dataset, image quality,
> and training setup.

## Project structure

```text
plant_health project/
├── app.py
├── model.py
├── recommendation.py
├── train_model.py
├── requirements.txt
├── .env.example
├── README.md
├── assets/
│   └── forest_background.jpg # forest backdrop for the app
├── dataset/
│   ├── Healthy/
│   ├── Leaf_Disease/
│   └── Severe_Disease/
└── models/
    └── plant_model.pkl       # created by training
```

The application uses a darkened forest photograph as its page background, with
neutral dark panels to keep text readable. The tomato photo is not used on the
Home page. The forest image is served from Unsplash's photo CDN
(`photo-1511497584788-876760111969`) and is included under the Unsplash License.

The `dataset` folders include eight CC0 tomato image thumbnails to make a first
training run possible. Their sources and important limitations are documented
in [dataset/SOURCES.md](./dataset/SOURCES.md). This tiny starter set is for
testing that the pipeline works, not for reliable health predictions.

## 1. Create the dataset

Use one subfolder per label. For the supplied demonstration categories:

```text
dataset/
├── Healthy/
├── Leaf_Disease/
└── Severe_Disease/
```

Put JPG, JPEG, or PNG leaf images directly in the matching folder. The folder
name becomes the training label. You can use disease-specific labels instead,
for example `Tomato_Early_Blight` and `Tomato_Healthy`; each category must have
at least two readable images. The included set technically meets that minimum,
but only two healthy examples are present. For a more useful college demonstration, aim for
**at least 50–100 varied images per class**, with different leaves, lighting,
backgrounds, and camera angles. More balanced, correctly labeled images are
better. Avoid near-duplicate photos and images of multiple classes in one
folder.

Do not place unrelated labels such as different plant species into broad
severity folders unless those photos represent the same problem you want the
classifier to learn. The prediction reports the folder label.

## 2. Train the model

Open PowerShell or Command Prompt and change to the project directory:

```powershell
cd "C:\Users\sanju\OneDrive\Desktop\plant_health project"
.\.venv\Scripts\python.exe train_model.py
```

Training checks the class folders and image files, extracts fixed-size image
features, creates a stratified held-out split, trains a Gaussian Naive Bayes classifier, and prints
accuracy and a per-class report. It then refits on all images and saves the
artifact to `models\plant_model.pkl`.

Check that it was created:

```powershell
Test-Path .\models\plant_model.pkl
```

The command should print `True`. If you see an empty-dataset message, add
readable JPG/JPEG/PNG files inside at least two class folders. Each class needs
at least two images to support training and evaluation.

## 3. Run the Streamlit application

The packages listed in `requirements.txt` include the packages requested for
this project plus Pillow for image decoding. If they are already installed, no
reinstallation is needed. Start the application from the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in a
browser. The model must exist before the app can make predictions; if missing,
the app gives the training command rather than returning a made-up result. If
`python -m streamlit` reports that Streamlit is not installed, `python` is not
the interpreter associated with your existing Streamlit installation. Check
which executable is being used with:

```powershell
Get-Command python, streamlit
python -m pip show streamlit folium streamlit-folium
```

If the project virtual environment was created elsewhere or is not present,
activate the environment where the packages are installed before training and
running, or replace `.venv\Scripts\python.exe` above with that environment's
Python path. Always use the same interpreter for both commands.

## 4. Add an optional Gemini API key

Copy the template to `.env` and edit the key:

```powershell
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `.env` to a key you obtained from Google AI Studio. Do
not commit or share `.env`. Restart Streamlit after changing environment values.
Gemini is only used when the user requests an explanation on the
Recommendations page; without a key, built-in recommendations continue to work.
If the API request fails, the app displays an explanation and retains the
built-in guidance. The uploaded image is not sent to Gemini.

## 5. How recommendations work

The classifier output is matched to a health category. Labels containing
`healthy` or `normal` map to Healthy; labels containing severe/advanced/critical
terms map to Severe Disease; other non-healthy labels map to Leaf Disease.
Built-in suggestions are provided for all categories. The optional Gemini
explanation uses the predicted label and confidence and must not be treated as a
confirmed diagnosis.

The displayed score is a simple **application-generated assessment** (90 for
Healthy, 65 for Leaf Disease, and 30 for Severe Disease), not a scientifically
validated measurement. Confidence is the classifier's largest class
probability; it is not a guarantee of correctness.

## 6. Location, map, and weather

Enter a city and country (state/region is optional) on **Location & Weather**.
Geopy's Nominatim geocoder resolves the place name; Folium displays it on a map.
The app then requests current temperature, humidity, precipitation, and weather
description from the public Open-Meteo API. No weather API key or GPS access is
needed. Network, geocoding, and weather failures are reported in the interface
without preventing leaf analysis.

## Common errors

- **Model not found:** Run `.\.venv\Scripts\python.exe train_model.py` from this
  directory after adding dataset images. The training command must use the same
  Python environment as Streamlit.
- **Empty dataset / fewer than two classes:** Add valid JPG/JPEG/PNG images to
  at least two class folders; put at least two readable images in each folder.
- **Could not open image:** Re-export the image as a valid JPG or PNG and upload
  again.
- **Gemini unavailable:** Confirm `.env` is in the project root, the key is
  valid, and restart Streamlit. Built-in recommendations remain available.
- **Location not found:** Check spelling and include the country; try a nearby
  larger city.
- **No module named `folium`:** Start the app with the Python environment where
  Folium and Streamlit-Folium are already installed. The rest of the app can
  still start without those map packages; the Location & Weather page will show
  coordinates and weather with a notice instead of rendering the map.
- **Weather unavailable:** Check internet access and retry later. Weather is
  optional.
- **`streamlit` is not recognized:** Activate the Python environment where
  Streamlit is installed, or use `python -m streamlit run app.py`.
