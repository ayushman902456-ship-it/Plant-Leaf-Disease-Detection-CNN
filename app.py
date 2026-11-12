import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PROFESSIONAL DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #f3f8f2;
}

.main-title {
    color: #176b45;
    font-size: 42px;
    font-weight: 800;
    text-align: center;
}

.subtitle {
    color: #607568;
    text-align: center;
    font-size: 17px;
}

.result-card {
    background-color: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #dce9dd;
    margin-top: 12px;
}

div.stButton > button {
    background-color: #176b45;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #0e5032;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #e6f0e5;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD TRAINED CNN MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "plant_leaf_model.keras"
    )


@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as file:
        return json.load(file)


try:
    model = load_model()
    class_names = load_class_names()

except Exception as error:
    st.error("Unable to load the trained model or class names.")
    st.code(str(error))
    st.stop()


# ==========================================
# 4. NORMALIZE CATEGORY NAMES
# ==========================================

def clean_name(name):
    """
    Removes underscores, spaces and symbols.
    Converts names into lowercase for matching.
    """

    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


# ==========================================
# 5. DISEASE INFORMATION
# ==========================================

disease_info = {

    "Tomato___healthy": {

        "name": "Healthy Tomato Leaf",

        "description": (
            "The tomato leaf appears healthy and does not "
            "show the common signs of early or late blight."
        ),

        "symptoms": [
            "Green and healthy leaf surface",
            "No visible dark spots",
            "No major yellow or brown patches"
        ],

        "treatment": [
            "No disease treatment is required.",
            "Continue regular watering.",
            "Provide enough sunlight and nutrients."
        ],

        "prevention": [
            "Water near the soil instead of wetting leaves.",
            "Maintain proper spacing between plants.",
            "Inspect plants regularly."
        ]
    },


    "Tomato___Early_blight": {

        "name": "Tomato Early Blight",

        "description": (
            "Early blight is a fungal disease that commonly "
            "affects tomato leaves."
        ),

        "symptoms": [
            "Brown spots on leaves",
            "Circular ring patterns inside spots",
            "Yellowing around infected areas",
            "Older leaves may dry and fall"
        ],

        "treatment": [
            "Remove badly infected leaves.",
            "Avoid watering directly over leaves.",
            "Ask an agricultural expert about suitable fungicides."
        ],

        "prevention": [
            "Maintain proper distance between plants.",
            "Remove infected plant material.",
            "Avoid excessive moisture on leaves."
        ]
    },


    "Tomato___Late_blight": {

        "name": "Tomato Late Blight",

        "description": (
            "Late blight is a serious disease that can spread "
            "quickly in cool and humid conditions."
        ),

        "symptoms": [
            "Dark brown or black patches",
            "Water-soaked leaf areas",
            "White growth may appear underneath leaves in humid conditions",
            "Leaves may become dry and damaged"
        ],

        "treatment": [
            "Remove infected plant material carefully.",
            "Avoid overhead watering.",
            "Seek local agricultural advice about disease management."
        ],

        "prevention": [
            "Use healthy planting material.",
            "Keep leaves dry when possible.",
            "Inspect plants frequently.",
            "Remove infected plant debris."
        ]
    }
}


# Create a flexible disease information lookup

disease_lookup = {
    clean_name(category): information
    for category, information in disease_info.items()
}


# ==========================================
# 6. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🌱 PlantCare AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart Tomato Leaf Disease Detection Using CNN'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.divider()


# ==========================================
# 7. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🌿 PlantCare AI")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "About Project",
        "Disease Guide"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Upload a tomato leaf image to predict "
    "whether it is healthy or affected by disease."
)

st.sidebar.caption(
    "Powered by Python, TensorFlow, CNN and Streamlit."
)


# ==========================================
# 8. HOME PAGE
# ==========================================

if page == "Home":

    st.header("🔍 Plant Leaf Disease Detection")

    st.write(
        "Upload a clear tomato leaf image. "
        "Our trained CNN model will analyse the image "
        "and predict its category."
    )

    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Uploaded Tomato Leaf",
                    use_container_width=True
                )

            with col2:

                st.subheader("📷 Image Details")

                st.write("Image uploaded successfully.")

                st.write("Image size:", image.size)

                st.write(
                    "Categories supported:",
                    len(class_names)
                )

            st.divider()

            if st.button("🔍 Analyse Leaf"):

                with st.spinner("Analysing your leaf image..."):

                    # Prepare image
                    resized_image = image.resize((128, 128))

                    image_array = np.array(
                        resized_image,
                        dtype=np.float32
                    )

                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    # CNN prediction
                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )

                    probabilities = predictions[0]

                    # Check model and class count
                    if len(probabilities) != len(class_names):

                        st.error(
                            "The number of model outputs does not "
                            "match the number of saved class names."
                        )

                        st.stop()

                    # Find predicted category
                    predicted_index = int(
                        np.argmax(probabilities)
                    )

                    predicted_class = str(
                        class_names[predicted_index]
                    ).strip()

                    confidence = float(
                        probabilities[predicted_index] * 100
                    )

                    # Flexible category matching
                    information = disease_lookup.get(
                        clean_name(predicted_class)
                    )

                # ==================================
                # 9. PREDICTION RESULT
                # ==================================

                st.divider()

                st.header("📊 Prediction Result")

                # Show the exact category returned by CNN
                with st.expander("View model prediction details"):

                    st.write(
                        "Predicted category:",
                        predicted_class
                    )

                    st.write(
                        "Predicted class index:",
                        predicted_index
                    )

                if information is not None:

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <h2 style="color:#176b45;">
                                🌱 {information['name']}
                            </h2>
                            <p>{information['description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Confidence score
                    st.subheader("🎯 Prediction Confidence")

                    st.progress(
                        max(0.0, min(confidence / 100, 1.0))
                    )

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    if confidence < 60:

                        st.warning(
                            "The model has low confidence. "
                            "Try uploading a clearer image."
                        )

                    # All class probabilities
                    st.subheader("📈 Category Probability")

                    probability_data = {
                        class_names[i].replace(
                            "Tomato___", ""
                        ).replace("_", " "): float(
                            probabilities[i] * 100
                        )
                        for i in range(len(class_names))
                    }

                    st.bar_chart(probability_data)

                    st.divider()

                    # Disease details
                    col3, col4 = st.columns(2)

                    with col3:

                        st.subheader("🔎 Symptoms")

                        for item in information["symptoms"]:

                            st.write("•", item)

                    with col4:

                        st.subheader("💊 Suggested Care")

                        for item in information["treatment"]:

                            st.write("•", item)

                    st.subheader("🛡️ Prevention")

                    for item in information["prevention"]:

                        st.write("•", item)

                    st.success(
                        "Image analysis completed successfully!"
                    )

                else:

                    st.error(
                        "The model predicted a category that "
                        "does not match the disease information."
                    )

                    st.write(
                        "Actual predicted category:",
                        predicted_class
                    )

                    st.write(
                        "Available categories:",
                        class_names
                    )

                    st.info(
                        "Please share these category names "
                        "so the mapping can be corrected."
                    )

                st.caption(
                    "Educational AI prediction only. "
                    "The confidence score is not a guarantee "
                    "of diagnosis. Consult an agricultural expert "
                    "for confirmation and treatment decisions."
                )

        except Exception as error:

            st.error("Something went wrong while analysing the image.")

            st.code(str(error))


# ==========================================
# 10. ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.title("📘 About PlantCare AI")

    st.write("""
    PlantCare AI is a machine learning project designed
    to detect three tomato leaf categories from uploaded images.

    It uses a Convolutional Neural Network (CNN)
    developed with TensorFlow and Keras.

    The model was trained using labelled tomato leaf images.

    Streamlit provides the user interface for uploading images
    and viewing prediction results.
    """)

    st.divider()

    st.subheader("⚙️ Technologies Used")

    technologies = [
        "Python",
        "TensorFlow",
        "Keras",
        "Convolutional Neural Network (CNN)",
        "NumPy",
        "Pillow",
        "Streamlit"
    ]

    for technology in technologies:

        st.write("✅", technology)

    st.divider()

    st.subheader("🌱 Categories Detected")

    for category in class_names:

        st.write(
            "🌿",
            category.replace("Tomato___", "").replace("_", " ")
        )

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write(
        "The main objective is to demonstrate how deep learning "
        "can be used to classify tomato leaf images and support "
        "early awareness of plant diseases."
    )


# ==========================================
# 11. DISEASE GUIDE PAGE
# ==========================================

elif page == "Disease Guide":

    st.title("🌿 Tomato Leaf Disease Guide")

    st.write(
        "Learn about the three tomato leaf categories "
        "supported by this application."
    )

    for category, information in disease_info.items():

        with st.expander(information["name"]):

            st.write(information["description"])

            st.markdown("### 🔎 Symptoms")

            for item in information["symptoms"]:

                st.write("•", item)

            st.markdown("### 💊 Suggested Care")

            for item in information["treatment"]:

                st.write("•", item)

            st.markdown("### 🛡️ Prevention")

            for item in information["prevention"]:

                st.write("•", item)


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#607568;">
        🌱 <b>PlantCare AI</b><br>
        Plant Leaf Disease Detection Using CNN<br>
        <small>Educational Machine Learning Project</small>
    </div>
    """,
    unsafe_allow_html=True
)