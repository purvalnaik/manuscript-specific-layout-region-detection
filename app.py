import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import cv2
import tempfile
import json

st.set_page_config(
    page_title="Manuscript Layout Detector",
    page_icon="??",
    layout="wide"
)

CLASS_NAMES = {
    0: "header",
    1: "footer",
    2: "main_text",
    3: "side_text",
    4: "filler"
}

st.title("?? Manuscript Specific Layout Region Detection")
st.write(
    "Upload a manuscript page to detect and classify layout regions."
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a manuscript image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

confidence = st.slider(
    "Confidence threshold",
    min_value=0.01,
    max_value=0.50,
    value=0.01,
    step=0.01
)

if uploaded_file is not None:

    file_bytes = uploaded_file.read()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(uploaded_file.name).suffix
    ) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    image = cv2.imread(temp_path)

    if image is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    height, width = image.shape[:2]

    results = model.predict(
        source=temp_path,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    predictions = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            score = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            label = CLASS_NAMES.get(
                class_id,
                str(class_id)
            )

            predictions.append({
                "label": label,
                "class_id": class_id,
                "confidence": round(score, 4),
                "bbox": [x1, y1, x2, y2]
            })

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"{label} {score:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    annotated = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(file_bytes, use_container_width=True)

    with col2:
        st.subheader("Detected Layout Regions")
        st.image(annotated, use_container_width=True)

    st.subheader("Predictions")

    if predictions:
        st.dataframe(
            predictions,
            use_container_width=True
        )
    else:
        st.warning(
            "No layout regions detected at this confidence threshold."
        )

    metadata = {
        "image": uploaded_file.name,
        "image_width": width,
        "image_height": height,
        "predictions": predictions
    }

    json_data = json.dumps(
        metadata,
        indent=4
    )

    st.download_button(
        label="Download JSON Metadata",
        data=json_data,
        file_name=f"{Path(uploaded_file.name).stem}.json",
        mime="application/json"
    )
