
from ultralytics import YOLO
from pathlib import Path
import argparse
import json
import cv2


CLASS_NAMES = {
    0: "header",
    1: "footer",
    2: "main_text",
    3: "side_text",
    4: "filler"
}


def process_image(model, image_path, output_dir, confidence=0.01):

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(image_path),
        conf=confidence,
        save=False,
        verbose=False
    )

    result = results[0]

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return

    height, width = image.shape[:2]

    predictions = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Keep bounding boxes inside image boundaries
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
                "confidence": round(confidence_score, 4),
                "bbox": [x1, y1, x2, y2]
            })

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = f"{label} {confidence_score:.2f}"

            cv2.putText(
                image,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Save annotated image
    output_image = output_dir / image_path.name
    cv2.imwrite(str(output_image), image)

    # Save JSON metadata
    output_json = output_dir / f"{image_path.stem}.json"

    metadata = {
        "image": image_path.name,
        "image_width": width,
        "image_height": height,
        "predictions": predictions
    }

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"Processed: {image_path.name} | "
        f"Detections: {len(predictions)}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Manuscript Specific Layout Region Detection"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image or folder"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder"
    )

    parser.add_argument(
        "--model",
        default="best.pt",
        help="YOLO model path"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="Confidence threshold"
    )

    args = parser.parse_args()

    model = YOLO(args.model)

    input_path = Path(args.input)

    if input_path.is_file():

        process_image(
            model,
            input_path,
            args.output,
            args.conf
        )

    elif input_path.is_dir():

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff"
        }

        images = [
            p for p in input_path.iterdir()
            if p.suffix.lower() in extensions
        ]

        if not images:
            print("No supported images found.")
            return

        for image_path in sorted(images):

            process_image(
                model,
                image_path,
                args.output,
                args.conf
            )

    else:

        print(
            f"Input path does not exist: {input_path}"
        )


if __name__ == "__main__":
    main()
