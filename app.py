import mlflow
import streamlit as st
from PIL import Image

from plant_classifier.config import TRAIN_CONFIG
from plant_classifier.pipelines.inference_pipeline import inference_pipeline

st.set_page_config(page_title="Plant Classifier", page_icon="🌱", layout="wide")

st.title("🌱 Plant Species Classifier")

# Sidebar configuration
with st.sidebar:
    st.markdown("# 🧑‍🏫About")
    st.markdown(
        """
        This app uses a Vision Transformer model to classify plant images
        and detect data drift in real-time.

        **Features:**
        - Plant classification
        - Data drift detection
        - EDA visualizations from training
        """
    )
    st.markdown("---")
    st.markdown("### 🧑‍💻Data Drift Detection")
    st.markdown(
        """
        - *Pixel Drift*: Measures pixel distribution changes in RGB channels
        - *Feature Drift*: Measures changes in deep feature representations
        - *Performance Drift*: Tracks prediction confidence degradation
        """
    )

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display image in first column
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", width="stretch")

    # Make prediction with drift detection
    with st.spinner("Running inference and drift detection..."):
        result = inference_pipeline(image, include_drift=True)

    # Display predictions - This stays static
    with col2:
        st.subheader("📊 Prediction Results")
        preds = result["predictions"]
        st.success(f"**Predicted Label:** {preds[0]['label']}")
        st.info(f"**Confidence Score:** {round(preds[0]['score'] * 100, 2)}%")

    # Tabs for Data Drift and EDA Figures
    st.markdown("---")
    tab1, tab2 = st.tabs(["🔍 Data Drift", "📊 EDA Figures"])

    # Tab 1: Data Drift Analysis
    with tab1:
        drift_metrics = result["drift_metrics"]

        # Check for errors
        if "error" in drift_metrics:
            st.error(f"⚠️ {drift_metrics['error']}")
        else:
            # Overall drift status
            overall_alert = drift_metrics.get("overall_alert", False)

            if overall_alert:
                st.error(
                    "🚨 **DRIFT DETECTED!** The uploaded image shows significant drift from training data."
                )
            else:
                st.success(
                    "✅ **No significant drift detected.** The image is similar to training data."
                )

            # Create three columns for different drift types
            drift_col1, drift_col2, drift_col3 = st.columns(3)

            # Visual Drift
            with drift_col1:
                st.markdown("### 👁️ Visual Drift")
                visual = drift_metrics["visual_drift"]
                pixel_drift = visual["overall_pixel_drift"]
                pixel_alert = visual["pixel_drift_alert"]

                if pixel_alert:
                    st.error("⚠️ **Alert**")
                else:
                    st.success("✅ **Normal**")

                st.metric("Pixel Drift Score", f"{pixel_drift:.4f}")

                with st.expander("Channel Details"):
                    st.write(f"Red Channel: {visual['drift_r']:.4f}")
                    st.write(f"Green Channel: {visual['drift_g']:.4f}")
                    st.write(f"Blue Channel: {visual['drift_b']:.4f}")

            # Feature Drift
            with drift_col2:
                st.markdown("### 🧠 Feature Drift")
                feature = drift_metrics["feature_drift"]
                feature_drift = feature["feature_drift"]
                feature_alert = feature["feature_drift_alert"]

                if feature_alert:
                    st.error("⚠️ **Alert**")
                else:
                    st.success("✅ **Normal**")

                st.metric("Feature Drift Score", f"{feature_drift:.4f}")

            # Performance Drift
            with drift_col3:
                st.markdown("### 📉 Performance Drift")
                performance = drift_metrics["performance_drift"]
                conf_drop = performance["confidence_drop"]
                conf_alert = performance["confidence_alert"]

                if conf_alert:
                    st.error("⚠️ **Alert**")
                else:
                    st.success("✅ **Normal**")

                st.metric("Confidence Drop", f"{conf_drop * 100:.2f}%")

                with st.expander("Confidence Details"):
                    st.write(f"Current: {performance['current_confidence']:.4f}")
                    st.write(f"Reference: {performance['reference_confidence']:.4f}")

            # Interpretation guide
            with st.expander("ℹ️ How to interpret drift metrics"):
                st.markdown(
                    """
                    **Visual Drift (Wasserstein Distance):**
                    - Measures pixel distribution changes in RGB channels
                    - Higher values indicate the image looks different from training data
                    - May indicate different lighting, color balance, or image quality

                    **Feature Drift:**
                    - Measures changes in deep feature representations
                    - Indicates if the model sees semantically different patterns
                    - High drift may mean new plant varieties or image conditions

                    **Performance Drift:**
                    - Tracks prediction confidence compared to validation data
                    - Low confidence may indicate model uncertainty
                    - Suggests the model may not perform well on this type of image

                    **When drift is detected:**
                    - Review the image for quality issues
                    - Consider if it represents a new distribution
                    - May need to retrain with similar images
                    """
                )

    # Tab 2: EDA Figures
    with tab2:
        st.subheader("Training Data Analysis")
        st.write(
            "These visualizations were generated during model training and logged to MLflow."
        )

        try:
            # Connect to MLflow
            mlflow.set_tracking_uri(TRAIN_CONFIG.mlflow_tracking_uri)

            # Get the latest run from the experiment
            experiment = mlflow.get_experiment_by_name(
                TRAIN_CONFIG.mlflow_experiment_name
            )

            if experiment is None:
                st.warning(
                    f"No experiment found with name: {TRAIN_CONFIG.mlflow_experiment_name}"
                )
            else:
                # Get all runs from the experiment, sorted by start time
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1,
                )

                if len(runs) == 0:
                    st.warning("No runs found in the experiment.")
                else:
                    run_id = runs.iloc[0]["run_id"]
                    st.info(f"Displaying figures from run: {run_id[:8]}...")

                    # List of figures to display
                    figures = [
                        (
                            "analyze_image_dimensions_train.png",
                            "Training Set - Image Dimensions",
                        ),
                        (
                            "analyze_image_dimensions_validation.png",
                            "Validation Set - Image Dimensions",
                        ),
                        (
                            "analyze_image_dimensions_test.png",
                            "Test Set - Image Dimensions",
                        ),
                        ("dist_labels_train.png", "Training Set - Label Distribution"),
                        (
                            "dist_labels_validation.png",
                            "Validation Set - Label Distribution",
                        ),
                        ("dist_labels_test.png", "Test Set - Label Distribution"),
                        ("visualize_train_data.png", "Training Data Samples"),
                        (
                            "visualize_test_predictions.png",
                            "Test Set - Model Predictions",
                        ),
                    ]

                    # Download and display figures in a grid
                    for i in range(0, len(figures), 2):
                        cols = st.columns(2)
                        for j, col in enumerate(cols):
                            if i + j < len(figures):
                                artifact_path, title = figures[i + j]
                                try:
                                    # Download artifact
                                    artifact_uri = f"runs:/{run_id}/{artifact_path}"
                                    local_path = mlflow.artifacts.download_artifacts(
                                        artifact_uri
                                    )

                                    # Display image
                                    with col:
                                        st.markdown(f"**{title}**")
                                        st.image(local_path, width="stretch")
                                except Exception as e:
                                    with col:
                                        st.error(f"Failed to load {title}: {str(e)}")

        except Exception as e:
            st.error(f"Error loading EDA figures: {str(e)}")
