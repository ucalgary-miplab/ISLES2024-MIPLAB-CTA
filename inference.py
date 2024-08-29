import os

from src.image_processing import postprocess, preprocess
from src.prediction import predict_infarct


def run():
    INPUT_PATH = "/input"
    OUTPUT_PATH = "/output"
    RESOURCE_PATH = "resources"

    # Find the images in challenge space and clinical data
    ncct_dir = os.path.join(INPUT_PATH, "images/non-contrast-ct")
    cta_dir = os.path.join(INPUT_PATH, "images/preprocessed-CT-angiography")

    ncct_fname = [f for f in os.listdir(ncct_dir) if f.endswith(".mha")][0]
    cta_fname = [f for f in os.listdir(cta_dir) if f.endswith(".mha")][0]

    ncct_file = os.path.join(ncct_dir, ncct_fname)
    cta_file = os.path.join(cta_dir, cta_fname)
    clinical_file = os.path.join(INPUT_PATH, "acute-stroke-clinical-information.json")

    # Extract data into ML space
    preprocess(ncct_file, cta_file, clinical_file, RESOURCE_PATH)

    # Process data
    predict_infarct(RESOURCE_PATH)

    # Convert back to image in challenge space; save to required path
    output_file = os.path.join(
        OUTPUT_PATH, "images", "stroke-lesion-segmentation", ncct_fname
    )
    postprocess(RESOURCE_PATH, output_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
