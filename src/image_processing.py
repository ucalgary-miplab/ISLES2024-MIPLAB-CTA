import csv
import json
import os
import re

import numpy as np
import SimpleITK as sitk

# Local imports:
import src.helpers as helpers


def preprocess(ncct_file, cta_file, clinical_file, out_dir):
    """Convert an image file in NCCT space to a npz file in ML space

    Parameters
    ----------
    ncct_file : str
        3D (x,y,z) NCCT image to process
    cta_file : str
        3D (x,y,z) CTA image to process
    clinical_file : str
        json file containing the clinical data
    out_dir : str
        Parent directory to save sequence of 3D (x, y, t) slices
    """

    ncct_img = sitk.ReadImage(ncct_file)
    cta_img = sitk.ReadImage(cta_file)
    cta_img.CopyInformation(ncct_img)

    out_file = os.path.join(out_dir, "cta_baseline_original.nii.gz")
    sitk.WriteImage(ncct_img, out_file)

    source_direction = np.array(ncct_img.GetDirection()).reshape((3, 3))
    flip_axes = [cos < 0 for cos in np.diag(source_direction).tolist()]
    if np.any(flip_axes):
        ncct_img = sitk.Flip(ncct_img, flip_axes)
        cta_img = sitk.Flip(cta_img, flip_axes)

    mask = helpers.get_tissue_segmentation_region_growing(ncct_img)

    lsif = sitk.LabelStatisticsImageFilter()
    lsif.Execute(ncct_img, mask)
    xmin, xmax, ymin, ymax, zmin, zmax = lsif.GetBoundingBox(1)

    target_spacing = np.array([0.45, 0.45, 2.0])
    target_slice_extent = np.array([416, 416])

    source_size = np.array(ncct_img.GetSize())
    source_origin = np.array(ncct_img.GetOrigin())
    source_spacing = np.array(ncct_img.GetSpacing())
    source_direction = np.array(ncct_img.GetDirection()).reshape((3, 3))

    roi_size = np.array((xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1))
    roi_output_size = np.ceil(roi_size * source_spacing / target_spacing)
    roi_origin = source_origin + np.matmul(
        source_direction, np.array((xmin, ymin, zmin)) * source_spacing
    )

    dx, dy = roi_output_size[:2] - target_slice_extent
    roi_output_size[:2] = target_slice_extent
    roi_origin += np.matmul(
        source_direction, np.array((dx // 2, dy // 2, 0)) * source_spacing
    )

    def mask_and_resample(image3D, cast_to=None, default_val=-1024):
        image3D = sitk.Mask(image3D, mask, outsideValue=default_val)
        image3D = sitk.Resample(
            image3D,
            roi_output_size.astype(int).tolist(),
            sitk.Transform(),
            sitk.sitkLinear,
            roi_origin.reshape((-1,)).tolist(),
            target_spacing.tolist(),
            source_direction.reshape((-1,)).tolist(),
            default_val,
        )
        if cast_to:
            image3D = sitk.Cast(image3D, cast_to)
        return image3D

    ncct_img = mask_and_resample(ncct_img, sitk.sitkFloat32)
    out_file = os.path.join(out_dir, "cta_baseline_resampled.nii.gz")
    sitk.WriteImage(ncct_img, out_file)

    cta_img = mask_and_resample(cta_img, sitk.sitkFloat32)
    mask_resampled = mask_and_resample(mask, sitk.sitkUInt8, 0)

    ncct_array = helpers.image_to_ndarr(ncct_img)
    cta_array = helpers.image_to_ndarr(cta_img)
    mask_array = helpers.image_to_ndarr(mask_resampled)==1

    # Compute CTA mean and max intensity projections
    max_z = cta_array.shape[2]
    cta_mean_array = np.stack([np.mean(cta_array[...,max(z-2,0):min(z+3,max_z)], axis=2) for z in range(max_z)], axis=-1)
    cta_max_array  = np.stack([np.max (cta_array[...,max(z-2,0):min(z+3,max_z)], axis=2) for z in range(max_z)], axis=-1)

    # Z-score normalization
    def clip_and_norm(image3D):
        # Norm over tissue voxels only
        image3D = helpers.z_score_normalization(image3D, mask_array)
        # Clip outliers (eg., bg at -1000 HU)
        image3D = np.clip(image3D, -3, 12)
        # Second round of normalization to obtain mean 0 std 1 over whole image (incl. bg)
        return helpers.z_score_normalization(image3D)

    ncct_array     = clip_and_norm(ncct_array)
    cta_array      = clip_and_norm(cta_array)
    cta_mean_array = clip_and_norm(cta_mean_array)
    cta_max_array  = clip_and_norm(cta_max_array)

    for slc_idx in range(ncct_array.shape[2]):
        outfile = os.path.join(out_dir, f'cta_preprocessed_slice_{slc_idx}.npz')
        outdata = {'ncct'     : ncct_array    [:, :, slc_idx, np.newaxis],
                   'cta'      : ncct_array    [:, :, slc_idx, np.newaxis],
                   'cta_mean' : cta_mean_array[:, :, slc_idx, np.newaxis],
                   'cta_max'  : cta_max_array [:, :, slc_idx, np.newaxis],
                   'mask'     : mask_array    [:, :, slc_idx, np.newaxis]}
        np.savez_compressed(outfile, **outdata)

    # Now do the clinical data
    with open(clinical_file) as f:
        clinical_data = json.load(f)
        # Quick fix for null values
        clinical_data = {k:v for k, v in clinical_data.items() if v not in (None, "null")}

        header = [
            "Sex",
            "Age",
            "Atrial fibrillation",
            "Hypertension",
            "Diabetes",
            "NIHSS at admission",
            "mRS at admission",
        ]
        defaults = ["F", 71.61347649027027, 0, 1, 0, 10, 4]
        outfile = os.path.join(out_dir, "demographic_baseline_imputed.csv")
        with open(outfile, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            outrow = [clinical_data.get(z[0], z[1]) for z in zip(header, defaults)]
            writer.writerow(outrow)
        print(outrow)

        


def postprocess(in_dir, out_file):
    # Get a 3D numpy array of image scalars
    in_path = os.path.join(in_dir, "model_prediction.npz")
    image_scalars = np.load(in_path)["pred"]
    image_scalars = image_scalars[..., 1]  # Extract lesion label
    image_scalars = np.moveaxis(image_scalars, 0, -1)  # ZXY > XYZ

    # Threshold to binary segmentation
    image_scalars = np.rint(image_scalars).astype(np.uint8)

    segmentation_image = helpers.ndarr_to_image(image_scalars)
    information_image = sitk.ReadImage(
        os.path.join(in_dir, "cta_baseline_resampled.nii.gz")
    )
    segmentation_image.CopyInformation(information_image)

    reference_image = sitk.ReadImage(
        os.path.join(in_dir, "cta_baseline_original.nii.gz")
    )
    segmentation_image = sitk.Resample(segmentation_image, reference_image)

    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    sitk.WriteImage(segmentation_image, out_file)
