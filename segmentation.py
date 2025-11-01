# --- SEGMENT BBBC020 IMAGES USING PRETRAINED CELLPOSE ---
import os, glob
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io as cp_io, plot

# Path to your image folder
base_dir = "BBBC020_v1_images"
out_dir = "cellpose_results"
os.makedirs(out_dir, exist_ok=True)

# Load the pre-trained model (cytoplasm general model)
model = models.CellposeModel(model_type='cyto2', gpu=False)

#Find all merged images (DAPI + CD11b)
image_paths = sorted(glob.glob(os.path.join(base_dir, "*", "*_(c1+c5).TIF")))


print(f"Found {len(image_paths)} images to process.\n")

#process each image
for path in image_paths:
    print(f"Processing: {path}")
    img = io.imread(path)

    #Convert RGB to grayscale if needed
    if img.ndim == 3:
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img

    # Run segmentation
    masks, flows, styles = model.eval(
        [img_gray],                # model expects a list of images
        channels=[0, 0],           # single-channel grayscale
        diameter=None,
        do_3D=False
    )

    masks = masks[0]
    flows = flows[0]
    print(f" → {np.max(masks)} cells segmented")

    # Save mask and flow images
    save_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0])
    cp_io.masks_flows_to_seg(img_gray, masks, flows, save_path)

    # # Optional: show result
    # fig = plt.figure(figsize=(7, 7))
    # plot.show_segmentation(fig, img_gray, masks, flows)
    # plt.title(os.path.basename(path))
    # plt.show()
    #Optional visualization (CellPose v4-compatible)
    fig = plt.figure(figsize=(7, 7))

    # Extract the flow field correctly from the dictionary
    flow_image = flows["flows"][0] if isinstance(flows, dict) and "flows" in flows else None

    if flow_image is not None:
        plot.show_segmentation(fig, img_gray, masks, flow_image)
    else:
    # Create a fake flow array (since new versions don't return it)
        flowi = np.zeros_like(img_gray)

        # Updated plotting call
        plot.show_segmentation(fig, img_gray, masks, flowi)
        plt.show()
    plt.title(os.path.basename(path))
    plt.show()
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io, measure, color
# from cellpose import models, io as cellpose_io
#
# # -------------------------
# # CONFIG
# # -------------------------
# image_dir = "BBBC020_v1_images"
# output_dir = "output_results"
# os.makedirs(output_dir, exist_ok=True)
#
# # -------------------------
# # LOAD MODEL
# # -------------------------
# print("\nLoading Cellpose model...")
# model = models.CellposeModel(gpu=False)  # v4+ compatible
#
# # -------------------------
# # PROCESS ALL IMAGES
# # -------------------------
# image_paths = cellpose_io.get_image_files(image_dir)
# print(f"Found {len(image_paths)} images to process.\n")
#
# for img_path in image_paths:
#     print(f"Processing: {img_path}")
#
#     # Load image and convert to grayscale
#     img = io.imread(img_path)
#     if img.ndim == 3:
#         img_gray = color.rgb2gray(img)
#     else:
#         img_gray = img
#
#     # Run segmentation
#     masks, _, _, _ = model.eval(img_gray, channels=[0, 0])
#     num_cells = len(np.unique(masks)) - 1
#     print(f"  → {num_cells} cells segmented")
#
#     # -------------------------
#     # SAVE SEGMENTATION IMAGE
#     # -------------------------
#     fig, axes = plt.subplots(1, 3, figsize=(8, 3))
#     axes[0].imshow(img_gray, cmap='gray')
#     axes[0].set_title('Original')
#
#     outlines = np.copy(img_gray)
#     axes[1].imshow(outlines, cmap='gray')
#     axes[1].imshow(masks, alpha=0.3)
#     axes[1].set_title('Segmented')
#
#     axes[2].imshow(color.label2rgb(masks, bg_label=0))
#     axes[2].set_title('Cell Mask')
#
#     for ax in axes:
#         ax.axis('off')
#
#     save_name = os.path.join(output_dir, os.path.basename(img_path).replace('.TIF','_segmented.png'))
#     plt.tight_layout()
#     plt.savefig(save_name, dpi=150)
#     plt.close(fig)
#
#     # -------------------------
#     # EXTRACT CELL FEATURES
#     # -------------------------
#     props = measure.regionprops_table(
#         masks,
#         intensity_image=img_gray,
#         properties=('label', 'area', 'perimeter', 'mean_intensity')
#     )
#     props = {k: v.tolist() for k, v in props.items()}
#
#     # Compute circularity: (4π × Area) / Perimeter²
#     circularity = []
#     for area, perimeter in zip(props['area'], props['perimeter']):
#         if perimeter > 0:
#             circ = (4 * np.pi * area) / (perimeter ** 2)
#         else:
#             circ = 0
#         circularity.append(circ)
#
#     # Add circularity
#     props['circularity'] = circularity
#     props['cell_count'] = num_cells
#
#     # Save to CSV
#     import pandas as pd
#     df = pd.DataFrame(props)
#     csv_name = save_name.replace('.png', '_features.csv')
#     df.to_csv(csv_name, index=False)
#
#     print(f"  → Saved mask and features to: {csv_name}\n")
#
# print("✅ Processing complete! Results saved in:", output_dir)
