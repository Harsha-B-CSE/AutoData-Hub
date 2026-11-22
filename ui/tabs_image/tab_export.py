"""Aspect-preserving resize and image/ZIP export tab."""

from __future__ import annotations

import io
from pathlib import PurePosixPath
import zipfile

import streamlit as st

from modules.image.resizer import letterbox_resize
from ui.tabs_image.common import (
    bgr_to_display,
    decode_image_bytes,
    encode_image,
)


RESOLUTION_PRESETS = {
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
    "224x224 (ML Baseline)": (224, 224),
}


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    red, green, blue = (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
    )
    return blue, green, red


def _export_name(source_name: str, extension: str) -> str:
    path = PurePosixPath(source_name.replace("\\", "/"))
    return path.with_suffix(extension).as_posix()


def render_tab_export(uploaded_file) -> None:
    """Resize the processed image and provide individual or ZIP downloads."""
    source_image = st.session_state.get("processed_image_bgr")
    if source_image is None:
        source_image = st.session_state.get("image_original_bgr")
    if source_image is None:
        st.info("Select and process an image before export.")
        return

    control1, control2 = st.columns(2)
    with control1:
        resolution = st.selectbox(
            "Target Resolution",
            options=[
                "512x512",
                "1024x1024",
                "224x224 (ML Baseline)",
                "Custom",
            ],
            key="image_export_resolution",
        )
    with control2:
        output_format = st.selectbox(
            "Output Format",
            options=[".png", ".jpg", ".webp"],
            key="image_export_format",
        )

    if resolution == "Custom":
        width_col, height_col = st.columns(2)
        target_width = int(
            width_col.number_input(
                "Custom Width",
                min_value=1,
                max_value=16_384,
                value=512,
                step=1,
                key="image_export_width",
            )
        )
        target_height = int(
            height_col.number_input(
                "Custom Height",
                min_value=1,
                max_value=16_384,
                value=512,
                step=1,
                key="image_export_height",
            )
        )
    else:
        target_width, target_height = RESOLUTION_PRESETS[resolution]

    option1, option2 = st.columns(2)
    interpolation = option1.selectbox(
        "Resize Interpolation",
        options=["lanczos", "bicubic", "bilinear", "nearest"],
        key="image_export_interpolation",
    )
    pad_color_hex = option2.color_picker(
        "Letterbox Color",
        value="#000000",
        key="image_export_pad_color",
    )
    pad_color_bgr = _hex_to_bgr(pad_color_hex)

    quality = 95
    png_compression = 4
    if output_format in {".jpg", ".webp"}:
        quality = st.slider(
            "Output Quality",
            min_value=50,
            max_value=100,
            value=95,
            key="image_export_quality",
        )
    else:
        png_compression = st.slider(
            "PNG Compression",
            min_value=0,
            max_value=9,
            value=4,
            key="image_export_png_compression",
        )

    if st.button(
        "Prepare Resized Image",
        width="stretch",
        key="image_prepare_export",
    ):
        with st.spinner("Resizing and encoding image..."):
            resized = letterbox_resize(
                source_image,
                target_size=(target_width, target_height),
                interpolation=interpolation,
                fill_color=pad_color_bgr,
            )
            encoded, extension, mime = encode_image(
                resized,
                output_format.lstrip("."),
                quality=quality,
                png_compression=png_compression,
                jpeg_background_bgr=pad_color_bgr,
            )
            selected_name = st.session_state.get(
                "image_selected_name", "processed_image"
            )
            st.session_state["image_export_result"] = {
                "bytes": encoded,
                "extension": extension,
                "mime": mime,
                "filename": PurePosixPath(selected_name).with_suffix(
                    extension
                ).name,
                "preview_bgr": resized,
            }

    export_result = st.session_state.get("image_export_result")
    if export_result:
        st.image(
            bgr_to_display(export_result["preview_bgr"]),
            caption=(
                f"Export preview: {target_width}x{target_height} "
                f"{output_format}"
            ),
            width="stretch",
        )
        st.download_button(
            "Download Processed Image",
            data=export_result["bytes"],
            file_name=export_result["filename"],
            mime=export_result["mime"],
            width="stretch",
            key="image_download_processed",
        )

    source_items = st.session_state.get("image_source_items", [])
    if len(source_items) > 1 and st.button(
        "Build ZIP Export",
        width="stretch",
        key="image_build_zip",
    ):
        processed_images = st.session_state.get("processed_images_bgr", {})
        archive_buffer = io.BytesIO()
        with st.spinner("Resizing and packaging image dataset..."):
            with zipfile.ZipFile(
                archive_buffer,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for item in source_items:
                    image = processed_images.get(item["name"])
                    if image is None:
                        image = decode_image_bytes(item["bytes"])
                    resized = letterbox_resize(
                        image,
                        target_size=(target_width, target_height),
                        interpolation=interpolation,
                        fill_color=pad_color_bgr,
                    )
                    encoded, extension, _ = encode_image(
                        resized,
                        output_format.lstrip("."),
                        quality=quality,
                        png_compression=png_compression,
                        jpeg_background_bgr=pad_color_bgr,
                    )
                    archive.writestr(
                        _export_name(item["name"], extension),
                        encoded,
                    )
        st.session_state["image_export_zip_bytes"] = archive_buffer.getvalue()

    if st.session_state.get("image_export_zip_bytes"):
        st.download_button(
            "Download Processed ZIP",
            data=st.session_state["image_export_zip_bytes"],
            file_name="processed_images.zip",
            mime="application/zip",
            width="stretch",
            key="image_download_zip",
        )
