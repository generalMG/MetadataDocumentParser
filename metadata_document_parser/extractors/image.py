import fitz
from typing import List
from pathlib import Path
from ..data_types import ImageData

class ImageExtractor:
    """
    Handles image extraction from PDF documents.
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def extract_pymupdf(self) -> List[ImageData]:
        """Extract images using PyMuPDF"""
        images = []
        doc = fitz.open(self.pdf_path)

        for page_num, page in enumerate(doc):
            image_list = page.get_images()

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                # Get image bbox
                img_rects = page.get_image_rects(xref)
                bbox = img_rects[0] if img_rects else (0, 0, 0, 0)

                # Extract image data
                base_image = doc.extract_image(xref)

                images.append(ImageData(
                    image_index=img_index,
                    page_num=page_num,
                    bbox=tuple(bbox),
                    width=base_image["width"],
                    height=base_image["height"],
                    colorspace=base_image.get("colorspace"),
                    image_bytes=base_image["image"],
                    ext=base_image["ext"]
                ))

        doc.close()
        return images
