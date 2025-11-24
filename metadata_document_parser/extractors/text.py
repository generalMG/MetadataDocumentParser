import fitz
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from ..data_types import TextBlock
from ..layout import LayoutAnalyzer

class TextExtractor:
    """
    Handles text extraction from PDF documents.
    """

    def __init__(self, pdf_path: Path, layout_analyzer: LayoutAnalyzer):
        self.pdf_path = pdf_path
        self.layout_analyzer = layout_analyzer

    def extract_pymupdf(self, layout_aware: bool = True, column_aware: bool = True) -> List[TextBlock]:
        """Extract text using PyMuPDF"""
        if column_aware and layout_aware:
            return self._extract_pymupdf_with_columns()
        else:
            return self._extract_pymupdf_simple(layout_aware)

    def _extract_pymupdf_simple(self, layout_aware: bool = True) -> List[TextBlock]:
        """Extract text using PyMuPDF with optional layout awareness (simple mode)"""
        text_blocks = []
        doc = fitz.open(self.pdf_path)

        for page_num, page in enumerate(doc):
            if layout_aware:
                # Extract text blocks with position and formatting info
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        bbox = tuple(block["bbox"])

                        # Extract text from lines
                        text_content = []
                        font_sizes = []
                        font_names = []

                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text_content.append(span["text"])
                                font_sizes.append(span.get("size", 0))
                                font_names.append(span.get("font", ""))

                        text = " ".join(text_content)
                        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else None
                        font_name = font_names[0] if font_names else None

                        # Determine block type
                        block_type = self.layout_analyzer.classify_block_type(bbox, avg_font_size, page.rect.height)

                        if text.strip():
                            text_blocks.append(TextBlock(
                                text=text,
                                bbox=bbox,
                                page_num=page_num,
                                font_size=avg_font_size,
                                font_name=font_name,
                                block_type=block_type
                            ))
            else:
                # Simple text extraction
                text = page.get_text()
                if text.strip():
                    text_blocks.append(TextBlock(
                        text=text,
                        bbox=(0, 0, page.rect.width, page.rect.height),
                        page_num=page_num
                    ))

        doc.close()
        return text_blocks

    def _extract_pymupdf_with_columns(self) -> List[TextBlock]:
        """
        Extract text using column-aware method.
        """
        text_blocks = []
        doc = fitz.open(self.pdf_path)

        for page_num, page in enumerate(doc):
            # Get column bounding boxes in reading order
            column_bboxes = self.layout_analyzer.get_column_boxes(page)

            if not column_bboxes:
                # Fallback to simple extraction
                text = page.get_text(sort=True)
                if text.strip():
                    text_blocks.append(TextBlock(
                        text=text,
                        bbox=(0, 0, page.rect.width, page.rect.height),
                        page_num=page_num,
                        block_type="text"
                    ))
                continue

            # Extract text from each column bbox in order
            for col_idx, col_bbox in enumerate(column_bboxes):
                # Use get_text with clip and sort for proper word reconstruction
                col_text = page.get_text(clip=col_bbox, sort=True)

                if col_text.strip():
                    # Try to get font info from the first block in this region
                    blocks = page.get_text("dict", clip=col_bbox)["blocks"]
                    avg_font_size = None
                    font_name = None

                    if blocks:
                        font_sizes = []
                        font_names = []
                        for block in blocks:
                            if block.get("type") == 0:  # Text block
                                for line in block.get("lines", []):
                                    for span in line.get("spans", []):
                                        font_sizes.append(span.get("size", 0))
                                        font_names.append(span.get("font", ""))

                        if font_sizes:
                            avg_font_size = sum(font_sizes) / len(font_sizes)
                        if font_names:
                            font_name = font_names[0]

                    # Determine block type
                    block_type = self.layout_analyzer.classify_block_type(
                        tuple(col_bbox),
                        avg_font_size,
                        page.rect.height
                    )

                    text_blocks.append(TextBlock(
                        text=col_text,
                        bbox=tuple(col_bbox),
                        page_num=page_num,
                        font_size=avg_font_size,
                        font_name=font_name,
                        block_type=block_type
                    ))

        doc.close()
        return text_blocks

    def extract_pdfplumber(self, layout_aware: bool = True) -> List[TextBlock]:
        """Extract text using pdfplumber with layout awareness"""
        if not PDFPLUMBER_AVAILABLE:
            print("Warning: pdfplumber not available")
            return []

        text_blocks = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if layout_aware:
                    # Extract words with position information
                    words = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False
                    )

                    if words:
                        # Group words into lines based on y-coordinate
                        lines = self._group_words_into_lines(words)

                        # Create text blocks from lines
                        for line_words in lines:
                            if line_words:
                                text = " ".join([w["text"] for w in line_words])
                                x0 = min(w["x0"] for w in line_words)
                                y0 = min(w["top"] for w in line_words)
                                x1 = max(w["x1"] for w in line_words)
                                y1 = max(w["bottom"] for w in line_words)

                                text_blocks.append(TextBlock(
                                    text=text,
                                    bbox=(x0, y0, x1, y1),
                                    page_num=page_num,
                                    block_type="text"
                                ))
                else:
                    # Simple text extraction
                    text = page.extract_text()
                    if text:
                        text_blocks.append(TextBlock(
                            text=text,
                            bbox=(0, 0, page.width, page.height),
                            page_num=page_num
                        ))

        return text_blocks

    def _group_words_into_lines(self, words: List[Dict], y_tolerance: float = 3) -> List[List[Dict]]:
        """Group words into lines based on y-coordinate proximity"""
        if not words:
            return []

        # Sort words by vertical position, then horizontal
        sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))

        lines = []
        current_line = [sorted_words[0]]
        current_y = sorted_words[0]["top"]

        for word in sorted_words[1:]:
            if abs(word["top"] - current_y) <= y_tolerance:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
                current_y = word["top"]

        if current_line:
            lines.append(current_line)

        return lines
