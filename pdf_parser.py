"""
PDF Metadata Document Parser

A comprehensive PDF parser that extracts text, images, and tables with layout awareness.
Supports multiple extraction libraries for quality comparison.
"""

import io
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    from toon_format import encode as toon_encode, count_tokens, compare_formats, EncodeOptions
    TOON_AVAILABLE = True
except ImportError:
    TOON_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False


@dataclass
class TextBlock:
    """Represents a block of text with layout information"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    block_type: str = "text"  # text, title, header, footer, etc.


@dataclass
class ImageData:
    """Represents an extracted image with metadata"""
    image_index: int
    page_num: int
    bbox: Tuple[float, float, float, float]
    width: int
    height: int
    colorspace: Optional[str] = None
    image_bytes: Optional[bytes] = None
    ext: str = "png"


@dataclass
class TableData:
    """Represents an extracted table"""
    table_index: int
    page_num: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    data: List[List[str]] = field(default_factory=list)
    extraction_method: str = "unknown"


@dataclass
class FormulaData:
    """Represents an extracted mathematical formula"""
    formula_index: int
    page_num: int
    bbox: Tuple[float, float, float, float]
    formula_text: str  # Original text representation
    latex: Optional[str] = None  # LaTeX conversion (if available)
    confidence: float = 0.0  # Detection confidence score
    image_bytes: Optional[bytes] = None  # Formula as image


@dataclass
class DocumentMetadata:
    """PDF document metadata"""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    num_pages: int = 0
    file_size: int = 0
    page_sizes: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Complete parsed document data"""
    metadata: DocumentMetadata
    text_blocks: List[TextBlock] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    formulas: List[FormulaData] = field(default_factory=list)
    extraction_method: str = "unknown"
    parsing_time: float = 0.0
    column_layout: Optional[str] = None  # 'single', 'double', 'multi'


class PDFMetadataParser:
    """
    Main PDF parser class with support for multiple extraction libraries.

    Supports:
    - PyMuPDF (fitz): Fast, comprehensive text + image + layout extraction
    - pdfplumber: Excellent text extraction with layout awareness
    - Camelot: Advanced table extraction
    - Tabula: Alternative table extraction
    """

    def __init__(self, pdf_path: str, footer_margin: int = 50, header_margin: int = 50,
                 fast_column_detection: bool = True):
        """
        Initialize the parser with a PDF file path.

        Args:
            pdf_path: Path to the PDF file
            footer_margin: Height (in points) of bottom stripe to ignore for column detection
            header_margin: Height (in points) of top stripe to ignore for column detection
            fast_column_detection: Use faster (less accurate) column detection (default: True)
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.file_size = self.pdf_path.stat().st_size
        self.footer_margin = footer_margin
        self.header_margin = header_margin
        self.fast_column_detection = fast_column_detection

    def parse(
        self,
        extract_text: bool = True,
        extract_images: bool = True,
        extract_tables: bool = True,
        extract_formulas: bool = False,
        text_method: str = "pymupdf",
        table_method: str = "camelot",
        layout_aware: bool = True,
        column_aware: bool = True
    ) -> ParsedDocument:
        """
        Parse the PDF document and extract all requested content.

        Args:
            extract_text: Whether to extract text content
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            extract_formulas: Whether to extract and detect mathematical formulas
            text_method: Method for text extraction ('pymupdf', 'pdfplumber')
            table_method: Method for table extraction ('camelot', 'tabula')
            layout_aware: Whether to preserve layout information
            column_aware: Whether to detect columns and fix reading order

        Returns:
            ParsedDocument containing all extracted data
        """
        import time
        start_time = time.time()

        # Extract metadata first
        metadata = self._extract_metadata()

        # Initialize result
        result = ParsedDocument(
            metadata=metadata,
            extraction_method=text_method
        )

        # Extract text
        if extract_text:
            if text_method == "pymupdf" and PYMUPDF_AVAILABLE:
                # Use column-aware extraction for better word reconstruction
                if column_aware and layout_aware:
                    result.text_blocks = self._extract_text_pymupdf_with_columns()
                    # Detect layout from the extracted blocks
                    result.column_layout = self._detect_column_layout(result.text_blocks)
                else:
                    result.text_blocks = self._extract_text_pymupdf(layout_aware)
                    # Apply column-aware reading order if requested
                    if column_aware and result.text_blocks:
                        result.column_layout = self._detect_column_layout(result.text_blocks)
                        result.text_blocks = self._apply_reading_order(result.text_blocks, result.column_layout)
            elif text_method == "pdfplumber" and PDFPLUMBER_AVAILABLE:
                result.text_blocks = self._extract_text_pdfplumber(layout_aware)
                # Apply column-aware reading order if requested
                if column_aware and result.text_blocks:
                    result.column_layout = self._detect_column_layout(result.text_blocks)
                    result.text_blocks = self._apply_reading_order(result.text_blocks, result.column_layout)
            else:
                print(f"Warning: {text_method} not available, skipping text extraction")

        # Extract images
        if extract_images and PYMUPDF_AVAILABLE:
            result.images = self._extract_images_pymupdf()

        # Extract tables
        if extract_tables:
            if table_method == "camelot" and CAMELOT_AVAILABLE:
                result.tables = self._extract_tables_camelot()
            elif table_method == "tabula" and TABULA_AVAILABLE:
                result.tables = self._extract_tables_tabula()
            else:
                print(f"Warning: {table_method} not available, skipping table extraction")

        # Extract formulas
        if extract_formulas and PYMUPDF_AVAILABLE:
            result.formulas = self._extract_formulas(result.text_blocks)

        result.parsing_time = time.time() - start_time
        return result

    def _extract_metadata(self) -> DocumentMetadata:
        """Extract PDF metadata using PyMuPDF"""
        if not PYMUPDF_AVAILABLE:
            return DocumentMetadata(file_size=self.file_size)

        doc = fitz.open(self.pdf_path)
        metadata_dict = doc.metadata

        metadata = DocumentMetadata(
            title=metadata_dict.get('title'),
            author=metadata_dict.get('author'),
            subject=metadata_dict.get('subject'),
            creator=metadata_dict.get('creator'),
            producer=metadata_dict.get('producer'),
            creation_date=metadata_dict.get('creationDate'),
            modification_date=metadata_dict.get('modDate'),
            num_pages=len(doc),
            file_size=self.file_size,
            page_sizes=[(page.rect.width, page.rect.height) for page in doc]
        )

        doc.close()
        return metadata

    def _extract_text_pymupdf(self, layout_aware: bool = True) -> List[TextBlock]:
        """Extract text using PyMuPDF with layout awareness"""
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

                        # Determine block type based on font size and position
                        block_type = self._classify_block_type(bbox, avg_font_size, page.rect.height)

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

    def _extract_text_pymupdf_with_columns(self) -> List[TextBlock]:
        """
        Extract text using column-aware method.

        This method first detects columns, then extracts text from each column
        in reading order. This ensures proper word reconstruction and reading order.
        """
        text_blocks = []
        doc = fitz.open(self.pdf_path)

        for page_num, page in enumerate(doc):
            # Get column bounding boxes in reading order
            if self.fast_column_detection:
                column_bboxes = self._column_boxes_fast(
                    page,
                    footer_margin=self.footer_margin,
                    header_margin=self.header_margin
                )
            else:
                column_bboxes = self._column_boxes(
                    page,
                    footer_margin=self.footer_margin,
                    header_margin=self.header_margin
                )

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
                    block_type = self._classify_block_type(
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

    def _extract_text_pdfplumber(self, layout_aware: bool = True) -> List[TextBlock]:
        """Extract text using pdfplumber with layout awareness"""
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

    def _classify_block_type(self, bbox: Tuple, font_size: Optional[float], page_height: float) -> str:
        """Classify block type based on position and font size"""
        x0, y0, x1, y1 = bbox

        # Header: top 10% of page
        if y0 < page_height * 0.1:
            return "header"

        # Footer: bottom 10% of page
        if y1 > page_height * 0.9:
            return "footer"

        # Title: large font size
        if font_size and font_size > 16:
            return "title"

        # Heading: medium-large font size
        if font_size and font_size > 12:
            return "heading"

        return "text"

    def _extract_images_pymupdf(self) -> List[ImageData]:
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

    def _extract_tables_camelot(self) -> List[TableData]:
        """Extract tables using Camelot"""
        tables = []

        try:
            # Camelot can extract from all pages
            extracted_tables = camelot.read_pdf(
                str(self.pdf_path),
                pages='all',
                flavor='lattice',  # Use 'stream' for tables without borders
                suppress_stdout=True
            )

            for idx, table in enumerate(extracted_tables):
                tables.append(TableData(
                    table_index=idx,
                    page_num=table.page - 1,  # Camelot uses 1-based indexing
                    bbox=tuple(table._bbox) if hasattr(table, '_bbox') else None,
                    data=table.df.values.tolist(),
                    extraction_method="camelot"
                ))
        except Exception as e:
            print(f"Camelot extraction error: {e}")

        return tables

    def _extract_tables_tabula(self) -> List[TableData]:
        """Extract tables using Tabula"""
        tables = []

        try:
            # Tabula extracts all tables from all pages
            extracted_tables = tabula.read_pdf(
                str(self.pdf_path),
                pages='all',
                multiple_tables=True,
                silent=True
            )

            for idx, df in enumerate(extracted_tables):
                tables.append(TableData(
                    table_index=idx,
                    page_num=0,  # Tabula doesn't easily provide page numbers
                    data=df.values.tolist(),
                    extraction_method="tabula"
                ))
        except Exception as e:
            print(f"Tabula extraction error: {e}")

        return tables

    def compare_extraction_methods(self) -> Dict[str, Any]:
        """
        Compare different extraction methods for quality assessment.

        Returns:
            Dictionary with comparison results
        """
        results = {
            "text_extraction": {},
            "table_extraction": {}
        }

        # Compare text extraction methods
        if PYMUPDF_AVAILABLE:
            import time
            start = time.time()
            pymupdf_blocks = self._extract_text_pymupdf(layout_aware=True)
            pymupdf_time = time.time() - start

            results["text_extraction"]["pymupdf"] = {
                "num_blocks": len(pymupdf_blocks),
                "total_chars": sum(len(b.text) for b in pymupdf_blocks),
                "time": pymupdf_time
            }

        if PDFPLUMBER_AVAILABLE:
            import time
            start = time.time()
            pdfplumber_blocks = self._extract_text_pdfplumber(layout_aware=True)
            pdfplumber_time = time.time() - start

            results["text_extraction"]["pdfplumber"] = {
                "num_blocks": len(pdfplumber_blocks),
                "total_chars": sum(len(b.text) for b in pdfplumber_blocks),
                "time": pdfplumber_time
            }

        # Compare table extraction methods
        if CAMELOT_AVAILABLE:
            import time
            start = time.time()
            camelot_tables = self._extract_tables_camelot()
            camelot_time = time.time() - start

            results["table_extraction"]["camelot"] = {
                "num_tables": len(camelot_tables),
                "time": camelot_time
            }

        if TABULA_AVAILABLE:
            import time
            start = time.time()
            tabula_tables = self._extract_tables_tabula()
            tabula_time = time.time() - start

            results["table_extraction"]["tabula"] = {
                "num_tables": len(tabula_tables),
                "time": tabula_time
            }

        return results

    def export_to_dict(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        """
        Export parsed document to dictionary format.

        Args:
            parsed_doc: Parsed document to export

        Returns:
            Dictionary representation of the document
        """
        return {
            "metadata": {
                "title": parsed_doc.metadata.title,
                "author": parsed_doc.metadata.author,
                "subject": parsed_doc.metadata.subject,
                "creator": parsed_doc.metadata.creator,
                "producer": parsed_doc.metadata.producer,
                "creation_date": parsed_doc.metadata.creation_date,
                "modification_date": parsed_doc.metadata.modification_date,
                "num_pages": parsed_doc.metadata.num_pages,
                "file_size": parsed_doc.metadata.file_size,
                "page_sizes": parsed_doc.metadata.page_sizes
            },
            "text_blocks": [
                {
                    "text": block.text,
                    "bbox": block.bbox,
                    "page_num": block.page_num,
                    "font_size": block.font_size,
                    "font_name": block.font_name,
                    "block_type": block.block_type
                }
                for block in parsed_doc.text_blocks
            ],
            "images": [
                {
                    "image_index": img.image_index,
                    "page_num": img.page_num,
                    "bbox": img.bbox,
                    "width": img.width,
                    "height": img.height,
                    "colorspace": img.colorspace,
                    "ext": img.ext
                }
                for img in parsed_doc.images
            ],
            "tables": [
                {
                    "table_index": tbl.table_index,
                    "page_num": tbl.page_num,
                    "bbox": tbl.bbox,
                    "data": tbl.data,
                    "extraction_method": tbl.extraction_method
                }
                for tbl in parsed_doc.tables
            ],
            "formulas": [
                {
                    "formula_index": formula.formula_index,
                    "page_num": formula.page_num,
                    "bbox": formula.bbox,
                    "formula_text": formula.formula_text,
                    "latex": formula.latex,
                    "confidence": formula.confidence
                }
                for formula in parsed_doc.formulas
            ],
            "extraction_method": parsed_doc.extraction_method,
            "parsing_time": parsed_doc.parsing_time,
            "column_layout": parsed_doc.column_layout
        }

    def export_to_json(self, parsed_doc: ParsedDocument, indent: int = 2) -> str:
        """
        Export parsed document to JSON format.

        Args:
            parsed_doc: Parsed document to export
            indent: JSON indentation (default: 2)

        Returns:
            JSON string representation
        """
        import json
        data = self.export_to_dict(parsed_doc)
        return json.dumps(data, indent=indent, default=str)

    def export_to_toon(self, parsed_doc: ParsedDocument, delimiter: str = ",") -> str:
        """
        Export parsed document to TOON format for token-efficient LLM input.

        TOON achieves 30-60% token reduction vs JSON for uniform arrays.
        Default export format for this parser.

        Args:
            parsed_doc: Parsed document to export
            delimiter: Array delimiter: ',' (comma), '\\t' (tab), or '|' (pipe)
                      Tab often provides best token efficiency

        Returns:
            TOON formatted string

        Raises:
            RuntimeError: If toon-format package is not installed

        Example:
            >>> parser = PDFMetadataParser("doc.pdf")
            >>> result = parser.parse()
            >>> toon_str = parser.export_to_toon(result, delimiter="\\t")
        """
        if not TOON_AVAILABLE:
            raise RuntimeError(
                "toon-format package is required for TOON export. "
                "Install it with: pip install toon-format"
            )

        data = self.export_to_dict(parsed_doc)
        options = EncodeOptions(indent=2, delimiter=delimiter, lengthMarker='#')
        return toon_encode(data, options=options)

    def export(self, parsed_doc: ParsedDocument, format: str = "toon",
              delimiter: str = ",", indent: int = 2) -> str:
        """
        Export parsed document to specified format.

        Args:
            parsed_doc: Parsed document to export
            format: Output format - "toon" (default, 30-60% fewer tokens) or "json"
            delimiter: TOON delimiter: ',' (comma), '\\t' (tab), or '|' (pipe)
            indent: JSON indentation (only used if format="json")

        Returns:
            Formatted string in requested format

        Example:
            >>> parser = PDFMetadataParser("doc.pdf")
            >>> result = parser.parse()
            >>> # TOON format (default - token-efficient for LLMs)
            >>> toon_output = parser.export(result)
            >>> # JSON format (explicit)
            >>> json_output = parser.export(result, format="json")
        """
        if format.lower() == "json":
            return self.export_to_json(parsed_doc, indent=indent)
        elif format.lower() == "toon":
            return self.export_to_toon(parsed_doc, delimiter=delimiter)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'toon' or 'json'")

    def compare_export_formats(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        """
        Compare token counts between JSON and TOON export formats.

        Args:
            parsed_doc: Parsed document to compare

        Returns:
            Dictionary with format comparison including token counts and savings

        Example:
            >>> parser = PDFMetadataParser("doc.pdf")
            >>> result = parser.parse()
            >>> comparison = parser.compare_export_formats(result)
            >>> print(f"TOON uses {comparison['toon_tokens']} tokens")
            >>> print(f"JSON uses {comparison['json_tokens']} tokens")
            >>> print(f"Savings: {comparison['savings_percent']}%")
        """
        if not TOON_AVAILABLE:
            return {
                "error": "toon-format package not installed",
                "message": "Install with: pip install toon-format"
            }

        # Get exports
        json_output = self.export_to_json(parsed_doc, indent=2)
        toon_output = self.export_to_toon(parsed_doc, delimiter=",")
        toon_tab_output = self.export_to_toon(parsed_doc, delimiter="\t")

        # Count tokens
        json_tokens = count_tokens(json_output)
        toon_tokens = count_tokens(toon_output)
        toon_tab_tokens = count_tokens(toon_tab_output)

        # Calculate savings
        savings_comma = ((json_tokens - toon_tokens) / json_tokens * 100) if json_tokens > 0 else 0
        savings_tab = ((json_tokens - toon_tab_tokens) / json_tokens * 100) if json_tokens > 0 else 0

        return {
            "json_tokens": json_tokens,
            "json_size_bytes": len(json_output.encode('utf-8')),
            "toon_comma_tokens": toon_tokens,
            "toon_comma_size_bytes": len(toon_output.encode('utf-8')),
            "toon_comma_savings_percent": round(savings_comma, 1),
            "toon_tab_tokens": toon_tab_tokens,
            "toon_tab_size_bytes": len(toon_tab_output.encode('utf-8')),
            "toon_tab_savings_percent": round(savings_tab, 1),
            "best_format": "toon_tab" if toon_tab_tokens < toon_tokens else "toon_comma",
            "best_savings_percent": round(max(savings_comma, savings_tab), 1)
        }

    def _detect_column_layout(self, text_blocks: List[TextBlock]) -> str:
        """
        Detect column layout of the document.

        Args:
            text_blocks: List of text blocks from the document

        Returns:
            'single', 'double', or 'multi' column layout
        """
        if not text_blocks:
            return 'single'

        # Group blocks by page
        pages_blocks = {}
        for block in text_blocks:
            if block.page_num not in pages_blocks:
                pages_blocks[block.page_num] = []
            pages_blocks[block.page_num].append(block)

        # Analyze each page to detect columns
        column_counts = []

        for page_num, blocks in pages_blocks.items():
            if not blocks:
                continue

            # Get page width from first block or use default
            page_width = max(block.bbox[2] for block in blocks)

            # Collect x-centers of all blocks
            x_centers = [(block.bbox[0] + block.bbox[2]) / 2 for block in blocks]

            if len(x_centers) < 3:
                column_counts.append(1)
                continue

            # Use clustering to detect columns
            # Simple approach: divide page into potential column regions
            # and see if blocks cluster around specific x-positions

            # Try to detect gaps in x-positions (column boundaries)
            x_centers_sorted = sorted(x_centers)

            # Calculate gaps between consecutive x-centers
            gaps = []
            for i in range(len(x_centers_sorted) - 1):
                gap = x_centers_sorted[i + 1] - x_centers_sorted[i]
                gaps.append((gap, x_centers_sorted[i]))

            # Find significant gaps (potential column boundaries)
            if gaps:
                avg_gap = sum(g[0] for g in gaps) / len(gaps)
                std_gap = (sum((g[0] - avg_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5

                # Significant gap is > mean + 1.5 * std
                threshold = avg_gap + 1.5 * std_gap
                significant_gaps = [g for g in gaps if g[0] > threshold]

                # Number of columns = number of significant gaps + 1
                num_columns = len(significant_gaps) + 1
                column_counts.append(min(num_columns, 3))  # Cap at 3 columns
            else:
                column_counts.append(1)

        # Determine overall layout
        if not column_counts:
            return 'single'

        avg_columns = sum(column_counts) / len(column_counts)

        if avg_columns < 1.5:
            return 'single'
        elif avg_columns < 2.5:
            return 'double'
        else:
            return 'multi'

    def _column_boxes_fast(self, page, footer_margin: int = 50, header_margin: int = 50) -> List[fitz.IRect]:
        """
        Fast column detection using simple bbox analysis (10-100x faster).

        Skips expensive operations like drawing/image detection and uses
        simple heuristics to find column boundaries.

        Args:
            page: PyMuPDF page object
            footer_margin: Height of bottom stripe to ignore
            header_margin: Height of top stripe to ignore

        Returns:
            List of IRect objects representing column bboxes, sorted by reading order
        """
        # Compute relevant page area
        clip = +page.rect
        clip.y1 -= footer_margin
        clip.y0 += header_margin

        # Get text blocks quickly
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT, clip=clip)["blocks"]

        if not blocks:
            return []

        # Extract bboxes of text blocks only
        text_bboxes = []
        for b in blocks:
            if b.get("type") == 0 and b.get("lines"):  # Text block with content
                bbox = fitz.IRect(b["bbox"])
                if not bbox.is_empty:
                    text_bboxes.append(bbox)

        if not text_bboxes:
            return []

        # Sort by position (top to bottom, left to right)
        text_bboxes.sort(key=lambda b: (b.y0, b.x0))

        # Find vertical gaps (potential column separators)
        page_width = int(page.rect.width)

        # Collect all x-ranges
        x_ranges = [(b.x0, b.x1) for b in text_bboxes]

        # Find the largest horizontal gap
        if len(x_ranges) < 2:
            # Single column - return whole page
            return [fitz.IRect(0, int(clip.y0), page_width, int(clip.y1))]

        # Simple approach: find x-positions where text is absent (column gaps)
        # Build coverage map
        x_coverage = [False] * page_width
        for x0, x1 in x_ranges:
            for x in range(max(0, int(x0)), min(page_width, int(x1))):
                x_coverage[x] = True

        # Find gaps
        gaps = []
        in_gap = False
        gap_start = 0

        for x in range(page_width):
            if not x_coverage[x]:
                if not in_gap:
                    gap_start = x
                    in_gap = True
            else:
                if in_gap:
                    gap_width = x - gap_start
                    if gap_width > 20:  # Significant gap (> 20 points)
                        gaps.append((gap_start, x, gap_width))
                    in_gap = False

        # No significant gaps = single column
        if not gaps:
            return [fitz.IRect(0, int(clip.y0), page_width, int(clip.y1))]

        # Find the largest gap (likely the column separator)
        largest_gap = max(gaps, key=lambda g: g[2])
        gap_mid = (largest_gap[0] + largest_gap[1]) // 2

        # Split into left and right columns
        left_blocks = [b for b in text_bboxes if b.x0 < gap_mid]
        right_blocks = [b for b in text_bboxes if b.x0 >= gap_mid]

        columns = []

        if left_blocks:
            left_x0 = min(b.x0 for b in left_blocks)
            left_x1 = max(b.x1 for b in left_blocks)
            left_y0 = min(b.y0 for b in left_blocks)
            left_y1 = max(b.y1 for b in left_blocks)
            columns.append(fitz.IRect(left_x0, left_y0, left_x1, left_y1))

        if right_blocks:
            right_x0 = min(b.x0 for b in right_blocks)
            right_x1 = max(b.x1 for b in right_blocks)
            right_y0 = min(b.y0 for b in right_blocks)
            right_y1 = max(b.y1 for b in right_blocks)
            columns.append(fitz.IRect(right_x0, right_y0, right_x1, right_y1))

        return columns

    def _column_boxes(self, page, footer_margin: int = 50, header_margin: int = 50,
                     no_image_text: bool = True) -> List[fitz.IRect]:
        """
        Advanced multi-column detection using PyMuPDF.

        Determines bboxes which wrap a column by intelligently detecting:
        - Text with different background colors
        - Text blocks and their boundaries
        - Vertical vs horizontal text
        - Text overlaying images

        Args:
            page: PyMuPDF page object
            footer_margin: Height of bottom stripe to ignore
            header_margin: Height of top stripe to ignore
            no_image_text: Whether to ignore text written on images

        Returns:
            List of IRect objects representing column bboxes, sorted by reading order
        """
        paths = page.get_drawings()
        bboxes = []
        path_rects = []
        img_bboxes = []
        vert_bboxes = []

        # Compute relevant page area
        clip = +page.rect
        clip.y1 -= footer_margin
        clip.y0 += header_margin

        def can_extend(temp, bb, bboxlist):
            """Check if temp can be extended by bb without intersecting bboxlist items."""
            for b in bboxlist:
                if not self._intersects_bboxes(temp, vert_bboxes) and (
                    b is None or b == bb or (temp & b).is_empty
                ):
                    continue
                return False
            return True

        def in_bbox(bb, bboxes):
            """Return 1-based number if a bbox contains bb, else return 0."""
            for i, bbox in enumerate(bboxes):
                if bb in bbox:
                    return i + 1
            return 0

        def extend_right(bboxes, width, path_bboxes, vert_bboxes, img_bboxes):
            """Extend bbox to the right page border where possible."""
            for i, bb in enumerate(bboxes):
                if in_bbox(bb, path_bboxes):
                    continue
                if in_bbox(bb, img_bboxes):
                    continue

                temp = +bb
                temp.x1 = width

                if self._intersects_bboxes(temp, path_bboxes + vert_bboxes + img_bboxes):
                    continue

                check = can_extend(temp, bb, bboxes)
                if check:
                    bboxes[i] = temp

            return [b for b in bboxes if b is not None]

        def clean_nblocks(nblocks):
            """Remove duplicates and fix sequence in special cases."""
            if len(nblocks) < 2:
                return nblocks

            # Remove duplicates
            for i in range(len(nblocks) - 1, 0, -1):
                if nblocks[i - 1] == nblocks[i]:
                    del nblocks[i]

            # Sort segments with same bottom value by x-coordinate
            if not nblocks:
                return nblocks

            y1 = nblocks[0].y1
            i0 = 0
            i1 = -1

            for i in range(1, len(nblocks)):
                b1 = nblocks[i]
                if abs(b1.y1 - y1) > 10:
                    if i1 > i0:
                        nblocks[i0:i1 + 1] = sorted(nblocks[i0:i1 + 1], key=lambda b: b.x0)
                    y1 = b1.y1
                    i0 = i
                i1 = i

            if i1 > i0:
                nblocks[i0:i1 + 1] = sorted(nblocks[i0:i1 + 1], key=lambda b: b.x0)

            return nblocks

        # Extract vector graphics
        for p in paths:
            path_rects.append(p["rect"].irect)
        path_bboxes = sorted(path_rects, key=lambda b: (b.y0, b.x0))

        # Get image bboxes
        for item in page.get_images():
            img_bboxes.extend(page.get_image_rects(item[0]))

        # Extract text blocks
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT, clip=clip)["blocks"]

        for b in blocks:
            bbox = fitz.IRect(b["bbox"])

            if no_image_text and in_bbox(bbox, img_bboxes):
                continue

            # Check if first line is horizontal
            if b["lines"]:
                line0 = b["lines"][0]
                if line0["dir"] != (1, 0):
                    vert_bboxes.append(bbox)
                    continue

                srect = fitz.EMPTY_IRECT()
                for line in b["lines"]:
                    lbbox = fitz.IRect(line["bbox"])
                    text = "".join([s["text"].strip() for s in line["spans"]])
                    if len(text) > 1:
                        srect |= lbbox
                bbox = +srect

                if not bbox.is_empty:
                    bboxes.append(bbox)

        # Sort by background, then position
        bboxes.sort(key=lambda k: (in_bbox(k, path_bboxes), k.y0, k.x0))

        # Extend bboxes to the right where possible
        bboxes = extend_right(bboxes, int(page.rect.width), path_bboxes, vert_bboxes, img_bboxes)

        if not bboxes:
            return []

        # Join bboxes to establish column structure
        nblocks = [bboxes[0]]
        bboxes = bboxes[1:]

        for i, bb in enumerate(bboxes):
            check = False

            for j in range(len(nblocks)):
                nbb = nblocks[j]

                # Never join across columns
                if bb is None or nbb.x1 < bb.x0 or bb.x1 < nbb.x0:
                    continue

                # Never join across different background colors
                if in_bbox(nbb, path_bboxes) != in_bbox(bb, path_bboxes):
                    continue

                temp = bb | nbb
                check = can_extend(temp, nbb, nblocks)
                if check:
                    break

            if not check:
                nblocks.append(bb)
                j = len(nblocks) - 1
                temp = nblocks[j]

            check = can_extend(temp, bb, bboxes)
            if not check:
                nblocks.append(bb)
            else:
                nblocks[j] = temp
            bboxes[i] = None

        # Clean and return
        nblocks = clean_nblocks(nblocks)
        return nblocks

    def _intersects_bboxes(self, bb, bboxes) -> bool:
        """Check if bb intersects any bbox in bboxes list."""
        for bbox in bboxes:
            if not (bb & bbox).is_empty:
                return True
        return False

    def _apply_reading_order(self, text_blocks: List[TextBlock], column_layout: str) -> List[TextBlock]:
        """
        Apply proper reading order based on detected column layout.

        For multi-column documents, sorts text left-to-right, top-to-bottom.
        For single-column documents, sorts top-to-bottom only.

        Args:
            text_blocks: List of text blocks
            column_layout: Detected column layout ('single', 'double', 'multi')

        Returns:
            Sorted list of text blocks in reading order
        """
        if not text_blocks or column_layout == 'single':
            # Simple top-to-bottom sorting
            return sorted(text_blocks, key=lambda b: (b.page_num, b.bbox[1]))

        # For multi-column layout, use advanced column detection if PyMuPDF is available
        if PYMUPDF_AVAILABLE:
            return self._apply_reading_order_pymupdf(text_blocks)

        # Fallback to simple column sorting
        return self._apply_reading_order_simple(text_blocks, column_layout)

    def _apply_reading_order_pymupdf(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Apply reading order using advanced PyMuPDF column detection.

        Args:
            text_blocks: List of text blocks

        Returns:
            Sorted list of text blocks in proper reading order
        """
        if not text_blocks:
            return []

        sorted_blocks = []
        doc = fitz.open(self.pdf_path)

        # Group blocks by page
        pages_blocks = {}
        for block in text_blocks:
            if block.page_num not in pages_blocks:
                pages_blocks[block.page_num] = []
            pages_blocks[block.page_num].append(block)

        # Process each page
        for page_num in sorted(pages_blocks.keys()):
            page = doc[page_num]
            page_blocks = pages_blocks[page_num]

            # Get column bboxes using advanced detection
            column_bboxes = self._column_boxes(
                page,
                footer_margin=self.footer_margin,
                header_margin=self.header_margin
            )

            if not column_bboxes:
                # Fallback to simple sorting
                page_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
                sorted_blocks.extend(page_blocks)
                continue

            # For each column bbox, extract text in reading order
            for col_bbox in column_bboxes:
                # Find text blocks that fall within this column bbox
                col_blocks = []
                for block in page_blocks:
                    block_rect = fitz.Rect(block.bbox)
                    # Check if block overlaps with column bbox
                    if not (block_rect & col_bbox).is_empty:
                        col_blocks.append(block)

                # Sort blocks within column by vertical position
                col_blocks.sort(key=lambda b: b.bbox[1])
                sorted_blocks.extend(col_blocks)

                # Remove processed blocks from page_blocks
                for block in col_blocks:
                    if block in page_blocks:
                        page_blocks.remove(block)

            # Add any remaining blocks (shouldn't happen, but safety check)
            if page_blocks:
                page_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
                sorted_blocks.extend(page_blocks)

        doc.close()
        return sorted_blocks

    def _apply_reading_order_simple(self, text_blocks: List[TextBlock],
                                    column_layout: str) -> List[TextBlock]:
        """
        Simple fallback column sorting (original implementation).

        Args:
            text_blocks: List of text blocks
            column_layout: Detected column layout ('single', 'double', 'multi')

        Returns:
            Sorted list of text blocks
        """
        sorted_blocks = []
        pages = {}
        for block in text_blocks:
            if block.page_num not in pages:
                pages[block.page_num] = []
            pages[block.page_num].append(block)

        for page_num in sorted(pages.keys()):
            page_blocks = pages[page_num]
            if not page_blocks:
                continue

            page_width = max(block.bbox[2] for block in page_blocks)

            if column_layout == 'double':
                mid_point = page_width / 2
                left_column = []
                right_column = []

                for block in page_blocks:
                    x_center = (block.bbox[0] + block.bbox[2]) / 2
                    if x_center < mid_point:
                        left_column.append(block)
                    else:
                        right_column.append(block)

                left_column.sort(key=lambda b: b.bbox[1])
                right_column.sort(key=lambda b: b.bbox[1])
                sorted_blocks.extend(left_column)
                sorted_blocks.extend(right_column)

            else:  # multi-column
                blocks_with_centers = [(b, (b.bbox[0] + b.bbox[2]) / 2) for b in page_blocks]
                blocks_with_centers.sort(key=lambda x: x[1])
                num_columns = 3
                column_width = page_width / num_columns
                columns = [[] for _ in range(num_columns)]

                for block, x_center in blocks_with_centers:
                    col_idx = min(int(x_center / column_width), num_columns - 1)
                    columns[col_idx].append(block)

                for column in columns:
                    column.sort(key=lambda b: b.bbox[1])
                    sorted_blocks.extend(column)

        return sorted_blocks

    def _extract_formulas(self, text_blocks: List[TextBlock]) -> List[FormulaData]:
        """
        Extract mathematical formulas from text blocks using heuristic detection.

        This uses classic ML/pattern matching (no deep learning) to identify
        formula-like text blocks based on:
        - Special mathematical characters
        - Font characteristics
        - Layout patterns

        Args:
            text_blocks: List of extracted text blocks

        Returns:
            List of detected formulas
        """
        formulas = []
        formula_index = 0

        # Mathematical symbols commonly found in formulas
        math_symbols = set('∫∑∏√±×÷≈≠≤≥∞∂∇αβγδεζηθλμπρσφψω')
        math_chars = set('+-*/=()[]{}^_∈∉⊂⊃∪∩')

        doc = fitz.open(self.pdf_path) if PYMUPDF_AVAILABLE else None

        for block in text_blocks:
            text = block.text.strip()

            if not text or len(text) < 2:
                continue

            # Calculate formula likelihood score
            score = 0.0

            # Check for mathematical symbols
            math_symbol_count = sum(1 for c in text if c in math_symbols)
            math_char_count = sum(1 for c in text if c in math_chars)

            if math_symbol_count > 0:
                score += math_symbol_count * 0.3

            if math_char_count > 0:
                score += math_char_count * 0.1

            # Check for common formula patterns
            if any(pattern in text for pattern in ['=', '∫', '∑', '∏', '√', '∂', '∇']):
                score += 0.5

            # Check for superscripts/subscripts patterns (^, _)
            if '^' in text or '_' in text:
                score += 0.3

            # Check for fraction-like patterns (a/b where a, b are short)
            import re
            if re.search(r'\w+/\w+', text):
                score += 0.2

            # Check font size (formulas are often in different size)
            if block.font_size and block.font_size < 10:
                score += 0.2

            # Check for isolated blocks (formulas are often standalone)
            if len(text) < 50 and math_char_count > len(text) * 0.1:
                score += 0.3

            # If score exceeds threshold, consider it a formula
            if score >= 0.7:
                # Try to extract formula image if document is available
                formula_image = None
                if doc:
                    try:
                        page = doc[block.page_num]
                        # Extract region as image
                        bbox = fitz.Rect(block.bbox)
                        pix = page.get_pixmap(clip=bbox, matrix=fitz.Matrix(2, 2))  # 2x resolution
                        formula_image = pix.tobytes("png")
                    except:
                        pass

                # Attempt basic LaTeX conversion
                latex = self._text_to_latex_heuristic(text)

                formulas.append(FormulaData(
                    formula_index=formula_index,
                    page_num=block.page_num,
                    bbox=block.bbox,
                    formula_text=text,
                    latex=latex,
                    confidence=min(score, 1.0),
                    image_bytes=formula_image
                ))

                formula_index += 1

        if doc:
            doc.close()

        return formulas

    def _text_to_latex_heuristic(self, text: str) -> str:
        """
        Convert text to LaTeX using heuristic rules (classic approach, no DL).

        This is a basic conversion and won't handle complex formulas perfectly.
        For better results, consider using external services or manual annotation.

        Args:
            text: Formula text

        Returns:
            LaTeX representation
        """
        latex = text

        # Greek letters (if already in Unicode)
        greek_map = {
            'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta',
            'ε': r'\epsilon', 'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta',
            'λ': r'\lambda', 'μ': r'\mu', 'π': r'\pi', 'ρ': r'\rho',
            'σ': r'\sigma', 'φ': r'\phi', 'ψ': r'\psi', 'ω': r'\omega',
            'Δ': r'\Delta', 'Σ': r'\Sigma', 'Π': r'\Pi', 'Ω': r'\Omega'
        }

        for greek, latex_greek in greek_map.items():
            latex = latex.replace(greek, latex_greek)

        # Mathematical symbols
        symbol_map = {
            '≈': r'\approx',
            '≠': r'\neq',
            '≤': r'\leq',
            '≥': r'\geq',
            '∞': r'\infty',
            '∂': r'\partial',
            '∇': r'\nabla',
            '∫': r'\int',
            '∑': r'\sum',
            '∏': r'\prod',
            '√': r'\sqrt',
            '±': r'\pm',
            '×': r'\times',
            '÷': r'\div',
            '∈': r'\in',
            '∉': r'\notin',
            '⊂': r'\subset',
            '⊃': r'\supset',
            '∪': r'\cup',
            '∩': r'\cap',
        }

        for symbol, latex_symbol in symbol_map.items():
            latex = latex.replace(symbol, latex_symbol)

        # Handle superscripts (simplified - only works for simple cases)
        # a^b -> a^{b}
        import re
        latex = re.sub(r'\^(\w)', r'^{\1}', latex)

        # Handle subscripts
        # a_b -> a_{b}
        latex = re.sub(r'_(\w)', r'_{\1}', latex)

        # Wrap in math mode if not already
        if not latex.startswith('$'):
            latex = f'${latex}$'

        return latex

    def save_images(self, parsed_doc: ParsedDocument, output_dir: str) -> List[str]:
        """
        Save extracted images to disk.

        Args:
            parsed_doc: Parsed document with images
            output_dir: Directory to save images

        Returns:
            List of saved image file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for img in parsed_doc.images:
            if img.image_bytes:
                filename = f"page_{img.page_num}_img_{img.image_index}.{img.ext}"
                filepath = output_path / filename

                with open(filepath, "wb") as f:
                    f.write(img.image_bytes)

                saved_paths.append(str(filepath))

        return saved_paths

    def visualize_columns(self, output_path: Optional[str] = None) -> str:
        """
        Create a visual representation of detected columns by drawing borders
        around detected column bboxes and numbering them.

        This is useful for debugging and understanding how the column detection works.

        Args:
            output_path: Optional output path for the annotated PDF.
                        If not provided, uses "<original_name>-columns.pdf"

        Returns:
            Path to the annotated PDF file
        """
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF is required for column visualization")

        if output_path is None:
            output_path = str(self.pdf_path).replace(".pdf", "-columns.pdf")

        doc = fitz.open(self.pdf_path)

        for page_num, page in enumerate(doc):
            # Remove any geometry issues
            page.wrap_contents()

            # Get column bboxes
            if self.fast_column_detection:
                column_bboxes = self._column_boxes_fast(
                    page,
                    footer_margin=self.footer_margin,
                    header_margin=self.header_margin
                )
            else:
                column_bboxes = self._column_boxes(
                    page,
                    footer_margin=self.footer_margin,
                    header_margin=self.header_margin
                )

            # Draw rectangles and numbers
            shape = page.new_shape()

            for i, rect in enumerate(column_bboxes):
                # Draw red border
                shape.draw_rect(rect)

                # Add sequence number
                shape.insert_text(
                    rect.tl + (5, 15),
                    str(i),
                    color=fitz.pdfcolor["red"],
                    fontsize=12
                )

            # Finish with red color
            shape.finish(color=fitz.pdfcolor["red"])
            shape.commit()

        # Save annotated document
        doc.ez_save(output_path)
        doc.close()

        return output_path
