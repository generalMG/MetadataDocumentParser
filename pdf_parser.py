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

    def __init__(self, pdf_path: str):
        """
        Initialize the parser with a PDF file path.

        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.file_size = self.pdf_path.stat().st_size

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
                result.text_blocks = self._extract_text_pymupdf(layout_aware)
            elif text_method == "pdfplumber" and PDFPLUMBER_AVAILABLE:
                result.text_blocks = self._extract_text_pdfplumber(layout_aware)
            else:
                print(f"Warning: {text_method} not available, skipping text extraction")

            # Apply column-aware reading order if requested
            if column_aware and result.text_blocks:
                result.column_layout = self._detect_column_layout(result.text_blocks)
                result.text_blocks = self._apply_reading_order(result.text_blocks, result.column_layout)

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
        """Export parsed document to dictionary format"""
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

        # For multi-column layout, we need to sort by columns
        sorted_blocks = []

        # Group by page
        pages = {}
        for block in text_blocks:
            if block.page_num not in pages:
                pages[block.page_num] = []
            pages[block.page_num].append(block)

        # Process each page
        for page_num in sorted(pages.keys()):
            page_blocks = pages[page_num]

            if not page_blocks:
                continue

            # Determine column boundaries
            page_width = max(block.bbox[2] for block in page_blocks)

            if column_layout == 'double':
                # Two-column layout: split at middle
                mid_point = page_width / 2

                # Assign blocks to columns based on their center x-coordinate
                left_column = []
                right_column = []

                for block in page_blocks:
                    x_center = (block.bbox[0] + block.bbox[2]) / 2

                    if x_center < mid_point:
                        left_column.append(block)
                    else:
                        right_column.append(block)

                # Sort each column by vertical position
                left_column.sort(key=lambda b: b.bbox[1])
                right_column.sort(key=lambda b: b.bbox[1])

                # Add left column first, then right column
                sorted_blocks.extend(left_column)
                sorted_blocks.extend(right_column)

            else:  # multi-column
                # For 3+ columns, use clustering approach
                # Extract x-centers
                blocks_with_centers = [(b, (b.bbox[0] + b.bbox[2]) / 2) for b in page_blocks]

                # Sort blocks by x-center to identify columns
                blocks_with_centers.sort(key=lambda x: x[1])

                # Group into columns (simple approach: divide into thirds)
                num_columns = 3
                column_width = page_width / num_columns

                columns = [[] for _ in range(num_columns)]

                for block, x_center in blocks_with_centers:
                    col_idx = min(int(x_center / column_width), num_columns - 1)
                    columns[col_idx].append(block)

                # Sort each column by vertical position and add to result
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
