from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

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
