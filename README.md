# PDF Metadata Document Parser

A comprehensive, layout-aware PDF parser for extracting text, images, and tables from digitally-born PDF documents. Supports multiple extraction libraries for quality comparison and optimal results.

## Features

- **Metadata Extraction**: Extracts complete PDF metadata (title, author, dates, page info, etc.)
- **Layout-Aware Text Extraction**: Preserves document structure, font information, and reading order
- **Advanced Column Detection**:
  - Intelligently detects multi-column layouts (research papers, newspapers, magazines)
  - **Fast mode**: 50-100x faster than detailed analysis
  - **Detailed mode**: Handles text with different background colors and images
  - Respects column boundaries and proper reading order
  - Configurable header/footer margins for improved accuracy
- **Column Visualization**: Debug tool to visualize detected column boundaries
- **Formula Detection & LaTeX Conversion**: Detects mathematical formulas and converts to LaTeX (classic ML, no GPU required)
- **Image Extraction**: Extracts embedded images with position and metadata
- **Table Extraction**: Advanced table detection and extraction
- **Multiple Extraction Methods**: Compare different libraries for optimal results
- **Token-Efficient Export**:
  - **TOON format** (default): 10-60% fewer tokens vs JSON - ideal for LLM input
  - **JSON format**: Standard JSON export when needed
  - Built-in token comparison to measure savings

## Supported Libraries

### Text Extraction
- **PyMuPDF (fitz)**: Fast, comprehensive extraction with layout awareness
- **pdfplumber**: Excellent text extraction with precise positioning

### Table Extraction
- **Camelot**: Advanced table extraction with lattice and stream modes
- **Tabula**: Java-based table extraction

### Image Extraction
- **PyMuPDF (fitz)**: Complete image extraction with metadata

## Installation

### Prerequisites

1. **Python 3.8+**

2. **Java Runtime Environment** (required for Tabula)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install default-jre

   # macOS
   brew install openjdk

   # Windows
   # Download from https://www.java.com/
   ```

3. **Ghostscript** (required for Camelot)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ghostscript

   # macOS
   brew install ghostscript

   # Windows
   # Download from https://www.ghostscript.com/
   ```

### Create Virtual Environment (Recommended)

It's recommended to use a virtual environment to isolate dependencies:

**Using venv (built-in):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using uv (creates venv automatically):**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

You can install dependencies using either classic pip or the faster uv package manager:

#### Option 1: Using pip (Classic)

```bash
pip install -r requirements.txt
```

#### Option 2: Using uv (Faster Alternative)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer written in Rust. It's 10-100x faster than pip.

```bash
# Install uv first (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
uv pip install -r requirements.txt
```

Or on Windows (PowerShell):
```powershell
# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install dependencies
uv pip install -r requirements.txt
```

**Benefits of using uv:**
- 10-100x faster installation and etc etc.

## Quick Start

### Basic Usage

```python
from pdf_parser import PDFMetadataParser

# Initialize parser
parser = PDFMetadataParser("document.pdf")

# Parse with all features
result = parser.parse(
    extract_text=True,
    extract_images=True,
    extract_tables=True,
    extract_formulas=True,  # NEW: Extract mathematical formulas
    layout_aware=True,
    column_aware=True  # NEW: Fix reading order for multi-column layouts
)

# Access results
print(f"Title: {result.metadata.title}")
print(f"Pages: {result.metadata.num_pages}")
print(f"Text blocks: {len(result.text_blocks)}")
print(f"Images: {len(result.images)}")
print(f"Tables: {len(result.tables)}")
```

### Run Examples

```bash
# Run all examples
python example_usage.py your_document.pdf

# Test multi-column extraction specifically
python example_multi_column.py research_paper.pdf

# Debug and visualize column detection
python test_column_detection.py research_paper.pdf --header-margin 50 --footer-margin 50

# Test TOON export with token comparison
python example_toon_export.py research_paper.pdf
```

## Usage Examples

### 1. Extract Text with Layout Awareness

```python
parser = PDFMetadataParser("document.pdf")

result = parser.parse(
    extract_text=True,
    text_method="pymupdf",  # or "pdfplumber"
    layout_aware=True
)

# Access text blocks with position and formatting
for block in result.text_blocks:
    print(f"Page {block.page_num}: {block.block_type}")
    print(f"Position: {block.bbox}")
    print(f"Font: {block.font_name} (Size: {block.font_size})")
    print(f"Text: {block.text}\n")
```

### 2. Extract Tables

```python
parser = PDFMetadataParser("document.pdf")

# Using Camelot (best for bordered tables)
result = parser.parse(
    extract_tables=True,
    table_method="camelot"
)

# Using Tabula (good for borderless tables)
result = parser.parse(
    extract_tables=True,
    table_method="tabula"
)

# Access table data
for table in result.tables:
    print(f"Table {table.table_index} on page {table.page_num}")
    print(f"Rows: {len(table.data)}")
    print(table.data)  # List of lists
```

### 3. Extract and Save Images

```python
parser = PDFMetadataParser("document.pdf")

result = parser.parse(extract_images=True)

# Save images to disk
saved_paths = parser.save_images(result, "output_images/")
print(f"Saved {len(saved_paths)} images")
```

### 4. Compare Extraction Methods

```python
parser = PDFMetadataParser("document.pdf")

comparison = parser.compare_extraction_methods()

# Results include performance metrics for each method
print(comparison)
```

### 5. Export Parsed Data (TOON/JSON)

**TOON format (default)** - 10-60% fewer tokens vs JSON, ideal for LLM input:

```python
parser = PDFMetadataParser("document.pdf")
result = parser.parse()

# Export to TOON format (default - token-efficient for LLMs)
toon_output = parser.export(result)  # format="toon" is default
print(toon_output)

# Save to file
with open("parsed_document.toon", "w") as f:
    f.write(toon_output)
```

**JSON format (explicit)** - use when you need standard JSON:

```python
# Export to JSON
json_output = parser.export(result, format="json")

# Or use dedicated method
json_output = parser.export_to_json(result, indent=2)

# Save to file
with open("parsed_document.json", "w") as f:
    f.write(json_output)
```

**Compare token counts** between formats:

```python
comparison = parser.compare_export_formats(result)

print(f"JSON tokens: {comparison['json_tokens']:,}")
print(f"TOON tokens: {comparison['toon_comma_tokens']:,}")
print(f"Savings: {comparison['toon_comma_savings_percent']}%")
```

### 6. Column-Aware Reading Order (for Research Papers, Newspapers)

```python
# Basic usage with default margins
parser = PDFMetadataParser("document.pdf")

# Enable column-aware reading order
result = parser.parse(
    extract_text=True,
    layout_aware=True,
    column_aware=True  # Automatically detects and fixes column order
)

# Check detected layout
print(f"Detected layout: {result.column_layout}")  # 'single', 'double', or 'multi'

# Text blocks are now in correct reading order (left column, then right column)
for block in result.text_blocks:
    print(block.text)
```

**Advanced: Custom Header/Footer Margins**

For PDFs with large headers or footers, adjust the margins to improve column detection:

```python
# Initialize with custom margins (in points, 72 points = 1 inch)
parser = PDFMetadataParser(
    "document.pdf",
    header_margin=100,  # Ignore top 100 points (large header)
    footer_margin=80    # Ignore bottom 80 points (large footer)
)

result = parser.parse(
    extract_text=True,
    layout_aware=True,
    column_aware=True
)
```

**Visualizing Column Detection**

Debug and verify column detection by creating an annotated PDF:

```python
parser = PDFMetadataParser("document.pdf")

# Creates a PDF with red borders around detected columns
output_path = parser.visualize_columns()
print(f"Annotated PDF saved to: {output_path}")
```

### 7. Formula Detection and LaTeX Conversion

```python
parser = PDFMetadataParser("document.pdf")

# Extract formulas with heuristic detection (no GPU/DL required)
result = parser.parse(
    extract_text=True,
    extract_formulas=True
)

# Access detected formulas
for formula in result.formulas:
    print(f"Formula: {formula.formula_text}")
    print(f"LaTeX:   {formula.latex}")
    print(f"Confidence: {formula.confidence:.2f}")

    # Save formula as image
    if formula.image_bytes:
        with open(f"formula_{formula.formula_index}.png", "wb") as f:
            f.write(formula.image_bytes)
```

### 8. Simple Reading Order (No Column Detection)

```python
parser = PDFMetadataParser("document.pdf")
result = parser.parse(
    layout_aware=True,
    column_aware=False  # Disable column detection
)

# Sort blocks by page and vertical position (simple top-to-bottom)
sorted_blocks = sorted(
    result.text_blocks,
    key=lambda b: (b.page_num, b.bbox[1])
)

# Print in reading order
for block in sorted_blocks:
    if block.block_type == "title":
        print(f"\nTITLE: {block.text}\n")
    elif block.block_type == "heading":
        print(f"\nHEADING: {block.text}\n")
    else:
        print(block.text)
```

## API Reference

### PDFMetadataParser

#### `__init__(pdf_path: str, footer_margin: int = 50, header_margin: int = 50, fast_column_detection: bool = True)`
Initialize the parser with a PDF file path.

**Parameters:**
- `pdf_path` (str): Path to the PDF file
- `footer_margin` (int): Height in points of bottom stripe to ignore for column detection (default: 50)
- `header_margin` (int): Height in points of top stripe to ignore for column detection (default: 50)
- `fast_column_detection` (bool): Use fast column detection algorithm—50-100x faster than detailed mode (default: True)

#### `parse(extract_text=True, extract_images=True, extract_tables=True, extract_formulas=False, text_method="pymupdf", table_method="camelot", layout_aware=True, column_aware=True) -> ParsedDocument`

Parse the PDF document.

**Parameters:**
- `extract_text` (bool): Extract text content
- `extract_images` (bool): Extract images
- `extract_tables` (bool): Extract tables
- `extract_formulas` (bool): Extract and detect mathematical formulas (NEW)
- `text_method` (str): Method for text extraction ("pymupdf" or "pdfplumber")
- `table_method` (str): Method for table extraction ("camelot" or "tabula")
- `layout_aware` (bool): Preserve layout information
- `column_aware` (bool): Detect columns and fix reading order (NEW)

**Returns:** `ParsedDocument` object containing all extracted data

#### `compare_extraction_methods() -> Dict`
Compare different extraction methods and return performance metrics.

#### `export_to_dict(parsed_doc: ParsedDocument) -> Dict`
Export parsed document to dictionary format.

#### `save_images(parsed_doc: ParsedDocument, output_dir: str) -> List[str]`
Save extracted images to disk.

#### `visualize_columns(output_path: Optional[str] = None) -> str`
Create a visual representation of detected columns by drawing red borders around detected column bboxes and numbering them. Useful for debugging and understanding column detection.

**Parameters:**
- `output_path` (str, optional): Output path for annotated PDF. If not provided, uses `<original_name>-columns.pdf`

**Returns:** Path to the annotated PDF file

### Data Classes

#### `DocumentMetadata`
Contains PDF metadata:
- `title`, `author`, `subject`, `creator`, `producer`
- `creation_date`, `modification_date`
- `num_pages`, `file_size`, `page_sizes`

#### `TextBlock`
Represents a text block with layout info:
- `text`: The text content
- `bbox`: Bounding box (x0, y0, x1, y1)
- `page_num`: Page number
- `font_size`, `font_name`: Font information
- `block_type`: "text", "title", "heading", "header", "footer"

#### `ImageData`
Represents an extracted image:
- `image_index`, `page_num`: Position info
- `bbox`: Bounding box
- `width`, `height`: Dimensions
- `colorspace`: Color space
- `image_bytes`: Raw image data
- `ext`: File extension

#### `TableData`
Represents an extracted table:
- `table_index`, `page_num`: Position info
- `bbox`: Bounding box (if available)
- `data`: Table data as list of lists
- `extraction_method`: Method used

#### `FormulaData` (NEW)
Represents a detected mathematical formula:
- `formula_index`: Index of the formula
- `page_num`: Page number
- `bbox`: Bounding box
- `formula_text`: Original text representation
- `latex`: LaTeX conversion (heuristic-based)
- `confidence`: Detection confidence score (0.0-1.0)
- `image_bytes`: Formula as image (optional)

#### `ParsedDocument`
Complete parsed document data:
- `metadata`: DocumentMetadata
- `text_blocks`: List of TextBlock
- `images`: List of ImageData
- `tables`: List of TableData
- `formulas`: List of FormulaData (NEW)
- `extraction_method`: Method used
- `parsing_time`: Time taken to parse
- `column_layout`: Detected layout ('single', 'double', 'multi') (NEW)

## Library Comparison

### Text Extraction

| Library | Speed | Layout Awareness | Font Info | Best For |
|---------|-------|------------------|-----------|----------|
| **PyMuPDF** | ⚡⚡⚡ Fast | ✅ Excellent | ✅ Yes | All-purpose, comprehensive extraction |
| **pdfplumber** | ⚡⚡ Moderate | ✅ Excellent | ❌ Limited | Precise text positioning |

**Recommendation**: Use **PyMuPDF** for most cases. Use **pdfplumber** when you need very precise character-level positioning.

### Table Extraction

| Library | Speed | Bordered Tables | Borderless Tables | Best For |
|---------|-------|-----------------|-------------------|----------|
| **Camelot** | ⚡⚡ Moderate | ✅ Excellent | ⚡ Good (stream mode) | Complex, well-structured tables |
| **Tabula** | ⚡⚡⚡ Fast | ✅ Good | ✅ Better | Simple tables, quick extraction |

**Recommendation**: Use **Camelot** with lattice mode for bordered tables. Use **Camelot** stream mode or **Tabula** for borderless tables.

## Performance Tips

1. **Fast column detection** (enabled by default): Optimized algorithm that's 50-100x faster than detailed mode
   ```python
   # Fast mode (default)
   parser = PDFMetadataParser("paper.pdf", fast_column_detection=True)

   # Detailed mode - only needed for PDFs with text on images or colored backgrounds
   parser = PDFMetadataParser("paper.pdf", fast_column_detection=False)
   ```

2. **Text only**: Disable image and table extraction for faster processing
   ```python
   result = parser.parse(extract_text=True, extract_images=False, extract_tables=False)
   ```

3. **Layout-aware off**: Set `layout_aware=False` for simple text extraction (fastest)
   ```python
   result = parser.parse(layout_aware=False, column_aware=False)
   ```

4. **Choose the right method**: PyMuPDF is generally fastest for text
5. **Batch processing**: Process multiple PDFs in parallel using multiprocessing
6. **Use uv for faster installs**: Install dependencies with `uv pip install` for 10-100x faster installation

## Package Manager Quick Reference

Here's a quick comparison of common commands between pip and uv:

| Task | pip | uv |
|------|-----|-----|
| Install from requirements.txt | `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| Install single package | `pip install package-name` | `uv pip install package-name` |
| Install with version | `pip install package==1.0.0` | `uv pip install package==1.0.0` |
| Upgrade package | `pip install --upgrade package` | `uv pip install --upgrade package` |
| Uninstall package | `pip uninstall package` | `uv pip uninstall package` |
| List installed packages | `pip list` | `uv pip list` |
| Freeze requirements | `pip freeze > requirements.txt` | `uv pip freeze > requirements.txt` |
| Create virtual env | `python -m venv venv` | `uv venv` |

**Note**: uv commands work exactly like pip commands but are significantly faster. You can simply replace `pip` with `uv pip` in most cases.

## Limitations

- **Scanned PDFs**: This parser is designed for digitally-born PDFs. For scanned PDFs, you'll need OCR (e.g., Tesseract)
- **Complex layouts**: Very complex multi-column layouts may require manual tuning
- **Encrypted PDFs**: Password-protected PDFs need to be decrypted first
- **Large files**: Very large PDFs (100+ MB) may require significant memory

## Troubleshooting

### Camelot not finding tables

Try both flavors:
```python
# For tables with borders
result = parser.parse(table_method="camelot")  # Uses 'lattice' by default

# For tables without borders, modify pdf_parser.py line 350:
# flavor='stream' instead of flavor='lattice'
```

### Java errors with Tabula

Ensure Java is installed and in your PATH:
```bash
java -version
```

### Ghostscript errors

Ensure Ghostscript is installed:
```bash
gs --version
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License

## Acknowledgments

This parser leverages several excellent open-source libraries:
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [Camelot](https://camelot-py.readthedocs.io/)
- [Tabula](https://tabula-py.readthedocs.io/)

## Support

For issues, questions, or contributions, please open an issue on GitHub.
