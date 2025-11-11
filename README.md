# PDF Metadata Document Parser

A comprehensive, layout-aware PDF parser for extracting text, images, and tables from digitally-born PDF documents. Supports multiple extraction libraries for quality comparison and optimal results.

## Features

- **Metadata Extraction**: Extracts complete PDF metadata (title, author, dates, page info, etc.)
- **Layout-Aware Text Extraction**: Preserves document structure, font information, and reading order
- **Image Extraction**: Extracts embedded images with position and metadata
- **Table Extraction**: Advanced table detection and extraction
- **Multiple Extraction Methods**: Compare different libraries for optimal results
- **Export Capabilities**: Save results as JSON or images to disk

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

### Install Dependencies

```bash
pip install -r requirements.txt
```

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
    layout_aware=True
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
python example_usage.py your_document.pdf
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

### 5. Export to JSON

```python
parser = PDFMetadataParser("document.pdf")
result = parser.parse()

# Convert to dictionary
data = parser.export_to_dict(result)

# Save to JSON
import json
with open("parsed_document.json", "w") as f:
    json.dump(data, f, indent=2)
```

### 6. Layout-Aware Reading Order

```python
parser = PDFMetadataParser("document.pdf")
result = parser.parse(layout_aware=True)

# Sort blocks by page and vertical position
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

#### `__init__(pdf_path: str)`
Initialize the parser with a PDF file path.

#### `parse(extract_text=True, extract_images=True, extract_tables=True, text_method="pymupdf", table_method="camelot", layout_aware=True) -> ParsedDocument`

Parse the PDF document.

**Parameters:**
- `extract_text` (bool): Extract text content
- `extract_images` (bool): Extract images
- `extract_tables` (bool): Extract tables
- `text_method` (str): Method for text extraction ("pymupdf" or "pdfplumber")
- `table_method` (str): Method for table extraction ("camelot" or "tabula")
- `layout_aware` (bool): Preserve layout information

**Returns:** `ParsedDocument` object containing all extracted data

#### `compare_extraction_methods() -> Dict`
Compare different extraction methods and return performance metrics.

#### `export_to_dict(parsed_doc: ParsedDocument) -> Dict`
Export parsed document to dictionary format.

#### `save_images(parsed_doc: ParsedDocument, output_dir: str) -> List[str]`
Save extracted images to disk.

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

#### `ParsedDocument`
Complete parsed document data:
- `metadata`: DocumentMetadata
- `text_blocks`: List of TextBlock
- `images`: List of ImageData
- `tables`: List of TableData
- `extraction_method`: Method used
- `parsing_time`: Time taken to parse

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

1. **Text only**: Disable image and table extraction for faster processing
2. **Layout-aware off**: Set `layout_aware=False` for simple text extraction
3. **Choose the right method**: PyMuPDF is generally fastest for text
4. **Batch processing**: Process multiple PDFs in parallel using multiprocessing

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
