"""
Example usage of the PDF Metadata Document Parser.

This script demonstrates how to use the parser to extract text, images,
and tables from PDF documents with layout awareness.
"""

import json
from pathlib import Path
from pdf_parser import PDFMetadataParser


def print_section(title: str):
    """Helper function to print section headers"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def example_basic_parsing(pdf_path: str):
    """Example: Basic PDF parsing with all features"""
    print_section("Example 1: Basic PDF Parsing")

    # Initialize parser
    parser = PDFMetadataParser(pdf_path)

    # Parse the document with default settings (PyMuPDF for text, Camelot for tables)
    result = parser.parse(
        extract_text=True,
        extract_images=True,
        extract_tables=True,
        layout_aware=True
    )

    # Display metadata
    print("\n📄 Document Metadata:")
    print(f"  Title: {result.metadata.title or 'N/A'}")
    print(f"  Author: {result.metadata.author or 'N/A'}")
    print(f"  Pages: {result.metadata.num_pages}")
    print(f"  File Size: {result.metadata.file_size:,} bytes")
    print(f"  Parsing Time: {result.parsing_time:.2f} seconds")

    # Display text blocks
    print(f"\n📝 Extracted {len(result.text_blocks)} text blocks")
    if result.text_blocks:
        print("\n  First 3 text blocks:")
        for i, block in enumerate(result.text_blocks[:3]):
            print(f"\n  Block {i + 1} (Page {block.page_num}, Type: {block.block_type}):")
            print(f"    Position: {block.bbox}")
            print(f"    Font: {block.font_name} (Size: {block.font_size})")
            print(f"    Text: {block.text[:100]}..." if len(block.text) > 100 else f"    Text: {block.text}")

    # Display images
    print(f"\n🖼️  Extracted {len(result.images)} images")
    if result.images:
        print("\n  Image details:")
        for img in result.images:
            print(f"    Page {img.page_num}, Image {img.image_index}: "
                  f"{img.width}x{img.height} ({img.ext})")

    # Display tables
    print(f"\n📊 Extracted {len(result.tables)} tables")
    if result.tables:
        print("\n  Table details:")
        for table in result.tables:
            print(f"    Page {table.page_num}, Table {table.table_index}: "
                  f"{len(table.data)} rows (Method: {table.extraction_method})")
            if table.data:
                print(f"      Preview: {table.data[0][:3]}...")

    return result


def example_text_only_pdfplumber(pdf_path: str):
    """Example: Text extraction only using pdfplumber"""
    print_section("Example 2: Text Extraction with pdfplumber")

    parser = PDFMetadataParser(pdf_path)

    # Extract only text using pdfplumber
    result = parser.parse(
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        text_method="pdfplumber",
        layout_aware=True
    )

    print(f"\n📝 Extracted {len(result.text_blocks)} text blocks using pdfplumber")
    print(f"⏱️  Parsing Time: {result.parsing_time:.2f} seconds")

    # Show text blocks by type
    by_type = {}
    for block in result.text_blocks:
        by_type.setdefault(block.block_type, []).append(block)

    print("\n  Text blocks by type:")
    for block_type, blocks in by_type.items():
        print(f"    {block_type}: {len(blocks)} blocks")

    return result


def example_tables_only(pdf_path: str):
    """Example: Table extraction using different methods"""
    print_section("Example 3: Table Extraction Comparison")

    parser = PDFMetadataParser(pdf_path)

    # Extract tables using Camelot
    print("\n🔧 Using Camelot...")
    result_camelot = parser.parse(
        extract_text=False,
        extract_images=False,
        extract_tables=True,
        table_method="camelot"
    )
    print(f"  Found {len(result_camelot.tables)} tables in {result_camelot.parsing_time:.2f}s")

    # Extract tables using Tabula
    print("\n🔧 Using Tabula...")
    result_tabula = parser.parse(
        extract_text=False,
        extract_images=False,
        extract_tables=True,
        table_method="tabula"
    )
    print(f"  Found {len(result_tabula.tables)} tables in {result_tabula.parsing_time:.2f}s")

    return result_camelot, result_tabula


def example_save_images(pdf_path: str, output_dir: str = "extracted_images"):
    """Example: Extract and save images to disk"""
    print_section("Example 4: Extract and Save Images")

    parser = PDFMetadataParser(pdf_path)

    # Parse with images only
    result = parser.parse(
        extract_text=False,
        extract_images=True,
        extract_tables=False
    )

    # Save images to disk
    saved_paths = parser.save_images(result, output_dir)

    print(f"\n💾 Saved {len(saved_paths)} images to '{output_dir}/'")
    for path in saved_paths:
        print(f"    {path}")

    return saved_paths


def example_compare_methods(pdf_path: str):
    """Example: Compare different extraction methods"""
    print_section("Example 5: Extraction Method Comparison")

    parser = PDFMetadataParser(pdf_path)

    print("\n⚡ Running comparison...")
    comparison = parser.compare_extraction_methods()

    print("\n📊 Comparison Results:\n")

    # Text extraction comparison
    if comparison["text_extraction"]:
        print("  Text Extraction:")
        for method, stats in comparison["text_extraction"].items():
            print(f"    {method.upper()}:")
            print(f"      Blocks: {stats['num_blocks']}")
            print(f"      Characters: {stats['total_chars']:,}")
            print(f"      Time: {stats['time']:.3f}s")

    # Table extraction comparison
    if comparison["table_extraction"]:
        print("\n  Table Extraction:")
        for method, stats in comparison["table_extraction"].items():
            print(f"    {method.upper()}:")
            print(f"      Tables: {stats['num_tables']}")
            print(f"      Time: {stats['time']:.3f}s")

    return comparison


def example_export_to_json(pdf_path: str, output_json: str = "parsed_document.json"):
    """Example: Export parsed document to JSON"""
    print_section("Example 6: Export to JSON")

    parser = PDFMetadataParser(pdf_path)

    # Parse the document
    result = parser.parse()

    # Export to dictionary
    data_dict = parser.export_to_dict(result)

    # Save to JSON (without image bytes to keep file size manageable)
    for img in data_dict["images"]:
        img.pop("image_bytes", None)  # Remove binary data

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Exported parsed document to '{output_json}'")
    print(f"   File size: {Path(output_json).stat().st_size:,} bytes")

    return output_json


def example_layout_aware_reading_order(pdf_path: str):
    """Example: Extract text in reading order with layout awareness"""
    print_section("Example 7: Layout-Aware Reading Order (OLD - Simple)")

    parser = PDFMetadataParser(pdf_path)

    result = parser.parse(
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        text_method="pymupdf",
        layout_aware=True,
        column_aware=False  # Disable column awareness for comparison
    )

    # Sort text blocks by page, then by vertical position (top to bottom)
    sorted_blocks = sorted(result.text_blocks, key=lambda b: (b.page_num, b.bbox[1]))

    print(f"\n📖 Document text in simple reading order (top-to-bottom):\n")

    current_page = -1
    for block in sorted_blocks[:10]:  # Show first 10 blocks
        if block.page_num != current_page:
            current_page = block.page_num
            print(f"\n{'─' * 80}")
            print(f"PAGE {current_page + 1}")
            print(f"{'─' * 80}\n")

        # Print with block type indicator
        type_icon = {
            "title": "📌",
            "heading": "▶",
            "header": "🔝",
            "footer": "🔻",
            "text": "  "
        }.get(block.block_type, "  ")

        print(f"{type_icon} {block.text}\n")

    return sorted_blocks


def example_column_aware_reading_order(pdf_path: str):
    """Example: Extract text with column-aware reading order (NEW)"""
    print_section("Example 8: Column-Aware Reading Order (NEW)")

    parser = PDFMetadataParser(pdf_path)

    result = parser.parse(
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        text_method="pymupdf",
        layout_aware=True,
        column_aware=True  # Enable column awareness
    )

    print(f"\n📰 Detected layout: {result.column_layout or 'unknown'}")
    print(f"📖 Document text in column-aware reading order:\n")

    current_page = -1
    for block in result.text_blocks[:10]:  # Show first 10 blocks
        if block.page_num != current_page:
            current_page = block.page_num
            print(f"\n{'─' * 80}")
            print(f"PAGE {current_page + 1} ({result.column_layout} layout)")
            print(f"{'─' * 80}\n")

        # Print with block type and position indicator
        type_icon = {
            "title": "📌",
            "heading": "▶",
            "header": "🔝",
            "footer": "🔻",
            "text": "  "
        }.get(block.block_type, "  ")

        # Show x-position to indicate column
        x_pos = int(block.bbox[0])
        print(f"{type_icon} [x={x_pos:3d}] {block.text[:80]}...\n" if len(block.text) > 80 else f"{type_icon} [x={x_pos:3d}] {block.text}\n")

    return result


def example_formula_extraction(pdf_path: str):
    """Example: Extract mathematical formulas from PDF"""
    print_section("Example 9: Formula Detection and Extraction")

    parser = PDFMetadataParser(pdf_path)

    result = parser.parse(
        extract_text=True,
        extract_formulas=True,  # Enable formula extraction
        extract_images=False,
        extract_tables=False
    )

    print(f"\n🔬 Detected {len(result.formulas)} mathematical formulas")

    if result.formulas:
        print("\n  Formula details:\n")
        for formula in result.formulas[:5]:  # Show first 5 formulas
            print(f"  Formula {formula.formula_index} (Page {formula.page_num}):")
            print(f"    Original text: {formula.formula_text}")
            print(f"    LaTeX:         {formula.latex}")
            print(f"    Confidence:    {formula.confidence:.2f}")
            print(f"    Position:      {formula.bbox}")
            print()

        # Save formula images if available
        output_dir = "extracted_formulas"
        Path(output_dir).mkdir(exist_ok=True)

        saved = 0
        for formula in result.formulas:
            if formula.image_bytes:
                filepath = Path(output_dir) / f"formula_{formula.formula_index}_page_{formula.page_num}.png"
                with open(filepath, "wb") as f:
                    f.write(formula.image_bytes)
                saved += 1

        if saved > 0:
            print(f"\n  💾 Saved {saved} formula images to '{output_dir}/'")
    else:
        print("\n  No formulas detected in this document.")
        print("  (Formulas require special mathematical characters or symbols)")

    return result


def main():
    """Main function to run examples"""
    import sys

    print("=" * 80)
    print(" PDF Metadata Document Parser - Examples")
    print("=" * 80)

    # Check if PDF path is provided
    if len(sys.argv) < 2:
        print("\n⚠️  Usage: python example_usage.py <path_to_pdf>")
        print("\nExample:")
        print("  python example_usage.py sample.pdf")
        print("\nThis will run all examples on the provided PDF file.")
        return

    pdf_path = sys.argv[1]

    # Verify file exists
    if not Path(pdf_path).exists():
        print(f"\n❌ Error: PDF file not found: {pdf_path}")
        return

    print(f"\n📄 Processing: {pdf_path}\n")

    try:
        # Run examples
        example_basic_parsing(pdf_path)
        example_text_only_pdfplumber(pdf_path)
        example_tables_only(pdf_path)
        example_save_images(pdf_path)
        example_compare_methods(pdf_path)
        example_export_to_json(pdf_path)
        example_layout_aware_reading_order(pdf_path)
        example_column_aware_reading_order(pdf_path)  # NEW
        example_formula_extraction(pdf_path)  # NEW

        print_section("✅ All Examples Completed Successfully")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
