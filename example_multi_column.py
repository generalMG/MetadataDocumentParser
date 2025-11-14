"""
Simple example demonstrating improved multi-column text extraction.

This example shows how to use the improved column detection for research papers
and other multi-column documents.
"""

import sys
from pathlib import Path
from pdf_parser import PDFMetadataParser


def extract_multi_column_text(pdf_path: str):
    """Extract text from a multi-column PDF with proper reading order"""

    print(f"Processing: {pdf_path}\n")

    # Initialize parser with default margins (50pt for header and footer)
    # Adjust these if your PDF has larger headers/footers
    parser = PDFMetadataParser(
        pdf_path,
        header_margin=50,  # Ignore top 50 points
        footer_margin=50   # Ignore bottom 50 points
    )

    # Parse with column awareness enabled
    result = parser.parse(
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        layout_aware=True,
        column_aware=True  # This enables the advanced column detection
    )

    # Display results
    print(f"✅ Successfully parsed!")
    print(f"   Layout type: {result.column_layout}")
    print(f"   Total pages: {result.metadata.num_pages}")
    print(f"   Text blocks: {len(result.text_blocks)}")
    print(f"   Processing time: {result.parsing_time:.2f}s\n")

    # Display text in proper reading order
    print("📖 Extracted text in reading order:\n")
    print("=" * 80)

    current_page = -1
    for block in result.text_blocks:
        # Show page headers
        if block.page_num != current_page:
            current_page = block.page_num
            print(f"\n\n{'=' * 80}")
            print(f"PAGE {current_page + 1} ({result.column_layout} layout)")
            print(f"{'=' * 80}\n")

        # Show text with type indicators
        type_indicator = {
            "title": "[TITLE] ",
            "heading": "[HEADING] ",
            "header": "[HEADER] ",
            "footer": "[FOOTER] ",
            "text": ""
        }.get(block.block_type, "")

        print(f"{type_indicator}{block.text}\n")

    return result


def compare_with_without_columns(pdf_path: str):
    """Compare extraction with and without column awareness"""

    print(f"\n{'=' * 80}")
    print(" Comparing extraction methods")
    print(f"{'=' * 80}\n")

    parser = PDFMetadataParser(pdf_path)

    # Without column awareness
    print("1️⃣  Extracting WITHOUT column awareness (simple top-to-bottom)...")
    result_simple = parser.parse(
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        column_aware=False
    )

    # With column awareness
    print("2️⃣  Extracting WITH column awareness (intelligent column detection)...")
    result_advanced = parser.parse(
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        column_aware=True
    )

    # Compare
    print(f"\n📊 Comparison:")
    print(f"   Simple method:   {len(result_simple.text_blocks)} blocks")
    print(f"   Advanced method: {len(result_advanced.text_blocks)} blocks")
    print(f"   Detected layout: {result_advanced.column_layout}")

    # Show first few blocks from each
    print(f"\n📄 First 3 blocks (Simple method):")
    for i, block in enumerate(result_simple.text_blocks[:3]):
        preview = block.text[:80] + "..." if len(block.text) > 80 else block.text
        print(f"   {i+1}. {preview}")

    print(f"\n📄 First 3 blocks (Advanced method):")
    for i, block in enumerate(result_advanced.text_blocks[:3]):
        preview = block.text[:80] + "..." if len(block.text) > 80 else block.text
        print(f"   {i+1}. {preview}")


def visualize_columns(pdf_path: str):
    """Create a visual representation of detected columns"""

    print(f"\n{'=' * 80}")
    print(" Visualizing Column Detection")
    print(f"{'=' * 80}\n")

    parser = PDFMetadataParser(pdf_path)

    print("Creating annotated PDF with column boundaries...")
    output_path = parser.visualize_columns()

    print(f"\n✅ Created visualization: {output_path}")
    print(f"   Open this file to see the detected column boundaries.")
    print(f"   Red boxes show the detected reading order (numbered 0, 1, 2...)")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python example_multi_column.py <path_to_pdf>")
        print("\nThis script demonstrates improved multi-column text extraction.")
        print("\nExamples:")
        print("  python example_multi_column.py research_paper.pdf")
        print("  python example_multi_column.py newspaper.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    try:
        # Example 1: Basic extraction with column awareness
        result = extract_multi_column_text(pdf_path)

        # Example 2: Compare methods
        compare_with_without_columns(pdf_path)

        # Example 3: Visualize columns
        visualize_columns(pdf_path)

        print(f"\n{'=' * 80}")
        print(" ✅ All examples completed successfully!")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
