"""
Debug script for testing and visualizing multi-column detection.

This script helps you:
1. Test the advanced column detection on your PDF
2. Visualize detected columns with red borders and numbers
3. Compare text extraction order before and after column-aware processing

Usage:
    python test_column_detection.py <path_to_pdf> [--header-margin 50] [--footer-margin 50]

Example:
    python test_column_detection.py research_paper.pdf
    python test_column_detection.py newspaper.pdf --header-margin 100 --footer-margin 80
"""

import sys
import argparse
from pathlib import Path
from pdf_parser import PDFMetadataParser


def print_separator(title: str = ""):
    """Print a separator line with optional title"""
    if title:
        print(f"\n{'=' * 80}")
        print(f" {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"{'─' * 80}")


def test_column_detection(pdf_path: str, header_margin: int = 50, footer_margin: int = 50):
    """Test and visualize column detection on a PDF"""

    print_separator("Multi-Column Detection Test")
    print(f"📄 PDF: {pdf_path}")
    print(f"📏 Header margin: {header_margin}pt")
    print(f"📏 Footer margin: {footer_margin}pt")

    # Initialize parser with custom margins
    parser = PDFMetadataParser(pdf_path, header_margin=header_margin, footer_margin=footer_margin)

    # Step 1: Visualize columns
    print_separator("Step 1: Visualizing Detected Columns")
    print("Creating annotated PDF with column boundaries...")

    output_path = parser.visualize_columns()
    print(f"✅ Created: {output_path}")
    print(f"   Open this file to see the detected column boundaries marked in red.\n")

    # Step 2: Extract text WITHOUT column awareness (old way)
    print_separator("Step 2: Extract Text WITHOUT Column Awareness (Simple Top-to-Bottom)")

    result_simple = parser.parse(
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        layout_aware=True,
        column_aware=False  # Disable column awareness
    )

    print(f"Layout detected: {result_simple.column_layout or 'single'}")
    print(f"Total text blocks: {len(result_simple.text_blocks)}")
    print(f"\nFirst 5 text blocks (simple top-to-bottom order):\n")

    for i, block in enumerate(result_simple.text_blocks[:5]):
        x_pos = int(block.bbox[0])
        y_pos = int(block.bbox[1])
        text_preview = block.text[:60] + "..." if len(block.text) > 60 else block.text
        print(f"  {i+1}. [Page {block.page_num}, x={x_pos:3d}, y={y_pos:3d}] {text_preview}")

    # Step 3: Extract text WITH column awareness (new way)
    print_separator("Step 3: Extract Text WITH Column Awareness (Column-by-Column)")

    result_column = parser.parse(
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        layout_aware=True,
        column_aware=True  # Enable column awareness
    )

    print(f"Layout detected: {result_column.column_layout or 'single'}")
    print(f"Total text blocks: {len(result_column.text_blocks)}")
    print(f"\nFirst 5 text blocks (column-aware order):\n")

    for i, block in enumerate(result_column.text_blocks[:5]):
        x_pos = int(block.bbox[0])
        y_pos = int(block.bbox[1])
        text_preview = block.text[:60] + "..." if len(block.text) > 60 else block.text
        print(f"  {i+1}. [Page {block.page_num}, x={x_pos:3d}, y={y_pos:3d}] {text_preview}")

    # Step 4: Compare reading orders
    print_separator("Step 4: Reading Order Comparison")

    if len(result_simple.text_blocks) > 0 and len(result_column.text_blocks) > 0:
        # Check if order changed
        simple_order = [b.text for b in result_simple.text_blocks]
        column_order = [b.text for b in result_column.text_blocks]

        if simple_order == column_order:
            print("⚠️  No difference in reading order detected.")
            print("   This could mean:")
            print("   - The document is single-column")
            print("   - The margins need adjustment")
            print("   - The column detection needs tuning for this specific PDF")
        else:
            differences = sum(1 for a, b in zip(simple_order, column_order) if a != b)
            print(f"✅ Reading order improved!")
            print(f"   {differences} text blocks reordered out of {len(simple_order)}")
            print(f"   Layout type: {result_column.column_layout}")

    # Step 5: Full text comparison
    print_separator("Step 5: Full Text Preview")

    print("📖 Text with column-aware reading order (first 20 blocks):\n")

    current_page = -1
    for i, block in enumerate(result_column.text_blocks[:20]):
        if block.page_num != current_page:
            current_page = block.page_num
            print(f"\n{'─' * 80}")
            print(f"PAGE {current_page + 1}")
            print(f"{'─' * 80}\n")

        type_icon = {
            "title": "📌",
            "heading": "▶",
            "header": "🔝",
            "footer": "🔻",
            "text": "  "
        }.get(block.block_type, "  ")

        x_pos = int(block.bbox[0])
        text_preview = block.text[:100] + "..." if len(block.text) > 100 else block.text
        print(f"{type_icon} [x={x_pos:3d}] {text_preview}\n")

    # Summary
    print_separator("Summary")
    print(f"📊 Results:")
    print(f"   - Detected layout: {result_column.column_layout or 'single'}")
    print(f"   - Total pages: {result_column.metadata.num_pages}")
    print(f"   - Text blocks extracted: {len(result_column.text_blocks)}")
    print(f"   - Processing time: {result_column.parsing_time:.2f}s")
    print(f"   - Annotated PDF: {output_path}")

    print_separator()
    print("💡 Tips:")
    print("   - Open the *-columns.pdf file to see detected column boundaries")
    print("   - Adjust --header-margin and --footer-margin if detection is off")
    print("   - Red-bordered boxes show the detected reading order (numbered 0, 1, 2...)")
    print(f"   - Current margins: header={header_margin}pt, footer={footer_margin}pt")
    print_separator()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test and visualize multi-column detection in PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_column_detection.py research_paper.pdf
  python test_column_detection.py newspaper.pdf --header-margin 100
  python test_column_detection.py document.pdf --header-margin 80 --footer-margin 80
        """
    )

    parser.add_argument("pdf_path", help="Path to the PDF file to test")
    parser.add_argument(
        "--header-margin",
        type=int,
        default=50,
        help="Height (in points) of top stripe to ignore (default: 50)"
    )
    parser.add_argument(
        "--footer-margin",
        type=int,
        default=50,
        help="Height (in points) of bottom stripe to ignore (default: 50)"
    )

    args = parser.parse_args()

    # Verify file exists
    if not Path(args.pdf_path).exists():
        print(f"❌ Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    try:
        test_column_detection(args.pdf_path, args.header_margin, args.footer_margin)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
