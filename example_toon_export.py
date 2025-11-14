"""
Example demonstrating TOON export for token-efficient PDF data serialization.

TOON achieves 30-60% token reduction vs JSON, making it ideal for LLM input.
"""

import sys
from pdf_parser import PDFMetadataParser


def main():
    if len(sys.argv) < 2:
        print("Usage: python example_toon_export.py <pdf_file>")
        print("\nDemonstrates TOON export with token count comparison.")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print("=" * 80)
    print("TOON Export Demo - Token-Efficient PDF Data Serialization")
    print("=" * 80)
    print(f"\nProcessing: {pdf_path}\n")

    # Parse PDF
    parser = PDFMetadataParser(pdf_path)
    result = parser.parse(
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_formulas=False
    )

    print(f"✅ Parsed {result.metadata.num_pages} pages in {result.parsing_time:.2f}s")
    print(f"   Text blocks: {len(result.text_blocks)}")
    print(f"   Column layout: {result.column_layout}\n")

    # Compare formats
    print("─" * 80)
    print("TOKEN COMPARISON")
    print("─" * 80)

    comparison = parser.compare_export_formats(result)

    if "error" in comparison:
        print(f"\n❌ {comparison['error']}")
        print(f"   {comparison['message']}\n")
        return

    # Display comparison
    print(f"\n📊 Format Comparison:\n")
    print(f"   JSON (formatted):")
    print(f"     Tokens: {comparison['json_tokens']:,}")
    print(f"     Size:   {comparison['json_size_bytes']:,} bytes\n")

    print(f"   TOON (comma delimiter):")
    print(f"     Tokens: {comparison['toon_comma_tokens']:,}")
    print(f"     Size:   {comparison['toon_comma_size_bytes']:,} bytes")
    print(f"     Savings: {comparison['toon_comma_savings_percent']}% fewer tokens\n")

    print(f"   TOON (tab delimiter):")
    print(f"     Tokens: {comparison['toon_tab_tokens']:,}")
    print(f"     Size:   {comparison['toon_tab_size_bytes']:,} bytes")
    print(f"     Savings: {comparison['toon_tab_savings_percent']}% fewer tokens\n")

    print(f"✨ Best format: {comparison['best_format'].replace('_', ' ').title()}")
    print(f"   {comparison['best_savings_percent']}% token reduction vs JSON\n")

    # Export examples
    print("─" * 80)
    print("EXPORT EXAMPLES")
    print("─" * 80)

    # TOON export (default)
    print("\n1️⃣  TOON Format (Default - Token Efficient):\n")
    toon_output = parser.export(result, format="toon")
    preview = toon_output[:500] + "..." if len(toon_output) > 500 else toon_output
    print(preview)

    # JSON export (explicit)
    print("\n\n2️⃣  JSON Format (Explicit):\n")
    json_output = parser.export(result, format="json")
    preview = json_output[:500] + "..." if len(json_output) > 500 else json_output
    print(preview)

    # Save to files
    print("\n\n─" * 80)
    print("SAVING TO FILES")
    print("─" * 80)

    # Save TOON
    toon_file = pdf_path.replace(".pdf", "_export.toon")
    with open(toon_file, "w") as f:
        f.write(parser.export(result, format="toon", delimiter="\t"))
    print(f"\n✅ Saved TOON format: {toon_file}")
    print(f"   ({comparison['toon_tab_tokens']:,} tokens)")

    # Save JSON
    json_file = pdf_path.replace(".pdf", "_export.json")
    with open(json_file, "w") as f:
        f.write(parser.export(result, format="json"))
    print(f"✅ Saved JSON format: {json_file}")
    print(f"   ({comparison['json_tokens']:,} tokens)")

    print(f"\n💡 For LLM input, use TOON format to save {comparison['best_savings_percent']}% tokens!")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTOON reduces tokens by {comparison['best_savings_percent']}% compared to JSON")
    print(f"This saves API costs and improves context window efficiency for LLMs.\n")
    print("Usage in your code:")
    print("  # Default: TOON format (token-efficient)")
    print("  output = parser.export(result)")
    print("\n  # Explicit JSON (if needed)")
    print("  output = parser.export(result, format='json')")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
