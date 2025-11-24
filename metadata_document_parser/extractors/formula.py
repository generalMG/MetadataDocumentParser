import fitz
from typing import List
from pathlib import Path
from ..data_types import TextBlock, FormulaData
from ..utils import text_to_latex_heuristic

class FormulaExtractor:
    """
    Handles formula extraction from text blocks.
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def extract_formulas(self, text_blocks: List[TextBlock]) -> List[FormulaData]:
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

        doc = fitz.open(self.pdf_path)

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
                latex = text_to_latex_heuristic(text)

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
