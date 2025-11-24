import fitz
import re
from typing import List, Optional
from pathlib import Path
from ..data_types import TextBlock, FormulaData
from ..utils import text_to_latex_heuristic
from .ocr import ExternalOCR

class FormulaExtractor:
    """
    Handles formula extraction from text blocks.
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def extract_formulas(self, text_blocks: List[TextBlock], strict_mode: bool = False, 
                        ocr_strategy: Optional[ExternalOCR] = None) -> List[FormulaData]:
        """
        Extract mathematical formulas from text blocks using heuristic detection.

        Args:
            text_blocks: List of extracted text blocks
            strict_mode: If True, requires stronger evidence to classify as formula
            ocr_strategy: Optional ExternalOCR strategy for high-quality LaTeX conversion

        Returns:
            List of detected formulas
        """
        formulas = []
        formula_index = 0

        # Mathematical symbols commonly found in formulas
        # Expanded set
        math_symbols = set('∫∑∏√±×÷≈≠≤≥∞∂∇αβγδεζηθλμπρσφψωΔΣΠΩ')
        math_chars = set('+-*/=()[]{}^_∈∉⊂⊃∪∩|<>;:!%')

        # Strong indicators (almost certainly math)
        strong_indicators = ['=', '∫', '∑', '∏', '√', '∂', '∇', '≈', '≠', '≤', '≥', '∞', '∈', '∉']
        
        # Regex patterns for math structures
        # Look for patterns like f(x) =, sum_i^n, etc.
        math_patterns = [
            r'[a-zA-Z]\s*\([a-zA-Z0-9,\s]+\)\s*=',  # f(x) = 
            r'\\[a-zA-Z]+',  # LaTeX commands like \alpha
            r'_\s*{[^}]+}',  # Subscript with braces _{...}
            r'\^\s*{[^}]+}', # Superscript with braces ^{...}
            r'[a-zA-Z]_[a-zA-Z0-9]', # Simple subscript x_i
            r'[0-9]+\s*[\+\-\*\/]\s*[0-9]+', # Arithmetic 1 + 2
        ]

        # Negative lookaheads/patterns to avoid
        # e.g. "Fig. 1", "Table 2", dates like "2023-01-01"
        non_math_patterns = [
            r'Fig\.\s*\d+',
            r'Table\s*\d+',
            r'\d{4}-\d{2}-\d{2}',
            r'Page\s*\d+',
            r'https?://',
            r'www\.',
            r'^[A-Z][a-z]+ \d+$', # "Chapter 1"
        ]

        doc = fitz.open(self.pdf_path) if (ocr_strategy or strict_mode) else None

        for block in text_blocks:
            text = block.text.strip()

            if not text or len(text) < 2:
                continue
                
            # Quick filter for non-math
            if any(re.search(p, text) for p in non_math_patterns):
                continue

            # Calculate formula likelihood score
            score = 0.0

            # Check for mathematical symbols
            math_symbol_count = sum(1 for c in text if c in math_symbols)
            math_char_count = sum(1 for c in text if c in math_chars)
            
            # Normalize counts by length to get density
            symbol_density = (math_symbol_count + math_char_count) / len(text)

            if math_symbol_count > 0:
                score += math_symbol_count * 0.4  # Increased weight

            if math_char_count > 0:
                score += math_char_count * 0.1

            # Check for strong indicators
            if any(ind in text for ind in strong_indicators):
                score += 0.6

            # Check for regex patterns
            if any(re.search(p, text) for p in math_patterns):
                score += 0.4

            # Check for superscripts/subscripts patterns (^, _)
            if '^' in text or '_' in text:
                score += 0.3

            # Check for fraction-like patterns (a/b where a, b are short)
            if re.search(r'\w+/\w+', text):
                score += 0.2

            # Check font size (formulas are often in different size)
            if block.font_size and block.font_size < 10:
                score += 0.2

            # Check for isolated blocks (formulas are often standalone)
            if len(text) < 50 and symbol_density > 0.2:
                score += 0.4
            
            # Strict mode adjustments
            threshold = 0.8 if strict_mode else 0.7
            
            if strict_mode:
                # In strict mode, we need at least one strong indicator or very high density
                has_strong = any(ind in text for ind in strong_indicators)
                if not has_strong and symbol_density < 0.3:
                    score = 0.0 # Penalize heavily

            # If score exceeds threshold, consider it a formula
            if score >= threshold:
                # Try to extract formula image if document is available
                formula_image = None
                if doc:
                    try:
                        page = doc[block.page_num]
                        # Extract region as image
                        bbox = fitz.Rect(block.bbox)
                        # Add small padding
                        bbox.x0 -= 2
                        bbox.y0 -= 2
                        bbox.x1 += 2
                        bbox.y1 += 2
                        
                        pix = page.get_pixmap(clip=bbox, matrix=fitz.Matrix(3, 3))  # 3x resolution for better OCR
                        formula_image = pix.tobytes("png")
                    except:
                        pass

                # Determine LaTeX
                latex = None
                if ocr_strategy and formula_image:
                    latex = ocr_strategy.image_to_latex(formula_image)
                
                if not latex:
                    # Fallback to heuristic
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
