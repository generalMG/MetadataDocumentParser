import re

def text_to_latex_heuristic(text: str) -> str:
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
    latex = re.sub(r'\^(\w)', r'^{\1}', latex)

    # Handle subscripts
    # a_b -> a_{b}
    latex = re.sub(r'_(\w)', r'_{\1}', latex)

    # Wrap in math mode if not already
    if not latex.startswith('$'):
        latex = f'${latex}$'

    return latex
