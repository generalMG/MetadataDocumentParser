from typing import List
from pathlib import Path
from ..data_types import TableData

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

class TableExtractor:
    """
    Handles table extraction from PDF documents.
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def extract_camelot(self) -> List[TableData]:
        """Extract tables using Camelot"""
        if not CAMELOT_AVAILABLE:
            print("Warning: Camelot not available")
            return []

        tables = []

        try:
            # Camelot can extract from all pages
            extracted_tables = camelot.read_pdf(
                str(self.pdf_path),
                pages='all',
                flavor='lattice',  # Use 'stream' for tables without borders
                suppress_stdout=True
            )

            for idx, table in enumerate(extracted_tables):
                tables.append(TableData(
                    table_index=idx,
                    page_num=table.page - 1,  # Camelot uses 1-based indexing
                    bbox=tuple(table._bbox) if hasattr(table, '_bbox') else None,
                    data=table.df.values.tolist(),
                    extraction_method="camelot"
                ))
        except Exception as e:
            print(f"Camelot extraction error: {e}")

        return tables

    def extract_tabula(self) -> List[TableData]:
        """Extract tables using Tabula"""
        if not TABULA_AVAILABLE:
            print("Warning: Tabula not available")
            return []

        tables = []

        try:
            # Tabula extracts all tables from all pages
            extracted_tables = tabula.read_pdf(
                str(self.pdf_path),
                pages='all',
                multiple_tables=True,
                silent=True
            )

            for idx, df in enumerate(extracted_tables):
                tables.append(TableData(
                    table_index=idx,
                    page_num=0,  # Tabula doesn't easily provide page numbers
                    data=df.values.tolist(),
                    extraction_method="tabula"
                ))
        except Exception as e:
            print(f"Tabula extraction error: {e}")

        return tables
