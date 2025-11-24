"""
PDF Metadata Document Parser

A comprehensive PDF parser that extracts text, images, and tables with layout awareness.
Supports multiple extraction libraries for quality comparison.

This file is now a wrapper around the modular `metadata_document_parser` package.
"""

from metadata_document_parser import PDFMetadataParser

# Re-export for backward compatibility
__all__ = ["PDFMetadataParser"]
