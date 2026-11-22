"""
Security Fixes Verification Tests
"""
import os
import tempfile
from pathlib import Path

import pytest

from core.dataset_manager import DatasetManager
from utils.ast_sandbox import validate_code, ASTSandboxError


def test_path_traversal_prevention():
    """Verify that save_scratch_file('../../etc/passwd', ...) raises ValueError."""
    dm = DatasetManager()
    with pytest.raises(ValueError):
        dm.save_scratch_file("../../etc/passwd", "test")


def test_ast_unapproved_import_rejection():
    """Verify that dynamic imports of unapproved modules trigger AST security violations."""
    with pytest.raises(ASTSandboxError):
        validate_code("import os")

    with pytest.raises(ASTSandboxError):
        validate_code("import subprocess")

    with pytest.raises(ASTSandboxError):
        validate_code("import sys")
