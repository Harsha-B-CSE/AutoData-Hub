"""
AutoData-Hub 4-Tier Dataset Manager.

Implements the multi-tier storage architecture defined in the system architecture:
- Tier 1 (In-Memory Versioning): Dict mapping version_id -> pd.DataFrame for 1-click rollback.
- Tier 2 (Transient Local Scratch): ./tmp/scratch/ for upload streams and intermediate plot assets.
- Tier 3 (Persistent Deliverables): ./output/ for cleaned_dataset.csv, pipeline.py, and audit_report.pdf with SHA-256 checksums.
- Tier 4 (External DB Sink): Optional SQLAlchemy write-back (to_sql()).

Enforces zero in-place mutations (deep copies on read/write) and strict UI decoupling (zero UI framework imports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import hashlib
import ntpath
import os
from pathlib import Path, PureWindowsPath
import tempfile
import time
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from core.ingestion_engine import DBConnectorConfig


# ---------------------------------------------------------------------------
# Metadata Contracts for Tiers 1 and 3
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetVersionInfo:
    """Metadata descriptor for an in-memory dataset version (Tier 1)."""
    version_id: str
    timestamp: str
    row_count: int
    column_count: int
    memory_bytes: int
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to serializable dictionary."""
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_bytes": self.memory_bytes,
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DeliverableArtifact:
    """Metadata descriptor for a persistent deliverable with SHA-256 integrity (Tier 3)."""
    artifact_type: str  # 'dataset', 'pipeline_code', 'audit_report'
    file_name: str
    file_path: str
    sha256_checksum: str
    size_bytes: int
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to serializable dictionary."""
        return {
            "artifact_type": self.artifact_type,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "sha256_checksum": self.sha256_checksum,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# 4-Tier Dataset Manager
# ---------------------------------------------------------------------------

class DatasetManager:
    """
    Central storage coordinator for AutoData-Hub.
    Coordinates In-Memory versioning, local scratch caching, persistent
    deliverables generation, and remote relational database export sinks.
    """

    def __init__(
        self,
        scratch_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        # Resolve filesystem roots relative to project root
        repo_root = Path(__file__).resolve().parent.parent
        self.scratch_dir = Path(scratch_dir) if scratch_dir else (repo_root / "tmp" / "scratch")
        self.output_dir = Path(output_dir) if output_dir else (repo_root / "output")

        # Guarantee physical existence of local directories
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tier 1 In-Memory Storage
        self._versions: Dict[str, pd.DataFrame] = {}
        self._version_meta: Dict[str, DatasetVersionInfo] = {}
        self._active_version_id: Optional[str] = None

    # -----------------------------------------------------------------------
    # Tier 1: In-Memory Versioning (1-Click Rollback)
    # -----------------------------------------------------------------------

    def register_version(
        self,
        version_id: str,
        df: pd.DataFrame,
        description: str = "",
        set_active: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetVersionInfo:
        """
        Registers a new dataset state into in-memory version storage.
        Strictly clones the DataFrame (deep copy) to preserve immutable states.
        """
        if df is None:
            raise ValueError("Cannot register None as a dataset version.")

        # Deep clone to prevent external mutations from altering the version ledger
        df_copy = df.copy(deep=True)
        mem_bytes = int(df_copy.memory_usage(deep=True).sum()) if len(df_copy) > 0 else 0

        info = DatasetVersionInfo(
            version_id=version_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            row_count=int(len(df_copy)),
            column_count=int(len(df_copy.columns)),
            memory_bytes=mem_bytes,
            description=description,
            metadata=dict(metadata) if metadata else {},
        )

        self._versions[version_id] = df_copy
        self._version_meta[version_id] = info

        if set_active or self._active_version_id is None:
            self._active_version_id = version_id

        return info

    def get_version(self, version_id: str) -> pd.DataFrame:
        """
        Retrieves a deep copy of a registered version DataFrame.
        """
        if version_id not in self._versions:
            raise KeyError(f"Dataset version '{version_id}' not found in in-memory storage.")
        return self._versions[version_id].copy(deep=True)

    def rollback_to(self, version_id: str) -> pd.DataFrame:
        """
        1-Click rollback: sets the designated version as active and returns a copy.
        """
        if version_id not in self._versions:
            raise KeyError(f"Cannot rollback: version '{version_id}' does not exist.")
        self._active_version_id = version_id
        return self.get_version(version_id)

    def get_active_version(self) -> Tuple[str, pd.DataFrame]:
        """
        Returns the active version identifier and a deep copy of its DataFrame.
        """
        if self._active_version_id is None or self._active_version_id not in self._versions:
            raise ValueError("No active dataset version currently registered.")
        return self._active_version_id, self.get_version(self._active_version_id)

    def get_active_version_id(self) -> Optional[str]:
        """Returns the identifier of the active version, if any."""
        return self._active_version_id

    def list_versions(self) -> List[DatasetVersionInfo]:
        """Returns metadata descriptors for all registered versions."""
        return list(self._version_meta.values())

    def has_version(self, version_id: str) -> bool:
        """Checks if a version exists in memory."""
        return version_id in self._versions

    def delete_version(self, version_id: str) -> bool:
        """Removes a specific version from in-memory cache."""
        if version_id in self._versions:
            del self._versions[version_id]
            del self._version_meta[version_id]
            if self._active_version_id == version_id:
                self._active_version_id = next(iter(self._versions.keys()), None)
            return True
        return False

    def clear_versions(self) -> None:
        """Purges all in-memory versions."""
        self._versions.clear()
        self._version_meta.clear()
        self._active_version_id = None

    # -----------------------------------------------------------------------
    # Tier 2: Transient Local Scratch (Upload Streams & Plot Assets)
    # -----------------------------------------------------------------------

    def get_scratch_dir(self) -> Path:
        """Returns the absolute path to the local scratch folder."""
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        return self.scratch_dir

    def save_scratch_file(self, filename: str, content: Union[bytes, str, BinaryIO]) -> Path:
        """Atomically publish data or a stream's remainder, including an empty payload."""
        target = self.get_scratch_file(filename)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)

                def write_chunk(chunk: Union[bytes, str]) -> None:
                    data = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                    if handle.write(data) != memoryview(data).nbytes:
                        raise OSError("Incomplete scratch file write")

                if isinstance(content, (bytes, str)):
                    write_chunk(content)
                elif hasattr(content, "read"):
                    try:
                        pos = content.tell()
                    except (AttributeError, OSError):
                        pos = None
                    try:
                        while (chunk := content.read(1024 * 1024)) not in (b"", ""):
                            if chunk is None:
                                raise BlockingIOError("Scratch stream would block before EOF")
                            write_chunk(chunk)
                    finally:
                        if pos is not None and hasattr(content, "seek"):
                            content.seek(pos)
                else:
                    raise TypeError(f"Unsupported content type for scratch file: {type(content)}")
            # Replace the directory entry; never truncate an existing linked file.
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target

    def get_scratch_file(self, filename: str) -> Path:
        """Strip benign upload folders; reject traversal, device names and redirected leaves."""
        client_path = PureWindowsPath(filename)  # Accept both client separator styles on any host.
        safe_name = Path(client_path.as_posix()).name
        reserved = (ntpath.isreserved(safe_name) if hasattr(ntpath, "isreserved")
                    else PureWindowsPath(safe_name).is_reserved())
        if (not safe_name or client_path.anchor or ".." in client_path.parts or reserved
                or ":" in safe_name or safe_name.endswith((" ", ".")) or "\x00" in filename):
            raise ValueError("Path traversal security violation")

        scratch_dir = self.get_scratch_dir().resolve()
        candidate = scratch_dir / safe_name
        if candidate.is_symlink():
            raise ValueError("Path traversal security violation")
        target = candidate.resolve()
        if scratch_dir not in target.parents or target != candidate:
            raise ValueError("Path traversal security violation")
        return target

    def create_session_scratch(self, session_id: str) -> Path:
        """Creates an isolated scratch subdirectory for a given session."""
        session_dir = self.get_scratch_file(session_id)
        if session_dir.name != session_id:
            raise ValueError("Path traversal security violation")
        session_dir.mkdir(exist_ok=True)
        return session_dir

    def cleanup_scratch(self, max_age_seconds: Optional[int] = None) -> int:
        """
        Purges transient files from the scratch folder.
        If max_age_seconds is specified, only deletes files older than the threshold.
        Returns the number of deleted files.
        """
        deleted_count = 0
        now = time.time()

        for item in self.get_scratch_dir().rglob("*"):
            if item.is_file():
                if max_age_seconds is None or (now - item.stat().st_mtime) >= max_age_seconds:
                    try:
                        item.unlink()
                        deleted_count += 1
                    except OSError:
                        pass

        return deleted_count

    # -----------------------------------------------------------------------
    # Tier 3: Persistent Deliverables (Output artifacts with SHA-256 Checksums)
    # -----------------------------------------------------------------------

    def get_output_dir(self) -> Path:
        """Returns the absolute path to the persistent deliverables folder."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    @staticmethod
    def compute_file_sha256(file_path: Union[str, Path]) -> str:
        """
        Computes the SHA-256 checksum of a file on disk using 64KB blocks.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found for SHA-256 computation: {file_path}")

        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def save_deliverable_dataset(
        self,
        df: pd.DataFrame,
        file_name: str = "cleaned_dataset.csv",
        format: str = "csv",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DeliverableArtifact:
        """
        Exports a cleaned dataset to Tier 3 persistent storage with SHA-256 verification.
        Zero in-place mutation on input DataFrame.
        """
        if df is None:
            raise ValueError("Cannot export None DataFrame to deliverable.")

        target_path = self.get_output_dir() / file_name
        fmt = format.lower().strip(".")

        if fmt == "csv":
            df.to_csv(target_path, index=False, **kwargs)
        elif fmt in ("parquet", "pq"):
            df.to_parquet(target_path, engine="pyarrow", index=False, **kwargs)
        elif fmt in ("feather", "ft"):
            df.to_feather(target_path, **kwargs)
        elif fmt in ("xlsx", "xls"):
            df.to_excel(target_path, index=False, engine="openpyxl", **kwargs)
        else:
            df.to_csv(target_path, index=False, **kwargs)

        sha256 = self.compute_file_sha256(target_path)
        size_bytes = target_path.stat().st_size

        artifact = DeliverableArtifact(
            artifact_type="dataset",
            file_name=file_name,
            file_path=str(target_path),
            sha256_checksum=sha256,
            size_bytes=size_bytes,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            metadata=dict(metadata) if metadata else {"rows": len(df), "cols": len(df.columns)},
        )
        return artifact

    def save_deliverable_code(
        self,
        code_str: str,
        file_name: str = "pipeline.py",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeliverableArtifact:
        """
        Exports generated standalone pipeline execution code to Tier 3 persistent storage.
        """
        target_path = self.get_output_dir() / file_name
        target_path.write_text(code_str, encoding="utf-8")

        sha256 = self.compute_file_sha256(target_path)
        size_bytes = target_path.stat().st_size

        return DeliverableArtifact(
            artifact_type="pipeline_code",
            file_name=file_name,
            file_path=str(target_path),
            sha256_checksum=sha256,
            size_bytes=size_bytes,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            metadata=dict(metadata) if metadata else {},
        )

    def save_deliverable_report(
        self,
        pdf_bytes: bytes,
        file_name: str = "audit_report.pdf",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeliverableArtifact:
        """
        Exports the ReportLab PDF audit report to Tier 3 persistent storage.
        """
        target_path = self.get_output_dir() / file_name
        target_path.write_bytes(pdf_bytes)

        sha256 = self.compute_file_sha256(target_path)
        size_bytes = target_path.stat().st_size

        return DeliverableArtifact(
            artifact_type="audit_report",
            file_name=file_name,
            file_path=str(target_path),
            sha256_checksum=sha256,
            size_bytes=size_bytes,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            metadata=dict(metadata) if metadata else {},
        )

    def verify_artifact_integrity(
        self,
        file_path: Union[str, Path],
        expected_sha256: str,
    ) -> bool:
        """
        Validates the SHA-256 integrity checksum of a file on disk against expected hash.
        """
        computed = self.compute_file_sha256(file_path)
        return computed.lower().strip() == expected_sha256.lower().strip()

    def list_deliverables(self) -> List[DeliverableArtifact]:
        """
        Scans the persistent output folder and returns descriptors for all deliverables.
        """
        artifacts: List[DeliverableArtifact] = []
        for item in self.get_output_dir().glob("*"):
            if item.is_file():
                sha256 = self.compute_file_sha256(item)
                size_bytes = item.stat().st_size
                suffix = item.suffix.lower()

                if suffix in (".csv", ".parquet", ".feather", ".xlsx"):
                    art_type = "dataset"
                elif suffix == ".py":
                    art_type = "pipeline_code"
                elif suffix == ".pdf":
                    art_type = "audit_report"
                else:
                    art_type = "file"

                artifacts.append(DeliverableArtifact(
                    artifact_type=art_type,
                    file_name=item.name,
                    file_path=str(item),
                    sha256_checksum=sha256,
                    size_bytes=size_bytes,
                    created_at=datetime.datetime.fromtimestamp(
                        item.stat().st_mtime,
                        tz=datetime.timezone.utc
                    ).isoformat(),
                    metadata={},
                ))
        return artifacts

    # -----------------------------------------------------------------------
    # Tier 4: External Relational DB Sink (SQLAlchemy to_sql)
    # -----------------------------------------------------------------------

    def export_to_sql(
        self,
        df: pd.DataFrame,
        connection_url_or_config: Union[str, DBConnectorConfig],
        table_name: str,
        if_exists: str = "replace",
        chunksize: int = 1000,
        **kwargs,
    ) -> int:
        """
        Writes dataset to an external relational database via SQLAlchemy to_sql().
        Zero in-place mutations on input DataFrame. Returns total rows exported.
        """
        if df is None:
            raise ValueError("Cannot export None DataFrame to SQL.")

        url = (
            connection_url_or_config
            if isinstance(connection_url_or_config, str)
            else connection_url_or_config.get_connection_url()
        )

        engine: Engine = create_engine(url, pool_pre_ping=True)
        # Deep copy to ensure zero caller mutation
        df_export = df.copy(deep=True)

        df_export.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            **kwargs,
        )

        return len(df_export)
