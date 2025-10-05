"""
Model Registry Management Utilities

Provides CLI tools for:
- Registering new model versions
- Managing model lifecycle (deploy, rollback, archive)
- Validating model files and metadata
- Generating checksums
- Model version comparisons

Author: System Architecture Team
Version: 1.0.0
Date: 2025-10-05
"""

import os
import json
import hashlib
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistryManager:
    """Manage model registry operations"""

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model_name: str,
        version: str,
        model_file: Path,
        metadata: Dict,
        preprocessing_file: Optional[Path] = None,
        set_as_latest: bool = True
    ):
        """
        Register a new model version

        Args:
            model_name: Name of the model (e.g., 'keras_cnn')
            version: Version string (e.g., '1.0.0')
            model_file: Path to model file
            metadata: Model metadata dictionary
            preprocessing_file: Optional preprocessing artifacts
            set_as_latest: Set this version as 'latest'
        """
        logger.info(f"Registering model: {model_name} v{version}")

        # Create version directory
        version_dir = self.registry_path / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        dest_model_file = version_dir / model_file.name
        if model_file.is_file():
            shutil.copy2(model_file, dest_model_file)
            logger.info(f"Copied model file to {dest_model_file}")
        elif model_file.is_dir():
            # For directories like .keras SavedModel
            shutil.copytree(model_file, dest_model_file, dirs_exist_ok=True)
            logger.info(f"Copied model directory to {dest_model_file}")

        # Copy preprocessing file if provided
        if preprocessing_file and preprocessing_file.exists():
            dest_preprocessing = version_dir / preprocessing_file.name
            shutil.copy2(preprocessing_file, dest_preprocessing)
            logger.info(f"Copied preprocessing file to {dest_preprocessing}")

        # Add metadata fields
        metadata.update({
            "model_name": model_name,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "registered_at": datetime.utcnow().isoformat()
        })

        # Save metadata
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")

        # Generate checksum
        checksum = self._generate_checksum(dest_model_file)
        checksum_file = version_dir / "checksum.sha256"
        with open(checksum_file, 'w') as f:
            f.write(f"{checksum}  {dest_model_file.name}\n")
        logger.info(f"Generated checksum: {checksum}")

        # Update 'latest' symlink
        if set_as_latest:
            self._update_latest_link(model_name, version)

        logger.info(f"Successfully registered {model_name} v{version}")

    def _generate_checksum(self, file_path: Path) -> str:
        """Generate SHA256 checksum for file or directory"""
        sha256_hash = hashlib.sha256()

        if file_path.is_file():
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        elif file_path.is_dir():
            # For directories, hash all files
            for root, dirs, files in os.walk(file_path):
                dirs.sort()  # Ensure consistent ordering
                for filename in sorted(files):
                    filepath = Path(root) / filename
                    with open(filepath, "rb") as f:
                        for byte_block in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def _update_latest_link(self, model_name: str, version: str):
        """Update 'latest' symlink to point to specified version"""
        model_dir = self.registry_path / model_name
        latest_link = model_dir / "latest"
        version_dir = model_dir / version

        # Remove existing symlink
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create new symlink
        try:
            latest_link.symlink_to(version, target_is_directory=True)
            logger.info(f"Updated 'latest' symlink for {model_name} -> {version}")
        except OSError as e:
            # On Windows, symlinks may require admin privileges
            # Fall back to a marker file
            logger.warning(f"Could not create symlink: {e}. Creating marker file instead.")
            with open(latest_link, 'w') as f:
                f.write(version)

    def list_versions(self, model_name: str) -> List[str]:
        """List all versions for a model"""
        model_dir = self.registry_path / model_name

        if not model_dir.exists():
            return []

        versions = []
        for item in model_dir.iterdir():
            if item.is_dir() and item.name not in ["latest", "stable"]:
                versions.append(item.name)

        return sorted(versions, reverse=True)

    def get_model_info(self, model_name: str, version: str = "latest") -> Optional[Dict]:
        """Get model metadata"""
        if version == "latest":
            version_dir = self._resolve_latest(model_name)
        else:
            version_dir = self.registry_path / model_name / version

        if not version_dir.exists():
            return None

        metadata_file = version_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                return json.load(f)

        return None

    def _resolve_latest(self, model_name: str) -> Optional[Path]:
        """Resolve 'latest' symlink to actual version directory"""
        latest_link = self.registry_path / model_name / "latest"

        if not latest_link.exists():
            return None

        if latest_link.is_symlink():
            return latest_link.resolve()
        else:
            # Marker file fallback
            with open(latest_link) as f:
                version = f.read().strip()
            return self.registry_path / model_name / version

    def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str
    ) -> Dict:
        """Compare two model versions"""
        info_a = self.get_model_info(model_name, version_a)
        info_b = self.get_model_info(model_name, version_b)

        if not info_a or not info_b:
            raise ValueError("One or both versions not found")

        comparison = {
            "model_name": model_name,
            "version_a": version_a,
            "version_b": version_b,
            "performance_delta": {},
            "metadata_changes": {}
        }

        # Compare performance metrics
        perf_a = info_a.get("performance", {})
        perf_b = info_b.get("performance", {})

        for metric in perf_a:
            if metric in perf_b:
                delta = perf_b[metric] - perf_a[metric]
                comparison["performance_delta"][metric] = {
                    "version_a": perf_a[metric],
                    "version_b": perf_b[metric],
                    "delta": delta,
                    "improvement": delta > 0
                }

        # Compare metadata
        if info_a.get("framework_version") != info_b.get("framework_version"):
            comparison["metadata_changes"]["framework_version"] = {
                "version_a": info_a.get("framework_version"),
                "version_b": info_b.get("framework_version")
            }

        return comparison

    def archive_version(self, model_name: str, version: str):
        """Archive a model version"""
        version_dir = self.registry_path / model_name / version
        archive_dir = self.registry_path / model_name / "archive" / version

        if not version_dir.exists():
            raise ValueError(f"Version not found: {model_name} v{version}")

        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(version_dir), str(archive_dir))
        logger.info(f"Archived {model_name} v{version} to {archive_dir}")

    def delete_version(self, model_name: str, version: str, confirm: bool = False):
        """Delete a model version (irreversible)"""
        if not confirm:
            raise ValueError("Must set confirm=True to delete model version")

        version_dir = self.registry_path / model_name / version

        if not version_dir.exists():
            raise ValueError(f"Version not found: {model_name} v{version}")

        shutil.rmtree(version_dir)
        logger.info(f"Deleted {model_name} v{version}")

    def validate_model(self, model_name: str, version: str = "latest") -> Dict:
        """Validate model files and metadata"""
        if version == "latest":
            version_dir = self._resolve_latest(model_name)
        else:
            version_dir = self.registry_path / model_name / version

        if not version_dir:
            return {"valid": False, "error": "Version not found"}

        validation_results = {
            "valid": True,
            "checks": {},
            "errors": []
        }

        # Check metadata exists
        metadata_file = version_dir / "metadata.json"
        if metadata_file.exists():
            validation_results["checks"]["metadata"] = "✓"
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)

                # Validate required fields
                required_fields = ["model_name", "version", "framework", "input_schema", "output_schema"]
                for field in required_fields:
                    if field not in metadata:
                        validation_results["errors"].append(f"Missing metadata field: {field}")
                        validation_results["valid"] = False
            except json.JSONDecodeError:
                validation_results["checks"]["metadata"] = "✗"
                validation_results["errors"].append("Invalid JSON in metadata.json")
                validation_results["valid"] = False
        else:
            validation_results["checks"]["metadata"] = "✗"
            validation_results["errors"].append("metadata.json not found")
            validation_results["valid"] = False

        # Check model file exists
        model_files = list(version_dir.glob("model.*"))
        if model_files:
            validation_results["checks"]["model_file"] = "✓"
        else:
            validation_results["checks"]["model_file"] = "✗"
            validation_results["errors"].append("No model file found")
            validation_results["valid"] = False

        # Check checksum
        checksum_file = version_dir / "checksum.sha256"
        if checksum_file.exists():
            validation_results["checks"]["checksum"] = "✓"
        else:
            validation_results["checks"]["checksum"] = "⚠"
            validation_results["errors"].append("No checksum file (warning)")

        return validation_results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Model Registry Management Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new model version")
    register_parser.add_argument("--name", required=True, help="Model name")
    register_parser.add_argument("--version", required=True, help="Version string")
    register_parser.add_argument("--model-file", required=True, help="Path to model file")
    register_parser.add_argument("--metadata", required=True, help="Path to metadata JSON file")
    register_parser.add_argument("--preprocessing", help="Path to preprocessing file")
    register_parser.add_argument("--registry-path", default="./models", help="Registry path")
    register_parser.add_argument("--set-latest", action="store_true", default=True, help="Set as latest")

    # List command
    list_parser = subparsers.add_parser("list", help="List model versions")
    list_parser.add_argument("--name", required=True, help="Model name")
    list_parser.add_argument("--registry-path", default="./models", help="Registry path")

    # Info command
    info_parser = subparsers.add_parser("info", help="Get model info")
    info_parser.add_argument("--name", required=True, help="Model name")
    info_parser.add_argument("--version", default="latest", help="Version")
    info_parser.add_argument("--registry-path", default="./models", help="Registry path")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two versions")
    compare_parser.add_argument("--name", required=True, help="Model name")
    compare_parser.add_argument("--version-a", required=True, help="First version")
    compare_parser.add_argument("--version-b", required=True, help="Second version")
    compare_parser.add_argument("--registry-path", default="./models", help="Registry path")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate model")
    validate_parser.add_argument("--name", required=True, help="Model name")
    validate_parser.add_argument("--version", default="latest", help="Version")
    validate_parser.add_argument("--registry-path", default="./models", help="Registry path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = ModelRegistryManager(args.registry_path)

    if args.command == "register":
        with open(args.metadata) as f:
            metadata = json.load(f)

        manager.register_model(
            model_name=args.name,
            version=args.version,
            model_file=Path(args.model_file),
            metadata=metadata,
            preprocessing_file=Path(args.preprocessing) if args.preprocessing else None,
            set_as_latest=args.set_latest
        )

    elif args.command == "list":
        versions = manager.list_versions(args.name)
        print(f"\nVersions for {args.name}:")
        for version in versions:
            print(f"  - {version}")

    elif args.command == "info":
        info = manager.get_model_info(args.name, args.version)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print(f"Model not found: {args.name} v{args.version}")

    elif args.command == "compare":
        comparison = manager.compare_versions(args.name, args.version_a, args.version_b)
        print(json.dumps(comparison, indent=2))

    elif args.command == "validate":
        results = manager.validate_model(args.name, args.version)
        print(f"\nValidation for {args.name} v{args.version}:")
        print(f"Valid: {results['valid']}")
        print("\nChecks:")
        for check, status in results["checks"].items():
            print(f"  {check}: {status}")
        if results["errors"]:
            print("\nErrors:")
            for error in results["errors"]:
                print(f"  - {error}")


if __name__ == "__main__":
    main()
