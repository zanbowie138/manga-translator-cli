"""Output path management for manga translation pipeline."""

from pathlib import Path
from typing import Optional


class OutputManager:
    """Manages output folder structure and file paths."""

    def __init__(self, base_folder: str, subfolder: Optional[str] = None):
        """Initialize output manager.

        Args:
            base_folder: Base output directory path
            subfolder: Optional subfolder to nest outputs under
        """
        self.base = Path(base_folder)
        self.subfolder = subfolder

    def setup(
        self,
        save_speech_bubbles: bool = False,
        save_bubble_interiors: bool = False,
        save_cleaned: bool = False,
        save_translated: bool = True
    ) -> None:
        """Create output directory structure based on save flags.

        Args:
            save_speech_bubbles: Create speech_bubbles folder
            save_bubble_interiors: Create bubble_masks folder
            save_cleaned: Create cleaned folder
            save_translated: Create translated folder
        """
        # Create base folder
        self.base.mkdir(parents=True, exist_ok=True)

        # Create subfolders for each output type
        if save_speech_bubbles:
            self._get_dir("speech_bubbles").mkdir(parents=True, exist_ok=True)
        if save_bubble_interiors:
            self._get_dir("bubble_masks").mkdir(parents=True, exist_ok=True)
        if save_cleaned:
            self._get_dir("cleaned").mkdir(parents=True, exist_ok=True)
        if save_translated:
            self._get_dir("translated").mkdir(parents=True, exist_ok=True)

    def get_path(self, output_type: str, filename: str) -> Path:
        """Get output file path for a specific output type.

        Args:
            output_type: Type of output ('translated', 'cleaned', 'bubble_masks', 'speech_bubbles')
            filename: Name of the output file

        Returns:
            Complete path to output file
        """
        return self._get_dir(output_type) / filename

    def _get_dir(self, folder_name: str) -> Path:
        """Get directory path for a specific output type.

        Args:
            folder_name: Name of the output folder

        Returns:
            Path to output directory
        """
        if self.subfolder:
            return self.base / folder_name / self.subfolder
        else:
            return self.base / folder_name
