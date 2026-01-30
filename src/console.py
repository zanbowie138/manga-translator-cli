"""Console output handler using rich library."""

from typing import Optional
from rich.console import Console as RichConsole
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.status import Status


class Console:
    """Wrapper around rich.console.Console for consistent messaging."""

    def __init__(self, quiet: bool = False):
        """Initialize console handler.

        Args:
            quiet: If True, suppress all non-error output
        """
        self.console = RichConsole()
        self.quiet = quiet

    def info(self, message: str) -> None:
        """Print info message with blue icon.

        Args:
            message: Message to display
        """
        if not self.quiet:
            self.console.print(f"[blue]ℹ[/blue]  {message}")

    def success(self, message: str) -> None:
        """Print success message with green checkmark.

        Args:
            message: Success message to display
        """
        if not self.quiet:
            self.console.print(f"[green]✓[/green] {message}")

    def error(self, message: str) -> None:
        """Print error message with red X (always shown, even in quiet mode).

        Args:
            message: Error message to display
        """
        self.console.print(f"[red]✗[/red] {message}", style="bold red")

    def section(self, title: str) -> None:
        """Print section header with separators.

        Args:
            title: Section title
        """
        if not self.quiet:
            separator = "=" * 50
            self.console.print(f"\n{separator}")
            self.console.print(title)
            self.console.print(separator)

    def progress_bar(
        self,
        total: int,
        description: str
    ) -> Optional[Progress]:
        """Create a progress bar for tracking operations.

        Args:
            total: Total number of items
            description: Description of the operation

        Returns:
            Progress object if not quiet, None otherwise
        """
        if self.quiet:
            return None

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console
        )

    def status(self, message: str) -> Optional[Status]:
        """Create a status spinner for long operations.

        Args:
            message: Status message to display

        Returns:
            Status object if not quiet, None otherwise
        """
        if self.quiet:
            return None

        return self.console.status(message, spinner="dots")

    def print(self, message: str, style: Optional[str] = None) -> None:
        """Print a plain message (respects quiet flag).

        Args:
            message: Message to print
            style: Optional rich style string
        """
        if not self.quiet:
            self.console.print(message, style=style)
