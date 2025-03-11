import os
from typing import List
from pathlib import Path
from loguru import logger
import shutil

class LogCleaner:
    def __init__(self, logs_dir: str, max_checkpoints: int):
        """
        Initialize LogCleaner with path to logs directory
        
        Args:
            logs_dir: Path to the logs directory
        """
        self.logs_dir = Path(logs_dir)
        self.max_checkpoints = max_checkpoints
        if not self.logs_dir.exists():
            raise ValueError(f"Logs directory does not exist: {self.logs_dir}")
    
    def find_model0_only_dirs(self) -> List[str]:
        """
        Scan through logs directory to find experiment directories that only have model_0.pt
        among all model checkpoint files (model_*.pt), but can have other non-checkpoint files.
        
        Returns:
            List of paths to directories containing only model_0.pt as checkpoint
        """
        model0_dirs = []
        
        # Iterate through project directories (MPPI, tairan, etc.)
        for project_dir in self.logs_dir.iterdir():
            if not project_dir.is_dir():
                continue
                
            # Iterate through experiment directories (20241023_111830-TEST-TEST-h1, etc.)
            for exp_dir in project_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                    
                # Get all model checkpoint files in the experiment directory
                model_files = list(exp_dir.glob('model_*.pt'))
                
                # Check if directory only contains model_0.pt among checkpoints
                if len(model_files) == 1 and model_files[0].name == 'model_0.pt':
                    model0_dirs.append(str(exp_dir))
        
        return model0_dirs
    
    def find_no_checkpoint_dirs(self) -> List[str]:
        """
        Scan through logs directory to find experiment directories that have no
        model checkpoint files (model_*.pt)
        
        Returns:
            List of paths to directories containing no checkpoints
        """
        no_checkpoint_dirs = []
        
        # Iterate through project directories (MPPI, tairan, etc.)
        for project_dir in self.logs_dir.iterdir():
            if not project_dir.is_dir():
                continue
                
            # Iterate through experiment directories (20241023_111830-TEST-TEST-h1, etc.)
            for exp_dir in project_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                    
                # Get all model checkpoint files in the experiment directory
                model_files = list(exp_dir.glob('model_*.pt'))
                
                # Check if directory has no checkpoints
                if len(model_files) == 0:
                    no_checkpoint_dirs.append(str(exp_dir))
        
        return no_checkpoint_dirs
    
    def find_many_checkpoints_dirs(self):
        """
        Scan through logs directory to find experiment directories that have too many
        model checkpoint files (model_*.pt)
        
        Returns:
            List of tuples (dir_path, checkpoint_count) for directories with many checkpoints
        """
        many_checkpoints_dirs = []
        
        # Iterate through project directories (MPPI, tairan, etc.)
        for project_dir in self.logs_dir.iterdir():
            if not project_dir.is_dir():
                continue
                
            # Iterate through experiment directories (20241023_111830-TEST-TEST-h1, etc.)
            for exp_dir in project_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                    
                # Get all model checkpoint files in the experiment directory
                model_files = list(exp_dir.glob('model_*.pt'))
                
                # Check if directory has too many checkpoints
                if len(model_files) > self.max_checkpoints:
                    many_checkpoints_dirs.append((str(exp_dir), len(model_files)))
        
        # Sort by number of checkpoints, descending
        many_checkpoints_dirs.sort(key=lambda x: x[1], reverse=True)
        return many_checkpoints_dirs
    
    def clean_failed_dirs(self, auto_confirm: bool = False) -> None:
        """
        Find and optionally delete directories with only model_0.pt
        
        Args:
            auto_confirm: If True, automatically delete without asking for confirmation
        """
        logger.info("Scanning for directories containing only model_0.pt...")
        model0_dirs = self.find_model0_only_dirs()
        no_checkpoint_dirs = self.find_no_checkpoint_dirs()
        model0_dirs.extend(no_checkpoint_dirs)
        
        if not model0_dirs:
            logger.info("No directories found containing only model_0.pt")
            return
            
        logger.info("\nFound the following directories containing only model_0.pt:")
        for dir_path in model0_dirs:
            logger.info(f"- {dir_path}")
            
        # Optional: Ask user if they want to delete these directories
        should_delete = auto_confirm
        if not auto_confirm:
            response = input("\nWould you like to delete these directories? (y/N): ")
            should_delete = response.lower() == 'y'
            
        if should_delete:
            for dir_path in model0_dirs:
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Deleted: {dir_path}")
                except Exception as e:
                    logger.error(f"Error deleting {dir_path}: {e}")

    def get_dir_checkpoints_size(self, dir_path: Path) -> float:
        """Get total size of all model_*.pt files in directory in MB"""
        total_bytes = sum(f.stat().st_size for f in dir_path.glob('model_*.pt'))
        return total_bytes / (1024 * 1024)  # Convert to MB

    def clean_many_checkpoints_dirs(self, auto_confirm: bool = False) -> None:
        """
        Find and optionally delete excess model checkpoints
        
        Args:
            auto_confirm: If True, automatically delete without asking for confirmation
        """
        logger.info(f"Scanning for directories with more than {self.max_checkpoints} checkpoints...")
        many_checkpoints_dirs = self.find_many_checkpoints_dirs()
        
        if not many_checkpoints_dirs:
            logger.info(f"No directories found with more than {self.max_checkpoints} checkpoints")
            return
        total_size = 0
        logger.info("\nFound the following directories with many checkpoints:")
        for dir_path, count in many_checkpoints_dirs:
            size = self.get_dir_checkpoints_size(Path(dir_path))
            logger.info(f"- {dir_path}: {count} checkpoints, {size:.2f} MB")
            total_size += size
            
        logger.info(f"Total size of all checkpoints: {total_size:.2f} MB")
            
        # Optional: Ask user if they want to clean these directories
        should_clean = auto_confirm
        if not auto_confirm:
            response = input("\nWould you like to clean these directories? (y/N): ")
            should_clean = response.lower() == 'y'
            
        if should_clean:
            for dir_path, _ in many_checkpoints_dirs:
                try:
                    exp_dir = Path(dir_path)
                    model_files = sorted(exp_dir.glob('model_*.pt'))
                    
                    # Function to extract step number from filename
                    def get_step_number(file_path: Path) -> int:
                        try:
                            # Extract number between 'model_' and '.pt'
                            return int(file_path.stem.split('_')[1])
                        except (IndexError, ValueError):
                            return 0
                    
                    # Always keep model_0.pt and files with steps divisible by 1000
                    files_to_keep = {
                        file for file in model_files 
                        if get_step_number(file) == 0 or  # Keep model_0.pt
                        (get_step_number(file) % 1000 == 0)  # Keep model_1000.pt, model_2000.pt, etc.
                    }
                    
                    files_to_delete = set(model_files) - files_to_keep
                    
                    # Log what we're keeping and deleting
                    logger.info(f"\nIn {dir_path}:")
                    logger.info("Keeping checkpoints: " + ", ".join(f.name for f in sorted(files_to_keep)))
                    
                    for file in files_to_delete:
                        file.unlink()
                        logger.info(f"Deleted: {file}")
                    
                    logger.info(f"Cleaned {dir_path}, kept {len(files_to_keep)} checkpoints")
                except Exception as e:
                    logger.error(f"Error cleaning {dir_path}: {e}")

    def clean_empty_dirs(self, auto_confirm: bool = False) -> None:
        """
        Find and optionally delete directories with no model checkpoints
        
        Args:
            auto_confirm: If True, automatically delete without asking for confirmation
        """
        logger.info("Scanning for directories with no checkpoints...")
        empty_dirs = self.find_no_checkpoint_dirs()
        
        if not empty_dirs:
            logger.info("No directories found with no checkpoints")
            return
            
        logger.info("\nFound the following directories with no checkpoints:")
        for dir_path in empty_dirs:
            logger.info(f"- {dir_path}")
            
        # Optional: Ask user if they want to delete these directories
        should_delete = auto_confirm
        if not auto_confirm:
            response = input("\nWould you like to delete these directories? (y/N): ")
            should_delete = response.lower() == 'y'
            
        if should_delete:
            for dir_path in empty_dirs:
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Deleted: {dir_path}")
                except Exception as e:
                    logger.error(f"Error deleting {dir_path}: {e}")

def main():
    # Assuming the script is in scripts/logging and logs is in the project root
    current_dir = Path(__file__).parent
    logs_dir = current_dir.parent.parent / 'logs'
    cleaner = LogCleaner(logs_dir, max_checkpoints=50)
    cleaner.clean_failed_dirs()
    # cleaner.clean_many_checkpoints_dirs()
if __name__ == "__main__":
    main()