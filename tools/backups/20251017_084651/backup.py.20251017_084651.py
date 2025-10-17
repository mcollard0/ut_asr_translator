#!/usr/bin/env python3
"""
Backup utility with ISO-8601 timestamps and rotation
Creates timestamped backups and maintains a rotation policy
"""

import os
import shutil
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def create_backup( file_path: str, backup_dir: Optional[str] = None, max_backups: Optional[int] = None ) -> str:
    """
    Create a timestamped backup of a file
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backups (defaults to ./backup)
        max_backups: Maximum number of backups to keep (auto-determined if None)
        
    Returns:
        Path to created backup file
    """
    source_path = Path( file_path );
    
    if not source_path.exists():
        raise FileNotFoundError( f"Source file not found: {file_path}" );
    
    # Determine backup directory;
    if backup_dir is None:
        backup_dir = source_path.parent / "backup";
    backup_path = Path( backup_dir );
    backup_path.mkdir( parents=True, exist_ok=True );
    
    # Generate timestamp and backup filename;
    timestamp = datetime.now().strftime( "%Y-%m-%dT%H:%M:%S" );
    backup_filename = f"{source_path.stem}.{timestamp}{source_path.suffix}";
    backup_full_path = backup_path / backup_filename;
    
    # Copy file;
    shutil.copy2( source_path, backup_full_path );
    print( f"📦 Backup created: {backup_full_path}" );
    
    # Determine rotation policy;
    if max_backups is None:
        file_size_mb = source_path.stat().st_size / ( 1024 * 1024 );
        max_backups = 50 if file_size_mb <= 0.15 else 25;
    
    # Rotate old backups;
    rotate_backups( source_path.name, backup_path, max_backups );
    
    return str( backup_full_path );


def rotate_backups( original_filename: str, backup_dir: Path, max_backups: int ):
    """
    Remove old backups to maintain rotation policy
    
    Args:
        original_filename: Name of original file (to match backup pattern)
        backup_dir: Directory containing backups
        max_backups: Maximum number to keep
    """
    # Find all backup files for this original file;
    stem = Path( original_filename ).stem;
    suffix = Path( original_filename ).suffix;
    pattern = f"{stem}.????-??-??T??:??:??{suffix}";
    
    backup_files = list( backup_dir.glob( pattern ) );
    backup_files.sort( key=lambda p: p.stat().st_mtime );  # Sort by modification time;
    
    # Remove excess backups;
    while len( backup_files ) > max_backups:
        old_backup = backup_files.pop( 0 );  # Remove oldest;
        old_backup.unlink();
        print( f"🗑️  Removed old backup: {old_backup}" );


def backup_project_files( project_root: str = "." ):
    """
    Backup key project files before major changes
    
    Args:
        project_root: Root directory of the project
    """
    project_path = Path( project_root );
    backup_dir = project_path / "backup";
    
    # Key files to backup;
    key_files = [
        "src/wa_asr_translator/config.py",
        "src/wa_asr_translator/asr.py", 
        "src/wa_asr_translator/translate.py",
        "src/wa_asr_translator/cli.py",
        "tools/run_whatsapp.py",
        "README.md",
        "docs/architecture.md",
        "requirements.txt"
    ];
    
    print( "📦 Creating backups for key project files..." );
    
    backed_up = [];
    for file_rel_path in key_files:
        file_path = project_path / file_rel_path;
        if file_path.exists():
            try:
                backup_path = create_backup( str( file_path ), str( backup_dir ) );
                backed_up.append( file_rel_path );
            except Exception as e:
                print( f"⚠️  Failed to backup {file_rel_path}: {e}" );
    
    print( f"✅ Backed up {len(backed_up)} files to {backup_dir}" );
    return backed_up;


if __name__ == "__main__":
    import sys
    
    if len( sys.argv ) < 2:
        print( "Usage: python backup.py <file_path> [backup_dir] [max_backups]" );
        print( "   or: python backup.py --project  # Backup all key project files" );
        sys.exit( 1 );
    
    if sys.argv[1] == "--project":
        backup_project_files();
    else:
        file_path = sys.argv[1];
        backup_dir = sys.argv[2] if len( sys.argv ) > 2 else None;
        max_backups = int( sys.argv[3] ) if len( sys.argv ) > 3 else None;
        
        try:
            backup_path = create_backup( file_path, backup_dir, max_backups );
            print( f"Backup created: {backup_path}" );
        except Exception as e:
            print( f"Error: {e}" );
            sys.exit( 1 );