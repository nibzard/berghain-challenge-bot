#!/usr/bin/env python3
"""
ABOUTME: Game logs cleanup script for removing underperforming strategy logs
ABOUTME: Safely deletes logs with >850 rejections while preserving best performers
"""

import json
import glob
import os
import shutil
import argparse
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Set


class GameLogsCleanup:
    def __init__(self, dry_run: bool = True, backup_dir: str = None, force: bool = False):
        self.dry_run = dry_run
        self.backup_dir = backup_dir
        self.force = force
        self.rejection_threshold = 850
        self.keep_strategies = {
            'rbcr', 'rbcr2', 'ultimate3', 'ultimate3h', 'perfect', 
            'apex', 'ultimate2', 'dual', 'optimal', 'ultimate'
        }
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'files_to_keep': 0,
            'files_to_delete': 0,
            'total_size': 0,
            'size_to_delete': 0,
            'strategies_analyzed': 0
        }
        
        self.files_to_delete: List[Tuple[str, str, int, int]] = []  # path, strategy, size, rejections
        self.files_to_keep: List[Tuple[str, str, int]] = []  # path, strategy, size
        
    def format_size(self, bytes_size: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f'{bytes_size:.1f}{unit}'
            bytes_size /= 1024.0
        return f'{bytes_size:.1f}TB'
    
    def extract_strategy_name(self, filename: str) -> str:
        """Extract strategy name from filename"""
        basename = os.path.basename(filename)
        parts = basename.replace('events_', '').split('_')
        
        # Find strategy name (everything before first number or timestamp)
        strategy_parts = []
        for part in parts:
            if part.isdigit() or (len(part) >= 3 and part[:3].isdigit()):
                break
            strategy_parts.append(part)
        
        return '_'.join(strategy_parts) if strategy_parts else 'unknown'
    
    def get_rejection_count(self, filepath: str) -> int:
        """Get rejection count from last line of game log"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return -1
                
                # Check last few lines for game completion
                for line in reversed(lines[-10:]):
                    try:
                        data = json.loads(line)
                        if 'data' in data and 'rejected' in data['data']:
                            return data['data']['rejected']
                    except:
                        continue
                        
                return -1  # No rejection count found
        except Exception:
            return -1
    
    def analyze_files(self):
        """Analyze all game log files and categorize them"""
        print("🔍 Analyzing game log files...")
        
        all_files = glob.glob('game_logs/events_*.jsonl')
        self.stats['total_files'] = len(all_files)
        
        strategy_summary = defaultdict(lambda: {
            'total_files': 0,
            'total_size': 0,
            'keep_count': 0,
            'delete_count': 0,
            'rejections': []
        })
        
        for i, file in enumerate(all_files):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(all_files)} files...")
            
            strategy = self.extract_strategy_name(file)
            file_size = os.path.getsize(file)
            
            self.stats['total_size'] += file_size
            strategy_summary[strategy]['total_files'] += 1
            strategy_summary[strategy]['total_size'] += file_size
            
            # Priority strategies - keep all files
            if strategy in self.keep_strategies:
                self.files_to_keep.append((file, strategy, file_size))
                strategy_summary[strategy]['keep_count'] += 1
                continue
            
            # Other strategies - check rejection count
            rejection_count = self.get_rejection_count(file)
            
            if rejection_count == -1:
                # Incomplete file - delete
                self.files_to_delete.append((file, strategy, file_size, -1))
                strategy_summary[strategy]['delete_count'] += 1
                self.stats['size_to_delete'] += file_size
            elif rejection_count > self.rejection_threshold:
                # Poor performance - delete
                self.files_to_delete.append((file, strategy, file_size, rejection_count))
                strategy_summary[strategy]['delete_count'] += 1
                strategy_summary[strategy]['rejections'].append(rejection_count)
                self.stats['size_to_delete'] += file_size
            else:
                # Good performance - keep
                self.files_to_keep.append((file, strategy, file_size))
                strategy_summary[strategy]['keep_count'] += 1
                strategy_summary[strategy]['rejections'].append(rejection_count)
        
        self.stats['files_to_keep'] = len(self.files_to_keep)
        self.stats['files_to_delete'] = len(self.files_to_delete)
        self.stats['strategies_analyzed'] = len(strategy_summary)
        
        # Print analysis summary
        print(f"\n{'='*80}")
        print("ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total files analyzed: {self.stats['total_files']:,}")
        print(f"  Files to keep:        {self.stats['files_to_keep']:,} ({self.format_size(self.stats['total_size'] - self.stats['size_to_delete'])})")
        print(f"  Files to delete:      {self.stats['files_to_delete']:,} ({self.format_size(self.stats['size_to_delete'])})")
        print(f"  Space savings:        {self.format_size(self.stats['size_to_delete'])}")
        print(f"  Retention rate:       {(self.stats['files_to_keep']/self.stats['total_files']*100):.1f}%")
        
        # Strategy breakdown
        print(f"\n📋 STRATEGY BREAKDOWN:")
        for strategy in sorted(strategy_summary.keys()):
            stats = strategy_summary[strategy]
            avg_rej = sum(stats['rejections']) / len(stats['rejections']) if stats['rejections'] else 0
            status = '✓ KEEP ALL' if strategy in self.keep_strategies else f'{stats["keep_count"]} keep, {stats["delete_count"]} delete'
            
            print(f"  {strategy:25} {stats['total_files']:4} files  {self.format_size(stats['total_size']):8}  {status}")
            if stats['rejections']:
                print(f"    └─ Avg rejections: {avg_rej:.0f}")
    
    def preview_deletions(self, limit: int = 20):
        """Show preview of files to be deleted"""
        if not self.files_to_delete:
            print("No files to delete.")
            return
        
        print(f"\n{'='*80}")
        print(f"DELETION PREVIEW (showing {min(limit, len(self.files_to_delete))} of {len(self.files_to_delete)} files)")
        print(f"{'='*80}")
        
        # Sort by rejection count (highest first), then by size
        sorted_deletions = sorted(self.files_to_delete, key=lambda x: (x[3] if x[3] > 0 else 9999, -x[2]))
        
        for file_path, strategy, size, rejections in sorted_deletions[:limit]:
            rej_str = f"{rejections:4} rej" if rejections > 0 else "incomplete"
            print(f"  {os.path.basename(file_path):50} {strategy:20} {self.format_size(size):8} {rej_str}")
        
        if len(self.files_to_delete) > limit:
            print(f"  ... and {len(self.files_to_delete) - limit} more files")
    
    def create_backup_dir(self):
        """Create backup directory if specified"""
        if self.backup_dir:
            os.makedirs(self.backup_dir, exist_ok=True)
            print(f"📁 Backup directory created: {self.backup_dir}")
    
    def execute_cleanup(self):
        """Execute the cleanup operation"""
        if not self.files_to_delete:
            print("✅ No files to delete.")
            return
        
        if self.dry_run:
            print(f"\n🔍 DRY RUN - No files will be deleted")
            self.preview_deletions()
            return
        
        print(f"\n⚠️  CLEANUP OPERATION")
        print(f"This will {'move' if self.backup_dir else 'DELETE'} {len(self.files_to_delete)} files ({self.format_size(self.stats['size_to_delete'])})")
        
        # Final confirmation
        if not self.force:
            response = input("Are you sure you want to proceed? (type 'yes' to continue): ")
            if response.lower() != 'yes':
                print("❌ Operation cancelled.")
                return
        else:
            print("🚀 Force flag enabled - proceeding automatically...")
        
        # Create backup directory if needed
        if self.backup_dir:
            self.create_backup_dir()
        
        # Create deletion manifest
        manifest_file = f"cleanup_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        manifest_data = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'move' if self.backup_dir else 'delete',
            'backup_dir': self.backup_dir,
            'stats': self.stats,
            'deleted_files': []
        }
        
        print(f"\n🗑️  Processing {len(self.files_to_delete)} files...")
        
        successful_operations = 0
        failed_operations = 0
        
        for i, (file_path, strategy, size, rejections) in enumerate(self.files_to_delete):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(self.files_to_delete)} files...")
            
            try:
                if self.backup_dir:
                    # Move to backup
                    backup_path = os.path.join(self.backup_dir, os.path.basename(file_path))
                    shutil.move(file_path, backup_path)
                    operation = 'moved'
                else:
                    # Delete file
                    os.remove(file_path)
                    operation = 'deleted'
                
                manifest_data['deleted_files'].append({
                    'original_path': file_path,
                    'strategy': strategy,
                    'size': size,
                    'rejections': rejections,
                    'operation': operation,
                    'backup_path': os.path.join(self.backup_dir, os.path.basename(file_path)) if self.backup_dir else None
                })
                
                successful_operations += 1
                
            except Exception as e:
                print(f"  ❌ Failed to process {file_path}: {e}")
                failed_operations += 1
        
        # Save manifest
        with open(manifest_file, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        print(f"\n✅ CLEANUP COMPLETED")
        print(f"  Files processed:      {successful_operations:,}")
        print(f"  Failed operations:    {failed_operations:,}")
        print(f"  Space {'freed' if not self.backup_dir else 'moved'}: {self.format_size(self.stats['size_to_delete'])}")
        print(f"  Manifest saved:       {manifest_file}")
        
        if self.backup_dir:
            print(f"  Backup location:      {self.backup_dir}")


def main():
    parser = argparse.ArgumentParser(description='Clean up underperforming game logs')
    parser.add_argument('--execute', action='store_true', 
                       help='Execute cleanup (default is dry-run)')
    parser.add_argument('--backup-dir', type=str, default=None,
                       help='Move files to backup directory instead of deleting')
    parser.add_argument('--preview-limit', type=int, default=20,
                       help='Number of files to show in preview (default: 20)')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompt when executing cleanup')
    
    args = parser.parse_args()
    
    print("🧹 Game Logs Cleanup Tool")
    print("="*50)
    
    # Initialize cleanup
    cleanup = GameLogsCleanup(dry_run=not args.execute, backup_dir=args.backup_dir, force=args.force)
    
    # Analyze files
    cleanup.analyze_files()
    
    # Show preview
    cleanup.preview_deletions(args.preview_limit)
    
    # Execute cleanup
    cleanup.execute_cleanup()
    
    if cleanup.dry_run:
        print(f"\n💡 To execute cleanup:")
        if args.backup_dir:
            print(f"   python cleanup_game_logs.py --execute --backup-dir {args.backup_dir}")
        else:
            print(f"   python cleanup_game_logs.py --execute")
        print(f"   python cleanup_game_logs.py --execute --backup-dir backup_logs  # Move instead of delete")


if __name__ == '__main__':
    main()