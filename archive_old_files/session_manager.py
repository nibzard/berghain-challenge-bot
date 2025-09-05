# ABOUTME: Session manager for coordinating parallel Berghain game execution
# ABOUTME: Manages bot processes, queues, resource allocation, and crash recovery

import time
import json
import subprocess
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, asdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, Future
import signal
import os
from enum import Enum

class SessionType(Enum):
    SINGLE_BOT_TEST = "single_bot_test"
    BOT_COMPARISON = "bot_comparison" 
    MARATHON = "marathon"
    STRESS_TEST = "stress_test"

class GameJobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class GameJob:
    job_id: str
    bot_type: str
    scenario: int
    run_number: int
    status: GameJobStatus = GameJobStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    process_id: Optional[int] = None
    result_file: Optional[str] = None
    error_message: Optional[str] = None
    timeout_seconds: int = 1800  # 30 minutes default
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None
        
    @property
    def is_active(self) -> bool:
        return self.status in [GameJobStatus.PENDING, GameJobStatus.RUNNING]

@dataclass
class SessionConfig:
    session_type: SessionType
    name: str
    bots: List[str]
    scenarios: List[int]
    runs_per_combination: int
    max_concurrent: int = 4
    timeout_minutes: int = 30
    output_dir: str = "game_logs"
    retry_failed: bool = True
    max_retries: int = 2
    
    def generate_jobs(self) -> List[GameJob]:
        """Generate all game jobs for this session"""
        jobs = []
        job_counter = 0
        
        for bot in self.bots:
            for scenario in self.scenarios:
                for run in range(self.runs_per_combination):
                    job_id = f"{self.name}_{bot}_s{scenario}_r{run:02d}"
                    jobs.append(GameJob(
                        job_id=job_id,
                        bot_type=bot,
                        scenario=scenario,
                        run_number=run,
                        timeout_seconds=self.timeout_minutes * 60
                    ))
                    job_counter += 1
                    
        return jobs
        
    @property
    def total_games(self) -> int:
        return len(self.bots) * len(self.scenarios) * self.runs_per_combination

def run_bot_game(bot_type: str, scenario: int, job_id: str, output_dir: str, timeout: int) -> Dict[str, Any]:
    """
    Execute a single bot game in a separate process
    Returns result dictionary with success/failure info
    """
    try:
        # Determine which bot script to run
        bot_scripts = {
            "optimal": "berghain_bot_optimal.py",
            "dynamic": "berghain_bot.py", 
            "simple": "berghain_bot_simple.py",
            "data_collector": "data_collector.py"
        }
        
        script = bot_scripts.get(bot_type, "berghain_bot.py")
        
        # Prepare command
        cmd = [
            "python", 
            script,
            "--scenario", str(scenario),
            "--job-id", job_id
        ]
        
        # Run the process
        start_time = datetime.now()
        process = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Check for result file
        result_files = list(Path(output_dir).glob(f"*{job_id}*.json"))
        result_file = str(result_files[0]) if result_files else None
        
        return {
            "success": process.returncode == 0,
            "duration": duration,
            "result_file": result_file,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "return_code": process.returncode,
            "error": None
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration": timeout,
            "result_file": None,
            "stdout": "",
            "stderr": f"Process timed out after {timeout} seconds",
            "return_code": -1,
            "error": "TIMEOUT"
        }
        
    except Exception as e:
        return {
            "success": False,
            "duration": 0,
            "result_file": None,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
            "error": str(e)
        }

class SessionManager:
    """Manages parallel execution of multiple bot games"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.executor: Optional[ProcessPoolExecutor] = None
        self.current_session: Optional[SessionConfig] = None
        self.jobs: List[GameJob] = []
        self.active_futures: Dict[str, Future] = {}
        self.completed_jobs: List[GameJob] = []
        self.failed_jobs: List[GameJob] = []
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Statistics
        self.stats = {
            "total_jobs": 0,
            "completed": 0,
            "failed": 0,
            "success_rate": 0.0,
            "average_duration": 0.0,
            "total_runtime": 0.0
        }
        
    def load_session_config(self, config_file: str) -> SessionConfig:
        """Load session configuration from JSON file"""
        with open(config_file, 'r') as f:
            data = json.load(f)
            
        return SessionConfig(
            session_type=SessionType(data["session_type"]),
            name=data["name"],
            bots=data["bots"],
            scenarios=data["scenarios"], 
            runs_per_combination=data["runs_per_combination"],
            max_concurrent=data.get("max_concurrent", 4),
            timeout_minutes=data.get("timeout_minutes", 30),
            output_dir=data.get("output_dir", "game_logs"),
            retry_failed=data.get("retry_failed", True),
            max_retries=data.get("max_retries", 2)
        )
        
    def create_session_from_dict(self, config_dict: Dict[str, Any]) -> SessionConfig:
        """Create session config from dictionary"""
        return SessionConfig(
            session_type=SessionType(config_dict["session_type"]),
            name=config_dict["name"],
            bots=config_dict["bots"],
            scenarios=config_dict["scenarios"],
            runs_per_combination=config_dict["runs_per_combination"],
            max_concurrent=config_dict.get("max_concurrent", 4),
            timeout_minutes=config_dict.get("timeout_minutes", 30),
            output_dir=config_dict.get("output_dir", "game_logs"),
            retry_failed=config_dict.get("retry_failed", True),
            max_retries=config_dict.get("max_retries", 2)
        )
        
    def start_session(self, config: SessionConfig) -> bool:
        """Start a new game session"""
        if self.is_running:
            print("❌ Session already running")
            return False
            
        self.current_session = config
        self.jobs = config.generate_jobs()
        self.completed_jobs = []
        self.failed_jobs = []
        self.active_futures = {}
        
        # Create output directory
        Path(config.output_dir).mkdir(exist_ok=True)
        
        # Initialize executor
        self.executor = ProcessPoolExecutor(max_workers=min(config.max_concurrent, self.max_workers))
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Reset stats
        self.stats = {
            "total_jobs": len(self.jobs),
            "completed": 0,
            "failed": 0,
            "success_rate": 0.0,
            "average_duration": 0.0,
            "total_runtime": 0.0
        }
        
        print(f"🚀 Starting session: {config.name}")
        print(f"   Jobs: {len(self.jobs)}")
        print(f"   Max concurrent: {config.max_concurrent}")
        print(f"   Timeout: {config.timeout_minutes}m per game")
        print(f"   Output: {config.output_dir}")
        
        # Start job execution thread
        self.execution_thread = threading.Thread(target=self._execute_jobs, daemon=True)
        self.execution_thread.start()
        
        return True
        
    def _execute_jobs(self):
        """Main job execution loop"""
        while self.is_running and (self.jobs or self.active_futures):
            # Submit new jobs up to max concurrent limit
            while (len(self.active_futures) < self.current_session.max_concurrent and 
                   self.jobs and self.is_running):
                
                job = self.jobs.pop(0)
                job.status = GameJobStatus.RUNNING
                job.start_time = datetime.now()
                
                # Submit to executor
                future = self.executor.submit(
                    run_bot_game,
                    job.bot_type,
                    job.scenario,
                    job.job_id,
                    self.current_session.output_dir,
                    job.timeout_seconds
                )
                
                self.active_futures[job.job_id] = future
                job.process_id = future.running()  # This might not work as expected
                
                print(f"🎯 Started: {job.job_id} ({job.bot_type} S{job.scenario})")
                
            # Check for completed jobs
            completed_job_ids = []
            for job_id, future in self.active_futures.items():
                if future.done():
                    completed_job_ids.append(job_id)
                    
            # Process completed jobs
            for job_id in completed_job_ids:
                future = self.active_futures.pop(job_id)
                job = self._find_job_by_id(job_id)
                
                if job:
                    try:
                        result = future.result()
                        job.end_time = datetime.now()
                        
                        if result["success"]:
                            job.status = GameJobStatus.COMPLETED
                            job.result_file = result["result_file"]
                            self.completed_jobs.append(job)
                            print(f"✅ Completed: {job.job_id} ({job.duration})")
                        else:
                            job.status = GameJobStatus.FAILED
                            job.error_message = result.get("error", "Unknown error")
                            self.failed_jobs.append(job)
                            print(f"❌ Failed: {job.job_id} - {job.error_message}")
                            
                    except Exception as e:
                        job.status = GameJobStatus.FAILED
                        job.error_message = str(e)
                        job.end_time = datetime.now()
                        self.failed_jobs.append(job)
                        print(f"💥 Exception: {job.job_id} - {e}")
                        
            # Update statistics
            self._update_stats()
            
            # Brief sleep to avoid busy waiting
            time.sleep(1)
            
        print("🏁 Session execution completed")
        
    def _find_job_by_id(self, job_id: str) -> Optional[GameJob]:
        """Find job by ID in all job lists"""
        # Check active futures first (most likely)
        for job in [*self.completed_jobs, *self.failed_jobs]:
            if job.job_id == job_id:
                return job
                
        # Check pending jobs
        for job in self.jobs:
            if job.job_id == job_id:
                return job
                
        return None
        
    def _update_stats(self):
        """Update session statistics"""
        total_completed = len(self.completed_jobs) + len(self.failed_jobs)
        
        self.stats.update({
            "completed": len(self.completed_jobs),
            "failed": len(self.failed_jobs),
            "success_rate": (len(self.completed_jobs) / max(total_completed, 1)) * 100,
            "total_runtime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        })
        
        # Calculate average duration
        completed_with_duration = [j for j in self.completed_jobs if j.duration]
        if completed_with_duration:
            avg_duration = sum(j.duration.total_seconds() for j in completed_with_duration) / len(completed_with_duration)
            self.stats["average_duration"] = avg_duration
            
    def stop_session(self):
        """Stop the current session"""
        if not self.is_running:
            return
            
        print("⏹️  Stopping session...")
        self.is_running = False
        
        # Cancel running jobs
        for job_id, future in self.active_futures.items():
            future.cancel()
            job = self._find_job_by_id(job_id)
            if job:
                job.status = GameJobStatus.CANCELLED
                
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=False)
            
        print("Session stopped")
        
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status"""
        if not self.current_session:
            return {"status": "no_session"}
            
        return {
            "status": "running" if self.is_running else "stopped",
            "session_name": self.current_session.name,
            "session_type": self.current_session.session_type.value,
            "stats": self.stats,
            "active_jobs": len(self.active_futures),
            "pending_jobs": len(self.jobs),
            "total_runtime": self.stats["total_runtime"]
        }
        
    def get_job_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all jobs"""
        all_jobs = [*self.completed_jobs, *self.failed_jobs, *self.jobs]
        
        # Add currently running jobs
        for job_id in self.active_futures.keys():
            job = self._find_job_by_id(job_id)
            if job and job not in all_jobs:
                all_jobs.append(job)
                
        return [asdict(job) for job in all_jobs]
        
    def save_session_report(self, output_file: str):
        """Save detailed session report"""
        report = {
            "session_config": asdict(self.current_session) if self.current_session else None,
            "session_stats": self.stats,
            "jobs": self.get_job_summary(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"📊 Session report saved: {output_file}")

# Predefined session configurations
PRESET_SESSIONS = {
    "quick_test": {
        "session_type": "single_bot_test",
        "name": "quick_test",
        "bots": ["optimal"],
        "scenarios": [1],
        "runs_per_combination": 3,
        "max_concurrent": 2,
        "timeout_minutes": 15
    },
    
    "bot_comparison": {
        "session_type": "bot_comparison",
        "name": "bot_comparison",
        "bots": ["optimal", "dynamic", "simple"],
        "scenarios": [1, 2, 3],
        "runs_per_combination": 5,
        "max_concurrent": 4,
        "timeout_minutes": 30
    },
    
    "marathon_s1": {
        "session_type": "marathon",
        "name": "marathon_scenario_1",
        "bots": ["optimal"],
        "scenarios": [1],
        "runs_per_combination": 100,
        "max_concurrent": 6,
        "timeout_minutes": 45
    },
    
    "stress_test": {
        "session_type": "stress_test", 
        "name": "stress_test_all",
        "bots": ["optimal", "dynamic", "simple"],
        "scenarios": [1, 2, 3],
        "runs_per_combination": 20,
        "max_concurrent": 8,
        "timeout_minutes": 60
    }
}

def main():
    """CLI interface for session manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Berghain Game Session Manager')
    parser.add_argument('--preset', choices=PRESET_SESSIONS.keys(), help='Use a preset session')
    parser.add_argument('--config', help='Load custom session config from JSON file')
    parser.add_argument('--list-presets', action='store_true', help='List available presets')
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available preset sessions:")
        for name, config in PRESET_SESSIONS.items():
            print(f"  {name}: {config['session_type']} - {len(config['bots'])} bots, "
                  f"{len(config['scenarios'])} scenarios, {config['runs_per_combination']} runs each")
        return
        
    manager = SessionManager()
    
    try:
        if args.preset:
            config_dict = PRESET_SESSIONS[args.preset]
            config = manager.create_session_from_dict(config_dict)
        elif args.config:
            config = manager.load_session_config(args.config)
        else:
            # Default quick test
            config_dict = PRESET_SESSIONS["quick_test"]
            config = manager.create_session_from_dict(config_dict)
            
        # Start session
        if manager.start_session(config):
            # Monitor progress
            while manager.is_running:
                time.sleep(5)
                status = manager.get_session_status()
                print(f"⏳ Progress: {status['stats']['completed']}/{status['stats']['total_jobs']} "
                      f"({status['stats']['success_rate']:.1f}% success) - "
                      f"{status['active_jobs']} active, {status['pending_jobs']} pending")
                      
            # Final report
            final_status = manager.get_session_status()
            print(f"\n🏆 Session completed!")
            print(f"   Total games: {final_status['stats']['total_jobs']}")
            print(f"   Successful: {final_status['stats']['completed']}")
            print(f"   Failed: {final_status['stats']['failed']}")
            print(f"   Success rate: {final_status['stats']['success_rate']:.1f}%")
            print(f"   Total runtime: {final_status['total_runtime']:.1f}s")
            
            # Save report
            report_file = f"session_report_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            manager.save_session_report(report_file)
            
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        manager.stop_session()

if __name__ == "__main__":
    main()