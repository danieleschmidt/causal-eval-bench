#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers and prioritizes high-value work items using advanced scoring algorithms.
"""

import json
import os
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValueItem:
    """Represents a discovered value item with comprehensive scoring."""
    id: str
    title: str
    description: str
    category: str
    source: str
    difficulty: str
    estimated_effort: float
    priority: str
    
    # Scoring components
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    
    # Metadata
    discovered_date: str
    file_paths: List[str]
    dependencies: List[str]
    tags: List[str]
    
    # Value metrics
    business_value: int
    time_criticality: int
    risk_reduction: int
    opportunity_enablement: int
    impact: int
    confidence: int
    ease: int

class ValueDiscoverySource(ABC):
    """Abstract base class for value discovery sources."""
    
    @abstractmethod
    def discover(self) -> List[ValueItem]:
        """Discover value items from this source."""
        pass

class GitHistoryAnalyzer(ValueDiscoverySource):
    """Analyzes git history for TODO comments, technical debt markers, and improvement opportunities."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        
    def discover(self) -> List[ValueItem]:
        """Discover value items from git history and code comments."""
        items = []
        
        # Find TODO/FIXME/HACK comments
        todo_items = self._find_todo_comments()
        items.extend(todo_items)
        
        # Analyze commit messages for quick fixes
        quick_fix_items = self._find_quick_fixes()
        items.extend(quick_fix_items)
        
        # Identify frequently changed files (hot spots)
        hotspot_items = self._find_hotspots()
        items.extend(hotspot_items)
        
        return items
    
    def _find_todo_comments(self) -> List[ValueItem]:
        """Find TODO, FIXME, HACK comments in codebase."""
        items = []
        patterns = [
            (r'TODO:?\s*(.*)', 'todo', 'medium', 2.0),
            (r'FIXME:?\s*(.*)', 'bug_fix', 'high', 1.5),
            (r'HACK:?\s*(.*)', 'technical_debt', 'high', 3.0),
            (r'XXX:?\s*(.*)', 'technical_debt', 'medium', 2.5),
            (r'DEPRECATED:?\s*(.*)', 'modernization', 'low', 4.0)
        ]
        
        try:
            # Use ripgrep for fast searching
            for pattern, category, priority, effort in patterns:
                result = subprocess.run(
                    ['rg', '-n', pattern, str(self.repo_path)],
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if ':' in line:
                            file_path, line_no, comment = line.split(':', 2)
                            match = re.search(pattern, comment, re.IGNORECASE)
                            if match:
                                description = match.group(1).strip() if match.group(1) else comment.strip()
                                
                                item = ValueItem(
                                    id=f"GIT-{category.upper()}-{len(items)+1:03d}",
                                    title=f"{category.replace('_', ' ').title()}: {description[:50]}...",
                                    description=description,
                                    category=category,
                                    source="git_history",
                                    difficulty=priority,
                                    estimated_effort=effort,
                                    priority=priority,
                                    wsjf_score=self._calculate_wsjf(category, priority),
                                    ice_score=self._calculate_ice(category, priority, effort),
                                    technical_debt_score=self._calculate_debt(category),
                                    composite_score=0.0,  # Will be calculated later
                                    discovered_date=datetime.now().isoformat(),
                                    file_paths=[file_path],
                                    dependencies=[],
                                    tags=[category, priority, "git_analysis"],
                                    business_value=self._estimate_business_value(category),
                                    time_criticality=self._estimate_time_criticality(priority),
                                    risk_reduction=self._estimate_risk_reduction(category),
                                    opportunity_enablement=self._estimate_opportunity(category),
                                    impact=self._estimate_impact(category, priority),
                                    confidence=self._estimate_confidence(category),
                                    ease=self._estimate_ease(effort)
                                )
                                items.append(item)
        except subprocess.TimeoutExpired:
            logger.warning("Git history analysis timed out")
        except Exception as e:
            logger.error(f"Error analyzing git history: {e}")
            
        return items
    
    def _find_quick_fixes(self) -> List[ValueItem]:
        """Analyze commit messages for 'quick fix', 'temporary', 'hack' indicators."""
        items = []
        try:
            result = subprocess.run(
                ['git', 'log', '--oneline', '--since=3 months ago', '--grep=quick', '--grep=temp', '--grep=hack'],
                cwd=self.repo_path, capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().split('\n')[:10]):  # Limit to 10 recent
                    if line.strip():
                        commit_hash, message = line.split(' ', 1)
                        item = ValueItem(
                            id=f"GIT-QUICKFIX-{i+1:03d}",
                            title=f"Review quick fix: {message[:50]}...",
                            description=f"Commit {commit_hash}: {message}. Quick fixes often indicate technical debt.",
                            category="technical_debt",
                            source="git_commits",
                            difficulty="medium",
                            estimated_effort=1.5,
                            priority="medium",
                            wsjf_score=15.0,
                            ice_score=120,
                            technical_debt_score=35,
                            composite_score=0.0,
                            discovered_date=datetime.now().isoformat(),
                            file_paths=[],
                            dependencies=[],
                            tags=["quick_fix", "technical_debt", "commit_analysis"],
                            business_value=4,
                            time_criticality=3,
                            risk_reduction=6,
                            opportunity_enablement=3,
                            impact=5,
                            confidence=7,
                            ease=8
                        )
                        items.append(item)
        except Exception as e:
            logger.error(f"Error analyzing commit messages: {e}")
            
        return items
    
    def _find_hotspots(self) -> List[ValueItem]:
        """Identify frequently changed files that might need refactoring."""
        items = []
        try:
            result = subprocess.run([
                'git', 'log', '--name-only', '--pretty=format:', '--since=6 months ago'
            ], cwd=self.repo_path, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Count file changes
                file_changes = {}
                for line in result.stdout.strip().split('\n'):
                    if line.strip() and not line.startswith('commit'):
                        file_changes[line] = file_changes.get(line, 0) + 1
                
                # Find top 5 most changed files
                hot_files = sorted(file_changes.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for i, (file_path, change_count) in enumerate(hot_files):
                    if change_count > 5:  # Only consider files changed more than 5 times
                        item = ValueItem(
                            id=f"GIT-HOTSPOT-{i+1:03d}",
                            title=f"Refactor hotspot: {Path(file_path).name}",
                            description=f"File {file_path} changed {change_count} times in 6 months. Consider refactoring.",
                            category="refactoring",
                            source="git_hotspots",
                            difficulty="high",
                            estimated_effort=min(change_count * 0.5, 8.0),
                            priority="medium",
                            wsjf_score=20.0,
                            ice_score=150,
                            technical_debt_score=change_count * 5,
                            composite_score=0.0,
                            discovered_date=datetime.now().isoformat(),
                            file_paths=[file_path],
                            dependencies=[],
                            tags=["hotspot", "refactoring", "high_churn"],
                            business_value=6,
                            time_criticality=4,
                            risk_reduction=8,
                            opportunity_enablement=5,
                            impact=7,
                            confidence=8,
                            ease=3
                        )
                        items.append(item)
        except Exception as e:
            logger.error(f"Error analyzing hotspots: {e}")
            
        return items
    
    def _calculate_wsjf(self, category: str, priority: str) -> float:
        """Calculate Weighted Shortest Job First score."""
        business_value = {"security": 9, "bug_fix": 8, "performance": 7, "technical_debt": 5, "feature": 6}.get(category, 5)
        time_criticality = {"high": 8, "medium": 5, "low": 2}.get(priority, 5)
        risk_reduction = {"security": 9, "bug_fix": 7, "technical_debt": 6}.get(category, 3)
        
        cost_of_delay = business_value + time_criticality + risk_reduction
        return cost_of_delay * 1.5  # Simplified WSJF
    
    def _calculate_ice(self, category: str, priority: str, effort: float) -> float:
        """Calculate Impact, Confidence, Ease score."""
        impact = {"security": 9, "bug_fix": 8, "performance": 7, "technical_debt": 5}.get(category, 5)
        confidence = {"high": 9, "medium": 7, "low": 5}.get(priority, 7)
        ease = max(1, 10 - int(effort * 2))  # Higher effort = lower ease
        
        return impact * confidence * ease
    
    def _calculate_debt(self, category: str) -> float:
        """Calculate technical debt score."""
        debt_weights = {
            "technical_debt": 50,
            "refactoring": 40,
            "bug_fix": 30,
            "security": 35,
            "performance": 25,
            "todo": 15
        }
        return debt_weights.get(category, 20)
    
    def _estimate_business_value(self, category: str) -> int:
        return {"security": 9, "bug_fix": 8, "performance": 7, "technical_debt": 5}.get(category, 5)
    
    def _estimate_time_criticality(self, priority: str) -> int:
        return {"high": 8, "medium": 5, "low": 2}.get(priority, 5)
    
    def _estimate_risk_reduction(self, category: str) -> int:
        return {"security": 9, "bug_fix": 7, "technical_debt": 6}.get(category, 3)
    
    def _estimate_opportunity(self, category: str) -> int:
        return {"performance": 8, "feature": 7, "refactoring": 5}.get(category, 4)
    
    def _estimate_impact(self, category: str, priority: str) -> int:
        base = {"security": 9, "bug_fix": 8, "performance": 7}.get(category, 5)
        modifier = {"high": 1.2, "medium": 1.0, "low": 0.8}.get(priority, 1.0)
        return int(base * modifier)
    
    def _estimate_confidence(self, category: str) -> int:
        return {"bug_fix": 9, "security": 8, "technical_debt": 7}.get(category, 6)
    
    def _estimate_ease(self, effort: float) -> int:
        return max(1, 10 - int(effort))

class SecurityScanner(ValueDiscoverySource):
    """Scans for security vulnerabilities and compliance issues."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        
    def discover(self) -> List[ValueItem]:
        """Discover security-related value items."""
        items = []
        
        # Check for missing security files
        security_files = [
            ("SECURITY.md", "Create security policy", "security_policy"),
            (".github/dependabot.yml", "Setup automated dependency updates", "dependency_management"),
            (".secrets.baseline", "Initialize secrets detection baseline", "secrets_management")
        ]
        
        for filename, title, category in security_files:
            file_path = self.repo_path / filename
            if not file_path.exists():
                item = ValueItem(
                    id=f"SEC-{category.upper().replace('_', '')}-001",
                    title=title,
                    description=f"Missing {filename} - important for security best practices",
                    category="security",
                    source="security_scan",
                    difficulty="low",
                    estimated_effort=0.5,
                    priority="high",
                    wsjf_score=45.0,
                    ice_score=360,
                    technical_debt_score=25,
                    composite_score=0.0,
                    discovered_date=datetime.now().isoformat(),
                    file_paths=[str(file_path)],
                    dependencies=[],
                    tags=["security", "compliance", "missing_file"],
                    business_value=8,
                    time_criticality=7,
                    risk_reduction=9,
                    opportunity_enablement=6,
                    impact=8,
                    confidence=9,
                    ease=9
                )
                items.append(item)
        
        return items

class TestCoverageAnalyzer(ValueDiscoverySource):
    """Analyzes test coverage gaps and testing improvements."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        
    def discover(self) -> List[ValueItem]:
        """Discover testing-related value items."""
        items = []
        
        # Check for missing test files
        source_files = list(Path(self.repo_path).rglob("*.py"))
        test_files = list(Path(self.repo_path).rglob("test_*.py"))
        
        # Simple heuristic: if we have more than 10 source files but fewer than 5 test files
        if len(source_files) > 10 and len(test_files) < 5:
            item = ValueItem(
                id="TEST-COVERAGE-001",
                title="Improve test coverage",
                description=f"Found {len(source_files)} source files but only {len(test_files)} test files. Increase test coverage.",
                category="testing",
                source="test_analysis",
                difficulty="medium",
                estimated_effort=5.0,
                priority="medium",
                wsjf_score=25.0,
                ice_score=210,
                technical_debt_score=40,
                composite_score=0.0,
                discovered_date=datetime.now().isoformat(),
                file_paths=[],
                dependencies=[],
                tags=["testing", "coverage", "quality"],
                business_value=6,
                time_criticality=4,
                risk_reduction=7,
                opportunity_enablement=5,
                impact=6,
                confidence=8,
                ease=5
            )
            items.append(item)
            
        return items

class ValueDiscoveryEngine:
    """Main engine for discovering and prioritizing value items."""
    
    def __init__(self, repo_path: str, config_path: Optional[str] = None):
        self.repo_path = Path(repo_path)
        self.config = self._load_config(config_path)
        self.sources = [
            GitHistoryAnalyzer(str(self.repo_path)),
            SecurityScanner(str(self.repo_path)),
            TestCoverageAnalyzer(str(self.repo_path))
        ]
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path) as f:
                    return yaml.safe_load(f)
            except ImportError:
                logger.warning("PyYAML not available, using default config")
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using defaults")
        
        # Default configuration for maturing repositories
        return {
            "scoring": {
                "weights": {
                    "wsjf": 0.6,
                    "ice": 0.1,
                    "technicalDebt": 0.2,
                    "security": 0.1
                },
                "thresholds": {
                    "minScore": 15,
                    "maxRisk": 0.7,
                    "securityBoost": 2.0
                }
            }
        }
    
    def discover_all(self) -> List[ValueItem]:
        """Discover value items from all sources."""
        all_items = []
        
        for source in self.sources:
            try:
                items = source.discover()
                all_items.extend(items)
                logger.info(f"Discovered {len(items)} items from {source.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error in {source.__class__.__name__}: {e}")
        
        # Calculate composite scores
        for item in all_items:
            item.composite_score = self._calculate_composite_score(item)
        
        # Sort by composite score (descending)
        all_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        logger.info(f"Total discovered items: {len(all_items)}")
        return all_items
    
    def _calculate_composite_score(self, item: ValueItem) -> float:
        """Calculate composite score using adaptive weights."""
        weights = self.config["scoring"]["weights"]
        
        normalized_wsjf = min(item.wsjf_score / 50.0, 1.0)  # Normalize to 0-1
        normalized_ice = min(item.ice_score / 500.0, 1.0)
        normalized_debt = min(item.technical_debt_score / 100.0, 1.0)
        
        composite = (
            weights["wsjf"] * normalized_wsjf +
            weights["ice"] * normalized_ice +
            weights["technicalDebt"] * normalized_debt
        )
        
        # Security boost
        if item.category == "security":
            composite *= self.config["scoring"]["thresholds"]["securityBoost"]
        
        return composite * 100  # Scale to 0-100
    
    def save_metrics(self, items: List[ValueItem], output_path: str):
        """Save discovery metrics and backlog."""
        metrics = {
            "discovery_timestamp": datetime.now().isoformat(),
            "total_items": len(items),
            "items_by_category": {},
            "items_by_priority": {},
            "items_by_source": {},
            "top_items": []
        }
        
        # Calculate category distributions
        for item in items:
            metrics["items_by_category"][item.category] = metrics["items_by_category"].get(item.category, 0) + 1
            metrics["items_by_priority"][item.priority] = metrics["items_by_priority"].get(item.priority, 0) + 1
            metrics["items_by_source"][item.source] = metrics["items_by_source"].get(item.source, 0) + 1
        
        # Top 10 items
        metrics["top_items"] = [asdict(item) for item in items[:10]]
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")

def main():
    """Main entry point for value discovery."""
    repo_path = os.getcwd()
    config_path = os.path.join(repo_path, ".terragon", "config.yaml")
    
    engine = ValueDiscoveryEngine(repo_path, config_path)
    items = engine.discover_all()
    
    # Save results
    terragon_dir = Path(repo_path) / ".terragon"
    terragon_dir.mkdir(exist_ok=True)
    
    metrics_path = terragon_dir / "latest-discovery.json"
    engine.save_metrics(items, str(metrics_path))
    
    # Print summary
    print(f"\n🔍 Value Discovery Complete")
    print(f"📊 Total Items Discovered: {len(items)}")
    print(f"🎯 Top Value Items:")
    
    for i, item in enumerate(items[:5], 1):
        print(f"  {i}. {item.title} (Score: {item.composite_score:.1f})")
    
    print(f"\n📁 Full results saved to: {metrics_path}")
    
    return items

if __name__ == "__main__":
    main()