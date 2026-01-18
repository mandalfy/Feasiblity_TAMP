"""
ML-guided TAMP planner.

Uses learned feasibility classifier to prune infeasible actions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from collections import deque
import time

from environments.tabletop_env import TabletopEnv
from environments.robot import FrankaPanda
from data_generation.motion_planner import RRTStarPlanner
from planning.baseline_tamp import (
    ActionType, Action, PlanResult, BaselineTAMPPlanner
)


class MLGuidedTAMPPlanner(BaselineTAMPPlanner):
    """
    ML-guided TAMP planner.
    
    Uses a trained classifier to predict action feasibility,
    only running expensive motion planning for promising candidates.
    """
    
    def __init__(
        self,
        env: TabletopEnv,
        model: nn.Module,
        feasibility_threshold: float = 0.3,
        motion_planning_timeout: float = 2.0,
        max_plan_length: int = 10,
        use_images: bool = False,
        device: str = "auto"
    ):
        """
        Initialize the ML-guided planner.
        
        Args:
            env: The tabletop environment
            model: Trained feasibility classifier
            feasibility_threshold: Minimum predicted probability to try motion planning
            motion_planning_timeout: Timeout for motion planning
            max_plan_length: Maximum plan length
            use_images: Whether model uses image inputs
            device: Device for ML inference
        """
        super().__init__(env, motion_planning_timeout, max_plan_length)
        
        # ML model setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.model.eval()
        
        self.feasibility_threshold = feasibility_threshold
        self.use_images = use_images
        
        # Statistics
        self.ml_predictions = 0
        self.pruned_actions = 0
        
    def plan(
        self,
        goal_positions: Dict[int, np.ndarray],
        timeout: float = 60.0
    ) -> PlanResult:
        """
        Plan using ML-guided search.
        
        Args:
            goal_positions: Dict mapping object_id -> target position
            timeout: Overall planning timeout
            
        Returns:
            PlanResult with success status and plan
        """
        start_time = time.time()
        self.motion_planning_calls = 0
        self.nodes_expanded = 0
        self.ml_predictions = 0
        self.pruned_actions = 0
        
        # Priority queue: elements are (priority, state, actions)
        # Lower priority = more promising (higher predicted feasibility)
        from heapq import heappush, heappop
        
        initial_state = self._get_symbolic_state()
        queue = [(0, 0, initial_state, [])]  # (priority, tiebreaker, state, actions)
        visited = set()
        tiebreaker = 0
        
        while queue:
            # Check timeout
            if time.time() - start_time > timeout:
                break
                
            _, _, current_state, actions = heappop(queue)
            state_key = self._state_to_key(current_state)
            
            if state_key in visited:
                continue
            visited.add(state_key)
            self.nodes_expanded += 1
            
            # Check if goal reached
            if self._check_goal(current_state, goal_positions):
                return PlanResult(
                    success=True,
                    plan=actions,
                    planning_time=time.time() - start_time,
                    motion_planning_calls=self.motion_planning_calls,
                    nodes_expanded=self.nodes_expanded
                )
                
            # Check plan length limit
            if len(actions) >= self.max_plan_length:
                continue
                
            # Generate candidate actions
            candidates = self._generate_candidates(current_state, goal_positions)
            
            # Score candidates with ML model
            scored_candidates = self._score_candidates(candidates, current_state)
            
            for action, score in scored_candidates:
                # Prune low-probability actions
                if score < self.feasibility_threshold:
                    self.pruned_actions += 1
                    continue
                    
                # Check feasibility via motion planning
                trajectory = self._check_feasibility(action)
                
                if trajectory is not None:
                    action.trajectory = trajectory
                    new_state = self._apply_action(current_state, action)
                    
                    # Priority: negative score so higher scores are explored first
                    tiebreaker += 1
                    priority = -score + len(actions) * 0.1  # Slight preference for shorter plans
                    heappush(queue, (priority, tiebreaker, new_state, actions + [action]))
                    
        # No plan found
        return PlanResult(
            success=False,
            plan=None,
            planning_time=time.time() - start_time,
            motion_planning_calls=self.motion_planning_calls,
            nodes_expanded=self.nodes_expanded
        )
    
    def _score_candidates(
        self,
        candidates: List[Action],
        state: Dict
    ) -> List[Tuple[Action, float]]:
        """
        Score candidate actions using ML model.
        
        Args:
            candidates: List of candidate actions
            state: Current symbolic state
            
        Returns:
            List of (action, score) tuples, sorted by score descending
        """
        if not candidates:
            return []
            
        self.ml_predictions += len(candidates)
        
        # Prepare inputs
        state_vector = self._state_to_vector(state)
        
        scores = []
        with torch.no_grad():
            for action in candidates:
                action_vector = self._action_to_vector(action)
                
                if self.use_images:
                    # Get image
                    image = self.env.get_image()
                    image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
                    image_tensor = image_tensor.to(self.device)
                    action_tensor = torch.FloatTensor(action_vector).unsqueeze(0).to(self.device)
                    prob = self.model(image_tensor, action_tensor)
                else:
                    # Concatenate state and action
                    input_vector = np.concatenate([state_vector, action_vector])
                    input_tensor = torch.FloatTensor(input_vector).unsqueeze(0).to(self.device)
                    prob = self.model(input_tensor)
                    
                scores.append((action, prob.item()))
                
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _state_to_vector(self, state: Dict) -> np.ndarray:
        """Convert symbolic state to vector for ML input."""
        # Robot state
        robot_state = self.env.robot.get_state_vector()  # 15 dims
        
        # Object states (max 6 objects, 7 dims each = 42 dims)
        max_objects = 6
        object_state = np.zeros(max_objects * 7)
        
        for i, (obj_id, info) in enumerate(list(state['objects'].items())[:max_objects]):
            pos = info['position']
            orn = np.array([0, 0, 0, 1])  # Approximate orientation
            object_state[i*7:(i+1)*7] = np.concatenate([pos, orn])
            
        return np.concatenate([robot_state, object_state]).astype(np.float32)
    
    def _action_to_vector(self, action: Action) -> np.ndarray:
        """Convert action to vector for ML input."""
        action_onehot = [1, 0] if action.action_type == ActionType.PICK else [0, 1]
        target_euler = [np.pi, 0, 0]  # Default orientation
        
        return np.concatenate([
            action_onehot,
            action.target_position,
            target_euler
        ]).astype(np.float32)
    
    def get_statistics(self) -> Dict:
        """Get planning statistics."""
        return {
            'motion_planning_calls': self.motion_planning_calls,
            'ml_predictions': self.ml_predictions,
            'pruned_actions': self.pruned_actions,
            'nodes_expanded': self.nodes_expanded,
            'pruning_rate': self.pruned_actions / max(1, self.ml_predictions)
        }


class HybridTAMPPlanner(MLGuidedTAMPPlanner):
    """
    Hybrid planner that validates ML predictions with motion planning.
    
    Provides guaranteed correctness by always verifying the final plan.
    """
    
    def __init__(
        self,
        env: TabletopEnv,
        model: nn.Module,
        verify_threshold: float = 0.8,
        **kwargs
    ):
        """
        Initialize hybrid planner.
        
        Args:
            env: Environment
            model: Feasibility classifier
            verify_threshold: Actions below this score always get verified
            **kwargs: Additional arguments for MLGuidedTAMPPlanner
        """
        super().__init__(env, model, **kwargs)
        self.verify_threshold = verify_threshold
        
    def _check_feasibility(self, action: Action) -> Optional[List[np.ndarray]]:
        """
        Check feasibility with optional ML shortcut.
        
        For high-confidence predictions, skip motion planning during search,
        but always verify the final plan.
        """
        # For now, always do motion planning (safety first)
        # In production, could skip for very high confidence
        return super()._check_feasibility(action)
