"""
RRT* Motion Planner for robot arm trajectory generation.

Provides collision-free path planning from current to target configuration.
"""

import numpy as np
from typing import Optional, List, Tuple
import pybullet as p
from dataclasses import dataclass
import time


@dataclass
class RRTNode:
    """Node in the RRT tree."""
    config: np.ndarray  # Joint configuration
    parent: Optional['RRTNode'] = None
    cost: float = 0.0


class RRTStarPlanner:
    """
    RRT* motion planner for collision-free trajectory generation.
    
    Uses PyBullet for collision checking.
    """
    
    def __init__(
        self,
        physics_client: int,
        robot_id: int,
        joint_indices: List[int],
        joint_limits_lower: np.ndarray,
        joint_limits_upper: np.ndarray,
        step_size: float = 0.1,
        goal_bias: float = 0.1,
        max_iterations: int = 1000,
        timeout: float = 5.0,
        neighbor_radius: float = 0.5
    ):
        """
        Initialize the RRT* planner.
        
        Args:
            physics_client: PyBullet physics client ID
            robot_id: Robot body ID in PyBullet
            joint_indices: List of joint indices to plan for
            joint_limits_lower: Lower joint limits
            joint_limits_upper: Upper joint limits
            step_size: Maximum step size between nodes
            goal_bias: Probability of sampling the goal
            max_iterations: Maximum planning iterations
            timeout: Maximum planning time in seconds
            neighbor_radius: Radius for finding nearby nodes in RRT*
        """
        self.physics_client = physics_client
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.joint_limits_lower = np.array(joint_limits_lower)
        self.joint_limits_upper = np.array(joint_limits_upper)
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.neighbor_radius = neighbor_radius
        
        self.n_joints = len(joint_indices)
        
    def plan(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        goal_threshold: float = 0.1
    ) -> Optional[List[np.ndarray]]:
        """
        Plan a collision-free path from start to goal.
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            goal_threshold: Distance threshold for reaching goal
            
        Returns:
            List of waypoint configurations, or None if planning failed
        """
        start_time = time.time()
        
        # Initialize tree with start node
        start_node = RRTNode(config=np.array(start_config), cost=0.0)
        nodes = [start_node]
        
        # Check if start and goal are valid
        if self._is_collision(start_config):
            return None
        if self._is_collision(goal_config):
            return None
            
        best_goal_node = None
        best_goal_cost = float('inf')
        
        for i in range(self.max_iterations):
            # Check timeout
            if time.time() - start_time > self.timeout:
                break
                
            # Sample random configuration (with goal bias)
            if np.random.random() < self.goal_bias:
                sample = goal_config
            else:
                sample = self._sample_random_config()
                
            # Find nearest node
            nearest_node = self._find_nearest(nodes, sample)
            
            # Steer towards sample
            new_config = self._steer(nearest_node.config, sample)
            
            # Check for collision
            if self._is_collision(new_config):
                continue
                
            # Check edge for collision
            if not self._check_edge(nearest_node.config, new_config):
                continue
                
            # RRT* improvement: find best parent in neighborhood
            new_cost = nearest_node.cost + self._distance(nearest_node.config, new_config)
            best_parent = nearest_node
            
            # Find neighbors within radius
            neighbors = self._find_neighbors(nodes, new_config)
            for neighbor in neighbors:
                if self._check_edge(neighbor.config, new_config):
                    cost = neighbor.cost + self._distance(neighbor.config, new_config)
                    if cost < new_cost:
                        new_cost = cost
                        best_parent = neighbor
                        
            # Create new node
            new_node = RRTNode(
                config=new_config,
                parent=best_parent,
                cost=new_cost
            )
            nodes.append(new_node)
            
            # Rewire neighbors
            for neighbor in neighbors:
                if neighbor is best_parent:
                    continue
                if self._check_edge(new_config, neighbor.config):
                    cost_through_new = new_cost + self._distance(new_config, neighbor.config)
                    if cost_through_new < neighbor.cost:
                        neighbor.parent = new_node
                        neighbor.cost = cost_through_new
                        
            # Check if goal reached
            dist_to_goal = self._distance(new_config, goal_config)
            if dist_to_goal < goal_threshold:
                goal_cost = new_cost + dist_to_goal
                if goal_cost < best_goal_cost:
                    best_goal_cost = goal_cost
                    best_goal_node = new_node
                    
        # Extract path if goal was reached
        if best_goal_node is not None:
            path = self._extract_path(best_goal_node)
            # Add final goal configuration
            path.append(goal_config)
            return path
            
        return None
    
    def _sample_random_config(self) -> np.ndarray:
        """Sample a random configuration within joint limits."""
        return np.random.uniform(
            self.joint_limits_lower,
            self.joint_limits_upper
        )
        
    def _find_nearest(self, nodes: List[RRTNode], config: np.ndarray) -> RRTNode:
        """Find the nearest node in the tree to the given configuration."""
        min_dist = float('inf')
        nearest = nodes[0]
        
        for node in nodes:
            dist = self._distance(node.config, config)
            if dist < min_dist:
                min_dist = dist
                nearest = node
                
        return nearest
    
    def _find_neighbors(self, nodes: List[RRTNode], config: np.ndarray) -> List[RRTNode]:
        """Find all nodes within the neighbor radius."""
        neighbors = []
        for node in nodes:
            if self._distance(node.config, config) < self.neighbor_radius:
                neighbors.append(node)
        return neighbors
    
    def _steer(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        """Steer from one configuration towards another, limited by step size."""
        direction = to_config - from_config
        dist = np.linalg.norm(direction)
        
        if dist <= self.step_size:
            return to_config
        else:
            return from_config + (direction / dist) * self.step_size
            
    def _distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        """Compute distance between two configurations."""
        return np.linalg.norm(config1 - config2)
    
    def _is_collision(self, config: np.ndarray) -> bool:
        """
        Check if a configuration is in collision.
        
        Uses PyBullet's collision detection.
        """
        # Save current state
        original_positions = []
        for joint_idx in self.joint_indices:
            state = p.getJointState(
                self.robot_id,
                joint_idx,
                physicsClientId=self.physics_client
            )
            original_positions.append(state[0])
            
        # Set to test configuration
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id,
                joint_idx,
                config[i],
                physicsClientId=self.physics_client
            )
            
        # Perform collision check
        p.performCollisionDetection(physicsClientId=self.physics_client)
        
        # Check for contacts (excluding floor)
        contact_points = p.getContactPoints(
            bodyA=self.robot_id,
            physicsClientId=self.physics_client
        )
        
        has_collision = False
        for contact in contact_points:
            body_b = contact[2]
            # Ignore contact with floor (body 0) and self
            if body_b != 0 and body_b != self.robot_id:
                # Ignore gripper contacts (expected during grasping)
                link_a = contact[3]
                if link_a < 9:  # Arm links, not gripper
                    has_collision = True
                    break
                    
        # Self-collision check
        self_contacts = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.robot_id,
            physicsClientId=self.physics_client
        )
        if len(self_contacts) > 0:
            has_collision = True
            
        # Restore original state
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id,
                joint_idx,
                original_positions[i],
                physicsClientId=self.physics_client
            )
            
        return has_collision
    
    def _check_edge(
        self,
        from_config: np.ndarray,
        to_config: np.ndarray,
        num_checks: int = 10
    ) -> bool:
        """
        Check if the edge between two configurations is collision-free.
        
        Performs interpolation and checks intermediate points.
        """
        for i in range(num_checks):
            t = (i + 1) / num_checks
            intermediate = from_config + t * (to_config - from_config)
            if self._is_collision(intermediate):
                return False
        return True
    
    def _extract_path(self, node: RRTNode) -> List[np.ndarray]:
        """Extract path from root to given node."""
        path = []
        current = node
        while current is not None:
            path.append(current.config)
            current = current.parent
        path.reverse()
        return path
    
    def smooth_path(
        self,
        path: List[np.ndarray],
        iterations: int = 50
    ) -> List[np.ndarray]:
        """
        Smooth the path using shortcutting.
        
        Args:
            path: Original path
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
            
        smoothed = list(path)
        
        for _ in range(iterations):
            if len(smoothed) <= 2:
                break
                
            # Pick two random points
            i = np.random.randint(0, len(smoothed) - 1)
            j = np.random.randint(i + 1, len(smoothed))
            
            # Try to shortcut
            if self._check_edge(smoothed[i], smoothed[j], num_checks=20):
                # Remove intermediate points
                smoothed = smoothed[:i+1] + smoothed[j:]
                
        return smoothed
    
    def interpolate_path(
        self,
        path: List[np.ndarray],
        num_points: int = 50
    ) -> List[np.ndarray]:
        """
        Interpolate path to fixed number of waypoints.
        
        Args:
            path: Original path (variable length)
            num_points: Number of output waypoints
            
        Returns:
            Interpolated path with exactly num_points waypoints
        """
        if len(path) < 2:
            return [path[0]] * num_points if path else []
            
        # Compute cumulative distances
        distances = [0.0]
        for i in range(1, len(path)):
            dist = self._distance(path[i-1], path[i])
            distances.append(distances[-1] + dist)
            
        total_dist = distances[-1]
        if total_dist == 0:
            return [path[0]] * num_points
            
        # Interpolate at uniform intervals
        interpolated = []
        for i in range(num_points):
            target_dist = (i / (num_points - 1)) * total_dist
            
            # Find segment containing target distance
            for j in range(1, len(distances)):
                if distances[j] >= target_dist:
                    # Interpolate within segment
                    segment_start = distances[j-1]
                    segment_length = distances[j] - segment_start
                    if segment_length > 0:
                        t = (target_dist - segment_start) / segment_length
                    else:
                        t = 0
                    point = path[j-1] + t * (path[j] - path[j-1])
                    interpolated.append(point)
                    break
                    
        return interpolated
