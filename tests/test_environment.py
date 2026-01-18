"""
Tests for the PyBullet environment.

Run with: pytest tests/test_environment.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.tabletop_env import TabletopEnv
from environments.robot import FrankaPanda


class TestFrankaPanda:
    """Tests for the Franka Panda robot controller."""
    
    @pytest.fixture
    def env(self):
        """Create a fresh environment."""
        env = TabletopEnv(render=False)
        yield env
        env.close()
        
    def test_robot_initialization(self, env):
        """Test robot loads correctly."""
        assert env.robot is not None
        assert env.robot.robot_id is not None
        
    def test_get_joint_positions(self, env):
        """Test reading joint positions."""
        positions = env.robot.get_joint_positions()
        assert len(positions) == 7
        assert all(isinstance(p, (int, float, np.floating)) for p in positions)
        
    def test_get_end_effector_pose(self, env):
        """Test end-effector pose retrieval."""
        pos, orn = env.robot.get_end_effector_pose()
        assert len(pos) == 3
        assert len(orn) == 4
        
    def test_inverse_kinematics(self, env):
        """Test IK solver."""
        target_pos = np.array([0.5, 0.0, 0.5])
        result = env.robot.inverse_kinematics(target_pos)
        # IK should return a valid config or None
        assert result is None or len(result) == 7
        
    def test_gripper_control(self, env):
        """Test gripper open/close."""
        env.robot.open_gripper()
        env.step_simulation(50)
        open_state = env.robot.get_gripper_state()
        
        env.robot.close_gripper()
        env.step_simulation(50)
        closed_state = env.robot.get_gripper_state()
        
        assert open_state >= closed_state
        
    def test_state_vector(self, env):
        """Test robot state vector extraction."""
        state = env.robot.get_state_vector()
        assert len(state) == 15  # 7 joints + 1 gripper + 3 pos + 4 orn


class TestTabletopEnv:
    """Tests for the tabletop environment."""
    
    @pytest.fixture
    def env(self):
        """Create environment."""
        env = TabletopEnv(render=False, num_objects=(2, 4))
        yield env
        env.close()
        
    def test_reset(self, env):
        """Test environment reset."""
        obs = env.reset()
        assert 'state_vector' in obs
        assert 'image' in obs
        
    def test_object_spawning(self, env):
        """Test objects spawn correctly."""
        env.reset(num_objects=3)
        assert len(env.object_ids) == 3
        
    def test_get_state(self, env):
        """Test state vector extraction."""
        env.reset()
        state = env.get_state()
        # 15 (robot) + 42 (6 objects * 7) = 57
        assert len(state) == 57
        
    def test_get_image(self, env):
        """Test image capture."""
        env.reset()
        image = env.get_image()
        assert image.shape == (128, 128, 3)
        
    def test_sample_grasp_target(self, env):
        """Test grasp target sampling."""
        env.reset(num_objects=3)
        result = env.sample_grasp_target()
        assert result is not None
        target_pos, obj_id = result
        assert len(target_pos) == 3
        assert obj_id in env.object_ids
        
    def test_sample_place_target(self, env):
        """Test place target sampling."""
        env.reset()
        target = env.sample_place_target()
        assert len(target) == 3
        # Check position is on table
        assert 0.3 <= target[0] <= 0.7
        assert -0.25 <= target[1] <= 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
