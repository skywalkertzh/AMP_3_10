import torch
from torch import Tensor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import json
from flask import send_file

from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.agents.ppo.ppo import PPO


class AnalysisPlotLocomotion(RL_EvalCallback):
    training_loop: PPO
    env: LeggedRobotBase

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        env: LeggedRobotBase = self.training_loop.env
        self.env = env
        self.policy = self.training_loop._get_inference_policy()
        self.num_envs = self.env.num_envs
        self.logger = WebLogger(self.config.sim_dt)
        self.reset_buffers()
        self.log_single_robot = self.config.log_single_robot

    def reset_buffers(self):
        self.obs_buf = [[] for _ in range(self.num_envs)]
        self.critic_obs_buf = [[] for _ in range(self.num_envs)]
        self.act_buf = [[] for _ in range(self.num_envs)]

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        self.robot_num_dofs = self.env.num_dofs
        self.log_dof_pos_limits = self.env.dof_pos_limits.cpu().numpy()
        self.log_dof_vel_limits = self.env.dof_vel_limits.cpu().numpy()
        self.log_dof_torque_limits = self.env.torque_limits.cpu().numpy()
        self.logger.set_robot_limits(self.log_dof_pos_limits, self.log_dof_vel_limits, self.log_dof_torque_limits)
        self.logger.set_robot_num_dofs(self.robot_num_dofs)

    def on_post_evaluate_policy(self):
        pass

    def on_pre_eval_env_step(self, actor_state):
        obs: Tensor = actor_state["obs"]["actor_obs"].cpu()
        critic_obs: Tensor = actor_state["obs"]["critic_obs"].cpu()
        actions: Tensor = actor_state["actions"].cpu()

        for i in range(self.num_envs):
            self.obs_buf[i].append(obs[i])
            self.critic_obs_buf[i].append(critic_obs[i])
            self.act_buf[i].append(actions[i])

        if self.log_single_robot:
            self.logger.log_states(
                {
                'dof_pos_target': actions[0].cpu().numpy(),
                'dof_pos': self.env.dof_pos[0].cpu().numpy(),
                'dof_vel': self.env.dof_vel[0].cpu().numpy(),
                'dof_torque': self.env.torques[0].cpu().numpy(),
                'command_x': self.env.commands[0, 0].item(),
                'command_y': self.env.commands[0, 1].item(),
                'command_yaw': self.env.commands[0, 2].item(),
                'base_vel_x': self.env.base_lin_vel[0, 0].item(),
                'base_vel_y': self.env.base_lin_vel[0, 1].item(),
                'base_vel_z': self.env.base_lin_vel[0, 2].item(),
                'base_vel_yaw': self.env.base_ang_vel[0, 2].item(),
                'contact_forces_z': self.env.contact_forces[0, self.env.feet_indices, 2].cpu().numpy()
                }
            )
        else:
            # log average of all robots
            self.logger.log_states(
                {
                    'dof_pos_target': actions.mean(dim=0).cpu().numpy(),
                    'dof_pos': self.env.simulator.dof_pos.mean(dim=0).cpu().numpy(),
                    'dof_vel': self.env.simulator.dof_vel.mean(dim=0).cpu().numpy(),
                    'dof_torque': self.env.torques.mean(dim=0).cpu().numpy(),
                    'command_x': self.env.simulator.commands[:, 0].mean().item(),
                    'command_y': self.env.simulator.commands[:, 1].mean().item(),
                    'command_yaw': self.env.simulator.commands[:, 2].mean().item(),
                    'base_vel_x': self.env.base_lin_vel[:, 0].mean().item(),
                    'base_vel_y': self.env.base_lin_vel[:, 1].mean().item(),
                    'base_vel_z': self.env.base_lin_vel[:, 2].mean().item(),
                    'base_vel_yaw': self.env.base_ang_vel[:, 2].mean().item(),
                    'contact_forces_z': self.env.simulator.contact_forces[:, self.env.feet_indices, 2].mean(dim=1).cpu().numpy()
                }
            )
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        step = actor_state["step"]
        if (step + 1) % self.config.plot_update_interval == 0:
            self.logger.plot_states()
        return actor_state

class WebLogger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.thread = None

    def set_robot_limits(self, dof_pos_limits, dof_vel_limits, dof_torque_limits):
        self.log_dof_pos_limits = dof_pos_limits
        self.log_dof_vel_limits = dof_vel_limits
        self.log_dof_torque_limits = dof_torque_limits

    def set_robot_num_dofs(self, num_dofs):
        self.robot_num_dofs = num_dofs

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run_server)
            self.thread.start()
        self._update_plot()
    
    def _run_server(self):
        @self.app.route('/')
        def index():
            return send_file('analysis_plot_template.html')

        self.socketio.run(self.app, debug=False, use_reloader=False)
    
    def _update_plot(self):
        log = self.state_log.copy()
        total_time = len(next(iter(log.values()))) * self.dt
        time = np.linspace(0, total_time, len(next(iter(log.values()))))

        for key in log:
            if isinstance(log[key], list):
                log[key] = np.array(log[key])

        BLUE = '#005A9D'
        RED = '#DA2513'
        YELLOW = '#EEDE70'

        num_dofs = self.robot_num_dofs

        def get_subplot_titles():
            titles = [
                'Base velocity x', 'Base velocity y', 'Base velocity yaw', 'Base velocity z',
                'Vertical Contact forces', 'Command velocities x', 'Command velocities y', 'Command velocities yaw'
            ]
            for i in range(num_dofs):
                titles.extend([f'DOF {i} Position', f'DOF {i} Velocity', f'DOF {i} Torque', f'DOF {i} Torque/Velocity'])
            return titles

        # Calculate number of rows needed
        num_rows = 2 + num_dofs

        fig = make_subplots(rows=num_rows, cols=4, subplot_titles=get_subplot_titles())

        def add_trace(x, y, color, row, col, name=None, show_legend=False):
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color), name=name, showlegend=show_legend), row=row, col=col)

        # Base velocities and commands
        add_trace(time, log["base_vel_x"], BLUE, 1, 1, "Base vel x")
        add_trace(time, log["command_x"], RED, 1, 1, "Command x")
        add_trace(time, log["base_vel_y"], BLUE, 1, 2, "Base vel y")
        add_trace(time, log["command_y"], RED, 1, 2, "Command y")
        add_trace(time, log["base_vel_yaw"], BLUE, 1, 3, "Base vel yaw")
        add_trace(time, log["command_yaw"], RED, 1, 3, "Command yaw")
        add_trace(time, log["base_vel_z"], BLUE, 1, 4, "Base vel z")

        # Vertical Contact forces
        forces = log["contact_forces_z"]
        for i in range(forces[0].shape[0]):
            add_trace(time, [force[i] for force in forces], BLUE, 2, 1, f"Force {i}")

        # Command velocities
        add_trace(time, log["command_x"], BLUE, 2, 2, "Command x", True)
        add_trace(time, log["command_y"], BLUE, 2, 3, "Command y", True)
        add_trace(time, log["command_yaw"], BLUE, 2, 4, "Command yaw", True)

        def add_limit_lines(row, col, lower, upper, color=YELLOW):
            fig.add_shape(type="rect", x0=time[0], x1=time[-1], y0=lower, y1=upper,
                        fillcolor=color, line=dict(width=0), layer='below', row=row, col=col)

        # DOF Positions, Velocities, and Torques
        for i in range(num_dofs):
            row = i + 3  # Start from the third row

            # Position
            add_trace(time, [pos[i] for pos in log["dof_pos"]], BLUE, row, 1, f"DOF {i} pos")
            add_trace(time, [pos[i] for pos in log["dof_pos_target"]], RED, row, 1, f"DOF {i} pos target")
            add_limit_lines(row, 1, self.log_dof_pos_limits[i, 0], self.log_dof_pos_limits[i, 1])
            # Velocity
            add_trace(time, [vel[i] for vel in log["dof_vel"]], BLUE, row, 2, f"DOF {i} vel")
            add_limit_lines(row, 2, -self.log_dof_vel_limits[i], self.log_dof_vel_limits[i])
            # Torque
            add_trace(time, [torque[i] for torque in log["dof_torque"]], BLUE, row, 3, f"DOF {i} torque")
            add_limit_lines(row, 3, -self.log_dof_torque_limits[i], self.log_dof_torque_limits[i])
            
            # Torque/Velocity curve
            fig.add_trace(go.Scatter(
                x=[vel[i] for vel in log["dof_vel"]], 
                y=[torque[i] for torque in log["dof_torque"]], 
                mode='markers', 
                marker=dict(color=BLUE, size=2), 
                showlegend=False,
                name=f"DOF {i} Torque/Velocity"
            ), row=row, col=4)

            # Add velocity limits
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=-self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            fig.add_shape(type="line", x0=self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

            # Add torque limits
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=-self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

        fig.update_layout(height=300*num_rows, width=1500, title_text="Robot State Plots", showlegend=True)
        
        # Update x and y axis labels
        for i in range(num_rows):
            for j in range(3):
                fig.update_xaxes(title_text="time [s]", row=i+1, col=j+1)
            fig.update_xaxes(title_text="", row=i+1, col=4)
        
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=1)
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=2)
        fig.update_yaxes(title_text="base ang vel [rad/s]", row=1, col=3)
        fig.update_yaxes(title_text="base lin vel [m/s]", row=1, col=4)
        fig.update_yaxes(title_text="Forces z [N]", row=2, col=1)
        fig.update_yaxes(title_text="Command vel", row=2, col=2)
        fig.update_yaxes(title_text="Command vel", row=2, col=3)
        fig.update_yaxes(title_text="Command vel", row=2, col=4)

        for i in range(3, num_rows + 1):
            fig.update_yaxes(title_text="Position [rad]", row=i, col=1)
            fig.update_yaxes(title_text="Velocity [rad/s]", row=i, col=2)
            fig.update_yaxes(title_text="Torque [Nm]", row=i, col=3)
            fig.update_yaxes(title_text="Torque/Velocity", row=i, col=4)

        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        self.socketio.emit('update_plots', plot_json)

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.thread:
            self.socketio.stop()
            self.thread.join()