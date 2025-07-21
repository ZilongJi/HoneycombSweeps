from typing import List, Optional

import numpy as np
from scipy.interpolate import interp1d

import brainpy.math as bm

from models import (
    HD_cell_L1, 
    PC_cell_L2, 
)


def simulate_honeycomb_movement(
    total_time=1, 
    dt=0.01, 
    move_prob=0.95, 
    drift_factor=0.0, 
    velocity=0.5, 
    rotation_speed=np.pi / 50,
    stick_length=0.2,
    goal_loc=[0, 0]
):
    
    env_size = 1.0
    time_steps = int(total_time / dt)

    pos_A = np.array([env_size / 2, env_size / 2])
    angle = 0
    pos_B = pos_A + stick_length * np.array([np.cos(angle), np.sin(angle)])
    move_angle = np.random.uniform(-np.pi, np.pi)

    A_trajectory = []   
    B_trajectory = []
    head_directions = []
    rotation_phases = []
    rotation_direction_log = []
    time_list = []
    breakTimepoints = []  # Initialise list


    step_count = 0
    current_time = 0
    just_rotated = False  # New flag to prevent back-to-back rotations

    while step_count < time_steps:
        if (np.random.rand() < move_prob and not just_rotated) or just_rotated:
            
            move_angle = (drift_factor * move_angle) + ((1 - drift_factor) * np.random.uniform(-np.pi, np.pi))
            step_size = velocity * dt
            new_pos_A = pos_A + step_size * np.array([np.cos(move_angle), np.sin(move_angle)])
            new_pos_A = np.clip(new_pos_A, 0, env_size)
            pos_A = new_pos_A
            pos_B = pos_A + stick_length * np.array([np.cos(angle), np.sin(angle)])

            A_trajectory.append(pos_A.copy())
            B_trajectory.append(pos_B.copy())
            head_directions.append(angle)
            rotation_phases.append(False)
            rotation_direction_log.append(0)
            time_list.append(current_time)

            step_count += 1
            current_time += dt
            just_rotated = False  # Reset flag after movement
        else:
            # Randomly choose rotation direction: +1 (CW) or -1 (CCW)
            direction_flag = np.random.choice([1, -1])
            rotation_duration = int(np.random.uniform(0.3, 1.0) / dt)

            for _ in range(min(rotation_duration, time_steps - step_count)):
                angle -= direction_flag * rotation_speed  # CW = -rotation
                angle = (angle + np.pi) % (2 * np.pi) - np.pi
                pos_B = pos_A + stick_length * np.array([np.cos(angle), np.sin(angle)])

                # Check alignment with goal direction
                vec_to_goal = goal_loc - pos_A
                dir_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])
                angle_diff = ((angle - dir_to_goal + np.pi) % (2 * np.pi)) - np.pi  # Wrap to [-π, π]

                # if np.abs(angle_diff) < np.deg2rad(10):  # e.g. 5° threshold
                #     breakTimepoints.append(current_time)  # Record the time when aligned
                #     break  # Stop rotation if aligned

                # if (np.abs(angle_diff) > np.deg2rad(170)):
                #     breakTimepoints.append(current_time)  # Record the time when aligned
                #     break  # Stop rotation if aligned

                A_trajectory.append((pos_A.copy()+pos_B.copy())/2)
                B_trajectory.append(pos_B.copy())
                head_directions.append(angle)
                rotation_phases.append(True)
                rotation_direction_log.append(direction_flag)
                time_list.append(current_time)

                step_count += 1
                current_time += dt
                if step_count >= time_steps:
                    break
                
            just_rotated = True  # Mark that a rotation just occurred

    A_trajectory = np.array(A_trajectory)
    B_trajectory = np.array(B_trajectory)
    head_directions = np.array(head_directions)
    rotation_phases = np.array(rotation_phases)
    rotation_direction_log = np.array(rotation_direction_log)
    time_original = np.array(time_list)

    displacement = np.linalg.norm(np.diff(B_trajectory, axis=0), axis=1)
    speeds = np.concatenate(([0], displacement / dt))

    dt_new = 0.001
    time_upsampled = np.arange(0, time_original[-1] + dt_new, dt_new)

    interp_A_x = interp1d(time_original, A_trajectory[:, 0], kind='linear', fill_value="extrapolate")
    interp_A_y = interp1d(time_original, A_trajectory[:, 1], kind='linear', fill_value="extrapolate")
    interp_B_x = interp1d(time_original, B_trajectory[:, 0], kind='linear', fill_value="extrapolate")
    interp_B_y = interp1d(time_original, B_trajectory[:, 1], kind='linear', fill_value="extrapolate")
    # this is potentially problematic due to the periodic nature of head directions
    # even if we are wrapping later
    interp_head_dir = interp1d(time_original, head_directions, kind='linear', fill_value="extrapolate")
    interp_speeds = interp1d(time_original, speeds, kind='linear', fill_value="extrapolate")

    # Interpolate all trajectories and parameters
    A_trajectory_upsampled = np.vstack((interp_A_x(time_upsampled), interp_A_y(time_upsampled))).T
    B_trajectory_upsampled = np.vstack((interp_B_x(time_upsampled), interp_B_y(time_upsampled))).T
    head_directions_upsampled = interp_head_dir(time_upsampled)
    speeds_upsampled = interp_speeds(time_upsampled)
    head_directions_upsampled = (head_directions_upsampled + np.pi) % (2 * np.pi) - np.pi

    interp_rotation_phase = interp1d(time_original, rotation_phases, kind='nearest', fill_value="extrapolate")
    rotation_phases_upsampled = interp_rotation_phase(time_upsampled).astype(bool)

    interp_rotation_dir = interp1d(time_original, rotation_direction_log, kind='nearest', fill_value="extrapolate")
    rotation_direction_upsampled = interp_rotation_dir(time_upsampled).astype(int)

    vec_to_goal = goal_loc - B_trajectory_upsampled
    Dir2Goal = np.arctan2(vec_to_goal[:, 1], vec_to_goal[:, 0])

    diff = np.diff(B_trajectory_upsampled, axis=0)
    Velocity = np.zeros_like(B_trajectory_upsampled)
    Velocity[:-1, :] = diff
    MVDirection = (np.arctan2(Velocity[:, 1], Velocity[:, 0]) + np.pi) % (2 * np.pi) - np.pi
    
    break_indices = np.searchsorted(time_upsampled, breakTimepoints)

    return (
        A_trajectory_upsampled,
        B_trajectory_upsampled,
        head_directions_upsampled,
        MVDirection,
        Dir2Goal,
        speeds_upsampled,
        rotation_phases_upsampled,
        time_upsampled,
        Velocity,
        rotation_direction_upsampled,
        break_indices
    )
    
    
def discontinuous_plot(ax, x, y, threshold=np.pi, label=None, **kwargs):
    """
    Plot y vs x on ax, breaking the line at discontinuities > threshold.
    Only applies 'label' to the first segment to avoid legend duplication.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    dy = np.abs(np.diff(y))
    breaks = np.where(dy > threshold)[0] + 1
    segments = np.split(np.column_stack([x, y]), breaks)

    for i, seg in enumerate(segments):
        ax.plot(seg[:, 0], seg[:, 1], label=label if i == 0 else None, **kwargs)
        
        
def run_model(
    # trajectory parameters
    goal_loc: List[float] = [0.5* np.pi, 0.5 * np.pi],
    total_time: float = 0.5, 
    dt: float = 0.01, # s
    move_prob: float = 0.5, 
    drift_factor: float = 0.0, 
    velocity: float = 20.0, 
    rotation_speed: float = np.pi / 50, 
    stick_length: float = 0.2, 
    animal_head_loc_all: Optional[np.ndarray] = None,
    md_all: Optional[np.ndarray] = None,
    speed_all: Optional[np.ndarray] = None,
    gd_all: Optional[np.ndarray] = None,
    # model parameters
    v0: float = bm.pi / 1000, # baseline speed
    mbar_hd: float = 10.0,
    noise_strength: float = 0.0, # noise
    phase_offset: float = 1.5, 
    theta_hd_modulation: float = 0.4, # head direction theta modulation strength
    theta_gc_modulation: float = 0.8, # grid cell theta modulation strength
    top_down_modulation: float = 4.0, # top-down modulation strength
    num_hd: int = 100, 
    num_pc: int = 50,
    tau_hd: float = 10.0, # ms time constant for head 
    tau_v_hd: float = 100.0, # time constant for adaptation
    a_hd: float = 0.4, # half-width of excitatory Gaussian inputs for head direction cells
    A_hd: float = 3.0, # magnitude of pre-synaptic inputs for head direction cells
    a_hd_goal: float = 0.4, # half-width of top-down goal-oriented excitatory Gaussian inputs
    A_hd_goal: float = 3.0, # magnitude of top-down goal-oriented excitatory inputs
    J0_hd: float = 4.0, # max connection strength for head direction cells
    top_down: bool = True, 
    tau_pc: float = 10.0, # ms time constant for place cells
    tau_v_pc: float = 100.0, # time constant for adaptation of place
    mbar_pc: float = 1.0,
    a_pc: float = 0.5, # half-width of excitatory Gaussian inputs for place cells
    A_pc: float = 5.0, # magnitude of pre-synaptic inputs for place cells
    A_td_pc: float = 5.0, # magnitude of top-down goal-oriented excitatory inputs to place cells
    J0_pc: float = 10.0, # max connection strength for place cells
    g: float = 1000.0, 
    time_chunk: int=50000, # perform simulation every 50s, in case of memory overflow
    device: str = "cpu", 
    seed: int = 111, 
):
    np.random.seed(seed)
    # simulate trajectory
    if animal_head_loc_all is None:
        goal_loc = np.array(goal_loc) / np.pi + 0.5
        (
            animal_centre_all, 
            animal_head_loc_all, 
            hd_all, 
            md_all, 
            gd_all, 
            speed_all, 
            rotation_phase, 
            t_upsampled, 
            velocity_all, 
            rotation_direction_flag_all, 
            break_inds, 
        ) = simulate_honeycomb_movement(
            total_time=total_time, 
            dt=dt, 
            move_prob=move_prob, 
            drift_factor=drift_factor, 
            velocity=velocity, 
            rotation_speed=rotation_speed, 
            stick_length=stick_length, 
            goal_loc=goal_loc, 
        )
        
        animal_head_loc_all = animal_head_loc_all * np.pi - 0.5 * np.pi
        speed_all = speed_all / 1000
    else:
        animal_centre_all = None
        hd_all = None
        t_upsampled = None
        rotation_phase = None
        rotation_direction_flag_all = None
        velocity_all = None
    
    # model initialisation
    bm.set_platform(device)
    bm.set_dt(1.0) # ms
    bm.clear_buffer_memory(device)
    
    HD_net = HD_cell_L1(
        num=num_hd,
        noise_stre=noise_strength/6, 
        tau=tau_hd, 
        tau_v=tau_v_hd, 
        k=1.0, 
        mbar=mbar_hd,
        a=a_hd, 
        A=A_hd, 
        J0=J0_hd, 
        z_min=-bm.pi, 
        z_max=bm.pi,
        goal_a=a_hd_goal,
        goal_A=A_hd_goal,
        topdown=top_down,
    )
    
    PC_net = PC_cell_L2(
        noise_stre=noise_strength, 
        num=num_pc, 
        tau=tau_pc, 
        tau_v=tau_v_pc,
        mbar=mbar_pc, 
        a=a_pc, 
        A=A_pc, 
        td_A=A_td_pc,
        J0=J0_pc,
        k=1.0, 
        g=g, 
        x_min=-bm.pi, 
        x_max=bm.pi, 
        num_hd=num_hd, 
        Phase_Offset=phase_offset, 
    )
    
    def run_coupled_net(
        i: float, 
        loc: float, 
        hd: float, 
        speed: float, 
        gd: float
    ):
        A_modulation = (2 + speed / v0) / 5
        theta_modulation_strength_hd = theta_hd_modulation * speed / v0
        theta_modulation_strength_gc = theta_gc_modulation * speed / v0
        
        time_theta_cycle = 100 # ms
        t = i * bm.dt
        theta_phase = bm.mod(t, time_theta_cycle) / time_theta_cycle * 2 * bm.pi
        theta_modulator_hd = 1 + theta_modulation_strength_hd * bm.cos(theta_phase)
        theta_modulator_gc = (1 + theta_modulation_strength_gc * bm.cos(theta_phase)) * A_modulation
        
        HD_net.step_run(
            i, 
            hd, 
            theta_modulator_hd, 
            top_down_modulation, 
            gd, 
        )
        hd_bump_center = HD_net.center
        hd_bump_activity = HD_net.r
        
        PC_net.step_run(
            i, 
            loc, 
            hd_bump_activity, 
            theta_modulator_gc, 
            hd, 
        )
        
        pc_bump_center = PC_net.center_bump
        pc_bump_activity = PC_net.r
        
        return (
            pc_bump_center, 
            hd_bump_center, 
            pc_bump_activity,
            hd_bump_activity,
            theta_phase, 
            theta_modulator_hd, 
        )
    
    time_steps = np.arange(len(animal_head_loc_all))
    
    print("Start running simulation.")
    
    @bm.jit
    def run(
        t: np.ndarray, 
        loc: np.ndarray,
        hd: np.ndarray,
        speed: np.ndarray,
        gd: np.ndarray, 
    ):
        return bm.for_loop(
            run_coupled_net, 
            (t, loc, hd, speed, gd),
        )

    pc_activity = np.empty((len(time_steps), num_pc, num_pc))
    hd_activity = np.empty((len(time_steps), num_hd))
    pc_bump_center = np.empty((len(time_steps), 2))
    hd_bump_center = np.empty((len(time_steps), 1))
    theta_phase = np.empty((len(time_steps), ))
    theta_rhythm = np.empty((len(time_steps), ))
    
    for i in range(0, len(time_steps), time_chunk):
        print(f"Simnulation progress: {i/len(time_steps)*100:.1f}%")
        (
            pc_bump_center_temp, 
            hd_bump_center_temp, 
            pc_bump_activity_temp,
            hd_bump_activity_temp,
            theta_phase_temp, 
            theta_modulator_hd_temp, 
        ) = run(
            time_steps[i:(i+time_chunk)],
            animal_head_loc_all[i:(i+time_chunk)],
            md_all[i:(i+time_chunk)],
            speed_all[i:(i+time_chunk)],
            gd_all[i:(i+time_chunk)],
        )
        
        pc_activity[i:(i+time_chunk), :, :] = pc_bump_activity_temp.reshape(-1, num_pc, num_pc)
        hd_activity[i:(i+time_chunk), :] = hd_bump_activity_temp
        pc_bump_center[i:(i+time_chunk), :] = pc_bump_center_temp
        hd_bump_center[i:(i+time_chunk), :] = hd_bump_center_temp
        theta_phase[i:(i+time_chunk)] = theta_phase_temp
        theta_rhythm[i:(i+time_chunk)] = theta_modulator_hd_temp
        
    print("Simulation completed.")
    
    return (
        HD_net, 
        PC_net,
        animal_centre_all,
        animal_head_loc_all,
        hd_all,
        md_all,
        gd_all,
        speed_all,
        velocity_all, 
        pc_activity,
        hd_activity,
        pc_bump_center,
        hd_bump_center,
        t_upsampled,
        theta_phase,
        theta_rhythm,
        rotation_phase,
        rotation_direction_flag_all,
    )
