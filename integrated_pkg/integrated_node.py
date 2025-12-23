import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, PoseStamped
from nav_msgs.msg import Odometry
import math
import numpy as np
import csv
import time
import os

# Modular imports
from integrated_pkg.utils import quaternion_to_euler

class PID:
    """
    PID Controller class.
    """
    def __init__(self, kp, ki, kd, min_out, max_out):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out = min_out
        self.max_out = max_out
        
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        """
        Calculates the PID output.
        :param error: target_value - current_value
        :param dt: time step in seconds
        """
        if dt <= 0.0:
            return 0.0

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Total output
        output = p_term + i_term + d_term
        
        # Save error for next step
        self.prev_error = error

        # Clamp output
        output = np.clip(output, self.min_out, self.max_out)
        
        return output

class PurePursuit:
    """
    Pure Pursuit controller class for calculating steering angles.
    """
    def __init__(self, wheelbase, steering_gain):
        self.L = wheelbase
        self.steering_gain = steering_gain

    def calculate_steering_angle(self, current_x, current_y, current_yaw, target_x, target_y):
        """
        Calculates the steering angle based on current state and target.
        """
        if current_x is None or target_x is None:
             return 0.0

        # 1. Calculate error vector in global frame
        dx = target_x - current_x
        dy = target_y - current_y
        
        # 2. Rotate into vehicle frame (Local Coordinates)
        local_x = dx * math.cos(current_yaw) + dy * math.sin(current_yaw)
        local_y = -dx * math.sin(current_yaw) + dy * math.cos(current_yaw)

        # 3. Calculate Steering using Equation
        alpha = math.atan2(local_y, local_x)
        actual_dist = math.hypot(local_x, local_y)
        
        if actual_dist < 1e-3: 
            return 0.0

        # Note: Our actual_dist is the Lookahead Distance (Constant)
        steering_angle = math.atan((2.0 * self.L * math.sin(alpha)) / actual_dist)
        steering_angle *= self.steering_gain

        return steering_angle

class IntegratedNode(Node):
    def __init__(self):
        super().__init__('integrated_node_position_error')

         # --- Parameters ---
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('steering_gain', 3.5) 
        
        # PID Parameters (For Position Error)
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.0) 
        self.declare_parameter('kd', 0.0)
        
        # Velocity Limits
        self.declare_parameter('min_velocity', 1.0)
        self.declare_parameter('max_velocity', 4.0)

        # Load parameters:
        self.L = self.get_parameter('wheelbase').value
        self.steering_gain = self.get_parameter('steering_gain').value
        
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        
        self.min_vel = self.get_parameter('min_velocity').value
        self.max_vel = self.get_parameter('max_velocity').value
       
         # --- State Variables ---
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.current_velocity = 0.0
        self.data_received = False
        self.last_time = time.time()
        
        self.dynamic_target = None

        self.pure_pursuit = PurePursuit(self.L, self.steering_gain)
        
        # Initialize PID Controller
        self.pid_controller = PID(self.kp, self.ki, self.kd, self.min_vel, self.max_vel)

        # --- Publishers & Subscribers ---
        
        self.pose_sub = self.create_subscription(
            PoseArray, '/pose_info', self.pose_callback, 10
        )
        
        self.target_sub = self.create_subscription(
            PoseStamped, '/pose_msg', self.target_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.drive_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        # --- LOGGING SETUP ---
        self.log_counter = 0
        csv_path = '/home/ubuntu/workspace/ros2_ws/src/integrated_pkg/data_logs/integrated_node_data.csv'
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write the Header Row ONCE
        self.csv_writer.writerow(['time', 'car_x', 'car_y', 'steering_angle', 'current_velocity', 'position_error', 'cmd_vel_out'])
        self.start_time = time.time()
        
        self.get_logger().info(f"LOGGING DATA TO: {csv_path}")
        self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Integrated Node Initialized")


    def pose_callback(self, msg):
        if not msg.poses or len(msg.poses) < 2: 
            return
        target_pose = msg.poses[1]
        self.current_x = target_pose.position.x
        self.current_y = target_pose.position.y
        q = target_pose.orientation
        _, _, self.current_yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        self.data_received = True

    def target_callback(self, msg):
        self.dynamic_target = (msg.pose.position.x, msg.pose.position.y)

    def odom_callback(self, msg):
        self.current_velocity = msg.twist.twist.linear.x
    
    def control_loop(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if not self.data_received:
            self.get_logger().warn("Waiting for vehicle pose...", throttle_duration_sec=2.0)
            return
        
        if self.dynamic_target is None:
            self.get_logger().warn("Waiting for Perception Target...", throttle_duration_sec=2.0)
            stop_msg = Twist()
            self.drive_pub.publish(stop_msg)
            return

        target = self.dynamic_target
        target_x, target_y = target

        # 1. Calculate Steering Angle
        steering_angle = self.pure_pursuit.calculate_steering_angle(
            self.current_x, self.current_y, self.current_yaw, target_x, target_y
        )
        max_steer = 0.6 
        steering_angle = np.clip(steering_angle, -max_steer, max_steer)

        # 2. Calculate Position Error (Distance to Target)
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance_error = math.hypot(dx, dy)
        
        # 3. Calculate Velocity Command using PID
        raw_cmd_velocity = self.pid_controller.compute(distance_error, dt)

        # 4. SCALING FACTOR (The Fix for Curves)
        # Even if position error is huge, we MUST slow down if steering is sharp.
        # We multiply the PID output by an exponential decay factor.
        decay_k = 5.0
        steering_scale = math.exp(-decay_k * abs(steering_angle))
        
        # Final Command = PID_Output * Scale
        cmd_velocity = raw_cmd_velocity * steering_scale
        
        # Ensure we don't drop below min_vel
        cmd_velocity = max(cmd_velocity, self.min_vel)
        
        # Clamp to max_vel
        cmd_velocity = np.clip(cmd_velocity, self.min_vel, self.max_vel)

        # 5. Construct Drive Message
        drive_msg = Twist()
        drive_msg.linear.x = float(cmd_velocity)
        drive_msg.angular.z = float(steering_angle)
        
        # 6. Publish Command
        self.drive_pub.publish(drive_msg)
        
        # --- 7. LOGGING to CSV file ---
        self.log_counter += 1

        if self.log_counter % 4 == 0:
            log_time = time.time() - self.start_time

            self.csv_writer.writerow([
                f"{log_time:.3f}",        
                f"{self.current_x:.3f}",      
                f"{self.current_y:.3f}",      
                f"{steering_angle:.4f}",      
                f"{self.current_velocity:.2f}",
                f"{distance_error:.2f}",
                f"{cmd_velocity:.2f}"
            ])

        # 8. Console Logging
        self.get_logger().info(
            f"Pos Error (Dist): {distance_error:.2f} m | "
            f"Cur Vel: {self.current_velocity:.2f} | "
            f"Cmd Vel: {cmd_velocity:.2f} | "
            f"Steer: {steering_angle:.2f}"
        )
    
    def destroy_node(self):
        try:
            self.csv_file.close()
            self.get_logger().info("Data Log Saved successfully!")
        except:
             pass
        super().destroy_node()

            
def main(args=None):
    rclpy.init(args=args)
    node = IntegratedNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
