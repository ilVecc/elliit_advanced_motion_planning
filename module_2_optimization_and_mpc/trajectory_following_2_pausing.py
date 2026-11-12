import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.wait_for_message import wait_for_message

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import numpy as np
import fastpathplanning as fpp


def generate_trajectory(X, T = 10):
    L = np.array([
        [-9, 1],
        [-13, 5],
        [-15, 5],
        [-12, 7],
    ])
    U = np.array([
        [-6, 6],
        [-7, 6],
        [-12, 9],
        [-7, 9],
    ])
    S = fpp.SafeSet(L, U)

    p_init = np.array([X[0], X[1]])
    p_term = np.array([-9.0, 8.0])
    alpha = [1, 10, 10]
    p = fpp.plan(S, p_init, p_term, T, alpha)
    
    return p

def query(p, t):
    dp = p.derivative()
    ddp = dp.derivative()

    p_t = p(t)
    dp_t = dp(t)
    ddp_t = ddp(t)

    # desired pose
    xy_t_d = p_t
    th_t_d = np.arctan2(dp_t[1], dp_t[0])
    # desired velocity
    v_t_d = np.sqrt(dp_t[1]**2+dp_t[0]**2)
    w_t_d = (ddp_t[1] * dp_t[0] - ddp_t[0] * dp_t[1]) / v_t_d**2

    X_d = np.array([xy_t_d[0], xy_t_d[1], th_t_d])
    V_d = np.array([v_t_d, w_t_d])
    
    return X_d, V_d


class Navigate(Node):
    
    def __init__(self):
        super().__init__('creep')
        self.cmd_pub = self.create_publisher(Twist, '/mir/cmd_vel', 10)
        # laser scans for safety
        self.create_subscription(LaserScan, '/mir/f_scan', self.on_scan, qos_profile_sensor_data)
        self.min_range = np.inf
        
        # odometry callbacks
        self.X = np.zeros([0, 0, 0])
        self.create_subscription(Odometry, "/mir/odom", self.on_odom, qos_profile_sensor_data)
        # msg = wait_for_message(self, Odometry, "/mir/odom", timeout_sec=5.0)
        success, msg = wait_for_message(Odometry, self, "/mir/odom", timeout_sec=5.0)
        self.on_odom(msg)
        # print initial pose
        print(self.X)
        
        # controller parameters
        ctrl_dt = 0.05  # 20 Hz control loop
        self.b = 10  # gain
        self.zeta = 1  # damping
        
        # trajectory generation (offline, static map) 
        self.T = 30
        self.traj = generate_trajectory(self.X, self.T)
        self.elapsed = 0
        self.t_init = self.get_clock().now()
        self.ok = True
        self.create_timer(ctrl_dt, self.on_timer)
    
    def on_scan(self, msg):
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        self.min_range = min(valid, default=np.inf)
        if self.min_range < 0.5:
            # too close, stop
            if self.ok:
                # on mode switch, update elapsed time in trajectory
                # this will be used when we resume control
                dt_msg = (self.get_clock().now() - self.t_init).to_msg()
                self.elapsed += (dt_msg.sec + dt_msg.nanosec * 1e-9) 
            self.ok = False
        else:
            # now it's fine, resume
            if not self.ok:
                # on mode switch, set the new initial time
                # we will add `self.elapsed` in the control loop
                self.t_init = self.get_clock().now()
            self.ok = True
    
    def on_odom(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        th = 2*np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        self.X = np.array([px, py, th])
    
    def on_timer(self):
        if not self.ok:
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            return

        # get the current time in the trajectory accounting for pauses
        start_dt_msg = (self.get_clock().now() - self.t_init).to_msg()
        dt = (start_dt_msg.sec + start_dt_msg.nanosec * 1e-9) + self.elapsed
        # stop if done
        if dt > self.T:
            print("Done")
            rclpy.shutdown()

        # find the current desired target in world coordinates
        X_d, V_d = query(self.traj, dt)
        
        
        
        # -------------- CONTROL LAW ---------------
        # [src: G. Oriolo, Control of Nonholonomic Systems - Lecture 6]
        #
        # controller gains (scheduled gains)
        a = np.sqrt(V_d[1]**2 + self.b*V_d[0]**2)
        k1 = 2 * self.zeta * a
        k2 = self.b*np.abs(V_d[0])
        k3 = k1
        # compute error in world coordinates
        c, s = np.cos(self.X[2]), np.sin(self.X[2])
        Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        e = Rz @ (X_d - self.X)  # [epx, epy, eth]
        # "linear" control law
        v1 = -k1 * e[0]
        v2 = -k2 * np.sign(V_d[0]) * e[1] - k3 * e[2]
        # local frame control : (vx,wz)
        u1 = V_d[0] * np.cos(e[2]) - v1
        u2 = V_d[1] - v2
        
        # print(dt, V_d)
        # print(dt, (u1, u2))
        # print(f"{dt:.3f} : {self.X[2]:.3f} -> {X_d[2]:.3f})
        
        cmd = Twist()
        cmd.linear.x = u1
        cmd.angular.z = u2
        self.cmd_pub.publish(cmd)
        
        
def main():
    # Keep the context alive on Ctrl+C so the final stop command can be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Navigate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())  # stop
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
