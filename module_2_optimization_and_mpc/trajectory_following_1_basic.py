import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# from trajectory_generator import *


import numpy as np
import fastpathplanning as fpp
import matplotlib.pyplot as plt


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
    # p_init = np.array([-7.49, 2.68])
    p_term = np.array([-9.0, 8.0])
    # p_init = np.array([1, 1])
    # p_term = np.array([9, 0])
    alpha = [1, 10, 10]
    # alpha = [1, 1, 1]
    p = fpp.plan(S, p_init, p_term, T, alpha)
    
    # plt.figure()
    # S.plot2d()
    # p.plot2d()
    # plt.show()
    
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




from rclpy.task import Future
from std_msgs.msg import String

def wait_for_message(node: Node, msg_type, topic: str, timeout_sec=None):
    future = Future()

    def callback(msg):
        if not future.done():
            future.set_result(msg)

    subscription = node.create_subscription(msg_type, topic, callback, 10)

    try:
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        return future.result() if future.done() else None
    finally:
        node.destroy_subscription(subscription)




class Navigate(Node):
    
    def __init__(self):
        super().__init__('creep')
        self.cmd_pub = self.create_publisher(Twist, '/mir/cmd_vel', 10)
        # MiR scans are best effort: a reliable subscriber gets nothing.
        # self.create_subscription(LaserScan, '/mir/f_scan', self.on_scan, qos_profile_sensor_data)
        # self.min_range = math.inf
        
        self.T = 30
        self.b = 10
        self.zeta = 1
        
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        
        self.X = np.zeros([0, 0, 0])
        self.create_subscription(Odometry, "/mir/odom", self.on_odom, qos_profile_sensor_data)
        msg = wait_for_message(self, Odometry, "/mir/odom", timeout_sec=5.0)
        self.on_odom(msg)
        
        print(self.X)
        
        self.traj = generate_trajectory(self.X, self.T)
        self.t_init = self.get_clock().now()
        self.create_timer(0.05, self.on_timer) # 20 Hz
    
    # def on_scan(self, msg):
    #     valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
    #     self.min_range = min(valid, default=math.inf)
    
    def on_odom(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        th = 2*np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        self.X = np.array([px, py, th])
    
    def on_timer(self):
        dt_msg = (self.get_clock().now() - self.t_init).to_msg()
        dt = dt_msg.sec + dt_msg.nanosec * 1e-9
        X_d, V_d = query(self.traj, dt)
        
        # controller gains
        a = np.sqrt(V_d[1]**2 + self.b*V_d[0]**2)
        self.k1 = 2 * self.zeta * a
        self.k3 = self.k1
        self.k2 = self.b*np.abs(V_d[0])
        
        # control law
        c, s = np.cos(self.X[2]), np.sin(self.X[2])
        Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        e = Rz @ (X_d - self.X)
        
        v1 = -self.k1 * e[0]
        v2 = -self.k2*np.sign(V_d[0])*e[1] - self.k3*e[2]
        
        u1 = V_d[0] * np.cos(e[2]) - v1
        u2 = V_d[1] - v2
        
        # print(dt, V_d)
        # print(dt, (u1, u2), self.X[2], X_d[2])
        
        cmd = Twist()
        cmd.linear.x = u1
        cmd.angular.z = u2
        self.cmd_pub.publish(cmd)
        
        if dt > self.T:
            print("Done")
            rclpy.shutdown()
        
        
def main():
    # Keep the context alive on Ctrl+C so the final stop command can be sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Navigate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist()) # stop
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
