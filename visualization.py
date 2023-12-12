import open3d as o3d
import numpy as np
import time
import sys, os, argparse, glob
import multiprocessing as mp
from tqdm import tqdm

class SimpleVO:
    def __init__(self, args):
        pass

    def create_cylinders_from_line(self, points, line, radius=1, color=[0, 1, 0]):
        cylinders = []

        # Line start and end points
        start_point, end_point = points[line[0]], points[line[1]]

        # Compute the length of the line segment
        length = np.linalg.norm(np.array(end_point) - np.array(start_point))

        # Create a cylinder between the points
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)

        # Color the cylinder
        cylinder.paint_uniform_color(color)

        # Move the cylinder to the midpoint
        mid_point = (np.array(start_point) + np.array(end_point)) / 2
        cylinder.translate(mid_point - np.array(cylinder.get_center()))

        # Orient the cylinder to align with the line segment
        # direction = np.array(end_point) - np.array(start_point)
        # direction = direction / np.linalg.norm(direction)
        # cylinder.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.arccos(direction[2]),
        #                                                         np.arctan2(direction[1], direction[0]),
        #                                                         0]), 
        #                 cylinder.get_center())
        direction = np.array(end_point) - np.array(start_point)
        direction = direction / np.linalg.norm(direction)
        up = [0, 0, 1]  # Assuming the cylinder is initially aligned along the Z-axis
        rotation_axis = np.cross(up, direction)
        rotation_angle = np.arccos(np.dot(up, direction))
        axis_angle = rotation_axis/np.linalg.norm(rotation_axis) * rotation_angle
        cylinder.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle),
                        center=np.array(cylinder.get_center()))


        cylinders.append(cylinder)

        return cylinders
    
   # def transform_cylinder_to_line(self, l):

    
    def create_plane(self, size):
      

        # Create lines to represent the pyramid edges
       
       
        # Create a triangle mesh for the blue bottom
        # Split the quadrilateral into two triangles
        triangles = [
            [1, 2, 3],
            [1, 3, 0],
            [3, 2, 1],
            [0, 3, 1]
        ]

      
        base = [[size, 0, size],
                [-size, 0, size],
                [-size, 0, -size],
                [size, 0, -size]]

        triangle_mesh = o3d.geometry.TriangleMesh()
        triangle_mesh.vertices = o3d.utility.Vector3dVector(base)
        triangle_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        triangle_mesh.vertex_colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in base])  # Blue color

        return triangle_mesh
    
    def create_ball_and_players(self, player_num, ball_radius):

        center = np.array([0, 0, 0])

        # Create the original sphere mesh
        ball_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=ball_radius)

        return ball_mesh

        # for i in range(player_num):
        #     player_position =  np.array([[0.0, 0.0, 0.0]])  # Replace with the coordinates of your point
        #     point_cloud = o3d.geometry.PointCloud()
        #     point_cloud.points = o3d.utility.Vector3dVector(player_position)
        #     point_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # Set point color (in this case, red)
        #     point_cloud = point_cloud.voxel_down_sample(voxel_size=0.1)

    def visualize_ball(self, ball_mesh, vis):

        T = np.identity(4)
        ball_mesh.transform(T)
        #frustum_base.transform(T)
        print("transform")

        # for cy in cylinders:
        #     vis.add_geometry(cy)
        vis.add_geometry(ball_mesh)

    def create_court(self, vis):
         vis.add_geometry(self.create_plane(100))

    def create_points_around_court(self, vis):

        points = [
            [100, 100, 100], [100, -100, 100], [-100, 100, 100], [100, 100, -100],
            [100, -100, -100], [-100, -100, 100], [-100, 100, -100], [-100, -100, -100]
        ]
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Save to a PLY file
        ply_file = 'points.ply'
        o3d.io.write_point_cloud(ply_file, pcd)

        # Reading and visualizing the point cloud from the PLY file
        pcd_read = o3d.io.read_point_cloud(ply_file)
       
        vis.add_geometry(pcd_read)

    

    def run(self):
        keep_running = True
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()


        self.create_points_around_court(vis)
        self.create_court(vis)
        ball_mesh= self.create_ball_and_players(1, 10)

        self.visualize_ball(ball_mesh, vis)
       

        keep_running = keep_running and vis.poll_events()
        vis.update_renderer()
        
        T = np.identity(4)
        last_T_inv = np.linalg.inv(T)
        while keep_running:
            try:
                players, ball = queue.get(block=False)
                if players is not None:
                    
                    # insert new camera pose here using vis.add_geometry()
                    
                    # Apply transformation
                    # print("create pose")
                    # print("R:", R)
                    # print("t:", t)
                    T = np.hstack([ball[0], ball[1]])
                    T = np.vstack([T, [0, 0, 0, 1]])

                    ball_mesh.transform(last_T_inv)
                    ball_mesh.transform(T)
                    vis.update_geometry(ball_mesh)
                    
                    # frustum_lines_temp.transform(T)
                    # frustum_base_temp.transform(T)
                    # print("transform")
                    # vis.add_geometry(frustum_lines_temp)
                    # vis.add_geometry(frustum_base_temp)


                    # frustum_base.transform(last_T_inv)
                    # frustum_base.transform(T)

                    # for cy in cylinders:
                    #     cy.transform(last_T_inv)
                    #     cy.transform(T)
                    #     vis.update_geometry(cy)

                    # frustum_lines.transform(last_T_inv)
                    # frustum_lines.transform(T)
                    # vis.update_geometry(frustum_base)
                    # vis.update_geometry(frustum_lines)

                    
                    last_T_inv = np.linalg.inv(T)
                   
                    # vis.update_renderer()
                    print("finish")
                    
                  
            except: pass
            
            keep_running = keep_running and vis.poll_events()
            vis.update_renderer()
        vis.destroy_window()
        p.join()

    def process_frames(self, queue):
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        for i in tqdm(range(100)):
            time.sleep(0.7)
            
            t = t + [[10],[10],[10]]
            queue.put(([(R,t)], (R, t)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('input', help='directory of sequential frames')
    # parser.add_argument('--distort', default='True')
    # parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
  
