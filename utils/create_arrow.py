import open3d as o3d
import numpy as np

def draw_geometries(pcds):
    """
    Draw Geometries
    Args:
        - pcds (): [pcd1,pcd2,...]
    """
    o3d.visualization.draw_geometries(pcds)

def get_o3d_FOR(origin=[0, 0, 0],size=10):
    """ 
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=size)
    mesh_frame.translate(origin)
    return(mesh_frame)

def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return magnitude


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return mesh_frame

def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return mesh

def create_arrow(point_a, point_b):
    """
    Creates an arrow from point A to point B using Open3D.

    :param point_a: The starting point of the arrow (x, y, z).
    :param point_b: The ending point of the arrow (x, y, z).
    :param cylinder_radius: The radius of the cylinder part of the arrow.
    :param cone_radius: The radius of the cone part of the arrow.
    :param cone_height: The height of the cone part of the arrow.
    :return: An Open3D arrow object.
    """
    # Create a vector from A to B
    vector = np.array(point_b) - np.array(point_a)

    # Compute the length of the vector
    length = np.linalg.norm(vector)
    cone_height = length*0.2
    cone_radius = length/10
    cylinder_radius = length/20
    # Create a cylinder that will act as the shaft of the arrow
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=length - cone_height)
    cylinder.translate(np.array(point_a))

    # Align the cylinder with the vector
    cylinder.rotate(cylinder.get_rotation_matrix_from_xyz((0, 0, np.arccos(vector[2] / length))), center=point_a)
    cylinder.rotate(cylinder.get_rotation_matrix_from_xyz((0, -np.arcsin(vector[1] / length), 0)), center=point_a)
    cylinder.rotate(cylinder.get_rotation_matrix_from_xyz((-np.arcsin(vector[0] / length), 0, 0)), center=point_a)

    # Create a cone that will act as the head of the arrow
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    cone.translate(np.array(point_b) - np.array([0, 0, cone_height]))

    # Align the cone with the vector
    cone.rotate(cone.get_rotation_matrix_from_xyz((0, 0, np.arccos(vector[2] / length))), center=point_b)
    cone.rotate(cone.get_rotation_matrix_from_xyz((0, -np.arcsin(vector[1] / length), 0)), center=point_b)
    cone.rotate(cone.get_rotation_matrix_from_xyz((-np.arcsin(vector[0] / length), 0, 0)), center=point_b)

    # Combine the cylinder and cone to make a complete arrow
    arrow = o3d.geometry.TriangleMesh()
    arrow += cylinder
    arrow += cone

    return arrow

def create_direct_arrow(point_a, point_b):
    """
    Creates a direct arrow from point A to point B using Open3D.

    :param point_a: The starting point of the arrow (x, y, z).
    :param point_b: The ending point of the arrow (x, y, z).
    :param cylinder_radius: The radius of the cylinder part of the arrow.
    :param cone_radius: The radius of the cone part of the arrow.
    :param cone_height: The height of the cone part of the arrow.
    :return: An Open3D arrow object.
    """
    # Calculate the transformation needed to align the arrow
    direction = np.array(point_b) - np.array(point_a)
    midpoint = (np.array(point_a) + np.array(point_b)) / 2
    length = np.linalg.norm(direction)

    # Create an arrow
    cone_height = length*0.2
    cone_radius = length/10
    cylinder_radius = length/20
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=cylinder_radius, 
                                                   cone_radius=cone_radius, 
                                                   cylinder_height=length - cone_height, 
                                                   cone_height=cone_height)

    # Normalize the direction
    if length > 0:
        direction /= length

    # Align the arrow with the direction
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_angle = np.arccos(np.dot(z_axis, direction))

    # Apply rotation if needed
    if np.linalg.norm(rotation_axis) > 0:
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        arrow.rotate(rotation_matrix, center=(0, 0, 0))

    # Translate the arrow to the midpoint
    arrow.translate(point_a)

    return arrow