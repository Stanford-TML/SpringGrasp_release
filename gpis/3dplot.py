import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data", type=str, default="gpis")
parser.add_argument("--stride", type=int, default=10)
parser.add_argument("--axis", type=str, default="z")
parser.add_argument("--isf_limit", type=float, default=0.2)
parser.add_argument("--quiver_spacing", type=int, default=5)
parser.add_argument("--query_point", type=float, nargs=3, default=None)

args = parser.parse_args()

data = np.load(f"../gpis_states/{args.data}.npz")
test_mean = data["mean"]
test_var = data["var"]
test_normal = data["normal"]
num_steps = test_mean.shape[0]


if args.axis == "x": 
    X, Y = np.meshgrid(np.linspace(data["lb"][1],data["ub"][1],num_steps),
                    np.linspace(data["lb"][2],data["ub"][2],num_steps), indexing="xy")
elif args.axis=="y":
    X, Y = np.meshgrid(np.linspace(data["lb"][0],data["ub"][0],num_steps),
                    np.linspace(data["lb"][2],data["ub"][2],num_steps), indexing="xy")
else:
    X, Y = np.meshgrid(np.linspace(data["lb"][0],data["ub"][0],num_steps),
                    np.linspace(data["lb"][1],data["ub"][1],num_steps), indexing="xy")

if args.query_point is not None:
    print(args.query_point)
    if args.axis == "x":
        q_point = np.array([args.query_point[1], args.query_point[2], 0.0])
        Z_index = np.argmin(np.abs(np.linspace(data["lb"][0],data["ub"][0],num_steps)-args.query_point[0]))
    elif args.axis == "y":
        q_point = np.array([args.query_point[0], args.query_point[2], 0.0])
        Z_index = np.argmin(np.abs(np.linspace(data["lb"][1],data["ub"][1],num_steps)-args.query_point[1]))
    else:
        q_point = np.array([args.query_point[0], args.query_point[1], 0.0])
        Z_index = np.argmin(np.abs(np.linspace(data["lb"][2],data["ub"][2],num_steps)-args.query_point[2]))


for i in range(0, num_steps, args.stride):
    fig = plt.figure(figsize=(9, 3))
    
    ax = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ax2 = fig.add_subplot(122, projection='3d')

    if args.query_point is not None:
        i = Z_index
    if args.axis == "x":
        Z = test_mean[i]
        color_dimension = np.log(test_var[i])
        normal_vec = test_normal[i]
    elif args.axis == "y":
        Z = test_mean[:,i]
        color_dimension = np.log(test_var[:,i])
        normal_vec = test_normal[:,i]
    else:
        Z = test_mean[:,:,i]
        color_dimension = np.log(test_var[:,:,i])
        normal_vec = test_normal[:,:,i]
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, minn + (maxx-minn)/2)
    m = cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)
    # Plot the surface.
    surf2 = ax.plot_wireframe(X, Y, np.zeros_like(Z), zorder=1, alpha=1, color="grey")
    surf = ax.plot_surface(X, Y, Z,facecolors=fcolors,
                        linewidth=0, antialiased=True, alpha=1, zorder=2)
    if args.query_point is not None:
        ax.scatter([q_point[0]],[q_point[1]], [q_point[2]], color="orange", s=100, zorder=10)
    normal_field = ax2.quiver(X[::args.quiver_spacing,::args.quiver_spacing],
                              Y[::args.quiver_spacing,::args.quiver_spacing], 
                              np.zeros_like(Z)[::args.quiver_spacing,::args.quiver_spacing], 
                              normal_vec[::args.quiver_spacing,::args.quiver_spacing,0], 
                              normal_vec[::args.quiver_spacing,::args.quiver_spacing,1], 
                              normal_vec[::args.quiver_spacing,::args.quiver_spacing,2], length=0.005, normalize=True)
    ax2.contourf(X, Y, Z, levels = np.linspace(-0.001, 0.001, 2),cmap="jet", alpha=0.5)
    if args.query_point is not None:
        ax2.scatter([q_point[0]],[q_point[1]], [q_point[2]], color="orange", s=100, zorder=10)

    # Customize the z axis.
    ax.set_zlim(-args.isf_limit, args.isf_limit)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    if args.query_point is not None:
        break