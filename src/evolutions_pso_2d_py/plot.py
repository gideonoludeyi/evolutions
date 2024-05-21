import io
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

from . import pso


def plot_pso(fitfn, c1, c2, maxiter, w=None, popsize=10, random_state=None, out=None):
    """
    plot the positions of the particles in the search space over iterations
    returns the generated plots in `.gif` format as a `io.BytesIO`.
    """
    generator = pso.pso(
        fitfn=lambda s: fitfn(s, min_=0, max_=5),
        c1=c1,
        c2=c2,
        w=w,
        popsize=popsize,
        maxiter=maxiter,
        random_state=random_state,
    )
    imgs = []
    x, y = np.array(np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100)))
    z = fitfn([x, y], min_=0, max_=5)
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]
    for i, positions in enumerate(generator):
        # plot only particles that within boundary
        positions = positions[np.all((positions >= 0.0) & (positions <= 5.0), axis=1)]
        # plot the particles' positions for the current iteration
        fig, ax = plt.subplots()
        img = ax.imshow(
            z, extent=[0, 5, 0, 5], origin="lower", cmap="viridis", alpha=0.5
        )
        fig.colorbar(img)
        ax.plot([x_min], [y_min], marker="x", markersize=4, color="white")
        for s in positions:
            ax.plot(s[0], s[1], marker=".", markersize=4, color="red")
        contours = ax.contour(x, y, z, 10, colors="black", alpha=0.4)
        ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        ax.set_title(f"Iteration {i}")
        with io.BytesIO() as buf:
            fig.savefig(buf, format="png")
            buf.seek(0)
            image = iio.imread(buf, extension=".png", index=None)
            imgs.append(image)
        plt.close(fig)
    # compile the plots together into a GIF format
    out = out or io.BytesIO()
    iio.imwrite(out, imgs, extension=".gif")
    return out
