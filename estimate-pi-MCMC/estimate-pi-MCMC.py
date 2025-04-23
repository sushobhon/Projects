

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    return mo, patches, plt, random


@app.cell
def _(random):
    # Defining a function to calculate probability
    # Defining a function to estimate the probability of a 2d point is in circle or not
    def calculate_prob_and_get_points(n: int):
        """
        This Function calculates the probability of a 2d point falling inside a circle
        and returns the points generated, categorized by whether they are inside or outside.

        param:
            n (int): number of points to consider

        returns:
            tuple: a tuple containing:
                - float: the estimated probability (count / n)
                - list: a list of points inside the circle ([x, y])
                - list: a list of points outside the circle ([x, y])
        """

        count = 0
        points_inside = []
        points_outside = []

        for i in range(n):
            # Generating 2 random number x and y from U(0,2)
            x = random.uniform(0, 2)
            y = random.uniform(0, 2)

            # Checking if the random point is within the circle
            if ((x - 1)**2 + (y - 1)**2) <= 1:
                count += 1
                points_inside.append([x, y])
            else:
                points_outside.append([x, y])

        return count / n, points_inside, points_outside
    return (calculate_prob_and_get_points,)


@app.cell
def _(mo):
    num_sim = mo.ui.slider(10, 10000, step= 10,label="**Select Simulation Number**", value=10000, full_width=True, show_value= True)
    num_sim
    return (num_sim,)


@app.cell
def _(mo, num_sim):
    mo.md(f"**Number of simulation:** {num_sim.value}")
    return


@app.cell
def _(calculate_prob_and_get_points, mo, num_sim):
    prob, points_inside, points_outside = calculate_prob_and_get_points(int(num_sim.value))
    estimated_pi = prob * 4
    mo.md(f"**Estimate of pi after {num_sim.value} iterations is:** {estimated_pi}")
    return estimated_pi, points_inside, points_outside


@app.cell
def _(estimated_pi, mo, num_sim, patches, plt, points_inside, points_outside):
    # Create Graphical Representation
    fig, ax = plt.subplots(1)

    # Draw the square (boundaries of the simulation area)
    square = patches.Rectangle((0, 0), 2, 2, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(square)

    # Draw the inscribed circle (centered at (1,1) with radius 1)
    circle = patches.Circle((1, 1), 1, linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(circle)

    # Plot the points inside the circle
    if points_inside:
        x_inside, y_inside = zip(*points_inside)
        ax.scatter(x_inside, y_inside, color='green', s=5, label='Inside Circle')

    # Plot the points outside the circle
    if points_outside:
        x_outside, y_outside = zip(*points_outside)
        ax.scatter(x_outside, y_outside, color='blue', s=5, label='Outside Circle')

    # Set plot limits and aspect ratio
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal', adjustable='box')

    # Set labels and title
    ax.set_xlabel(f'Pi := {estimated_pi}')
    ax.set_title(f'MC Estimation of Pi with {num_sim.value} points')
    # ax.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()
    mo.as_html(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
