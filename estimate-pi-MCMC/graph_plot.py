import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        if ((x - 1)**2 + (y - 1)**2)**0.5 <= 1:
            count += 1
            points_inside.append([x, y])
        else:
            points_outside.append([x, y])

    return count / n, points_inside, points_outside

if __name__ == "__main__":
    n = int(input("Enter how many points you want to consider: "))

    prob, points_inside, points_outside = calculate_prob_and_get_points(n)
    estimated_pi = prob * 4

    print(f"Estimate of pi after {n} iterations is: {estimated_pi}")

    # --- Create the graphical representation ---

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
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_title(f'Monte Carlo Estimation of Pi with {n} points')
    # ax.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()