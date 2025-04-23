import random

# Defining a function too estimate the probability of a 2d point is in circle or not
def calculate_prob(n:int):
    """
    This Function calculates the probability of a 2d point falling inside a circel

    param:
        n (int): number of points to conside
    """

    # Defining a Counter to keep track of number of points
    count = 0

    for i in range(n):
        # Generating 2 random number x and y from U(0,2)
        x = random.uniform(0,2)
        y = random.uniform(0,2)

        # Increasing the counter if the random numer is within the circle
        if (((x-1)**2) + ((y-1)**2))**0.5 <= 1:
            count += 1
        # print(f"Iteration {i+1}: {count/(i+1)}")

    return count/n

if __name__ == "__main__":
    n = int(input("Enter how many points you want to consider: "))
    prob = calculate_prob(n)
    print(f"Eastimate of pi after {n} iteration is: {prob*4}")  