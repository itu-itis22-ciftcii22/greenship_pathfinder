import math
import numpy as np
import matplotlib.pyplot as plt
import heapq

class Domain:
    class Cell:
        def __init__(self):
            self.contains = "empty"
            self.blocked = False
            self.attractval = 0
            self.repelval = 0
            # Parent cell's row index
            self.parent_i = 0
            # Parent cell's column index
            self.parent_j = 0

    def __init__(self, row, col):
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError("Row and column must be integers.")
        if row <= 0 or col <= 0:
            raise ValueError("Parameters 'row' and 'col' must be positive numbers.")
        self.map = [[self.Cell() for _ in range(col)] for _ in range(row)]
        self.nrow = row
        self.ncol = col
        self.containables = ("pon", "target", "waypoint")

    class Coordinate:
        def __init__(self, row, col):
            if not isinstance(row, int) or not isinstance(col, int):
                raise TypeError("Row and column must be integers.")
            self.row = row
            self.col = col

    class Cost:
        def __init__(self):
            # Total cost of the cell (g + h)
            self.f = float('inf')
            # Cost from start to this cell
            self.g = float('inf')
            # Heuristic cost from this cell to destination
            self.h = 0

    def isValid(self, coord: Coordinate):
        return (coord.row >= 0) and (coord.row < self.nrow) and (coord.col >= 0) and (coord.col < self.ncol)

    def updateCell(self, coord: Coordinate, contains=None, blocked=None, value=None, is_attractor=False, is_repellor=False):
        if self.isValid(coord):
            if value is not None:
                if is_attractor:
                    self.map[coord.row][coord.col].attractval = value
                elif is_repellor:
                    self.map[coord.row][coord.col].repelval = value
            if contains is not None:
                self.map[coord.row][coord.col].contains = contains
            if blocked is not None:
                self.map[coord.row][coord.col].blocked = blocked

    def updateMap(self, coord: Coordinate, contains=None, blocked=None, value=None, radius=None, is_attractor=False, is_repellor=False):
        self.updateCell(coord, contains, blocked, value, is_attractor, is_repellor)

        if (value is not None) and (is_attractor or is_repellor):
            if radius is None:
                radius = 1
            i = coord.row
            j = coord.col
            # Loop through the weight distance, using a circular pattern around the obstacle
            # Iterate over neighbors within the radius
            for x in range(max(0, i - radius), min(self.nrow, i + radius + 1)):
                for y in range(max(0, j - radius), min(self.ncol, j + radius + 1)):
                    distance = ((i - x) ** 2 + (j - y) ** 2) ** 0.5  # Euclidean distance
                    if distance <= radius:
                        # Apply a falloff formula to calculate the value for the neighbor
                        new_value = value * (1 - distance / radius)
                        # Add the value to the map (or modify existing value)
                        if (new_value > self.map[x][y].attractval and is_attractor) or (new_value > self.map[x][y].repelval and is_repellor):
                            self.updateCell(self.Coordinate(x, y), value=new_value, is_attractor=is_attractor, is_repellor=is_repellor)
        """def findRelativePoint(distance, angle, heading, x_base=0, y_base=0):
        # Calculate the total angle in degrees (account for clockwise direction)
        total_angle = -(angle + heading)  # Negative to convert clockwise to counterclockwise
        
        # Convert angle to radians
        total_angle_rad = math.radians(total_angle)
        
        # Calculate displacements
        x_displacement = distance * math.cos(total_angle_rad)
        y_displacement = distance * math.sin(total_angle_rad)
        
        # Add displacements to the base-point coordinates
        x = x_base + x_displacement
        y = y_base + y_displacement
        
        return int(x), int(y)"""

    # Trace the path from source to destination
    def trace_path(self, dest: Coordinate):
        path = []
        row = dest.row
        col = dest.col

        # Trace the path from destination to source using parent cells
        while not (self.map[row][col].parent_i == row and self.map[row][col].parent_j == col):
            path.append(self.Coordinate(row, col))
            temp_row = self.map[row][col].parent_i
            temp_col = self.map[row][col].parent_j
            row = temp_row
            col = temp_col

        # Add the source cell to the path
        path.append(self.Coordinate(row, col))
        # Reverse the path to get the path from source to destination
        path.reverse()

        return path

    # Implement the A* search algorithm
    def a_star_search(self, src: Coordinate, dest: Coordinate):
        # Check if the source and destination are valid
        if not self.isValid(src) or not self.isValid(dest):
            print("Source or destination is invalid")
            return

        # Check if we are already at the destination
        if src.row == dest.row and src.col == dest.col:
            print("We are already at the destination")
            return

        # Initialize the closed list (visited cells)
        closed_list = [[False for _ in range(self.ncol)] for _ in range(self.nrow)]
        # Initialize the details of each cell
        cell_costs = [[self.Cost() for _ in range(self.ncol)] for _ in range(self.nrow)]

        # Initialize the start cell details
        i = src.row
        j = src.col
        cell_costs[i][j].f = 0
        cell_costs[i][j].g = 0
        cell_costs[i][j].h = 0
        self.map[i][j].parent_i = i
        self.map[i][j].parent_j = j

        # Initialize the open list (cells to be visited) with the start cell
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))

        # Initialize the flag for whether destination is found
        found_dest = False

        # Main loop of A* search algorithm
        while len(open_list) > 0:
            # Pop the cell with the smallest f value from the open list
            p = heapq.heappop(open_list)

            # Mark the cell as visited
            i = p[1]
            j = p[2]
            closed_list[i][j] = True

            # For each direction, check the successors
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for direction in directions:
                new_i = i + direction[0]
                new_j = j + direction[1]

                # If the successor is valid, unblocked, and not visited
                if self.isValid(self.Coordinate(new_i, new_j)) and not closed_list[new_i][new_j]:
                    # If the successor is the destination
                    if new_i == dest.row and new_j == dest.col:
                        # Set the parent of the destination cell
                        self.map[new_i][new_j].parent_i = i
                        self.map[new_i][new_j].parent_j = j
                        print("The destination cell is found")
                        # Trace and print the path from source to destination
                        path = self.trace_path(dest)
                        found_dest = True
                        return path
                    else:
                        distance = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
                        # Calculate the new f, g, and h values
                        g_new = cell_costs[i][j].g + distance
                        h_new = ((new_i - dest.row) ** 2 + (new_j - dest.col) ** 2) ** 0.5
                        f_new = g_new + h_new

                        #FARK BURADA, haritadaki ağırlığı eklememin sayesinde engellere yapışmadan gidebiliyor
                        f_new += - self.map[new_i][new_j].attractval + self.map[new_i][new_j].repelval

                        # If the cell is not in the open list or the new f value is smaller
                        if not self.map[new_i][new_j].blocked and (cell_costs[new_i][new_j].f == float('inf') or cell_costs[new_i][new_j].f > f_new):
                            # Add the cell to the open list
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            # Update the cell details
                            cell_costs[new_i][new_j].f = f_new
                            cell_costs[new_i][new_j].g = g_new
                            cell_costs[new_i][new_j].h = h_new
                            self.map[new_i][new_j].parent_i = i
                            self.map[new_i][new_j].parent_j = j

        # If the destination is not found after visiting all cells
        if not found_dest:
            print("Failed to find the destination cell")

    def writeMapToFile(self, filename):
        with open(filename, "w") as file:
            for x in range(self.nrow):
                for y in range(self.ncol):
                    file.write(str(int(self.map[x][y].attractval)))
                    file.write("\t")
                file.write("\n")

    def plotMap(self):
        # Initialize the grid for the entire domain based on the number of rows and columns
        grid_color = np.zeros((self.nrow, self.ncol, 3))  # RGB color space

        # Iterate over all cells in the domain
        for i in range(self.nrow):
            for j in range(self.ncol):
                cell = self.map[i][j]  # Access the Cell object
                normalized_attractval = (cell.attractval) / 50
                normalized_attractval = 1 - max(0, min(normalized_attractval, 1))  # Clamp values to [0, 1]
                normalized_repelval = (cell.repelval) / 50
                normalized_repelval = 1- max(0, min(normalized_repelval, 1))
                # Assign color based on `contains`
                if cell.contains == self.containables[0]:
                    grid_color[j, i] = [0, 0, 0]  # Black for pon
                elif cell.contains == self.containables[1]:
                    grid_color[j, i] = [1, 0, 0]  # Red for target
                elif cell.contains == self.containables[2]:
                    grid_color[j, i] = [0, 1, 0]  # Green for waypoint
                else:
                    """if normalized_weight != 1:
                        print(f"Warning: Cell at ({i}, {j}) has invalid weight: {cell.weight}")"""
                    # Handle cells that are neither targets nor blockages:
                    # Normalize weight to a range of 0 to 1 for grayscale representation
                    grid_color[j, i] = [1, normalized_repelval, normalized_attractval]  # Grayscale

        # Create the plot
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_color, interpolation="nearest")
        # Add labels to the axes
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Invert the Y-axis to place origin (0, 0) at the bottom-left
        plt.gca().invert_yaxis()

        # Add title and axis ticks
        plt.title("Domain Visualization")

        # Show plot
        plt.show()
