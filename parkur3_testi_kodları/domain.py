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

    def __init__(self, nrow, ncol, containables=None):
        if not isinstance(nrow, int) or not isinstance(ncol, int):
            raise TypeError("Row and column must be integers.")
        if nrow <= 0 or ncol <= 0:
            raise ValueError("Row and column  must be positive numbers.")
        self.map = [[self.Cell() for _ in range(ncol)] for _ in range(nrow)]
        self.nrow = nrow
        self.ncol = ncol
        if containables is None:
            self.containables = ("pon", "target", "waypoint")
        else:
            self.containables = containables

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
                        new_value = np.exp(np.log(value)*(radius-distance)/(radius-1))
                        # Add the value to the map (or modify existing value)
                        if (new_value > self.map[x][y].attractval and is_attractor) or (new_value > self.map[x][y].repelval and is_repellor):
                            self.updateCell(self.Coordinate(x, y), value=new_value, is_attractor=is_attractor, is_repellor=is_repellor)

    # Trace the path from source to destination
    def trace_path(self, dest: Coordinate):
        path = []
        row = self.map[dest.row][dest.col].parent_i
        col = self.map[dest.row][dest.col].parent_j

        # Trace the path from destination to source using parent cells
        while not (self.map[row][col].parent_i == row and self.map[row][col].parent_j == col):
            path.append(self.Coordinate(row, col))
            temp_row = self.map[row][col].parent_i
            temp_col = self.map[row][col].parent_j
            row = temp_row
            col = temp_col

        # Do not add the source cell to the path
        # path.append(self.Coordinate(row, col))
        # Reverse the path to get the path from source to destination
        path.reverse()

        return path

    # Implement the A* search algorithm
    def a_star_search(self, src: Coordinate, dest: Coordinate, corridor=None, moving_obstacles=None):
        if moving_obstacles is None:
            moving_obstacles = []
        # Check if the source and destination are valid
        if not self.isValid(src) or not self.isValid(dest):
            print("Source or destination is invalid")
            print("Source:")
            print(src.row, src.col)
            print("Destination:")
            print(dest.row, dest.col)
            return None

        # Check if we are already at the destination
        if src.row == dest.row and src.col == dest.col:
            print("We are already at the destination")
            return None

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

        # Main loop of A* search algorithm
        while len(open_list) > 0:
            # Pop the cell with the smallest f value from the open list
            p = heapq.heappop(open_list)

            # Mark the cell as visited
            i = p[1]
            j = p[2]
            closed_list[i][j] = True

            # For each direction, check the successors
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]#, (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for direction in directions:
                new_i = i + direction[0]
                new_j = j + direction[1]

                # If the successor is valid, unblocked, and not visited
                if self.isValid(self.Coordinate(new_i, new_j)) and not self.map[new_i][new_j].blocked and not closed_list[new_i][new_j]:
                    # If the successor is the destination or out of bounds
                    if new_i == dest.row and new_j == dest.col:
                        # Set the parent of the destination cell
                        self.map[new_i][new_j].parent_i = i
                        self.map[new_i][new_j].parent_j = j
                        path = self.trace_path(self.Coordinate(new_i, new_j))
                        return path
                    else:
                        distance = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
                        # Calculate the new f, g, and h values
                        g_new = cell_costs[i][j].g + distance
                        h_new = ((new_i - dest.row) ** 2 + (new_j - dest.col) ** 2) ** 0.5
                        f_new = g_new + h_new

                        f_new += -self.map[new_i][new_j].attractval + self.map[new_i][new_j].repelval

                        if corridor is not None and len(corridor) > 0:
                            query_point = np.array([new_i, new_j])
                            lcf_penalty = float('inf')

                            for idx in range(len(corridor)):
                                d = np.linalg.norm(query_point - np.array([corridor[idx][0], corridor[idx][1]]))
                                if d < lcf_penalty:
                                    lcf_penalty = d

                            f_new += lcf_penalty ** 2

                        for moving_obstacle in moving_obstacles:
                            distance = ((new_i - moving_obstacle.row) ** 2 + (new_j - moving_obstacle.col) ** 2) ** 0.5
                            if distance <= 4:
                                penalty = np.exp(np.log(50) * (4 - distance) / (4 - 1))
                                f_new += penalty



                        # If the cell is not in the open list or the new f value is smaller
                        if (cell_costs[new_i][new_j].f > f_new):
                            # Add the cell to the open list
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            # Update the cell details
                            cell_costs[new_i][new_j].f = f_new
                            cell_costs[new_i][new_j].g = g_new
                            cell_costs[new_i][new_j].h = h_new
                            self.map[new_i][new_j].parent_i = i
                            self.map[new_i][new_j].parent_j = j

        path = self.trace_path(self.Coordinate(i, j))
        return path

    def writeMapToFile(self, filename):
        with open(filename, "w") as file:
            for x in range(self.nrow):
                for y in range(self.ncol):
                    file.write(str(int(self.map[x][y].attractval)))
                    file.write("\t")
                file.write("\n")

    def plotMap(self, maxvalue, colors=None ):
        # Use default values if parameters are not provided
        if colors is None:
            colors = {
                "pon": [0, 0, 0],  # Black
                "target": [0, 0, 1],  # Blue
                "waypoint": [0, 1, 0]  # Green
            }

        # Initialize the grid for the entire domain based on the number of rows and columns
        grid_color = np.zeros((self.ncol, self.nrow, 3))  # RGB color space

        # Iterate over all cells in the domain
        for i in range(self.nrow):
            for j in range(self.ncol):
                cell = self.map[i][j]  # Access the Cell object
                # Assign color based on `contains`
                if cell.contains in colors:
                    grid_color[j, i] = colors[cell.contains]
                else:
                    normalized_attractval = (cell.attractval) / maxvalue
                    normalized_attractval = 1 - max(0, min(normalized_attractval, 1))  # Clamp values to [0, 1]
                    normalized_repelval = (cell.repelval) / maxvalue
                    normalized_repelval = 1 - max(0, min(normalized_repelval, 1))
                    # Handle cells that are neither targets nor blockages
                    grid_color[j, i] = [1, normalized_repelval, normalized_repelval]  # Grayscale

        # Create the plot
        plt.figure(figsize = (10, 15 / (self.nrow / self.ncol)))
        plt.imshow(grid_color)#, interpolation="nearest")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.gca().invert_yaxis()
        plt.title("Domain Visualization")
        plt.show()