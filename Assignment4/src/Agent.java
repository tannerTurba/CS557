/*
 * Tanner Turba
 * December 10, 2024
 * CS 557 - Machine Learning
 * 
 * This class defines an agent, which moves through the grid environment.
 */
public class Agent {
    private Cell currentCell;
    private int score = 0;
    private int totalActions = 0;
    private boolean isTerminated = false;
    private double actionSuccessProbability;
    private Grid grid;

    /**
     * Instantiates a new agent.
     * @param actionSuccessProbability
     * @param grid
     */
    public Agent(double actionSuccessProbability, Grid grid) {
        this.actionSuccessProbability = actionSuccessProbability;
        this.grid = grid;
        this.currentCell = grid.getStartingCell();
    }

    /**
     * Sets the agent back at start and resets the score. 
     */
    public void reset() {
        currentCell = grid.getStartingCell();
        score = 0;
        isTerminated = false;
    }

    /**
     * Moves the agent based on the specified. Also accounts for drift.
     * @param a the action to take.
     * @return
     */
    public double move(Action a) {
        if (currentCell.getType() == CellType.CLIFF) {
            // Acting in a cliff -> return to start
            currentCell = grid.getStartingCell();
            score -= 10;
            return -10;
        }

        // Calculate drift and find potential new location.
        int drift = calcDrift();
        Cell potentialLocation = null;
        switch (a) {
            case UP:
                potentialLocation = moveUp(grid, currentCell, drift);
                break;

            case DOWN:
                potentialLocation = moveDown(grid, currentCell, drift);
                break;

            case LEFT:
                potentialLocation = moveLeft(grid, currentCell, drift);
                break;

            case RIGHT:
                potentialLocation = moveRight(grid, currentCell, drift);
                break;
        
            default:
                break;
        }

        if (potentialLocation != null && !potentialLocation.getType().isBlocking()) {
            // If the agent can move to the new location, set currentCell
            currentCell = potentialLocation;

            if (potentialLocation.getType() == CellType.CLIFF) {
                // Entering a cliff
                score -= 20;
                return -20;
            }
            else if (potentialLocation.getType().isTerminal()) {
                // No futher action is possible
                isTerminated = true;

                if (potentialLocation.getType() == CellType.GOAL) {
                    // Entering a goal cell
                    score += 10;
                    return 10;
                }
                else if (potentialLocation.getType() == CellType.MINE) {
                    // Entering a mine cell
                    score -= 100;
                    return -100;
                }
            }
            else {
                // Any other cell
                score--;
                return -1;
            }
        }
        return 0;
    }

    /**
     * Moves the agent up, with possible drift.
     * @param grid the grid the agent is operating in.
     * @param current the current cell that the agent is in.
     * @param drift the drift integer, which determines which direction the agent will drift, if any.
     * @return
     */
    private Cell moveUp(Grid grid, Cell current, int drift) {
        int x = current.getXCoordinate();
        int y = current.getYCoordinate();

        if (y > 0) {
            Cell result = grid.get(y - 1)[x];
            if (drift < 0) {
                // Drift left
                return moveLeft(grid, result, 0);
            }
            else if (drift > 0) {
                // Drift right
                return moveRight(grid, result, 0);
            }
            return result;
        }
        return null;
    }

    /**
     * Moves the agent down, with possible drift.
     * @param grid the grid the agent is operating in.
     * @param current the current cell that the agent is in.
     * @param drift the drift integer, which determines which direction the agent will drift, if any.
     * @return
     */
    private Cell moveDown(Grid grid, Cell current, int drift) {
        int x = current.getXCoordinate();
        int y = current.getYCoordinate();

        if (y < grid.size() - 1) {
            Cell result = grid.get(y + 1)[x];
            if (drift < 0) {
                // Drift left
                return moveLeft(grid, result, 0);
            }
            else if (drift > 0) {
                // Drive right
                return moveRight(grid, result, 0);
            }
            return result;
        }
        return null;
    }

    /**
     * Moves the agent left, with possible drift.
     * @param grid the grid the agent is operating in.
     * @param current the current cell that the agent is in.
     * @param drift the drift integer, which determines which direction the agent will drift, if any.
     * @return
     */
    private Cell moveLeft(Grid grid, Cell current, int drift) {
        int x = current.getXCoordinate();
        int y = current.getYCoordinate();

        if (x > 0) {
            Cell result = grid.get(y)[x - 1];
            if (drift < 0) {
                // Drift down
                return moveDown(grid, result, 0);    
            }
            else if (drift > 0) {
                // Drift up
                return moveUp(grid, result, 0);
            }
            return result;
        }
        return null;
    }
    
    /**
     * Moves the agent right, with possible drift.
     * @param grid the grid the agent is operating in.
     * @param current the current cell that the agent is in.
     * @param drift the drift integer, which determines which direction the agent will drift, if any.
     * @return
     */
    private Cell moveRight(Grid grid, Cell current, int drift) {
        int x = current.getXCoordinate();
        int y = current.getYCoordinate();

        Cell[] row = grid.get(y);
        if (x < row.length - 1) {
            Cell result = grid.get(y)[x + 1];
            if (drift < 0) {
                // Drift down
                return moveDown(grid, result, 0);
            }
            else if (drift > 0) {
                // Drift up
                return moveUp(grid, result, 0);
            }
            return result;
        }
        return null;
    }

    /**
     * Calculates an integer that determines if the agent will drift and 
     * in which direction.
     * @return the drift integer.
     */
    private int calcDrift() {
        double drift = (1 - actionSuccessProbability) / 2.0;
        double x = Math.random();

        if (0 <= x && x <= drift) {
            // [0, (1 - p)/2] -> drift left/down
            return -1;
        }
        else if (drift + actionSuccessProbability <= x && x <= 1) {
            // [p + (1 - p)/2, 1] -> drift right/up
            return 1;
        }
        else {
            // ((1 - p)/2, p + (1 - p)/2) -> intended action
            return 0;
        }
    }

    /**
     * @return Cell return the currentCell
     */
    public Cell getCurrentCell() {
        return currentCell;
    }

    /**
     * @return int return the score
     */
    public int getScore() {
        return score;
    }

    /**
     * @return int return the totalActions
     */
    public int getTotalActions() {
        return totalActions;
    }

    /**
     * @return boolean return the isTerminated
     */
    public boolean isTerminated() {
        return isTerminated;
    }

    /**
     * @return double return the actionSuccessProbability
     */
    public double getActionSuccessProbability() {
        return actionSuccessProbability;
    }
}
