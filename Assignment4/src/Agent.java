public class Agent {
    private Cell currentCell;
    private int score = 0;
    private int totalActions = 0;
    private boolean isTerminated = false;
    private double actionSuccessProbability;
    private Grid grid;

    public Agent(double actionSuccessProbability, Grid grid) {
        this.actionSuccessProbability = actionSuccessProbability;
        this.grid = grid;
        this.currentCell = grid.getStartingCell();
    }

    public void reset() {
        currentCell = grid.getStartingCell();
        score = 0;
        isTerminated = false;
    }

    public double move(Action a) {
        if (currentCell.getType() == CellType.CLIFF) {
            // Acting in a cliff -> return to start
            currentCell = grid.getStartingCell();
            score -= 10;
            return -10;
        }

        int drift = calcDrift();
        Cell potentialLocation = null;
        switch (a) {
            case UP:
                // System.out.println("moving up");
                potentialLocation = moveUp(grid, currentCell, drift);
                break;

            case DOWN:
                // System.out.println("moving down");
                potentialLocation = moveDown(grid, currentCell, drift);
                break;

            case LEFT:
                // System.out.println("moving left");
                potentialLocation = moveLeft(grid, currentCell, drift);
                break;

            case RIGHT:
                // System.out.println("moving right");
                potentialLocation = moveRight(grid, currentCell, drift);
                break;
        
            default:
                break;
        }

        if (potentialLocation != null && !potentialLocation.getType().isBlocking()) {
            currentCell = potentialLocation;

            if (potentialLocation.getType() == CellType.CLIFF) {
                // Entering a cliff
                score -= 20;
                return -20;
            }
            else if (potentialLocation.getType().isTerminal()) {
                // no futher action is possible
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

    private Cell moveUp(Grid grid, Cell current, int drift) {
        int x = current.getXCoordinate();
        int y = current.getYCoordinate();

        if (y > 0) {
            Cell result = grid.get(y - 1)[x];
            if (drift < 0) {
                // Drift left
                // System.out.println("drifting left");
                return moveLeft(grid, result, 0);
            }
            else if (drift > 0) {
                // Drift right
                // System.out.println("drifting right");
                return moveRight(grid, result, 0);
            }
            // System.out.println();
            return result;
        }
        return null;
    }

    private Cell moveDown(Grid grid, Cell current, int drift) {
        int x = current.getXCoordinate();
        int y = current.getYCoordinate();

        if (y < grid.size() - 1) {
            Cell result = grid.get(y + 1)[x];
            if (drift < 0) {
                // Drift left
                // System.out.println("drifting left");
                return moveLeft(grid, result, 0);
            }
            else if (drift > 0) {
                // Drive right
                // System.out.println("drifting right");
                return moveRight(grid, result, 0);
            }
            // System.out.println();
            return result;
        }
        return null;
    }

    private Cell moveLeft(Grid grid, Cell current, int drift) {
        int x = current.getXCoordinate();
        int y = current.getYCoordinate();

        if (x > 0) {
            Cell result = grid.get(y)[x - 1];
            if (drift < 0) {
                // Drift down
                // System.out.println("drifting down");
                return moveDown(grid, result, 0);    
            }
            else if (drift > 0) {
                // Drift up
                // System.out.println("drifting up");
                return moveUp(grid, result, 0);
            }
            // System.out.println();
            return result;
        }
        return null;
    }
    
    private Cell moveRight(Grid grid, Cell current, int drift) {
        int x = current.getXCoordinate();
        int y = current.getYCoordinate();

        Cell[] row = grid.get(y);
        if (x < row.length - 1) {
            Cell result = grid.get(y)[x + 1];
            if (drift < 0) {
                // Drift down
                // System.out.println("drifting down");
                return moveDown(grid, result, 0);
            }
            else if (drift > 0) {
                // Drift up
                // System.out.println("drifting up");
                return moveUp(grid, result, 0);
            }
            // System.out.println();
            return result;
        }
        return null;
    }

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
