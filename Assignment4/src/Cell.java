import java.util.*;

/*
 * Tanner Turba
 * December 10, 2024
 * CS 557 - Machine Learning
 * 
 * This class defines a singular cell in a grid, which is used to keep track of Q values
 * and other operations.
 */
public class Cell {
    private CellType type;
    private int xCoordinate;
    private int yCoordinate;
    private Map<Action, Double> q = new HashMap<>();

    /**
     * Instantiates a new instance of a cell.
     * @param c The celltype
     * @param xCoordinate the x-coordinate
     * @param yCoordinate the y-coordinate
     */
    public Cell(char c, int xCoordinate, int yCoordinate) {
        type = CellType.getCellType(c);
        this.xCoordinate = xCoordinate;
        this.yCoordinate = yCoordinate;

        // Initialize all actions to zero.
        for (Action action : Action.values()) {
            q.put(action, 0.0);
        }
    }

    /**
     * Gets the best action based on highest reward.
     * @return
     */
    public ArrayList<Action> getGreedyAction() {
        ArrayList<Action> bestActions = new ArrayList<>();
        double bestReward = -Double.MAX_VALUE;
        
        for (Map.Entry<Action, Double> pair : q.entrySet()) {
            if (bestActions.size() == 0 || bestReward < pair.getValue()) {
                bestActions.clear();
                bestActions.add(pair.getKey());
                bestReward = pair.getValue();
            }
            else if (bestReward == pair.getValue()) {
                bestActions.add(pair.getKey());
            }
        }
        return bestActions;
    }

    /**
     * Returns the string representation of the type. 
     */
    public String toString() {
        return type.toString();
    }

    /**
     * @return CellType return the type
     */
    public CellType getType() {
        return type;
    }

    /**
     * @return int return the xCoordinate
     */
    public int getXCoordinate() {
        return xCoordinate;
    }

    /**
     * @return int return the yCoordinate
     */
    public int getYCoordinate() {
        return yCoordinate;
    }

    /**
     * @return Map<Action, Double> return the q
     */
    public Map<Action, Double> getQ() {
        return q;
    }
}
