import java.util.*;

public class Cell {
    private CellType type;
    private int xCoordinate;
    private int yCoordinate;
    private Map<Action, Double> q = new HashMap<>();

    public Cell(char c, int xCoordinate, int yCoordinate) {
        type = CellType.getCellType(c);
        this.xCoordinate = xCoordinate;
        this.yCoordinate = yCoordinate;

        for (Action action : Action.values()) {
            q.put(action, 0.0);
        }
    }

    public Action getGreedyAction() {
        Action bestAction = null;
        double bestReward = -Double.MAX_VALUE;
        
        for (Map.Entry<Action, Double> pair : q.entrySet()) {
            if (bestAction == null || bestReward < pair.getValue()) {
                bestAction = pair.getKey();
                bestReward = pair.getValue();
            }
        }
        return bestAction;
    }

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
