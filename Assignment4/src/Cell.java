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
}
