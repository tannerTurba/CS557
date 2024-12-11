/*
 * Tanner Turba
 * December 10, 2024
 * CS 557 - Machine Learning
 * 
 * This is the enumeration that defines a cell's type and aids in 
 * managing a cell's type-based attributes and printing.
 */
public enum CellType {
    START,
    GOAL,
    EMPTY,
    BLOCK,
    MINE,
    CLIFF;

    /**
     * Converts a string to its activation function.
     * @param func
     * @return
     */
    public static CellType getCellType(char c) {
        switch (c) {
            case 's':
                return START;

            case 'g':
                return GOAL;

            case 'b':
                return BLOCK;

            case 'm':
                return MINE;

            case 'c':
                return CLIFF;
        
            default:
                return EMPTY;
        }
    }

    /**
     * Determines if an instance of celltype is terminal.
     * @return
     */
    public boolean isTerminal() {
        return this == MINE || this == GOAL;
    }

    /**
     * Determines if an instance of celltype is blocking.
     * @return
     */
    public boolean isBlocking() {
        return this == BLOCK;
    }

    /**
     * Converts a celltype to its string representation.
     */
    public String toString() {
        switch (this) {
            case START:
                return "S";

            case GOAL:
                return "G";

            case BLOCK:
                return "B";

            case MINE:
                return "M";

            case CLIFF:
                return "C";

            default:
                return "_";
        }
    }
}
