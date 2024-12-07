/*
 * Tanner Turba
 * December 3, 2024
 * CS 557 - Machine Learning
 * 
 * This is the enumeration that supports the extra credit activation function options. 
 * For each available option, the activation function, g, can be calculated as well as its derivative, g'.
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

    public boolean isTerminal() {
        return this == MINE || this == GOAL;
    }

    public boolean isBlocking() {
        return this == BLOCK;
    }

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
