public class Cell {
    private CellType type;

    public Cell(char c) {
        type = CellType.getCellType(c);
    }

    public String toString() {
        return type.toString();
    }
}
