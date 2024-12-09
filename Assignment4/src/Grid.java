import java.util.*;

public class Grid extends ArrayList<Cell[]> {
    private Cell startingCell;
    
    public Grid(Scanner scanner) {
        super();

        int y = 0;
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine().trim().toLowerCase();
            // Not a comment, so continue
            if (line.charAt(0) != '#') {
                Cell[] row = new Cell[line.length()];
                for (int x = 0; x < line.length(); x++) {
                    char c = line.charAt(x);
                    if (Character.isAlphabetic(c) || c == '_') {
                        Cell newCell = new Cell(c, x, y);
                        row[x] = newCell;

                        if (newCell.getType() == CellType.START) {
                            startingCell = newCell;
                        }
                    }
                }
                this.add(row);
                y++;
            }
        }
    }

    /**
     * @return Cell return the startingCell
     */
    public Cell getStartingCell() {
        return startingCell;
    }

    public String printPolicy() {
        StringBuilder sb = new StringBuilder();

        for (Cell[] row : this) {
            for (Cell cell : row) {
                CellType type = cell.getType();
                if (type.isTerminal() || type.isBlocking()) {
                    sb.append(type);
                }
                else {
                    sb.append(cell.getGreedyAction());
                }
            }
            sb.append("\n");
        }

        return sb.toString();
    }
}