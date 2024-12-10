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

    public String printGrid() {
        StringBuilder sb = new StringBuilder();

        for (Cell[] row : this) {
            for (int i = 0; i < row.length; i++) {
                sb.append("------------");
            }
            sb.append("\n");

            for (Action action : Action.values()) {
                for (Cell cell : row) {
                    if (action == Action.UP) {
                        sb.append(String.format("|  %6.1f  ", cell.getQ().get(action)));
                    }
                    else if (action == Action.LEFT) {
                        sb.append(String.format("|%-6.1f    ", cell.getQ().get(action)));
                    }
                    else if (action == Action.RIGHT) {
                        sb.append(String.format("|    %6.1f", cell.getQ().get(action)));
                    }
                    else {
                        sb.append(String.format("|  %6.1f  ", cell.getQ().get(action)));
                    }
                    
                }
                sb.append("|\n");
            }
        }

        for (int i = 0; i < this.get(0).length; i++) {
            sb.append("------------");
        }
        sb.append("\n");

        return sb.toString();
    }
}