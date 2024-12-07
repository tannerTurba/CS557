import java.io.*;
import java.util.*;;

public class Driver {
    private String fileName = null;
    private double learningRate = 0.9;
    private double epsilon = 0.9;
    private double discountRate = 0.9;
    private int learningRateDecay = 1000;
    private int epsilonDecay = 200;
    private double actionSuccessProbability = 0.8;
    private boolean isQLearning = false;
    private int trials = 10000;
    private boolean isUnicode = false;
    private int verbosity = 1;

    private ArrayList<Cell[]> grid = new ArrayList<>();
    
    /**
     * Creates a driver instance from the user's input parameters.
     * @param args
     */
    public Driver(String[] args) {
        // Reads command line args
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-f":
                    fileName = args[++i];
                    break;
                
                case "-a":
                    learningRate = Double.parseDouble(args[++i]);
                    break;

                case "-e":
                    epsilon = Integer.parseInt(args[++i]);
                    break;

                case "-g":
                    discountRate = Double.parseDouble(args[++i]);
                    break;

                case "-na":
                    learningRateDecay = Integer.parseInt(args[++i]);
                    break;

                case "-ne":
                    epsilonDecay = Integer.parseInt(args[++i]);
                    break;

                case "-p":
                    actionSuccessProbability = Double.parseDouble(args[++i]);
                    break;

                case "-q":
                    isQLearning = true;
                    break;

                case "-T":
                    trials = Integer.parseInt(args[++i]);
                    break;

                case "-u":
                    isUnicode = false;
                    break;
                    
                case "-v":
                    verbosity = Integer.parseInt(args[++i]);
                    break;

                default:
                    break;
            }
        }
        readFile();
    }

    private void readFile() {
        // Load grid from file.
        try {
            Scanner scanner = new Scanner(new File(fileName));
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim().toLowerCase();
                // Not a comment, so continue
                if (line.charAt(0) != '#') {
                    Cell[] row = new Cell[line.length()];
                    for (int i = 0; i < line.length(); i++) {
                        char c = line.charAt(i);
                        if (Character.isAlphabetic(c) || c == '_') {
                            row[i] = new Cell(c);
                        }
                    }
                    grid.add(row);
                }
            }
            scanner.close();
        } 
        catch (FileNotFoundException e) {
            System.err.println("No such file or directory: " + fileName);
        }
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Cell[] cells : grid) {
            for (Cell cell : cells) {
                sb.append(cell);
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        Driver driver = new Driver(args);
        System.out.println(driver);
    }
}
