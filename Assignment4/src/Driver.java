import java.io.*;
import java.util.*;;

public class Driver {
    private String fileName = null;
    private double alpha = 0.9;
    private double epsilon = 0.9;
    private double discountRate = 0.9;
    private int alphaDecay = 1000;
    private int epsilonDecay = 200;
    private double actionSuccessProbability = 0.8;
    private boolean isQLearning = false;
    private int trials = 10000;
    private boolean isUnicode = false;
    private int verbosity = 1;

    private Grid grid = null;
    private Agent agent = null;
    private int maxActions = 0;
    
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
                    alpha = Double.parseDouble(args[++i]);
                    break;

                case "-e":
                    epsilon = Integer.parseInt(args[++i]);
                    break;

                case "-g":
                    discountRate = Double.parseDouble(args[++i]);
                    break;

                case "-na":
                    alphaDecay = Integer.parseInt(args[++i]);
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
            grid = new Grid(scanner);
            scanner.close();
        } 
        catch (FileNotFoundException e) {
            System.err.println("No such file or directory: " + fileName);
        }
        maxActions = grid.size() * grid.get(0).length;
        agent = new Agent(actionSuccessProbability, grid);
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

    public void play() {
        agent.move(Action.DOWN);
        System.out.println(agent.getCurrentCell());
    }

    public void learn() {
        double alpha = this.alpha;
        double epsilon = this.epsilon;
        for (int t = 0; t < trials; t++) {
            if (t % alphaDecay == 0) {
                alpha = this.alpha / 1 + (t / alphaDecay);
            }
            if (t % epsilonDecay == 0) {
                epsilon = this.epsilon / 1 + (t / epsilonDecay);
            }
            
            agent.reset();
            for (int actionCount = 0; actionCount < maxActions && agent.isTerminated(); actionCount++) {
                // perform actions based on epsilon-greedy policy.
    
                if (isQLearning) {
                    // update Q values using off-policy/Q-learning updates.
                }
                else {
                    // update Q values using on-policy/SARSA updates.
                }
            }
        }
    }

    public static void main(String[] args) {
        Driver driver = new Driver(args);
        System.out.println(driver);

        driver.play();

    }
}
