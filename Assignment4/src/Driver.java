import java.io.*;
import java.util.*;;

/*
 * Tanner Turba
 * December 10, 2024 
 * CS 557 - Machine Learning
 * 
 * The main class in charge of program flow. This program implements the 
 * SARSA and Q-learning algorithms for a navigation task in a simple grid world environment.
 */
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
    private StringBuilder sb = new StringBuilder();
    
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

    /**
     * Reads data from a file
     */
    private void readFile() {
        sb.append(String.format("* Reading %s...\n", fileName));

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

    /**
     * Returns the string from the StringBuilder.
     */
    public String toString() {
        return sb.toString();
    }

    /**
     * Performs training episodes or evaluation episodes, based on the set flags.
     * @param isTraining true if training.
     * @param midTraining true if printing during midTraining.
     * @return
     */
    public double play(boolean isTraining, boolean midTraining) {
        double alpha = this.alpha;
        double epsilon = this.epsilon;
        int reward = 0;
        int trials = this.trials;

        // Printing commands
        if (!isTraining && !midTraining) {
            trials = 50;
            sb.append("* Beginning 50 evaluation episodes...\n");
        }
        else if (isTraining) {
            String learnType = "SARSA";
            if (isQLearning) {
                learnType = "Q-Learning";
            }
            sb.append(String.format("* Beginning %d learning episodes with %s...\n", trials, learnType));
            if (verbosity >= 3) {
                sb.append("  * After     Avg. Total Reward for\n");
                sb.append("  * Episode   Current Greedy Policy\n");
            } 
        }

        // For each episode.
        for (int t = 0; t < trials; t++) {
            // Calc alpha and epsilon decay.
            if ((t) % alphaDecay == 0) {
                alpha = this.alpha / (1 + (t / alphaDecay));
                if (verbosity >= 4 && !midTraining && t != 0) {
                    sb.append(String.format("    (after episode %d, alpha to %.5f)\n", t, alpha));
                }
            }
            if ((t) % epsilonDecay == 0) {
                epsilon = this.epsilon / (1 + (t / epsilonDecay));
                if (verbosity >= 4 && !midTraining && t != 0) {
                    sb.append(String.format("    (after episode %d, epsilon to %.5f)\n", t, epsilon));
                }
            }
            
            // Place agent at start.
            agent.reset();

            // Action to take.
            Cell s = agent.getCurrentCell();
            Action a = epsilonGreedyPolicy(epsilon, isTraining);
            for (int actionCount = 0; actionCount < maxActions && !agent.isTerminated(); actionCount++) {
                // Perform actions based on epsilon-greedy policy.
                double r = agent.move(a);
                Cell sPrime = agent.getCurrentCell();
                Action aPrime = epsilonGreedyPolicy(epsilon, isTraining);

                if (isTraining) {
                    Map<Action, Double> qS = s.getQ();
                    Map<Action, Double> qSPrime = sPrime.getQ();
                    double policyUpdate = 0.0;
                    if (isQLearning) {
                        // update Q values using off-policy/Q-learning updates.
                        policyUpdate = 0;
                        double bestReward = -Double.MAX_VALUE;
                        for (Map.Entry<Action, Double> pair : qSPrime.entrySet()) {
                            if (bestReward < pair.getValue()) {
                                bestReward = pair.getValue();
                            }
                        }
    
                        policyUpdate = bestReward;
                    }
                    else {
                        // update Q values using on-policy/SARSA updates.
                        policyUpdate = qSPrime.get(aPrime);
                    }
    
                    double newTempDiff = qS.get(a) + alpha * (r + (discountRate * policyUpdate) - qS.get(a));
                    qS.put(a, newTempDiff);
                }
                s = sPrime;
                a = aPrime;
            }
            reward += agent.getScore();

            // Printing commands.
            if (verbosity >= 3 && isTraining && (t + 1) % (trials / 10) == 0) {
                double rwrd = play(false, true);
                sb.append(String.format("    %7d     %6.3f\n", t+1, rwrd));
            }
        }

        double avgReward = reward / (double)trials;
        if (!midTraining) {
            if (isTraining) {
                sb.append("  Done with learning!\n");
            }
            else {
                sb.append(String.format("  Avg. Total Reward of Learned Policy: %.3f\n", avgReward));
                sb.append(String.format("* Learned greedy policy:\n%s", grid.printPolicy(isUnicode)));
                if (verbosity >= 2) {
                    sb.append(String.format("* Learned Q values:\n%s", grid.printGrid()));
                }
            }
        }
        return avgReward;
    }
    
    /**
     * Performs the epsilon-greedy policy to determine which action to take.
     * @param epsilon the threshold for using random or greedy action.
     * @param isTraining if false, deactivates the random action.
     * @return
     */
    private Action epsilonGreedyPolicy(double epsilon, boolean isTraining) {
        Cell currentState = agent.getCurrentCell();

        if (isTraining && Math.random() <= epsilon) {
            // Random action
            switch ((int)(Math.random() * 4)) {
                case 0:
                    return Action.UP;

                case 1:
                    return Action.DOWN;

                case 2:
                    return Action.LEFT;
            
                default:
                    return Action.RIGHT;
            }
        }
        else {
            // Best action
            return currentState.getGreedyAction().get(0);
        }
    }

    public static void main(String[] args) {
        Driver driver = new Driver(args);
        
        // Train
        driver.play(true, false);
        
        // Evaluate
        driver.play(false, false);
        
        // Print
        System.out.println(driver);
    }
}
