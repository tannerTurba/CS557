import java.util.*;

/*
 * Tanner Turba
 * December 10, 2024
 * CS 557 - Machine Learning
 * 
 * An enum to distinguish between the different possible actions
 * and agent can take.
 */
public enum Action {
    UP, LEFT, RIGHT, DOWN;

    /**
     * Converts an instance of Action to its string value.
     */
    public String toString() {
        switch (this) {
            case UP:
                return "^";

            case DOWN:
                return "v";

            case LEFT:
                return "<";

            case RIGHT:
                return ">";

            default:
                return "";
        }
    }

    /**
     * Converts a set of best-actions to it's string representation. 
     * @param actions the set of best-actions
     * @param isUnicode true if using unicode character
     * @return
     */
    public static String toString(ArrayList<Action> actions, boolean isUnicode) {
        if (actions.size() == 1) {
            if (isUnicode) {
                switch (actions.get(0)) {
                    case LEFT:
                        return "\u2190";

                    case UP:
                        return "\u2191";

                    case RIGHT:
                        return "\u2192";

                    case DOWN:
                        return "\u2193";
                
                    default:
                        return "";
                }
            }
            return actions.get(0).toString();
        }
        else if (actions.size() == 2) {
            if (actions.contains(Action.LEFT) && actions.contains(Action.RIGHT)) {
                if (isUnicode) {
                    return "\u2194";
                }
                return "-";
            }
            else if (actions.contains(Action.UP) && actions.contains(Action.DOWN)) {
                if (isUnicode) {
                    return "\u2195";
                }
                return "|";
            }
            else if (actions.contains(Action.UP) && actions.contains(Action.LEFT)) {
                if (isUnicode) {
                    return "\u2196";
                }
                return "\\";
            }
            else if (actions.contains(Action.UP) && actions.contains(Action.RIGHT)) {
                if (isUnicode) {
                    return "\u2197";
                }
                return "/";
            }
            else if (actions.contains(Action.DOWN) && actions.contains(Action.RIGHT)) {
                if (isUnicode) {
                    return "\u2198";
                }
                return "\\";
            }
            else if (actions.contains(Action.DOWN) && actions.contains(Action.LEFT)) {
                if (isUnicode) {
                    return "\u2199";
                }
                return "/";
            }
        }
        else if (actions.size() == 3) {
            if (actions.contains(Action.UP) && actions.contains(Action.DOWN) && actions.contains(Action.RIGHT)) {
                if (isUnicode) {
                    return "\u22a2";
                }
                return ">";
            }
            else if (actions.contains(Action.UP) && actions.contains(Action.DOWN) && actions.contains(Action.LEFT)) {
                if (isUnicode) {
                    return "\u22a3";
                }
                return "<";
            }
            else if (actions.contains(Action.DOWN) && actions.contains(Action.LEFT) && actions.contains(Action.RIGHT)) {
                if (isUnicode) {
                    return "\u22a4";
                }
                return "v";
            }
            else if (actions.contains(Action.UP) && actions.contains(Action.LEFT) && actions.contains(Action.RIGHT)) {
                if (isUnicode) {
                    return "\u22a5";
                }
                return "^";
            }
        }
        else if (actions.size() == 4) {
            return "+";
        }
        return "";
    }
}
