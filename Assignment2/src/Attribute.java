import java.util.*;

/**
 * Tanner Turba
 * October 30, 2024
 * CS 557 - Machine Learning
 * 
 * This class represents an Attribute that is included in the input data
 * from the file. 
 */
public class Attribute {
    private String name = null;
    private Map<Character, String> valMap = new HashMap<>();
    
    /**
     * Creates an attribute from a line in the input file.
     * @param line the file line
     */
    public Attribute(String line) {
        Scanner scanner = new Scanner(line);

        // Read the name
        scanner.useDelimiter(":");
        name = scanner.next();
        scanner.useDelimiter(" ");

        // Consume ':' and continue
        scanner.next();
        while (scanner.hasNext()) {
            String val = scanner.next();
            String[] valPair = val.split("=");

            // Populate the value map
            if (valPair.length == 1) {
                valMap.put(valPair[0].trim().charAt(0), null);
            }
            else if (valPair.length > 1) {
                valMap.put(valPair[0].trim().charAt(0), valPair[1]);
            }
        }
        scanner.close();
    }

    /**
     * Gets the value map 
     * @return
     */
    public Map<Character, String> getValMap() {
        return valMap;
    }

    /**
     * Gets the name of the attribute
     * @return
     */
    public String getName() {
        return name;
    }

    /**
     * Returns a string representation of the attribute
     */
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(String.format("\n%s\n", name));
        for (Map.Entry<Character, String> entry : valMap.entrySet()) {
            sb.append(String.format("%s=%s\n", entry.getKey(), entry.getValue()));
        }

        return sb.toString();
    }
}
