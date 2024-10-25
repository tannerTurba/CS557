import java.util.*;

public class Attribute {
    private String name = null;
    private Map<Character, String> valMap = new HashMap<>();
    
    public Attribute(String line) {
        Scanner scanner = new Scanner(line);
        scanner.useDelimiter(":");
        name = scanner.next();
        
        scanner.useDelimiter(" ");
        // consume ':'
        scanner.next();
        while (scanner.hasNext()) {
            String val = scanner.next();
            String[] valPair = val.split("=");

            if (valPair.length == 1) {
                valMap.put(valPair[0].trim().charAt(0), null);
            }
            else if (valPair.length > 1) {
                valMap.put(valPair[0].trim().charAt(0), valPair[1]);
            }
        }

        scanner.close();
    }

    public Map<Character, String> getValMap() {
        return valMap;
    }

    public String getName() {
        return name;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(String.format("\n%s\n", name));
        for (Map.Entry<Character, String> entry : valMap.entrySet()) {
            sb.append(String.format("%s=%s\n", entry.getKey(), entry.getValue()));
        }

        return sb.toString();
    }
}
