public enum Action {
    UP, DOWN, LEFT, RIGHT;

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
}
